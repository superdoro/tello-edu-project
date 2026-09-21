"""
身分辨識門面：抽特徵 -> 比對決策 -> 自動登錄 -> 自動存檔。

流程上的關鍵約束：decide() 必須是「不碰影像」的 —— 它只吃已經帶著特徵向量的候選人 dict。
這讓整套比對與狀態機能用合成向量在沒有 GPU 的環境離線測試，
而那正是這個系統裡最容易寫錯、也最難在飛行中除錯的部分。

torch 與 cv2 都是延遲載入：不按 R 建檔就完全不會佔用 VRAM，也不會拖慢啟動。
"""

import atexit
import math
import time

import numpy as np

from vision.identity.gallery import IdentityGallery, FACE, BODY
from vision.identity.enroller import AutoEnroller, IDLE, ARMING, ACTIVE

# ==========================================
# 分數校準
# 人臉與體態的 cosine 尺度完全不同，不能直接加權相加：
#   facenet    同人 0.65~0.85 / 異人 0.20~0.45  (動態範圍大)
#   ImageNet   同人 0.85~0.97 / 異人 0.60~0.85  (全擠在高位)
# 直接 0.6*face + 0.4*body 會讓 body 永遠貢獻一個接近常數的值，形同無效。
# 所以先各自線性映射到共同的「證據」尺度 (0~1) 再融合。
# 下面四個數值是有依據的起始值，用實際影像實測後應該回來調整。
# ==========================================
FACE_LO, FACE_HI = 0.45, 0.80
BODY_LO, BODY_HI = 0.70, 0.95
W_FACE, W_BODY = 0.60, 0.40
FACE_ONLY_TRUST = 0.95       # 只有臉時的折扣
BODY_ONLY_TRUST = 0.85       # 只有體態時的折扣 (證據本質較弱，折扣較重)
FACE_OVERRIDE_HI = 0.80      # 臉很確定
BODY_OVERRIDE_LO = 0.25      # 但衣著對不上 (換外套、背光) -> 讓臉有權否決衣著

# ==========================================
# 鎖定狀態機
# ==========================================
UNLOCKED, LOCKED, LOST = "UNLOCKED", "LOCKED", "LOST"
LOCK_ON = 0.55               # 重新找回目標所需的分數
LOCK_KEEP = 0.42             # 維持鎖定所需的分數 (0.13 的遲滯帶防止抖動)
RIVAL_MARGIN = 0.08          # 領先亞軍的最小幅度
LOST_CONFIRM_FRAMES = 3      # 連續幾格達不到才宣告失聯 (容忍單格關鍵點掉幀)
REACQUIRE_FRAMES = 2         # 連續幾格達標才算重新找回

# ==========================================
# 連續性加分：小幅、有界、會隨時間衰減
# 上限 0.10 小於遲滯帶寬 0.13，所以連續性可以打破兩人分數接近時的平手，
# 但永遠無法推翻明顯的外觀差距 —— 這正是「外觀為主 + 短期連續性加分」的意思。
# ==========================================
CONT_MAX = 0.10
CONT_POS_SCALE = 0.12
CONT_SCALE_SCALE = 0.35
CONT_TIME_SCALE = 0.50
CONT_MAX_DT = 2.0

AUTOSAVE_INTERVAL = 10.0     # 特徵庫自動存檔的最短間隔 (秒)
BUDGET_STRIDE2_MS = 55.0     # 一格超過這個時間就隔格抽特徵
BUDGET_STRIDE3_MS = 90.0
FRAME_MS_ALPHA = 0.2


def calib(cos_val, lo, hi):
    """把某個模態的 cosine 映射到共同的 0~1 證據尺度"""
    if cos_val is None:
        return None
    return float(min(1.0, max(0.0, (cos_val - lo) / max(1e-6, hi - lo))))


class IdentityTracker:
    def __init__(self, gallery_dir="data/identity", name="target",
                 device=None, fp16=True, frame_is_rgb=True, auto_lock_on_load=True):
        self.gallery = IdentityGallery(root=gallery_dir, name=name)
        self.enroller = AutoEnroller(self.gallery)
        self.device = device
        self.fp16 = fp16
        self.frame_is_rgb = frame_is_rgb

        self.state = UNLOCKED
        self.hit_streak = 0
        self.miss_streak = 0
        self.prev = None
        self.last_info = {}

        self.raw_frame = None
        self.disabled = False
        self.face_emb = None
        self.body_emb = None
        self.require_models = True   # 離線測試時設為 False，可直接注入特徵向量

        self.frame_no = 0
        self.stride = 1
        self.frame_ms_ema = 0.0
        self.ms_reports = 0
        self.timing = {'pre_ms': 0.0, 'face_ms': 0.0, 'body_ms': 0.0, 'n_face': 0, 'n_body': 0}
        self.debug_dump_dir = None   # 設定後會把對齊完的人臉 chip 存出來 (只給離線除錯用)

        if self.gallery.load() and auto_lock_on_load:
            # 特徵庫裡已經有身分 -> 直接進入辨識狀態，不必重新按 R。
            # 起始狀態是 LOST 而不是 LOCKED，代表必須先以較高的門檻 (LOCK_ON) 認出人才會開始跟。
            self.enroller.state = ACTIVE
            self.state = LOST
            print("[身分辨識] 偵測到既有身分，看到本人就會自動鎖定（按 R 可清除重建）")

        atexit.register(self._atexit_save)

    # ==========================================
    # 對外
    # ==========================================

    def set_raw_frame(self, raw):
        """接收縮放前的原生畫面 (960x720)。人臉裁切用它能多拿到約 33% 的水平解析度。"""
        self.raw_frame = raw

    WARMUP_FRAMES = 5            # 前幾格包含模型暖機，不列入效能統計

    def note_frame_ms(self, ms):
        """由呼叫端回報整格耗時，用來決定要不要降頻抽特徵"""
        self.ms_reports += 1
        if self.ms_reports <= self.WARMUP_FRAMES:
            return          # 前幾格包含模型暖機，不代表穩態效能
        if self.frame_ms_ema <= 0:
            self.frame_ms_ema = ms
        else:
            self.frame_ms_ema = FRAME_MS_ALPHA * ms + (1 - FRAME_MS_ALPHA) * self.frame_ms_ema
        old = self.stride
        if self.frame_ms_ema > BUDGET_STRIDE3_MS:
            self.stride = 3
        elif self.frame_ms_ema > BUDGET_STRIDE2_MS:
            self.stride = 2
        else:
            self.stride = 1
        if self.stride != old:
            print(f"[身分辨識] 影格耗時 {self.frame_ms_ema:.0f}ms，特徵抽取降頻為每 {self.stride} 格一次")

    def request_enroll(self, candidates, now=None):
        now = time.time() if now is None else now
        if self.enroller.state == IDLE and not self._ensure_models():
            return
        msg = self.enroller.request(candidates, now)
        if self.enroller.state == IDLE:
            self.state = UNLOCKED
            self.prev = None
        print(msg)

    def status(self):
        st = self.enroller.status()
        st.update({
            "state": self.state,
            "score": self.last_info.get('fused'),
            "face_score": self.last_info.get('face_cos'),
            "body_score": self.last_info.get('body_cos'),
            "stride": self.stride,
            "disabled": self.disabled,
        })
        return st

    def update(self, frame, candidates, now=None, w_img=720, h_img=480):
        """每格呼叫一次，回傳 (目標候選人 或 None, 狀態 dict)"""
        now = time.time() if now is None else now
        self.frame_no += 1

        if self.disabled or self.enroller.state == IDLE:
            target, info = self.decide(candidates, now, w_img, h_img)
            self.last_info = info
            return target, info

        skip = (self.stride > 1) and (self.frame_no % self.stride != 0)
        if skip:
            # 降頻時不重新抽特徵，僅以連續性沿用上一格的指派，也不做任何登錄判斷
            target = self._sticky_pick(candidates, now, math.hypot(w_img, h_img))
            info = dict(self.last_info)
            info['skipped'] = True
            self._remember(target, now)
            return target, info

        self.extract(frame, candidates)
        target, info = self.decide(candidates, now, w_img, h_img)

        for msg in self.enroller.step(candidates, target, info, now):
            print(msg)

        if self.gallery.dirty and (now - self.gallery.last_save) > AUTOSAVE_INTERVAL:
            self.gallery.save_if_dirty(now=now)

        self.last_info = info
        return target, info

    def save_if_dirty(self, force=False):
        self.gallery.save_if_dirty(force=force)

    # ==========================================
    # 特徵抽取 (需要 cv2 與 torch，延遲載入)
    # ==========================================

    def _ensure_models(self):
        # 以物件本身判斷而不是額外的旗標，避免旗標與實際狀態不一致
        if not self.require_models:
            return True
        if self.face_emb is not None and self.body_emb is not None:
            return True
        if self.disabled:
            return False
        try:
            from vision.identity.embedders import FaceEmbedder, BodyEmbedder
            self.face_emb = FaceEmbedder(device=self.device, fp16=self.fp16)
            self.body_emb = BodyEmbedder(device=self.device, fp16=self.fp16)
            if not (self.face_emb.available() and self.body_emb.available()):
                self.disabled = True
                print("[身分辨識] 特徵模型無法載入，本次飛行停用身分辨識（仍可跟隨最靠近的人）")
                return False
        except Exception as e:
            self.disabled = True
            print(f"[身分辨識] 特徵模型初始化失敗，本次飛行停用身分辨識：{e}")
            return False
        return True

    def _source(self, frame):
        """挑選要用哪張影像裁切，並算出關鍵點座標需要的縮放比例"""
        h_img, w_img = frame.shape[:2]
        src = frame
        if self.raw_frame is not None and self.raw_frame.size > 0:
            src = self.raw_frame
            rh, rw = src.shape[:2]
            return src, (rw / float(w_img), rh / float(h_img))
        return src, (1.0, 1.0)

    def extract(self, frame, candidates):
        """把每個候選人的影像轉成特徵向量，結果直接寫回 candidate dict"""
        if self.disabled or not candidates or not self.require_models:
            return
        if not self._ensure_models():
            return
        try:
            from vision.identity import preprocess as pp
            src, scale_xy = self._source(frame)
            t0 = time.time()

            chips, chip_owner = [], []
            crops, crop_owner = [], []

            for c in candidates:
                kpts = c.get('keypoints') or []
                yaw_err = c.get('face_yaw_error', 0.0)

                # 人臉：align_face 內部會先做純算術的品質預篩，
                # 人背對或距離太遠時會在 warp 之前就回傳 None，成本趨近於零
                chip, q = pp.align_face(src, kpts, yaw_err, scale_xy)
                if chip is not None:
                    nose_x = q.get('nose', (0, 0))[0]
                    eye_mid_x = (q['eye_l'][0] + q['eye_r'][0]) / 2.0
                    c['face_ok_match'] = True
                    c['face_ok_enroll'] = bool(q.get('ok_enroll'))
                    c['face_quality'] = float(q.get('quality', 0.5))
                    c['face_bins'] = (pp.face_view_bin(q['asym'], nose_x, eye_mid_x),
                                      pp.scale_bin(c['body_scale']),
                                      pp.bright_bin(q.get('mean', 128.0)))
                    chips.append(chip)
                    chip_owner.append(c)
                else:
                    c['face_ok_match'] = False
                    c['face_ok_enroll'] = False
                    c['face_reason'] = q.get('reason', 'no_face')

                # 體態：便宜，每個候選人都算
                crop = pp.body_crop(src, c.get('box'), kpts, scale_xy)
                if crop is not None:
                    blur = pp.chip_blur(crop)
                    bright = pp.crop_brightness(crop)
                    c['body_quality'] = float(
                        0.5 * min(1.0, c['body_scale'] / 220.0) + 0.5 * min(1.0, blur / 150.0))
                    c['body_ok_enroll'] = bool(c['body_scale'] >= pp.BODY_SCALE_ENROLL
                                               and blur >= pp.BODY_BLUR_ENROLL)
                    c['body_bins'] = (pp.body_view_bin(kpts, yaw_err),
                                      pp.scale_bin(c['body_scale']),
                                      pp.bright_bin(bright))
                    crops.append(crop)
                    crop_owner.append(c)
                else:
                    c['body_ok_enroll'] = False

            t1 = time.time()
            if chips:
                vecs = self.face_emb.embed_batch(chips)
                if vecs is not None:
                    for c, v in zip(chip_owner, vecs):
                        c['face_vec'] = v
            t2 = time.time()
            if crops:
                vecs = self.body_emb.embed_batch(crops)
                if vecs is not None:
                    for c, v in zip(crop_owner, vecs):
                        c['body_vec'] = v
            t3 = time.time()

            self.timing = {'pre_ms': (t1 - t0) * 1000, 'face_ms': (t2 - t1) * 1000,
                           'body_ms': (t3 - t2) * 1000, 'n_face': len(chips), 'n_body': len(crops)}
            if self.debug_dump_dir and chips:
                self._dump_chips(chips, chip_owner)
        except Exception as e:
            self.disabled = True
            print(f"[身分辨識] 特徵抽取發生錯誤，本次飛行停用身分辨識：{e}")

    def _dump_chips(self, chips, owners):
        """把對齊後的人臉存成檔案：肉眼看 20 張就能確認對齊模板與通道順序是不是對的"""
        import os
        import cv2
        os.makedirs(self.debug_dump_dir, exist_ok=True)
        for chip, c in zip(chips, owners):
            name = (f"{self.frame_no:05d}_q{c.get('face_quality', 0):.2f}"
                    f"_{'E' if c.get('face_ok_enroll') else 'm'}.png")
            cv2.imwrite(os.path.join(self.debug_dump_dir, name),
                        cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))

    # ==========================================
    # 決策 (不碰影像，可離線測試)
    # ==========================================

    def decide(self, candidates, now=None, w_img=720, h_img=480):
        now = time.time() if now is None else now
        diag = max(1.0, math.hypot(w_img, h_img))
        info = {
            'state': self.state, 'enroll_state': self.enroller.state,
            'fused': None, 'margin': 1.0, 'face_cos': None, 'body_cos': None,
            'stable_frames': self.hit_streak, 'skipped': False,
        }

        # 建檔中：用短期幾何連續性追住 anchor (此時特徵庫還是空的)
        if self.enroller.state == ARMING:
            target = self.enroller.arming_pick(candidates, diag)
            info['state'] = ARMING
            self._remember(target, now)
            return target, info

        # 還沒有身分：維持原本「跟畫面中最大的人」的行為
        if self.enroller.state == IDLE or self.gallery.slot.empty():
            target = max(candidates, key=lambda c: c['body_scale']) if candidates else None
            self.state = UNLOCKED
            info['state'] = UNLOCKED
            self._remember(target, now)
            return target, info

        scored = []
        for c in candidates:
            fused, fcos, bcos = self._fuse(c)
            if fused is None:
                continue
            c['_fused'], c['_face_cos'], c['_body_cos'] = fused, fcos, bcos
            c['_final'] = fused + self._continuity(c, now, diag)
            scored.append(c)
        scored.sort(key=lambda c: -c['_final'])

        best = scored[0] if scored else None
        runner = scored[1] if len(scored) > 1 else None
        margin = (best['_final'] - runner['_final']) if (best and runner) else 1.0
        ambiguous = margin < RIVAL_MARGIN

        if best is not None:
            info.update({'fused': best['_fused'], 'margin': margin,
                         'face_cos': best['_face_cos'], 'body_cos': best['_body_cos']})

        if self.state == LOCKED:
            # 模稜兩可時沿用原目標 (分數仍要過 KEEP)，但 margin 會讓登錄那邊擋下收樣本
            ok = best is not None and best['_final'] >= LOCK_KEEP
        else:
            ok = best is not None and best['_final'] >= LOCK_ON and not ambiguous

        target = None
        if ok:
            self.hit_streak += 1
            self.miss_streak = 0
            if self.state == LOCKED:
                target = best
            elif self.hit_streak >= REACQUIRE_FRAMES:
                self.state = LOCKED
                target = best
                print(f"[身分辨識] 已鎖定目標（分數 {best['_fused']:.2f}）")
        else:
            self.miss_streak += 1
            self.hit_streak = 0
            if self.state == LOCKED and self.miss_streak >= LOST_CONFIRM_FRAMES:
                self.state = LOST
                print("[身分辨識] 目標失聯，原地懸停持續比對中")

        info['state'] = self.state
        info['stable_frames'] = self.hit_streak
        self._remember(target, now)
        return target, info

    def _fuse(self, c):
        slot = self.gallery.slot
        fcos = slot.score(c.get('face_vec'), FACE) if c.get('face_ok_match') else None
        bcos = slot.score(c.get('body_vec'), BODY)

        fe = calib(fcos, FACE_LO, FACE_HI)
        be = calib(bcos, BODY_LO, BODY_HI)

        if fe is not None and be is not None:
            fused = W_FACE * fe + W_BODY * be
            if fe > FACE_OVERRIDE_HI and be < BODY_OVERRIDE_LO:
                # 臉很確定但衣著對不上 (換外套、背光)：讓臉有權否決衣著
                fused = max(fused, fe * FACE_ONLY_TRUST)
        elif fe is not None:
            fused = fe * FACE_ONLY_TRUST
        elif be is not None:
            fused = be * BODY_ONLY_TRUST
        else:
            fused = None
        return fused, fcos, bcos

    def _continuity(self, c, now, diag):
        p = self.prev
        if p is None or c.get('box') is None or p.get('box') is None:
            return 0.0
        dt = now - p['t']
        if dt <= 0 or dt > CONT_MAX_DT:
            return 0.0

        vx, vy = p['vel']
        pb = p['box']
        pred = (pb[0] + vx * dt, pb[1] + vy * dt, pb[2] + vx * dt, pb[3] + vy * dt)
        iou = _iou(c['box'], pred)
        if iou <= 0:
            return 0.0

        dpos = math.hypot(c['cx'] - (p['cx'] + vx * dt), c['cy'] - (p['cy'] + vy * dt)) / diag
        dscl = abs(math.log(max(c['body_scale'], 1.0) / max(p['scale'], 1.0)))
        return (CONT_MAX * min(1.0, iou)
                * math.exp(-dpos / CONT_POS_SCALE)
                * math.exp(-dscl / CONT_SCALE_SCALE)
                * math.exp(-dt / CONT_TIME_SCALE))

    def _sticky_pick(self, candidates, now, diag):
        best, best_c = None, 0.0
        for c in candidates:
            score = self._continuity(c, now, diag)
            if score > best_c:
                best, best_c = c, score
        return best if best_c > 0.02 else None

    def _remember(self, target, now):
        if target is None:
            return
        vel = (0.0, 0.0)
        p = self.prev
        if p is not None:
            dt = now - p['t']
            if 0 < dt <= CONT_MAX_DT:
                vel = ((target['cx'] - p['cx']) / dt, (target['cy'] - p['cy']) / dt)
        self.prev = {'cx': target['cx'], 'cy': target['cy'],
                     'scale': target['body_scale'], 'box': target.get('box'),
                     't': now, 'vel': vel}

    # ==========================================
    # 畫面顯示
    # ==========================================

    def draw(self, annotated_frame, org=(10, 100)):
        import cv2
        st = self.status()
        x, y = org

        if st['disabled']:
            cv2.putText(annotated_frame, "ID: DISABLED", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        if self.enroller.state == IDLE:
            cv2.putText(annotated_frame, "ID: OFF (press R to enroll)", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            return

        if self.enroller.state == ARMING:
            cv2.putText(annotated_frame, "ID: ENROLLING... stay in view", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return

        score = st['score']
        if self.state == LOCKED:
            color, head = (0, 0, 255), f"ID: LOCKED {score:.2f}" if score is not None else "ID: LOCKED"
        else:
            color, head = (0, 140, 255), "ID: SEARCHING"

        detail = ""
        if st['face_score'] is not None:
            detail += f" F{st['face_score']:.2f}"
        if st['body_score'] is not None:
            detail += f" B{st['body_score']:.2f}"
        cv2.putText(annotated_frame, head + detail, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated_frame,
                    f"GALLERY: face {st['face_n']} / body {st['body_n']}  [{st['last_reason']}]",
                    (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 2)

    # ==========================================

    def _atexit_save(self):
        try:
            self.gallery.save_if_dirty()
        except Exception:
            pass


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
