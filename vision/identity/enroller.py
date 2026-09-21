"""
自動登錄狀態機：按一次 R 鍵之後，自動把目標的特徵持續蒐集進特徵庫。

    IDLE ──(按 R)──> ARMING(建立種子) ──(種子自洽)──> ACTIVE(持續自動蒐集)
      ^                                                    │
      └──────────────────(再按一次 R)──────────────────────┘

純 numpy，不碰 torch 與 cv2：候選人的特徵向量與品質分數由 tracker 事先算好再傳進來，
因此整套准入規則可以用合成向量離線測試。

設計上最需要謹慎的是「防污染」。自動登錄最怕把別人收進特徵庫，而且一旦收進去會自我強化：
壞樣本讓錯的人拿高分 -> 收進更多壞樣本。所以准入條件遠比追蹤條件嚴格，
任何一條不滿足就不收樣本 —— 但追蹤照常繼續，收樣本失敗絕不影響飛行。
"""

import math
import time

import numpy as np

from vision.identity.gallery import FACE, BODY, Sample

IDLE, ARMING, ACTIVE = "IDLE", "ARMING", "ACTIVE"


class AutoEnroller:
    # ---------- 種子階段 ----------
    SEED_WINDOW = 3.0            # 種子蒐集時間 (秒)
    SEED_FRAMES = 40             # 種子蒐集的影格上限
    SEED_MIN_FACE = 3            # 種子最少要有幾張一致的臉
    SEED_MIN_BODY = 6
    SEED_KEEP_FACE = 4           # 種子最多保留幾筆 (種子是永久樣本，不能無限制累積)
    SEED_KEEP_BODY = 6
    SEED_FACE_AGREE = 0.70       # 種子人臉彼此的 cosine 至少要這麼像
    SEED_BODY_AGREE = 0.80
    EXCLUSIVE_RATIO = 1.6        # 按 R 當下，目標必須比第二大的人大這麼多倍
    EXCLUSIVE_IOU = 0.05
    ANCHOR_GATE = 0.22           # 種子期間追住 anchor 的位置門檻 (畫面對角線比例)
    ANCHOR_SCALE_RATIO = 1.5

    # ---------- 准入規則 ----------
    ENROLL_SCORE_MIN = 0.62      # 絕對信心 (比對只要 0.42/0.55)
    ENROLL_MARGIN = 0.15         # 領先亞軍的幅度 (比對只要 0.08)
    FACE_ENROLL_CENTROID = 0.62  # 對質心的信心
    FACE_ENROLL_TOPK = 0.58
    FACE_ANCHOR_MIN = 0.65       # 人臉要多確定才夠格替體態樣本背書
    CHAIN_FRAMES = 12            # 鏈式例外：連續匹配幾格才算連續性未中斷
    CHAIN_SECONDS = 1.5          # 鏈式例外：距離上一次人臉背書的時限
    ENROLL_STABLE_FRAMES = 5     # 時間穩定性
    OCCL_IOU = 0.15              # 與其他人重疊超過此值就不收
    OCCL_CENTER_RATIO = 0.5      # 其他人的中心距離小於此倍率的體型時不收
    ENROLL_MIN_INTERVAL = 0.6    # 每種模態最短收樣間隔 (秒)

    # ---------- 隔離期 (後悔藥) ----------
    REGRET_SCORE = 0.35          # 目標分數崩塌的門檻
    REGRET_SECONDS = 2.0         # 崩塌持續多久才判定剛剛收錯人
    REGRET_WINDOW = 3.0          # 撤銷崩塌前多久之內收的樣本

    def __init__(self, gallery):
        self.gallery = gallery
        self.state = IDLE
        self.reset_runtime()

    # ==========================================
    # 狀態
    # ==========================================

    def reset_runtime(self):
        self.arm_t0 = 0.0
        self.arm_frames = 0
        self.anchor_state = None
        self.seed = {FACE: [], BODY: []}
        self.last_admit = {FACE: 0.0, BODY: 0.0}
        self.last_face_anchor_ts = -1e9
        self.low_score_since = None
        self.last_reason = ""

    def status(self):
        n_face, n_body = self.gallery.slot.counts()
        return {
            "enroll_state": self.state,
            "face_n": n_face,
            "body_n": n_body,
            "last_reason": self.last_reason,
        }

    # ==========================================
    # R 鍵
    # ==========================================

    def request(self, candidates, now=None):
        """R 鍵：IDLE 時開始建檔，其他狀態時清除身分回到自動選擇最大的人"""
        now = time.time() if now is None else now

        if self.state != IDLE:
            self.gallery.save_if_dirty(force=True, now=now)
            self.state = IDLE
            self.reset_runtime()
            return "[身分登錄] 已清除目標身分，恢復自動選擇畫面中最大的人"

        if not candidates:
            return "[身分登錄] 畫面中沒有偵測到人，無法建檔"

        anchor = max(candidates, key=lambda c: c['body_scale'])
        others = [c for c in candidates if c is not anchor]
        if others:
            second = max(others, key=lambda c: c['body_scale'])
            overlap = any(_iou(anchor.get('box'), o.get('box')) > self.EXCLUSIVE_IOU for o in others)
            # 獨佔性檢查：這是 bootstrap 階段最關鍵的防線。
            # 種子樣本決定整個特徵庫的地基，此時還沒有任何外觀模型可以分辨誰是誰，
            # 所以只在「畫面中有一個明顯主導的人」時才允許開始建檔。
            if overlap or anchor['body_scale'] < self.EXCLUSIVE_RATIO * second['body_scale']:
                return ("[身分登錄] 畫面中有多人且大小相近，"
                        "請靠近鏡頭或請其他人退出畫面後再按 R")

        self.gallery.reset()
        self.state = ARMING
        self.reset_runtime()
        self.arm_t0 = now
        self.anchor_state = _state_of(anchor)
        return "[身分登錄] 開始建檔，請留在鏡頭前約 3 秒（可慢慢轉身讓系統看到不同角度）"

    # ==========================================
    # 種子階段：用短期幾何連續性追住 anchor
    # 這裡用幾何是正當的 —— 前提就是「畫面中只有一個明顯主導的人」，
    # 而且此時特徵庫還是空的，根本沒有外觀模型可用。
    # ==========================================

    def arming_pick(self, candidates, diag):
        if not candidates or self.anchor_state is None:
            return None
        best, best_cost = None, None
        for c in candidates:
            dist = math.hypot(c['cx'] - self.anchor_state['cx'],
                              c['cy'] - self.anchor_state['cy']) / max(diag, 1.0)
            s_new, s_old = c['body_scale'], self.anchor_state['scale']
            ratio = max(s_new, s_old) / max(1.0, min(s_new, s_old))
            if dist > self.ANCHOR_GATE or ratio > self.ANCHOR_SCALE_RATIO:
                continue
            cost = dist + 0.5 * (ratio - 1.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = c, cost
        if best is not None:
            self.anchor_state = _state_of(best)
        return best

    # ==========================================
    # 每格更新
    # ==========================================

    def step(self, candidates, target, info, now=None):
        """回傳這一格要印出來的訊息 (通常是空的)"""
        now = time.time() if now is None else now
        if self.state == IDLE:
            return []
        if self.state == ARMING:
            return self._step_arming(target, now)
        return self._step_active(candidates, target, info, now)

    # ---------- ARMING ----------

    def _step_arming(self, target, now):
        self.arm_frames += 1
        if target is not None:
            if target.get('face_vec') is not None and target.get('face_ok_enroll'):
                self.seed[FACE].append(target)
            if target.get('body_vec') is not None and target.get('body_ok_enroll'):
                self.seed[BODY].append(target)

        if (now - self.arm_t0) < self.SEED_WINDOW and self.arm_frames < self.SEED_FRAMES:
            return []
        return self._finalize_seed(now)

    def _finalize_seed(self, now):
        face_ok = self._consistent(self.seed[FACE], 'face_vec', self.SEED_FACE_AGREE)
        body_ok = self._consistent(self.seed[BODY], 'body_vec', self.SEED_BODY_AGREE)

        # 種子自洽性：彼此不夠像代表這 3 秒內框到了不同的人，或畫面本身太差。
        # 寧可建檔失敗要求重來，也不要讓一個有問題的地基污染之後所有的比對。
        if len(face_ok) < self.SEED_MIN_FACE or len(body_ok) < self.SEED_MIN_BODY:
            self.state = IDLE
            self.gallery.reset()
            self.reset_runtime()
            return [f"[身分登錄] 種子樣本不足或彼此不一致"
                    f"（臉 {len(face_ok)}/{self.SEED_MIN_FACE}、"
                    f"體態 {len(body_ok)}/{self.SEED_MIN_BODY}），建檔取消。"
                    f"請站在畫面中央、確認只有你一個人入鏡後重新按 R"]

        # 種子是永久樣本 (不受配額與汰換限制)，所以必須自己限量：
        # 3 秒 x 20fps 可能累積 40 格，全部留下來會塞滿特徵庫而且全是近乎重複的樣本。
        for c in self._pick_seeds(face_ok, 'face_vec', self.SEED_KEEP_FACE,
                                  0.97, self.SEED_MIN_FACE):
            self.gallery.slot.admit(_make_sample(c, FACE, now, protected=True), FACE)
        for c in self._pick_seeds(body_ok, 'body_vec', self.SEED_KEEP_BODY,
                                  0.97, self.SEED_MIN_BODY):
            self.gallery.slot.admit(_make_sample(c, BODY, now, protected=True), BODY)

        self.gallery.mark_dirty()
        self.gallery.save_if_dirty(force=True, now=now)
        self.state = ACTIVE
        self.last_face_anchor_ts = now
        n_face, n_body = self.gallery.slot.counts()
        return [f"[身分登錄] 已建立目標身分（臉 {n_face} / 體態 {n_body}），"
                f"之後會自動持續蒐集樣本，不必再按 R"]

    @staticmethod
    def _pick_seeds(items, key, keep, diversity, min_keep=1):
        """
        從候選種子中挑出品質最好、又彼此不重複的幾筆。

        如果多樣性過濾後數量不足 (人站著不動時每一格幾乎一模一樣)，
        就用品質次佳的補齊 —— 種子的意義在於「多個樣本互相背書」，
        寧可留下幾筆相似的樣本，也不要讓整個身分只靠單一一筆撐著。
        """
        quality_key = 'face_quality' if key == 'face_vec' else 'body_quality'
        ranked = sorted(items, key=lambda c: -c.get(quality_key, 0.5))
        picked, rest = [], []
        for c in ranked:
            if len(picked) >= keep:
                break
            if picked:
                sims = np.asarray([p[key] for p in picked], dtype=np.float32) @ c[key]
                if float(sims.max()) >= diversity:
                    rest.append(c)
                    continue
            picked.append(c)
        while len(picked) < min_keep and rest:
            picked.append(rest.pop(0))
        return picked

    @staticmethod
    def _consistent(items, key, threshold):
        """取彼此最一致的最大子集：以與其他樣本平均相似度最高者為中心，收進所有夠像的"""
        if len(items) < 2:
            return list(items)
        vecs = np.asarray([c[key] for c in items], dtype=np.float32)
        sims = vecs @ vecs.T
        center = int(np.argmax(sims.mean(axis=1)))
        keep = [items[i] for i in range(len(items)) if sims[center, i] >= threshold]
        return keep

    # ---------- ACTIVE ----------

    def _step_active(self, candidates, target, info, now):
        msgs = []
        fused = info.get('fused')
        self._check_regret(fused, now, msgs)

        if target is None:
            self.last_reason = "no_target"
            return msgs

        face_cos = info.get('face_cos')
        if face_cos is not None and face_cos >= self.FACE_ANCHOR_MIN:
            self.last_face_anchor_ts = now

        if fused is None or fused < self.ENROLL_SCORE_MIN:
            self.last_reason = "score_low"
            return msgs
        if info.get('margin', 0.0) < self.ENROLL_MARGIN:
            # 兩人分數接近時照樣鎖定，但絕不收樣本。單這一條就擋掉大部分污染。
            self.last_reason = "rival_close"
            return msgs
        if info.get('stable_frames', 0) < self.ENROLL_STABLE_FRAMES:
            self.last_reason = "not_stable"
            return msgs
        if self._occluded(candidates, target):
            self.last_reason = "occluded"
            return msgs

        admitted = []
        slot = self.gallery.slot

        # --- 人臉樣本 ---
        if (target.get('face_vec') is not None and target.get('face_ok_enroll')
                and (now - self.last_admit[FACE]) >= self.ENROLL_MIN_INTERVAL):
            vec = target['face_vec']
            centroid = slot.centroid_score(vec, FACE)
            topk = slot.score(vec, FACE)
            if ((centroid is None or centroid >= self.FACE_ENROLL_CENTROID)
                    and (topk is None or topk >= self.FACE_ENROLL_TOPK)):
                ok, why = slot.admit(_make_sample(target, FACE, now), FACE)
                if ok:
                    self.last_admit[FACE] = now
                    admitted.append("face")
                else:
                    self.last_reason = f"face_{why}"
            else:
                self.last_reason = "face_vs_centroid_low"

        # --- 體態樣本：需要人臉背書 ---
        face_anchored = face_cos is not None and face_cos >= self.FACE_ANCHOR_MIN
        # 鏈式例外：背對鏡頭時永遠不會有臉，改用外觀連續性把「剛剛被臉驗證過」的信任傳遞過來。
        # 有嚴格的連續格數與時間上限，不是無條件放行。
        chain_ok = (info.get('stable_frames', 0) >= self.CHAIN_FRAMES
                    and (now - self.last_face_anchor_ts) <= self.CHAIN_SECONDS)
        if (target.get('body_vec') is not None and target.get('body_ok_enroll')
                and (now - self.last_admit[BODY]) >= self.ENROLL_MIN_INTERVAL):
            if face_anchored or chain_ok:
                ok, why = slot.admit(_make_sample(target, BODY, now), BODY)
                if ok:
                    self.last_admit[BODY] = now
                    admitted.append("body")
                else:
                    self.last_reason = f"body_{why}"
            else:
                self.last_reason = "no_face_anchor"

        if admitted:
            self.gallery.mark_dirty()
            n_face, n_body = slot.counts()
            self.last_reason = "admitted:" + "+".join(admitted)
            msgs.append(f"[身分登錄] 收錄新樣本 {'+'.join(admitted)}"
                        f"（臉 {n_face} / 體態 {n_body}）")
        return msgs

    def _occluded(self, candidates, target):
        for c in candidates:
            if c is target:
                continue
            if _iou(c.get('box'), target.get('box')) > self.OCCL_IOU:
                return True
            d = math.hypot(c['cx'] - target['cx'], c['cy'] - target['cy'])
            if d < self.OCCL_CENTER_RATIO * max(target['body_scale'], 1.0):
                return True
        return False

    def _check_regret(self, fused, now, msgs):
        """目標分數在收樣本後短時間內崩塌，代表剛剛很可能收錯人 -> 撤銷那段時間的樣本"""
        if fused is not None and fused >= self.REGRET_SCORE:
            self.low_score_since = None
            return
        if self.low_score_since is None:
            self.low_score_since = now
            return
        if (now - self.low_score_since) < self.REGRET_SECONDS:
            return
        t0 = self.low_score_since - self.REGRET_WINDOW
        removed = self.gallery.slot.revoke_since(t0)
        self.low_score_since = None
        if removed:
            self.gallery.mark_dirty()
            msgs.append(f"[身分登錄] 目標分數異常崩塌，已撤銷最近收錄的 {removed} 筆樣本")


# ==========================================
# 小工具
# ==========================================

def _state_of(c):
    return {"cx": c['cx'], "cy": c['cy'], "scale": c['body_scale']}


def _iou(a, b):
    if a is None or b is None:
        return 0.0
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


def _make_sample(c, modality, now, protected=False):
    bins = c.get(f'{modality}_bins', (0, 1, 1))
    return Sample(
        vec=c[f'{modality}_vec'],
        quality=float(c.get(f'{modality}_quality', 0.5)),
        ts=now,
        view_bin=int(bins[0]), scale_bin=int(bins[1]), bright_bin=int(bins[2]),
        protected=protected,
    )
