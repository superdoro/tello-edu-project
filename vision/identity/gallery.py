"""
目標特徵庫 (gallery)：樣本、分箱配額、汰換與持久化。

純 numpy，不依賴 torch 與 cv2，因此可以在沒有 GPU 的環境用合成向量單獨測試 ——
這裡的邏輯是整套身分辨識中最容易寫錯、又最難在飛行中除錯的部分。

核心設計是「分箱配額」：樣本依 (視角, 距離, 亮度) 分箱，每箱各有上限，汰換只在同箱內發生。
這一個決定同時解決三件事：從不同角度與距離蒐集樣本、避免收進一堆重複樣本、
以及特徵庫滿了之後該丟哪一個 —— 塞滿的正面箱永遠不會擠掉唯一的背面樣本。
"""

import json
import os
import time
from dataclasses import dataclass

import numpy as np

# ==========================================
# 版本與前處理識別碼
# embedding 只有在「同一個模型 + 同一套前處理」下才可以互相比較。
# 改了對齊模板卻沿用舊特徵庫，辨識會靜默地失準，是最難除錯的一類 bug，
# 所以這些識別碼會寫進檔案並在載入時逐項驗證。
# ==========================================
GALLERY_VERSION = 1
FACE_MODEL_ID = "facenet-vggface2"
BODY_MODEL_ID = "mnv3s576+hsv144"
FACE_TEMPLATE_ID = "eyes2pt_160_y0.38_iod0.36_v1"
BODY_CROP_ID = "kpt_headtorso_128x160_v1"
SOURCE_ID = "raw960x720"

FACE = "face"
BODY = "body"

# 容量上限不是為了省記憶體 (24x512 的浮點數只有 49KB)，而是為了限制特徵庫隨時間漂移
FACE_PER_BIN, FACE_MAX = 3, 24
BODY_PER_BIN, BODY_MAX = 2, 48
FACE_DIVERSITY, BODY_DIVERSITY = 0.92, 0.90   # 與同箱既有樣本太像就不收
# top-k 的 k 絕不能大於每箱配額：稀有視角 (例如背面) 在同一箱內最多只有 per_bin 筆樣本，
# k 超過就必然會把其他視角的低分平均進來，讓背對鏡頭時的分數被系統性地稀釋。
FACE_TOPK, BODY_TOPK = 2, 2
CENTROID_MIN_QUALITY = 0.70                    # 只有高品質樣本能參與質心計算
BODY_TTL_SECONDS = 12 * 3600                   # 體態樣本的保存期限 (衣服每天換，臉不會)


@dataclass(eq=False)   # 內含 numpy 陣列，改用物件識別比較
class Sample:
    vec: np.ndarray
    quality: float
    ts: float
    view_bin: int
    scale_bin: int
    bright_bin: int
    protected: bool = False

    @property
    def bin_key(self):
        return (self.view_bin, self.scale_bin, self.bright_bin)


class ModalityConfig:
    def __init__(self, dim, per_bin, max_total, diversity, topk):
        assert topk <= per_bin, "top-k 不可大於每箱配額，否則稀有視角的分數會被稀釋"
        self.dim = dim
        self.per_bin = per_bin
        self.max_total = max_total
        self.diversity = diversity
        self.topk = topk


FACE_CFG = ModalityConfig(512, FACE_PER_BIN, FACE_MAX, FACE_DIVERSITY, FACE_TOPK)
BODY_CFG = ModalityConfig(720, BODY_PER_BIN, BODY_MAX, BODY_DIVERSITY, BODY_TOPK)


def _remove(arr, victim):
    """以物件識別移除 (不能用 list.remove，它會觸發 numpy 陣列的逐元素比較)"""
    for i, s in enumerate(arr):
        if s is victim:
            del arr[i]
            return True
    return False


class IdentitySlot:
    """單一身分的人臉與體態樣本集合"""

    def __init__(self, name="target", face_cfg=FACE_CFG, body_cfg=BODY_CFG):
        self.name = name
        self.cfg = {FACE: face_cfg, BODY: body_cfg}
        self.samples = {FACE: [], BODY: []}
        self._mean_cache = {FACE: None, BODY: None}

    # ---------- 查詢 ----------

    def counts(self):
        return len(self.samples[FACE]), len(self.samples[BODY])

    def empty(self):
        return not self.samples[FACE] and not self.samples[BODY]

    def has(self, modality):
        return bool(self.samples[modality])

    def score(self, vec, modality):
        """
        逐格比對用的相似度：每個分箱各自取前 k 名的平均，再取所有分箱中的最高分。

        為什麼不是「整個特徵庫取 top-k」：分箱代表不同視角，背面樣本與正面樣本幾乎正交。
        人背對鏡頭時，整庫 top-k 必然會把不相干的正面樣本平均進來，把分數系統性地拉低 ——
        而背對鏡頭正是環繞模式最需要鎖得住的時候。改成「和最像的那個視角比」就沒有這個問題。

        為什麼不是單純的 max：同一箱內有多筆樣本時要取平均，代表必須多個樣本共同背書，
        單一個僥倖的離群樣本沒辦法自己撐起高分 —— 那正是污染的放大器。
        """
        arr = self.samples[modality]
        if not arr or vec is None:
            return None
        vec = np.asarray(vec, dtype=np.float32)
        k = self.cfg[modality].topk

        bins = {}
        for s in arr:
            bins.setdefault(s.bin_key, []).append(s)

        best = None
        for group in bins.values():
            sims = np.asarray([s.vec for s in group], dtype=np.float32) @ vec
            kk = min(k, sims.size)
            m = float(np.mean(np.sort(sims)[-kk:]))
            if best is None or m > best:
                best = m
        return best

    def mean(self, modality):
        """
        質心：只由受保護的種子樣本與高品質樣本計算。

        質心是雜訊平均後的穩定錨點，單一個壞樣本無法挾持它，所以最適合拿來當登錄的「准入官」。
        """
        cached = self._mean_cache[modality]
        if cached is not None:
            return cached
        arr = self.samples[modality]
        if not arr:
            return None
        core = [s for s in arr if s.protected or s.quality >= CENTROID_MIN_QUALITY]
        if not core:
            core = [s for s in arr if s.protected] or arr
        v = np.mean(np.asarray([s.vec for s in core], dtype=np.float32), axis=0)
        n = float(np.linalg.norm(v))
        v = (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)
        self._mean_cache[modality] = v
        return v

    def centroid_score(self, vec, modality):
        m = self.mean(modality)
        if m is None or vec is None:
            return None
        return float(np.dot(m, np.asarray(vec, dtype=np.float32)))

    def max_sim_in_bin(self, vec, modality, bin_key):
        arr = [s for s in self.samples[modality] if s.bin_key == bin_key]
        if not arr:
            return 0.0
        sims = np.asarray([s.vec for s in arr], dtype=np.float32) @ np.asarray(vec, dtype=np.float32)
        return float(sims.max())

    # ---------- 新增與汰換 ----------

    def admit(self, sample, modality):
        """
        嘗試把樣本收進特徵庫，回傳 (有沒有收, 原因)。

        受保護的種子樣本 (protected) 不受多樣性與配額限制，也永遠不會被汰換 ——
        這保證特徵庫裡永遠存在一個不可污染的身分錨點，是整套防污染設計的地基。
        """
        cfg = self.cfg[modality]
        arr = self.samples[modality]

        if sample.vec is None or sample.vec.shape[-1] != cfg.dim:
            return False, "dim_mismatch"

        if sample.protected:
            arr.append(sample)
            self._mean_cache[modality] = None
            return True, "seed"

        # 多樣性：與同箱既有樣本幾乎一樣的樣本沒有資訊量，只會擠掉未來有用的樣本
        if self.max_sim_in_bin(sample.vec, modality, sample.bin_key) >= cfg.diversity:
            return False, "too_similar"

        # 配額只計算非種子樣本：種子是永久保留的，如果讓它們佔用配額，
        # 種子所在的那一箱 (通常就是正面、中距離那一箱) 會永遠無法再收新樣本
        same_bin = [s for s in arr if s.bin_key == sample.bin_key and not s.protected]
        if len(same_bin) >= cfg.per_bin:
            victim = self._worst(same_bin)
            if victim is None or victim.quality >= sample.quality:
                return False, "bin_full"
            _remove(arr, victim)

        if len(arr) >= cfg.max_total:
            victim = self._worst_from_largest_bin(arr)
            if victim is None:
                return False, "gallery_full"
            _remove(arr, victim)

        arr.append(sample)
        self._mean_cache[modality] = None
        return True, "admitted"

    @staticmethod
    def _worst(samples):
        cand = [s for s in samples if not s.protected]
        if not cand:
            return None
        return min(cand, key=lambda s: (s.quality, -s.ts))

    def _worst_from_largest_bin(self, arr):
        bins = {}
        for s in arr:
            if not s.protected:
                bins.setdefault(s.bin_key, []).append(s)
        if not bins:
            return None
        largest = max(bins.values(), key=len)
        return self._worst(largest)

    # ---------- 維護 ----------

    def prune_body(self, now=None, ttl=BODY_TTL_SECONDS):
        """丟掉過期的體態樣本；人臉樣本永遠保留"""
        now = time.time() if now is None else now
        before = len(self.samples[BODY])
        self.samples[BODY] = [s for s in self.samples[BODY] if (now - s.ts) <= ttl]
        self._mean_cache[BODY] = None
        return before - len(self.samples[BODY])

    def revoke_since(self, t0):
        """
        撤銷某個時間點之後收進來的非種子樣本。

        用途是「後悔藥」：如果目標分數在收樣本後短時間內崩塌，代表剛剛很可能收錯人，
        把那段時間的樣本撤掉可以救回偶發的鏈式污染。
        """
        removed = 0
        for modality in (FACE, BODY):
            keep = [s for s in self.samples[modality] if s.protected or s.ts < t0]
            removed += len(self.samples[modality]) - len(keep)
            self.samples[modality] = keep
            self._mean_cache[modality] = None
        return removed

    def clear(self):
        self.samples = {FACE: [], BODY: []}
        self._mean_cache = {FACE: None, BODY: None}


class IdentityGallery:
    """特徵庫的持久化容器"""

    def __init__(self, root="data/identity", name="target",
                 face_cfg=FACE_CFG, body_cfg=BODY_CFG):
        self.root = root
        self.name = name
        self.path = os.path.join(root, f"{name}.npz")
        self.slot = IdentitySlot(name, face_cfg, body_cfg)
        self.dirty = False
        self.last_save = 0.0

    # ---------- 讀寫 ----------

    def info_dict(self):
        return {
            "version": GALLERY_VERSION,
            "name": self.name,
            "face_model": FACE_MODEL_ID,
            "face_dim": self.slot.cfg[FACE].dim,
            "body_model": BODY_MODEL_ID,
            "body_dim": self.slot.cfg[BODY].dim,
            "face_template": FACE_TEMPLATE_ID,
            "body_crop": BODY_CROP_ID,
            "source": SOURCE_ID,
            "updated": time.time(),
        }

    def _validate(self, info):
        expected = self.info_dict()
        for key in ("version", "face_model", "face_dim", "body_model", "body_dim",
                    "face_template", "body_crop"):
            if info.get(key) != expected[key]:
                return False, key
        if info.get("source") != expected["source"]:
            # 裁切來源不同 (例如用離線工具的縮放畫面建檔、卻在飛行時用原生畫面比對)
            # 會讓相似度整體下降，但不至於完全不可用，所以只提醒不拒絕
            print(f"[身分辨識] 提醒：特徵庫是用 {info.get('source')} 建立的，"
                  f"目前是 {expected['source']}，辨識分數可能偏低")
        return True, None

    def load(self, now=None):
        if not os.path.exists(self.path):
            return False
        try:
            with np.load(self.path, allow_pickle=False) as z:
                info = json.loads(str(z["info"]))
                ok, bad_key = self._validate(info)
                if not ok:
                    print(f"[身分辨識] 特徵庫格式不符 ({bad_key} 不一致)，已忽略舊檔案。"
                          f"請重新按 R 建檔。")
                    return False
                self.slot.clear()
                for modality in (FACE, BODY):
                    vecs = z[f"{modality}_vecs"]
                    meta = z[f"{modality}_meta"]
                    for v, m in zip(vecs, meta):
                        self.slot.samples[modality].append(Sample(
                            vec=np.asarray(v, dtype=np.float32),
                            quality=float(m[0]), ts=float(m[1]),
                            view_bin=int(m[2]), scale_bin=int(m[3]), bright_bin=int(m[4]),
                            protected=bool(m[5]),
                        ))
                self.slot._mean_cache = {FACE: None, BODY: None}
        except Exception as e:
            print(f"[身分辨識] 特徵庫讀取失敗，已忽略：{e}")
            return False

        dropped = self.slot.prune_body(now=now)
        n_face, n_body = self.slot.counts()
        if not self.slot.empty():
            msg = f"[身分辨識] 已載入特徵庫：臉 {n_face} / 體態 {n_body}"
            if dropped:
                msg += f"（體態樣本過期丟棄 {dropped} 筆）"
            print(msg)
        self.dirty = bool(dropped)
        return not self.slot.empty()

    def save_if_dirty(self, force=False, now=None):
        if not self.dirty and not force:
            return False
        now = time.time() if now is None else now
        try:
            os.makedirs(self.root, exist_ok=True)
            payload = {"info": np.str_(json.dumps(self.info_dict()))}
            for modality in (FACE, BODY):
                arr = self.slot.samples[modality]
                dim = self.slot.cfg[modality].dim
                if arr:
                    payload[f"{modality}_vecs"] = np.asarray(
                        [s.vec for s in arr], dtype=np.float32)
                    payload[f"{modality}_meta"] = np.asarray(
                        [[s.quality, s.ts, s.view_bin, s.scale_bin, s.bright_bin,
                          1.0 if s.protected else 0.0] for s in arr], dtype=np.float32)
                else:
                    payload[f"{modality}_vecs"] = np.zeros((0, dim), dtype=np.float32)
                    payload[f"{modality}_meta"] = np.zeros((0, 6), dtype=np.float32)
                mean = self.slot.mean(modality)
                payload[f"{modality}_mean"] = (mean if mean is not None
                                               else np.zeros(dim, dtype=np.float32))

            # 原子寫入：飛行中斷電也不會留下半個檔案
            tmp = self.path + ".tmp"
            np.savez_compressed(tmp, **payload)
            if not os.path.exists(tmp) and os.path.exists(tmp + ".npz"):
                tmp = tmp + ".npz"      # numpy 會自動補副檔名
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[身分辨識] 特徵庫寫入失敗：{e}")
            return False

        self.dirty = False
        self.last_save = now
        return True

    def mark_dirty(self):
        self.dirty = True

    def reset(self):
        self.slot.clear()
        self.dirty = True
