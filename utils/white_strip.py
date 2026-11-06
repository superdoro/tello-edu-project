import cv2
import numpy as np

# 判定結果。刻意是三值而不是布林 —— 理由見 WhiteStripVerifier 的類別說明。
WHITE = 1
UNKNOWN = 0
NOT_WHITE = -1

# 與目標顏色無關的通用名稱：TissueDetector 可以切換「撞白色 / 撞彩色」
# (彩色驗證器見 utils/colour_strip.py)，投票只在乎「符不符合目前的目標」
MATCH = WHITE
NO_MATCH = NOT_WHITE

_VERDICT_NAME = {WHITE: "WHITE", UNKNOWN: "UNKNOWN", NOT_WHITE: "NOT_WHITE"}


class WhiteStripVerifier:
    """
    白色紙條驗證器：對 YOLO 框內做門檻化 + 分割，確認它真的是一條白色的紙條。

    為什麼需要這一關：YOLO 會把白牆、窗簾、牆上的畫框認成紙條，而且誤判框的
    信心值跟真紙條幾乎一樣 (實測 0.62 vs 0.66)，調信心門檻沒有用。形狀關卡
    (min_aspect) 擋掉了一部分，但畫框那種細長的邊框一樣過得去。

    ---------------------------------------------------------------
    判準：亮 + 不飽和 + 細長，其中「不飽和」是主力
    ---------------------------------------------------------------
    用 Otsu 把框內切成明暗兩半，再要求亮的那一半「夠不飽和」(夠白)，
    最後看切出來的最大連通塊夠不夠細長。

    三道條件裡真正有鑑別力的是飽和度。實測 (0923 錄影，誤判 n=335 為牆上的
    藍色格狀結構，真紙條 n=803)，固定 blob_aspect_min=3.0 只調 sat_max：

        sat_max   255      120      80       50       40       30
        擋掉誤判    0%      41%      67%      97%      98%      99%
        保留真目標  87%      86%      86%      86%      85%      85%

    形狀門檻反而可以放得很鬆 —— 在 sat_max=40 下，blob_aspect_min 從 4.0
    降到 2.0 會讓保留率從 75% 升到 95%，擋掉誤判只從 99% 掉到 98%。
    這對近距離是好消息：逼近時團塊形狀必然劣化，而我們並不依賴它。

    注意 pad_ratio 要維持 0：往外擴框會把背景一起算進來，實測 0.3 就讓保留率
    從 95% 掉到 64%。

    絕對亮度完全不能用：紙條 V 中位數 124、天花板 134，紙條反而比較暗。
    面積類特徵也不能用 —— Otsu 是自適應的，在任何 crop 裡都一定會切出一個
    分界，即使那是一片均勻的牆，所以「白色像素比例」永遠只是在描述
    「這個 crop 比較亮的那一半」，量到的百分位證實兩者完全重疊。
    面積只拿來當「分割是不是失敗了」的守衛，不當鑑別器。

    ---------------------------------------------------------------
    為什麼回傳三值而不是 True/False
    ---------------------------------------------------------------
    UNKNOWN 是這個設計的支點。逼近到最後紙條會脹滿整個框，crop 裡再也沒有
    背景可以對比，Otsu 的可分離度自然崩潰 —— 這時候給出 NOT_WHITE 會在最後
    關頭把已經鎖定的目標丟掉。改成回 UNKNOWN，呼叫端就知道「分割器這一格
    無法判斷」，把它排除在投票之外，既有的判定原封不動保留。

    ---------------------------------------------------------------
    analyze() = features() + verdict_from()
    ---------------------------------------------------------------
    刻意拆成兩段：features() 做影像處理算出特徵，verdict_from() 只是套門檻。
    離線掃參數時可以把特徵倒出來一次，再用不同門檻重算判定 —— otsu_min_sep、
    blob_aspect_min、min_blob_area_frac、colour_frac_min 四個參數的掃描因此
    是瞬間完成的，不必為每組參數重跑一次影像處理。
    (sat_max 與 pad_ratio 會改變影像處理本身，必須在 dump 階段就算好。)

    參數是用 tests/replay_tissue.py 在 recordings/ 的兩段錄影上掃出來的，
    改動前請先重跑：
        python tests/replay_tissue.py dump  recordings/<raw>.mp4 --out /tmp/d.json
        python tests/replay_tissue.py sweep /tmp/d.json --fp 10-20,63-70 --true 0-8,30-48

    這個模組刻意不 import torch / ultralytics，才能單獨 import 來做離線分析。
    """

    def __init__(self,
                 sat_max: int = 40,
                 blob_aspect_min: float = 2.0,
                 min_blob_area_frac: float = 0.02,
                 otsu_min_sep: float = 20.0,
                 colour_frac_min: float = 0.25,
                 pad_ratio: float = 0.0,
                 min_crop_w: int = 6,
                 min_crop_h: int = 18):
        """
        :param sat_max: 飽和度上限，超過的像素不算白色。**整個驗證器最關鍵的參數**
                        (實測：255->擋掉 0%，120->41%，80->67%，50->97%，40->98%，30->99%)。
                        設 40 是高原區靠內一格，留一點餘裕給偏暖光源下的紙條。
        :param blob_aspect_min: 最大連通塊的長寬比下限。實測在 sat_max=40 下
                        2.0 -> 保留 95% / 擋掉 98%   <-- 預設值，高原區
                        3.0 -> 保留 85% / 擋掉 98%
                        4.0 -> 保留 75% / 擋掉 99%
                        刻意放鬆：鑑別力來自飽和度，而逼近時團塊形狀必然劣化，
                        門檻設緊只會在最需要偵測的近距離把目標擋掉。
        :param min_blob_area_frac: 團塊面積比下限，低於此視為分割只切出雜訊 (UNKNOWN)。
                        純粹是守衛不是鑑別器 —— 調高它只會把真目標變成 UNKNOWN，
                        在「保留率」上看起來變好，實際是不下結論而已。
        :param otsu_min_sep: Otsu 切出來的明暗兩半，平均亮度至少要差這麼多。
                        差太少代表 crop 幾乎是均勻的、分割沒有意義 -> UNKNOWN。
                        ⚠️ 在現有的兩段錄影上這道守衛從未觸發過 (0 與 20 的結果
                        完全相同)，它是為了「紙條脹滿整個框」這個手上沒有資料的
                        近距離情況留的保險，合成測試確認它在那種情況下會正確觸發。
        :param colour_frac_min: 亮的那一半裡，飽和度夠低 (夠白) 的比例下限。
                        低於此代表「確實有個亮的東西，但它是有顏色的」-> NOT_WHITE。
                        牆上藍色畫框那類誤判就是在這裡被擋下的。
                        實測在 sat_max=40 下 0.25 以上是高原 (保留 95% / 擋掉 99%)。
        :param pad_ratio: 往外擴框的比例。**維持 0** —— 實測 0.3 會讓保留率從
                        95% 掉到 64%，因為擴框會把背景一起算進分割。
        :param min_crop_w / min_crop_h: 小於這個尺寸的 crop 不做判斷 (UNKNOWN)。
                        寬度小於 6px 代表紙條在 2.3 公尺外，分割沒有意義。
        """
        self.sat_max = sat_max
        self.blob_aspect_min = blob_aspect_min
        self.min_blob_area_frac = min_blob_area_frac
        self.otsu_min_sep = otsu_min_sep
        self.colour_frac_min = colour_frac_min
        self.pad_ratio = pad_ratio
        self.min_crop_w = min_crop_w
        self.min_crop_h = min_crop_h

    def reset(self):
        """本身無狀態，保留這個方法是為了與 DarkRegionAnalyzer 的介面一致"""
        pass

    @staticmethod
    def _empty(reason, x0, y0, crop_w, crop_h, med_s=0.0, med_v=0.0, otsu_thr=0.0):
        """算不出特徵時的回傳值。所有數值欄位都要在，下游取值才不會 KeyError。"""
        return {
            "ok": False, "reason": reason,
            "crop_w": crop_w, "crop_h": crop_h,
            "crop_box": (x0, y0, crop_w, crop_h),
            "otsu_thr": otsu_thr, "otsu_sep": 0.0, "colour_frac": 0.0,
            "blob_aspect": 0.0, "blob_area_frac": 0.0, "blob_w_ratio": 0.0,
            "blob_box": None, "med_s": med_s, "med_v": med_v, "mask": None,
        }

    # ==========================================
    # 第一段：影像處理，算特徵
    # ==========================================
    def features(self, frame, box):
        """
        對單一個框做門檻化 + 分割，回傳所有原始特徵。

        :param frame: 原始 BGR 影像
        :param box: (cx, cy, w, h)，YOLO 的框 (中心點 + 寬高)
        :return: dict。ok=False 代表連特徵都算不出來 (crop 太小或分割全空)，
                 此時只有 reason 欄位有意義。
        """
        h_img, w_img = frame.shape[:2]
        cx, cy, w, h = box

        pad_x = int(w * self.pad_ratio)
        pad_y = int(h * self.pad_ratio)
        x0 = max(0, int(cx - w / 2) - pad_x)
        y0 = max(0, int(cy - h / 2) - pad_y)
        x1 = min(w_img, int(cx + w / 2) + pad_x)
        y1 = min(h_img, int(cy + h / 2) + pad_y)

        crop_w, crop_h = x1 - x0, y1 - y0
        if crop_w < self.min_crop_w or crop_h < self.min_crop_h:
            return self._empty("crop-small", x0, y0, crop_w, crop_h)

        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Otsu 門檻化亮度。cv2 回傳 0/255 的 uint8，轉成布林方便做邏輯運算。
        otsu_thr, bright_u8 = cv2.threshold(v_ch, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = bright_u8 > 0
        n_bright = int(bright.sum())
        n_dark = int(bright.size - n_bright)
        if n_bright == 0 or n_dark == 0:
            # crop 裡只有單一亮度 —— 框太貼合紙條、裡面完全沒有背景可以對比。
            # 這是「不知道」，不是「不是白的」。pad_ratio 調大可以減少這種情況。
            return self._empty("flat", x0, y0, crop_w, crop_h,
                               med_s=float(np.median(s_ch)), med_v=float(np.median(v_ch)),
                               otsu_thr=float(otsu_thr))

        # 明暗兩半的平均亮度差：太小代表 crop 幾乎均勻，這個分割不能信
        otsu_sep = float(v_ch[bright].mean() - v_ch[~bright].mean())

        # 亮 + 不飽和 = 白。colour_frac 是「亮的那一半裡有多少是真的白」。
        white_mask = bright & (s_ch <= self.sat_max)
        colour_frac = float(white_mask.sum()) / float(n_bright)

        blob_aspect = 0.0
        blob_area_frac = 0.0
        blob_w_ratio = 0.0
        blob_box = None
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(
            white_mask.astype(np.uint8), connectivity=8)
        if n_lab > 1:
            # 第 0 個是背景，從 1 開始找面積最大的連通塊
            areas = stats[1:, cv2.CC_STAT_AREA]
            k = int(np.argmax(areas)) + 1
            bx, by = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
            bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
            blob_aspect = bh / max(bw, 1)
            blob_area_frac = float(stats[k, cv2.CC_STAT_AREA]) / float(crop_w * crop_h)
            blob_w_ratio = bw / max(crop_w, 1)
            blob_box = (x0 + bx, y0 + by, bw, bh)   # 換算回整張影像的座標

        return {
            "ok": True,
            "reason": "",
            "crop_w": crop_w, "crop_h": crop_h,
            "crop_box": (x0, y0, crop_w, crop_h),
            "otsu_thr": float(otsu_thr),
            "otsu_sep": otsu_sep,
            "colour_frac": colour_frac,
            "blob_aspect": blob_aspect,
            "blob_area_frac": blob_area_frac,
            "blob_w_ratio": blob_w_ratio,
            "blob_box": blob_box,
            "med_s": float(np.median(s_ch)),
            "med_v": float(np.median(v_ch)),
            "mask": white_mask,
        }

    # ==========================================
    # 第二段：套門檻，得出判定
    # ==========================================
    def verdict_from(self, feat, otsu_min_sep=None, blob_aspect_min=None,
                     min_blob_area_frac=None, colour_frac_min=None):
        """
        對 features() 的結果套門檻。參數留空就用 __init__ 設定的值；
        離線掃參數時傳入不同的值，就能用同一份特徵重算判定。

        :return: (verdict, reason)
        """
        otsu_min_sep = self.otsu_min_sep if otsu_min_sep is None else otsu_min_sep
        blob_aspect_min = self.blob_aspect_min if blob_aspect_min is None else blob_aspect_min
        min_blob_area_frac = (self.min_blob_area_frac if min_blob_area_frac is None
                              else min_blob_area_frac)
        colour_frac_min = self.colour_frac_min if colour_frac_min is None else colour_frac_min

        if not feat.get("ok"):
            return UNKNOWN, feat.get("reason", "no-features")

        # crop 幾乎均勻 -> 分割沒有意義。這是「不知道」，不是「不是白的」。
        if feat["otsu_sep"] < otsu_min_sep:
            return UNKNOWN, "flat"

        # 亮的東西存在，但它是有顏色的 -> 確定不是白紙條 (牆上畫框那類)
        if feat["colour_frac"] < colour_frac_min:
            return NOT_WHITE, "col"

        # 只切出雜訊 -> 分割失敗，不下結論
        if feat["blob_box"] is None or feat["blob_area_frac"] < min_blob_area_frac:
            return UNKNOWN, "speckle"

        if feat["blob_aspect"] >= blob_aspect_min:
            return WHITE, ""
        return NOT_WHITE, f"a{feat['blob_aspect']:.1f}"

    def analyze(self, frame, box):
        """
        features() + verdict_from() 的組合，飛行時用這個。

        :return: dict，除了 features() 的所有欄位外再加上
                 verdict : WHITE(+1) / UNKNOWN(0) / NOT_WHITE(-1)
                 reason  : 判定理由，會直接印在畫面上
        """
        feat = self.features(frame, box)
        verdict, reason = self.verdict_from(feat)
        feat["verdict"] = verdict
        feat["reason"] = reason or feat.get("reason", "")
        return feat

    # ==========================================
    @staticmethod
    def draw(annotated_frame, info, colour=(255, 255, 255)):
        """把分割出來的最大連通塊畫成細框，一眼就看得出分割器抓到了什麼"""
        if not info or not info.get("blob_box"):
            return annotated_frame
        bx, by, bw, bh = info["blob_box"]
        cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), colour, 1)
        return annotated_frame

    @staticmethod
    def draw_mask_pip(annotated_frame, info, width=80, org=None):
        """
        把二值遮罩縮成小圖疊在畫面角落 (picture-in-picture)。
        現場調 blob_aspect_min 時這是最有用的一個畫面 —— 直接看到分割結果。
        """
        if not info or info.get("mask") is None or not info["mask"].size:
            return annotated_frame
        mask = (info["mask"].astype(np.uint8)) * 255
        mh, mw = mask.shape[:2]
        scale = width / max(mw, 1)
        pip_w, pip_h = width, max(1, int(mh * scale))

        h_img, w_img = annotated_frame.shape[:2]
        pip_h = min(pip_h, h_img - 20)
        if pip_h < 1:
            return annotated_frame
        pip = cv2.cvtColor(cv2.resize(mask, (pip_w, pip_h)), cv2.COLOR_GRAY2BGR)

        x, y = org if org else (w_img - pip_w - 10, 90)
        x = max(0, min(x, w_img - pip_w))
        y = max(0, min(y, h_img - pip_h))
        annotated_frame[y:y + pip_h, x:x + pip_w] = pip
        cv2.rectangle(annotated_frame, (x, y), (x + pip_w, y + pip_h), (0, 255, 255), 1)
        return annotated_frame


def verdict_name(v):
    return _VERDICT_NAME.get(v, "?")
