import cv2
import numpy as np

from utils.white_strip import WhiteStripVerifier, MATCH, UNKNOWN, NO_MATCH


class ColourStripVerifier:
    """
    彩色紙條驗證器：確認一條紙條「有沒有顏色」。

    與 WhiteStripVerifier 介面相同 (features / verdict_from / analyze / draw)，
    回傳同樣的三值判定，TissueDetector 用 C 鍵在兩者之間切換；
    白色模式也用它反過來檢查 —— 有顏色的紙條就算白色驗證通過也要擋掉。

    ---------------------------------------------------------------
    「有顏色」的定義：沿著紙條，有一段以上的彩色條紋
    ---------------------------------------------------------------
    實際場地的彩色紙條是「白色衛生紙上畫了幾段紫色條紋」(10/13 18:11 錄影)，
    不是整條都是彩色 —— 紫色只佔紙條長度的 19~35%。
    所以判斷的是「紙條上有多少比例的列是彩色的」(coloured_rows)，
    而不是「紙條大多是彩色」；整條都是彩色的紙條當然也會通過。

    ---------------------------------------------------------------
    只看紙條本身的像素 (YOLO 的分割輪廓)
    ---------------------------------------------------------------
    紙條會扭轉、彎曲，常常不在框的正中央；框裡也一定會包到背景
    (藍色抽屜櫃、紅沙發)。只看框中央那一欄的話，同一段錄影裡有一成的偵測
    會錯過歪到旁邊的條紋；改用 YOLO 切出來的輪廓 (內縮 1px 去掉邊緣的背景色) 之後，
    247 個偵測裡只剩 4 個量不到，而且其中 3 個是誤判 (白牆邊緣、椅腳)。
    拿不到輪廓時 (偵測模型不是分割模型) 退回看框中央那一欄。

        coloured_rows  紙條像素裡彩色超過 ROW_COLOUR_MIN 的列，佔整條紙條的比例
        hue_conc       彩色像素色相的集中度 (0~1)，太低代表多種顏色混在一起
                       (YOLO 偶爾會把牆上的畫框認成紙條)

    ---------------------------------------------------------------
    門檻的校正 (10/20 五趟錄影，判斷距離內 (110cm 以內) 人工標記 101 個偵測)
    ---------------------------------------------------------------
    很多條紋是麥克筆塗的淡紫色斜線，在這個場地的光線下飽和度只有 40~60，
    舊門檻 SAT_MIN 70 / ROWS_MIN 0.08 每格只抓得到 54% 的條紋紙條 (22:47 那趟
    條紋紙條被確認成白色衝過去)。逐格判斷 (投票之前)：

                                   白紙條被當成彩色    條紋紙條抓得到
        SAT_MIN 70, ROWS_MIN 0.08       0/31              54%   <-- 舊
        SAT_MIN 40, ROWS_MIN 0.08       0/31              96%   (白紙條最高 0.08，貼著門檻)
        SAT_MIN 40, ROWS_MIN 0.12       0/31              89%   <-- 採用，留 0.04 餘裕

    在 50~110cm 各距離都差不多 (79~94%)。牆角、門框誤判的 coloured_rows 最多 0.06，照樣算沒顏色。
    黑斑紙條 (偏棕色) 在新門檻下也算彩色 (86~100%)，白色模式一樣不撞、會閃。
    ⚠️ 白色驗證器把飽和度 40 以下算白，兩邊的門檻現在是接在一起的 ——
    換到偏暖光源的場地 (白紙條飽和度可能到 40~60) 請先錄影重新校正。

    這個模組刻意不 import torch / ultralytics，才能單獨 import 來做離線分析。
    """

    def __init__(self,
                 sat_min: int = 40,
                 val_min: int = 50,
                 row_colour_min: float = 0.3,
                 rows_min: float = 0.12,
                 hue_conc_min: float = 0.7,
                 min_px: int = 40,
                 centre_band: float = 0.5):
        """
        :param sat_min: 飽和度下限，超過才算彩色像素 (校正見類別說明：淡紫色麥克筆條紋只有 40~60)。
        :param val_min: 亮度下限，太暗的像素色相不穩定，不列入。
        :param row_colour_min: 一列裡紙條像素有這個比例以上是彩色，這一列才算彩色列。
        :param rows_min: 彩色列佔整條紙條的比例下限，超過就算「有顏色的紙條」。
        :param hue_conc_min: 彩色像素色相集中度下限 (圓形統計的 mean resultant length)。
                        單色條紋通常 > 0.9；兩種顏色各半約 0~0.5。
        :param min_px: 紙條像素少於此 (太遠、太小) 就不下結論 (UNKNOWN)。
        :param centre_band: 沒有分割輪廓時，改看框中央多寬的一欄 (框寬的比例)。
        """
        self.sat_min = sat_min
        self.val_min = val_min
        self.row_colour_min = row_colour_min
        self.rows_min = rows_min
        self.hue_conc_min = hue_conc_min
        self.min_px = min_px
        self.centre_band = centre_band

    def reset(self):
        pass

    @staticmethod
    def _empty(reason, x0, y0, crop_w, crop_h, source=""):
        return {
            "ok": False, "reason": reason, "source": source,
            "crop_w": crop_w, "crop_h": crop_h, "crop_box": (x0, y0, crop_w, crop_h),
            "n_px": 0, "colour_px_frac": 0.0, "coloured_rows": 0.0,
            "hue_conc": 0.0, "hue_deg": 0.0,
            "blob_aspect": 0.0, "blob_box": None, "mask": None,
        }

    # ==========================================
    def _strip_mask(self, crop_h, crop_w, x0, y0, poly):
        """紙條像素的遮罩 (crop 座標)：有分割輪廓就用輪廓，否則用框中央那一欄"""
        if poly is not None and len(poly) >= 3:
            mk = np.zeros((crop_h, crop_w), np.uint8)
            pts = (np.asarray(poly, dtype=np.float32) - np.array([x0, y0], np.float32))
            cv2.fillPoly(mk, [pts.round().astype(np.int32)], 1)
            if crop_w >= 6:
                # 內縮 1px：輪廓邊緣常常包到一點背景，背景剛好是彩色時會被誤算
                mk = cv2.erode(mk, np.ones((3, 3), np.uint8))
            return mk.astype(bool), "mask"
        mk = np.zeros((crop_h, crop_w), bool)
        half = max(1, int(round(crop_w * self.centre_band / 2.0)))
        c = crop_w // 2
        mk[:, max(0, c - half):min(crop_w, c + half)] = True
        return mk, "centre"

    def features(self, frame, box, poly=None):
        """
        :param frame: 原始 BGR 影像
        :param box: (cx, cy, w, h)，YOLO 的框 (中心點 + 寬高)
        :param poly: YOLO 分割輪廓 (整張影像座標的 Nx2 點)，None 就改看框中央
        """
        h_img, w_img = frame.shape[:2]
        cx, cy, w, h = box
        x0, y0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
        x1, y1 = min(w_img, int(cx + w / 2)), min(h_img, int(cy + h / 2))
        crop_w, crop_h = x1 - x0, y1 - y0
        if crop_w < 2 or crop_h < 2:
            return self._empty("crop-small", x0, y0, crop_w, crop_h)

        strip, source = self._strip_mask(crop_h, crop_w, x0, y0, poly)
        n_px = int(strip.sum())
        if n_px < self.min_px:
            return self._empty("small", x0, y0, crop_w, crop_h, source)

        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        coloured = strip & (hsv[:, :, 1] >= self.sat_min) & (hsv[:, :, 2] >= self.val_min)
        n_col = int(coloured.sum())

        row_px = strip.sum(axis=1)
        row_col = coloured.sum(axis=1)
        valid = row_px >= 2
        coloured_rows = (float(((row_col[valid] / row_px[valid]) >= self.row_colour_min).mean())
                         if valid.any() else 0.0)

        feat = self._empty("", x0, y0, crop_w, crop_h, source)
        feat.update(ok=True, n_px=n_px, colour_px_frac=n_col / n_px,
                    coloured_rows=coloured_rows, mask=coloured)

        ys, xs = np.nonzero(strip)
        feat["blob_aspect"] = (np.ptp(ys) + 1) / max(np.ptp(xs) + 1, 1)
        if n_col:
            # 色相是角度 (OpenCV 0~179 對應 0~358 度)，要用圓形統計：
            # 紅色橫跨 0 度兩側，直接取平均會得到青色
            ang = hsv[:, :, 0][coloured].astype(np.float32) * (2.0 * np.pi / 180.0)
            c, s = float(np.cos(ang).mean()), float(np.sin(ang).mean())
            feat["hue_conc"] = float(np.hypot(c, s))
            feat["hue_deg"] = float(np.degrees(np.arctan2(s, c)) % 360.0)
            cy_, cx_ = np.nonzero(coloured)
            feat["blob_box"] = (x0 + int(cx_.min()), y0 + int(cy_.min()),
                                int(np.ptp(cx_)) + 1, int(np.ptp(cy_)) + 1)
        return feat

    def has_colour(self, feat):
        """這條紙條看得出有彩色條紋 (給白色模式擋掉用)。看不清楚 (ok=False) 時回 False"""
        return bool(feat.get("ok")) and feat["coloured_rows"] >= self.rows_min

    def verdict_from(self, feat):
        """彩色模式的判定。:return: (verdict, reason)"""
        if not feat.get("ok"):
            return UNKNOWN, feat.get("reason", "no-features")
        if feat["coloured_rows"] < self.rows_min:
            return NO_MATCH, "white"
        if feat["hue_conc"] < self.hue_conc_min:
            return NO_MATCH, "multi"
        return MATCH, ""

    def analyze(self, frame, box, poly=None):
        feat = self.features(frame, box, poly)
        verdict, reason = self.verdict_from(feat)
        feat["verdict"] = verdict
        feat["reason"] = reason or feat.get("reason", "")
        return feat

    # 畫圖與白色驗證器共用
    draw = staticmethod(WhiteStripVerifier.draw)
    draw_mask_pip = staticmethod(WhiteStripVerifier.draw_mask_pip)
