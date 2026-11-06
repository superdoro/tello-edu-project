import cv2

from vision.base import VisionProcessor, VisionData
from vision.plate_backends import sharpness


class PlateVision(VisionProcessor):
    """
    PLATE READER 的視覺模組：深度 (漫遊避障用) + 車牌偵測、鎖定、估距、截圖。

    深度、黑區、白牆整段交給 DepthExplorerVision，欄位 (depth_L/C/R、dark_*、depth_unreliable)
    與 FLUID EXPLORER 完全一樣，PlateReadControl 搜尋時可以直接餵給 FluidExploreControl。
    這裡只在它的結果上疊車牌。

    車牌偵測與 OCR 都是可替換的後端 (vision/plate_backends.py)。OCR 不在這裡跑 ——
    什麼時候讀 (停穩了沒) 只有飛行邏輯知道，這裡每格只附上鎖定那個車牌的截圖與清晰度。

    ==========================================================
    原生畫面
    ==========================================================
    同 TissueDetector：有原生 960x720 畫面 (set_native_frame) 就拿它偵測與截圖，
    座標再換回 720x480。OCR 吃的截圖解析度多 1.33 倍，遠處的小車牌也偵測得比較好。

    ==========================================================
    估距
    ==========================================================
    dist = 實際尺寸 x 焦距 / 框的像素大小，寬、高各算一次取比較近的那個。
    斜著看車牌時框寬會縮小 (高度不會)，只用寬會高估距離、飛得比預期近；取較近的比較安全。
    """

    # 720x480 畫面的焦距 (px)。水平同 TissueChargeControl.FOCAL_PX；
    # 垂直 = 水平 x (480/720)/(720/960) —— 960x720 -> 720x480 的縮放是非等向的
    F_H = 461.0
    F_V = F_H * ((480.0 / 720.0) / (720.0 / 960.0))     # 約 410

    def __init__(self, detector=None, depth=None, to_bgr=True,
                 conf_threshold=0.5, crop_margin=0.10):
        """
        :param detector: PlateDetectorBackend，None = 沒有偵測器 (只漫遊)
        :param depth: 提供深度欄位的視覺模組，None 就建立 DepthExplorerVision (會載入深度模型)。
                      離線測試傳假的，才不用載入模型
        :param to_bgr: Tello 串流其實是 RGB (djitellopy 2.5)，交給後端前先轉成 BGR。
                       如果模型是用本專案的錄影檔訓練的 (錄影檔與串流同順序)，改成 False
        :param conf_threshold: 偵測信心門檻
        :param crop_margin: 截圖時框的四周多留這個比例，OCR 才不會切到字
        """
        print("========================================")
        print("[系統訊息] 啟動車牌辨識視覺 (車牌偵測 + 深度避障)...")
        if depth is None:
            from vision.fluid_explorer_vision import DepthExplorerVision
            depth = DepthExplorerVision()
        self.depth = depth
        self.detector = detector
        if detector is None:
            print("[警告] 沒有車牌偵測器 -> 本模式只會漫遊避障。")
        print("========================================")

        self.to_bgr = to_bgr
        self.conf_threshold = conf_threshold
        self.crop_margin = crop_margin

        # ---------- 目標黏著 (同 TissueDetector) ----------
        # 畫面裡有好幾個車牌時，每格都挑最大的會在兩個之間跳來跳去，機頭跟著甩。
        # 鎖定之後只認離上一格最近的；連續 LOCK_MISS_MAX 格找不到才放掉重挑。
        self.LOCK_GATE = 120
        self.LOCK_MISS_MAX = 5
        self.lock_cx = self.lock_cy = None
        self.lock_miss = 0
        self.hits = 0                  # 鎖定的這個車牌累計看到幾格 (擋單格誤判)

        self.show_all = False          # F 鍵：連被信心門檻擋掉的框也畫出來
        self._native = None

    # ==========================================
    # 對外
    # ==========================================
    @property
    def available(self):
        return self.detector is not None and getattr(self.detector, "available", True)

    def set_native_frame(self, frame):
        """主迴圈在 process_frame 之前呼叫，交給這一格的原生畫面。用一次就清掉"""
        self._native = frame

    def toggle_tracking_mode(self):
        """F 鍵：切換「顯示被信心門檻擋掉的偵測」"""
        self.show_all = not self.show_all
        print(f"[車牌辨識] 低信心偵測顯示：{'開啟' if self.show_all else '關閉'}")

    def reset(self):
        """切換飛行模式時清掉深度的平滑狀態與目標鎖定"""
        if hasattr(self.depth, 'reset'):
            self.depth.reset()
        self._drop_lock()

    def _drop_lock(self):
        self.lock_cx = self.lock_cy = None
        self.lock_miss = 0
        self.hits = 0

    # ==========================================
    def process_frame(self, frame) -> VisionData:
        data = self.depth.process_frame(frame)
        data.is_detected = False
        data.plate = None
        data.n_plates = data.n_low_conf = 0

        annotated = data.annotated_frame
        if annotated is None or annotated is frame:
            annotated = frame.copy()
        h_img, w_img = frame.shape[:2]

        native, self._native = self._native, None
        best = None
        if self.available:
            src = frame
            if native is not None and native.ndim == 3 and native.shape[1] > w_img:
                src = native
            sx, sy = w_img / src.shape[1], h_img / src.shape[0]
            det_in = cv2.cvtColor(src, cv2.COLOR_RGB2BGR) if self.to_bgr else src
            src_h, src_w = src.shape[:2]

            cands = []
            for d in self.detector.detect(det_in) or []:
                x1, y1, x2, y2 = d["box"]
                x1, x2 = max(0.0, x1), min(float(src_w), x2)
                y1, y2 = max(0.0, y1), min(float(src_h), y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                conf = float(d.get("conf", 1.0))
                c = {"x1": x1 * sx, "y1": y1 * sy, "x2": x2 * sx, "y2": y2 * sy, "conf": conf}
                c["cx"], c["cy"] = (c["x1"] + c["x2"]) / 2.0, (c["y1"] + c["y2"]) / 2.0
                c["w"], c["h"] = c["x2"] - c["x1"], c["y2"] - c["y1"]
                if conf < self.conf_threshold:
                    data.n_low_conf += 1
                    if self.show_all:
                        self._draw_box(annotated, c, (120, 120, 120), f"{conf:.2f}", 1)
                    continue
                c["dist_cm"] = min(self.detector.plate_width_cm * self.F_H / c["w"],
                                   self.detector.plate_height_cm * self.F_V / c["h"])
                # 原生畫面的像素是正方形，長寬比在這裡量 (斜看時會變小，記進 log 供之後做正對)
                c["aspect"] = (x2 - x1) / (y2 - y1)
                c["src_box"] = (x1, y1, x2, y2)
                # 後端多給的欄位 (例如 ArUco 的 marker_id、四角點) 原樣帶出去
                c.update({k: v for k, v in d.items() if k not in ("box", "conf")})
                cands.append(c)
            data.n_plates = len(cands)

            best = self._pick(cands)
            if best is not None:
                best["hits"] = self.hits
                best["crop"] = self._crop(det_in, best["src_box"])
                best["sharp"] = sharpness(best["crop"])
                data.plate = best
                data.is_detected = True
                data.cx, data.cy = int(best["cx"]), int(best["cy"])
                data.w, data.h = int(best["w"]), int(best["h"])

            for c in cands:
                if c is best:
                    label = f"PLATE {c['conf']:.2f} d{c['dist_cm']:.0f}cm"
                    self._draw_box(annotated, c, (0, 255, 0), label, 3)
                    cv2.drawMarker(annotated, (int(c["cx"]), int(c["cy"])), (0, 255, 0),
                                   cv2.MARKER_CROSS, 20, 2)
                else:
                    self._draw_box(annotated, c, (255, 255, 0), f"d{c['dist_cm']:.0f}", 1)

        cv2.putText(annotated, self._status_text(data, best), (10, h_img - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        data.annotated_frame = annotated
        return data

    # ==========================================
    # 內部
    # ==========================================
    def _pick(self, cands):
        """已鎖定就只認離上一格最近的 (LOCK_GATE 內)；沒鎖定就挑最大的 (最近的)"""
        if self.lock_cx is not None:
            near = [c for c in cands
                    if abs(c["cx"] - self.lock_cx) <= self.LOCK_GATE
                    and abs(c["cy"] - self.lock_cy) <= self.LOCK_GATE]
            if near:
                best = min(near, key=lambda c: abs(c["cx"] - self.lock_cx) + abs(c["cy"] - self.lock_cy))
                self.lock_cx, self.lock_cy = best["cx"], best["cy"]
                self.lock_miss = 0
                self.hits += 1
                return best
            # 鎖定的那個這一格沒看到：先撐幾格，不要馬上改追別的車牌
            self.lock_miss += 1
            if self.lock_miss <= self.LOCK_MISS_MAX:
                return None
            self._drop_lock()
        if not cands:
            return None
        best = max(cands, key=lambda c: c["w"] * c["h"])
        self.lock_cx, self.lock_cy = best["cx"], best["cy"]
        self.lock_miss = 0
        self.hits = 1
        return best

    def _crop(self, img, box):
        """從偵測用的畫面 (原生、已轉 BGR) 切出車牌，四周多留 crop_margin。回傳複本 (OCR 在背景執行緒讀)"""
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * self.crop_margin, (y2 - y1) * self.crop_margin
        h, w = img.shape[:2]
        x0, y0 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x3, y3 = min(w, int(x2 + mx + 0.5)), min(h, int(y2 + my + 0.5))
        return img[y0:y3, x0:x3].copy()

    @staticmethod
    def _draw_box(img, c, colour, label, thick):
        p1, p2 = (int(c["x1"]), int(c["y1"])), (int(c["x2"]), int(c["y2"]))
        cv2.rectangle(img, p1, p2, colour, thick)
        if label:
            cv2.putText(img, label, (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    def _status_text(self, data, best):
        if not self.available:
            return "NO PLATE MODEL (roam only)"
        s = f"PLATE {'1' if best else '0'}/{data.n_plates}"
        if best:
            s += f"  d{best['dist_cm']:.0f}cm  hits {best['hits']}  sharp {best['sharp']:.0f}"
        if data.n_low_conf:
            s += f"  low-conf {data.n_low_conf}"
        if self.show_all:
            s += "  [SHOW ALL]"
        return s
