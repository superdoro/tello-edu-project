import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData
from utils.dark_region import DarkRegionAnalyzer


class TissueDetector(VisionProcessor):
    """
    衛生紙條偵測器：YOLO 找目標 + YOLO depth 供漫遊避障。

    這個模組同時餵養兩套飛行邏輯，所以輸出分成兩組互不相干的資料：

        data.tissue                        -> 給 TissueChargeControl 對準與衝撞
        data.depth_L/C/R
        data.depth_unreliable / dark_*     -> 給 FluidExploreControl 漫遊避障

    第二組欄位刻意與 fluid_explorer_vision.py 完全同名同單位 (公分)，
    因為 TissueChargeControl 的搜尋階段是直接把指令委派給 FluidExploreControl 的 ——
    同一份視覺資料要能無縫餵進去，那邊調好的 300/250/150 也才能直接沿用。
    """

    def __init__(self,
                 yolo_path="model/yolo26/runs/detect/yolo_tissue_collect/best.pt",
                 depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt",
                 tissue_class_id=0, conf_threshold=0.5,
                 depth_band=(0.35, 0.45)):
        """
        :param yolo_path: 衛生紙條偵測模型
        :param depth_model_path: 深度模型，供漫遊避障使用
        :param tissue_class_id: 紙條在模型中的類別編號
        :param conf_threshold: 信心門檻 (抓不到就調低，誤判多就調高)
        :param depth_band: 深度取樣的水平帶，與 fluid_explorer_vision.py 一致
        """
        print("========================================")
        print("[系統訊息] 啟動衛生紙條獵手視覺 (YOLO 偵測 + 深度避障)...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"[GPU 加速啟動] 模型已載入至 {torch.cuda.get_device_name(0)}！")
        else:
            print("[警告] 未偵測到 CUDA，推論將使用 CPU。")

        # 先確認檔案存在再交給 YOLO()。
        # 直接把不存在的路徑丟進去，ultralytics 可能會當成模型名稱跑去網路上下載 ——
        # 飛行時連的是 Tello 熱點、沒有外網，那個同步下載會卡住主迴圈，
        # Tello 超過約 15 秒收不到指令就會自動降落。
        self.model = None
        if os.path.exists(yolo_path):
            self.model = YOLO(yolo_path)
            self.model.to(self.device)
            print(f"[系統訊息] 紙條模型: {yolo_path}")
        else:
            print(f"[警告] 找不到紙條模型 {yolo_path}")
            print("        -> 本模式只會漫遊避障，不會偵測也不會衝撞。")
            print("        -> 請用 model/yolo26/train_yoly.py 訓練後放到該路徑。")

        self.depth_model = None
        if depth_model_path and os.path.exists(depth_model_path):
            self.depth_model = YOLO(depth_model_path)
            self.depth_model.to(self.device)
        elif depth_model_path:
            print(f"[警告] 找不到深度模型 {depth_model_path} -> 漫遊將沒有避障能力。")
        print("========================================")

        self.tissue_class_id = tissue_class_id
        self.conf_threshold = conf_threshold
        self.depth_band = depth_band

        # 黑區分析：純黑牆面會讓深度模型誤判成「很遠」，漫遊時要知道深度不可信。
        # 與 FLUID EXPLORER 用同一個分析器，行為一致。
        self.dark_analyzer = DarkRegionAnalyzer()

        # F 鍵切換：連低於信心門檻的偵測也畫出來，方便現場調 conf_threshold
        self.show_all = False

        # ---------- 目標黏著 ----------
        # 場地裡通常不只一條紙條。如果每一格都選「面積最大」的那條，兩條大小接近時
        # 選擇會在兩條之間來回跳 —— 實測錄影裡 389 格跳了 33 次 (368 -> 710 -> 325 -> 671)，
        # 無人機的轉向就跟著在 ±60 之間甩來甩去，永遠對不準。
        # 改成：已經鎖定一條就優先選離它最近的，只有它消失夠久才重新挑最大的。
        self.lock_cx = None
        self.lock_miss = 0
        self.LOCK_GATE = 120       # 與上一格目標水平距離在這個範圍內才算同一條
        self.LOCK_MISS_MAX = 15    # 連續這麼多格找不到鎖定的那條，就放棄重新挑

    def toggle_tracking_mode(self):
        """F 鍵：切換「顯示所有偵測 (含低信心)」"""
        self.show_all = not self.show_all
        print(f"[視覺系統] 低信心偵測顯示：{'開啟' if self.show_all else '關閉'}")

    def reset(self):
        """切換飛行模式時清掉黑區分析的平滑狀態與目標鎖定"""
        self.dark_analyzer.reset()
        self.lock_cx = None
        self.lock_miss = 0

    def _pick(self, candidates):
        """
        從候選中選出要追的那一條 (黏著策略，理由見 __init__)。
        """
        if self.lock_cx is not None:
            near = [c for c in candidates if abs(c["cx"] - self.lock_cx) <= self.LOCK_GATE]
            if near:
                best = min(near, key=lambda c: abs(c["cx"] - self.lock_cx))
                self.lock_cx, self.lock_miss = best["cx"], 0
                return best
            # 鎖定的那條這一格不見了：先撐幾格 (可能只是單格漏偵測)，不要馬上換目標
            self.lock_miss += 1
            if self.lock_miss <= self.LOCK_MISS_MAX:
                return None
        # 沒有鎖定，或鎖定的那條消失太久 -> 重新挑面積最大 (最近) 的
        best = max(candidates, key=lambda c: c["area"])
        self.lock_cx, self.lock_miss = best["cx"], 0
        return best

    # ==========================================
    # 深度 (供漫遊避障)
    # ==========================================
    def _depth_map(self, frame):
        """跑 YOLO depth 並取出與原圖同尺寸的深度圖，失敗時回傳 None"""
        if self.depth_model is None:
            return None

        res = self.depth_model(frame, verbose=False, device=self.device)[0]
        depth_map = None
        try:
            if hasattr(res, 'depth') and res.depth is not None:
                depth_map = res.depth.data.cpu().numpy().squeeze() * 100.0
            elif hasattr(res, 'masks') and res.masks is not None:
                depth_map = res.masks.data[0].cpu().numpy().squeeze() * 100.0
        except Exception:
            return None

        if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.ndim < 2:
            return None

        h_img, w_img = frame.shape[:2]
        if depth_map.shape[:2] != (h_img, w_img):
            depth_map = cv2.resize(depth_map, (w_img, h_img))
        return depth_map

    # ==========================================
    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.tissue = None

        # 預設「前方無限遠」：深度模型缺席時，漫遊會退化成直線前進而不是卡住不動
        data.depth_L = data.depth_C = data.depth_R = 999.0

        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]

        # ---------- 黑區分析 (必須在深度判讀之前) ----------
        dark_info = self.dark_analyzer.analyze(frame)
        data.dark_ratio = dark_info["ratio"]
        data.dark_L, data.dark_C, data.dark_R = dark_info["L"], dark_info["C"], dark_info["R"]
        data.depth_unreliable = dark_info["is_dark"]
        self.dark_analyzer.draw(annotated_frame, dark_info, warn_org=(10, 100))

        # ---------- 深度：左/中/右三段 (比照 fluid_explorer_vision) ----------
        depth_map = self._depth_map(frame)
        if depth_map is not None:
            y1 = int(h_img * self.depth_band[0])
            y2 = int(h_img * self.depth_band[1])
            w3 = w_img // 3
            if y2 > y1:
                data.depth_L = float(np.mean(depth_map[y1:y2, 0:w3]))
                data.depth_C = float(np.mean(depth_map[y1:y2, w3:2 * w3]))
                data.depth_R = float(np.mean(depth_map[y1:y2, 2 * w3:w_img]))

                cv2.rectangle(annotated_frame, (0, y1), (w3, y2), (255, 200, 0), 1)
                cv2.rectangle(annotated_frame, (w3, y1), (2 * w3, y2), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (2 * w3, y1), (w_img, y2), (255, 200, 0), 1)
                cv2.putText(annotated_frame, f"L: {int(data.depth_L)}cm", (10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                color_c = (0, 255, 0) if data.depth_C > 150 else (0, 0, 255)
                cv2.putText(annotated_frame, f"C: {int(data.depth_C)}cm", (w3 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_c, 2)
                cv2.putText(annotated_frame, f"R: {int(data.depth_R)}cm", (2 * w3 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---------- YOLO 偵測紙條 ----------
        n_low = 0
        if self.model is not None:
            results = self.model(frame, verbose=False, device=self.device)
            candidates = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) != self.tissue_class_id:
                        continue
                    conf = float(box.conf[0])
                    cx, cy, w, h = [int(v) for v in box.xywh[0].tolist()]
                    if conf < self.conf_threshold:
                        n_low += 1
                        if self.show_all:   # 除錯用：畫出被信心門檻擋掉的框
                            bx, by = cx - w // 2, cy - h // 2
                            cv2.rectangle(annotated_frame, (bx, by), (bx + w, by + h), (120, 120, 120), 1)
                            cv2.putText(annotated_frame, f"{conf:.2f}", (bx, max(12, by - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
                        continue
                    candidates.append({
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "area": w * h, "conf": conf,
                    })

            best = self._pick(candidates) if candidates else None
            if not candidates and self.lock_cx is not None:
                # 整格都沒偵測到也要累計，否則鎖定永遠不會過期
                self.lock_miss += 1
                if self.lock_miss > self.LOCK_MISS_MAX:
                    self.lock_cx = None
            if best:
                data.tissue = best
                data.is_detected = True

                bx, by = best["cx"] - best["w"] // 2, best["cy"] - best["h"] // 2
                cv2.rectangle(annotated_frame, (bx, by), (bx + best["w"], by + best["h"]),
                              (0, 255, 0), 3)
                cv2.drawMarker(annotated_frame, (best["cx"], best["cy"]), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(annotated_frame,
                            f"TISSUE {best['conf']:.2f} A:{best['area']}",
                            (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---------- 狀態列 ----------
        if self.model is None:
            status = "NO TISSUE MODEL (roam only)"
        else:
            status = f"TISSUE {'1' if data.tissue else '0'}"
            if n_low:
                status += f"  low-conf {n_low}"
            if self.show_all:
                status += "  [SHOW ALL]"
        cv2.putText(annotated_frame, status, (10, h_img - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        data.annotated_frame = annotated_frame
        return data
