import cv2
import numpy as np
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class DepthExplorerVision(VisionProcessor):
    def __init__(self, depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt"):
        print("========================================")
        print("👁️ [系統訊息] 啟動深度視覺 (精準中央取樣/去除天花板與地板干擾)...")
        print("========================================")
        self.depth_model = YOLO(depth_model_path)

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=True, annotated_frame=frame)
        data.depth_L = 999.0
        data.depth_C = 999.0
        data.depth_R = 999.0

        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]

        results = self.depth_model(frame, verbose=False)
        res = results[0]
        depth_map = None

        try:
            if hasattr(res, 'depth') and res.depth is not None:
                # YOLO 輸出通常為公尺 (m)，乘以 100 轉換為公分 (cm)
                depth_map = res.depth.data.cpu().numpy().squeeze() * 100.0
            elif hasattr(res, 'masks') and res.masks is not None:
                depth_map = res.masks.data[0].cpu().numpy().squeeze() * 100.0
        except Exception:
            pass

        if depth_map is not None and isinstance(depth_map, np.ndarray) and len(depth_map.shape) >= 2:
            if depth_map.shape[:2] != (h_img, w_img):
                depth_map = cv2.resize(depth_map, (w_img, h_img))

            # 修正取樣區域：只取畫面高度的 55% ~ 65% (一條極窄的水平雷射帶)
            # 這樣能完美避開天花板與地板的雜訊
            roi_y1 = int(h_img * 0.35)
            roi_y2 = int(h_img * 0.45)
            w3 = w_img // 3

            # 計算左、中、右三個狹長區塊的平均值
            data.depth_L = float(np.mean(depth_map[roi_y1:roi_y2, 0:w3]))
            data.depth_C = float(np.mean(depth_map[roi_y1:roi_y2, w3:2*w3]))
            data.depth_R = float(np.mean(depth_map[roi_y1:roi_y2, 2*w3:w_img]))

            # --- UI 繪製 ---
            # 畫出這三個狹長的取樣框，讓你在畫面上能明確看到無人機在「看」哪裡
            cv2.rectangle(annotated_frame, (0, roi_y1), (w3, roi_y2), (255, 200, 0), 1)
            cv2.rectangle(annotated_frame, (w3, roi_y1), (2*w3, roi_y2), (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (2*w3, roi_y1), (w_img, roi_y2), (255, 200, 0), 1)

            # 顯示文字資訊
            cv2.putText(annotated_frame, f"L: {int(data.depth_L)}cm", (10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            color_c = (0, 255, 0) if data.depth_C > 150 else (0, 0, 255)
            cv2.putText(annotated_frame, f"C: {int(data.depth_C)}cm", (w3 + 10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_c, 2)

            cv2.putText(annotated_frame, f"R: {int(data.depth_R)}cm", (2*w3 + 10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        data.annotated_frame = annotated_frame
        return data