import cv2
import numpy as np
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class RunningVision(VisionProcessor):
    def __init__(self, depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt"):
        print("========================================")
        print("🧭 [系統訊息] 啟動室內環繞視覺 (沿牆空間感知)...")
        print("========================================")
        self.depth_model = YOLO(depth_model_path)

    def process_frame(self, frame) -> VisionData:
        # 標記 is_detected 為 True，因為這個模式不需要辨識特定物件，只要有深度就能飛
        data = VisionData(is_detected=True, annotated_frame=frame)
        
        # 預設深度為無限遠
        data.depth_L = 999.0
        data.center_depth = 999.0
        data.depth_R = 999.0

        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]

        # 執行 YOLO Depth
        results = self.depth_model(frame, verbose=False)
        res = results[0]
        depth_map = None
        
        try:
            if hasattr(res, 'depth') and res.depth is not None:
                depth_map = res.depth.data.cpu().numpy().squeeze()
            elif hasattr(res, 'masks') and res.masks is not None:
                depth_map = res.masks.data[0].cpu().numpy().squeeze()
        except Exception as e:
            print(f"[警告] 深度圖提取失敗: {e}")

        # 切割畫面並計算區域平均深度
        if depth_map is not None and isinstance(depth_map, np.ndarray) and len(depth_map.shape) >= 2:
            if depth_map.shape[:2] != (h_img, w_img):
                depth_map = cv2.resize(depth_map, (w_img, h_img))
            
            # 取中間 1/3 到 2/3 的高度帶，避免天花板和地板的雜訊干擾
            strip_h_start, strip_h_end = h_img // 3, 2 * h_img // 3
            
            roi_L = depth_map[strip_h_start:strip_h_end, 0:w_img//3]
            roi_C = depth_map[strip_h_start:strip_h_end, w_img//3:2*w_img//3]
            roi_R = depth_map[strip_h_start:strip_h_end, 2*w_img//3:w_img]
            
            data.depth_L = float(np.mean(roi_L))
            data.center_depth = float(np.mean(roi_C))
            data.depth_R = float(np.mean(roi_R))
            
            # 視覺化 UI 介面：畫出三個區塊的輔助線與深度數值
            cv2.line(annotated_frame, (w_img//3, strip_h_start), (w_img//3, strip_h_end), (0, 255, 0), 2)
            cv2.line(annotated_frame, (2*w_img//3, strip_h_start), (2*w_img//3, strip_h_end), (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"L: {int(data.depth_L)}", (10, strip_h_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(annotated_frame, f"C: {int(data.center_depth)}", (w_img//3 + 10, strip_h_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"R: {int(data.depth_R)}", (2*w_img//3 + 10, strip_h_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        data.annotated_frame = annotated_frame
        return data