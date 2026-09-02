import cv2
import numpy as np
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class DroneDetector(VisionProcessor):
    def __init__(self, yolo_model_path="model/yolo26/runs/detect/yolo_drone_collect/best.pt"):
        print("========================================")
        print("[系統訊息] 啟動無人機獵手視覺模組...")
        print("========================================")
        
        self.yolo_drone = YOLO(yolo_model_path)
        self.drone_class_id = 0

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.drone = None
        
        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2] 

        # ==========================================
        # 步驟 A: 第一階段 - 尋找無人機 (Detection)
        # ==========================================
        results_drone = self.yolo_drone(frame, verbose=False)
        drones = []
        
        if results_drone and results_drone[0].boxes is not None:
            for box in results_drone[0].boxes:
                conf = float(box.conf[0])
                if conf > 0.5 and int(box.cls[0]) == self.drone_class_id:
                    cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
                    drones.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})

        # ==========================================
        # 步驟 B: 第二階段 - 選擇最佳無人機 (Selection)
        # ==========================================
        if drones:
            best_drone = max(drones, key=lambda x: x['area'])
            data.drone = best_drone
            data.is_detected = True
            
            bx = max(0, best_drone['cx'] - best_drone['w'] // 2)
            by = max(0, best_drone['cy'] - best_drone['h'] // 2)
            bx_end = min(w_img, bx + best_drone['w'])
            by_end = min(h_img, by + best_drone['h'])
            
            cv2.rectangle(annotated_frame, (bx, by), (bx_end, by_end), (0, 255, 0), 2)

        data.annotated_frame = annotated_frame
        return data