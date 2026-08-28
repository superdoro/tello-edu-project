import cv2
from ultralytics import YOLO
from vision.farneback_detector import FarnebackDetector
from vision.base import VisionProcessor, VisionData

class HybridDetector(VisionProcessor):
    def __init__(self, yolo_path="model/yolo26/runs/detect/yolo26_train4/weights/best.pt", target_id=0):
        self.target_id = target_id
        print("========================================")
        print("🤖 [系統訊息] 啟動極速複合視覺 (自訂 YOLO + 傳統 Farneback)...")
        print("========================================")
        
        self.yolo = YOLO(yolo_path)
        
        # 👉 將原本的 self.raft 改成 self.flow_alg
        self.flow_alg = FarnebackDetector(resize_dim=(128, 128))

    def process_frame(self, frame) -> VisionData:
        # 1. 取得 Farneback 傳統光流資料
        data = self.flow_alg.process_frame(frame)
        
        # 2. 執行 YOLO 辨識 (在原始畫面上尋找)
        results = self.yolo(frame, verbose=False)
        
        target_list = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.5 and int(box.cls[0]) == self.target_id:
                cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
                target_list.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})
                
        # 3. 視覺合成
        annotated_frame = data.annotated_frame
        data.target = None
        
        if target_list:
            target = max(target_list, key=lambda x: x['area'])
            data.target = target
            
            x = target['cx'] - target['w']//2
            y = target['cy'] - target['h']//2
            cv2.rectangle(annotated_frame, (x, y), (x+target['w'], y+target['h']), (0, 255, 255), 3)
            cv2.putText(annotated_frame, "TARGET", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        data.annotated_frame = annotated_frame
        return data