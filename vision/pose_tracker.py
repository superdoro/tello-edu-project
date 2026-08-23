import cv2
import math
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class BodyPoseTracker(VisionProcessor):
    def __init__(self, yolo_path="model/yolo26/runs/detect/yolo_pose_collect/yolo26n-pose.pt", target_class_id=0, conf_threshold=0.6):
        print("========================================")
        print("[系統訊息] 啟動人體姿態跟追...")
        print("========================================")
        self.model = YOLO(yolo_path) 
        self.target_class_id = target_class_id
        self.conf_threshold = conf_threshold

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.target = None
        
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        target_list = []
        
        if results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes
            keypoints = results[0].keypoints.data 
            
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.conf_threshold or int(box.cls[0]) != self.target_class_id:
                    continue
                
                kpts = keypoints[i].tolist()
                
                # 確保偵測到了足夠的節點 (至少要有肩膀: 節點 5 和 6)
                if len(kpts) > 6:
                    left_shoulder = kpts[5]
                    right_shoulder = kpts[6]
                    
                    # 必須確保肩膀節點的信心度夠高 (索引 2 是該節點的 confidence)
                    if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                        # 1. 計算「胸腔中心點」(最穩定的追蹤錨點)
                        chest_cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
                        chest_cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
                        
                        # 2. 計算「雙肩像素寬度」(用來取代 Area，作為極度穩定的距離指標)
                        shoulder_width = math.hypot(left_shoulder[0] - right_shoulder[0], 
                                                    left_shoulder[1] - right_shoulder[1])
                        
                        target_list.append({
                            "chest_cx": chest_cx,
                            "chest_cy": chest_cy,
                            "shoulder_width": shoulder_width,
                            "keypoints": kpts
                        })
                
        # 尋找距離最近 (肩膀最寬) 的人作為主目標
        if target_list:
            best_target = max(target_list, key=lambda x: x['shoulder_width'])
            data.target = best_target
            data.is_detected = True
            
            # UI 繪製：在胸腔中心畫一個明顯的綠色十字星
            tx, ty = best_target['chest_cx'], best_target['chest_cy']
            sw = int(best_target['shoulder_width'])
            
            cv2.drawMarker(annotated_frame, (tx, ty), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
            cv2.circle(annotated_frame, (tx, ty), sw//2, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"W: {sw}px", (tx - 30, ty - sw//2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        data.annotated_frame = annotated_frame
        return data