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

        self.tracking_mode = "chest"

    def toggle_tracking_mode(self):
        self.tracking_mode = "hand" if self.tracking_mode == "chest" else "chest"
        print(f"[視覺系統] 追蹤模式已切換為: {'手部指揮' if self.tracking_mode == 'hand' else '胸腔鎖定'}")

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

                # 確保陣列長度夠長 (至少要有手腕)
                if len(kpts) > 10:
                    nose = kpts[0]
                    left_shoulder = kpts[5]
                    right_shoulder = kpts[6]
                    left_elbow = kpts[7]
                    right_elbow = kpts[8]
                    left_wrist = kpts[9]
                    right_wrist = kpts[10]

                    body_scale = 0

                    # ==========================================
                    # 🔥 第一防線：主軀幹指標 (雙肩/軀幹/頭肩)
                    # ==========================================
                    if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                        mid_sh_x = (left_shoulder[0] + right_shoulder[0]) / 2
                        mid_sh_y = (left_shoulder[1] + right_shoulder[1]) / 2

                        shoulder_width = math.hypot(left_shoulder[0] - right_shoulder[0], 
                                                    left_shoulder[1] - right_shoulder[1])
                        body_scale = max(body_scale, shoulder_width)

                        # 軀幹高度 (肩膀到臀部)
                        if len(kpts) > 12:
                            left_hip, right_hip = kpts[11], kpts[12]
                            if left_hip[2] > 0.5 and right_hip[2] > 0.5:
                                mid_hip_x = (left_hip[0] + right_hip[0]) / 2
                                mid_hip_y = (left_hip[1] + right_hip[1]) / 2
                                torso_height = math.hypot(mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y)
                                body_scale = max(body_scale, torso_height * 0.8)

                        # 頭肩高度
                        if nose[2] > 0.5:
                            neck_height = math.hypot(nose[0] - mid_sh_x, nose[1] - mid_sh_y)
                            body_scale = max(body_scale, neck_height * 1.5)

                    # ==========================================
                    # 🔥 第二防線：前臂長度降級 (視野溢出時的備案)
                    # ==========================================
                    l_forearm = 0
                    if left_elbow[2] > 0.5 and left_wrist[2] > 0.5:
                        l_forearm = math.hypot(left_elbow[0] - left_wrist[0], left_elbow[1] - left_wrist[1])
                        
                    r_forearm = 0
                    if right_elbow[2] > 0.5 and right_wrist[2] > 0.5:
                        r_forearm = math.hypot(right_elbow[0] - right_wrist[0], right_elbow[1] - right_wrist[1])
                        
                    max_forearm = max(l_forearm, r_forearm)
                    if max_forearm > 0:
                        # 將前臂長度放大 1.5 倍，使其數值無縫接軌肩膀寬度
                        body_scale = max(body_scale, max_forearm * 1.1)

                    # 如果連前臂都看不到，直接略過這個目標
                    if body_scale == 0:
                        continue

                    # ==========================================
                    # 決定追蹤準星 (cx, cy) - 加入防呆降級
                    # ==========================================
                    track_cx, track_cy = 0, 0
                    valid_wrists = []
                    if left_wrist[2] > 0.5: valid_wrists.append(left_wrist)
                    if right_wrist[2] > 0.5: valid_wrists.append(right_wrist)

                    if self.tracking_mode == "hand" and valid_wrists:
                        # 找舉得最高的手
                        highest_wrist = min(valid_wrists, key=lambda w: w[1])
                        track_cx, track_cy = int(highest_wrist[0]), int(highest_wrist[1])
                    else:
                        # 胸腔模式，或切換到手部但手腕卻出框時
                        if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                            track_cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
                            track_cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
                        elif valid_wrists:
                            # 胸腔出框，但手還在畫面上，強制降級用手來追蹤
                            highest_wrist = min(valid_wrists, key=lambda w: w[1])
                            track_cx, track_cy = int(highest_wrist[0]), int(highest_wrist[1])
                        else:
                            continue # 都找不到，放棄

                    target_list.append({
                        "cx": track_cx,
                        "cy": track_cy,
                        "body_scale": body_scale, 
                        "keypoints": kpts
                    })

        if target_list:
            best_target = max(target_list, key=lambda x: x['body_scale'])
            data.target = best_target
            data.is_detected = True

            tx, ty = best_target['cx'], best_target['cy']
            scale_val = int(best_target['body_scale'])

            color = (0, 165, 255) if self.tracking_mode == "hand" else (0, 255, 0) 
            marker = cv2.MARKER_SQUARE if self.tracking_mode == "hand" else cv2.MARKER_CROSS

            cv2.drawMarker(annotated_frame, (tx, ty), color, markerType=marker, markerSize=20, thickness=3)
            cv2.circle(annotated_frame, (tx, ty), scale_val//2, color, 2)

            mode_str = "HAND" if self.tracking_mode == "hand" else "CHEST"
            cv2.putText(annotated_frame, f"[{mode_str}] SCALE: {scale_val}px", (tx - 50, ty - scale_val//2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        data.annotated_frame = annotated_frame
        return data