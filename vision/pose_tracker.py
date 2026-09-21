import cv2
import math
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class BodyPoseTracker(VisionProcessor):
    def __init__(self, yolo_path="model/yolo26/runs/detect/yolo_pose_collect/yolo26n-pose.pt", target_class_id=0, conf_threshold=0.6):
        print("========================================")
        print("[系統訊息] 啟動人體姿態跟追 (主動對齊正臉)...")
        print("========================================")
        self.model = YOLO(yolo_path) 
        self.target_class_id = target_class_id
        self.conf_threshold = conf_threshold
        self.tracking_mode = "chest"

        # ==========================================
        # ByteTrack 目標鎖定 (按 R 鍵切換)
        #
        # ByteTrack 只做「逐格關聯」：用 IoU + Kalman 把這一格的偵測接上一格的軌跡，
        # 不跑額外的神經網路 (實測約 0.2ms/格)。它能撐過短暫遮擋與兩人交會，
        # 但人離開畫面超過 track_buffer (預設 30 格 ≈ 1 秒) 再回來就會換新 ID ——
        # 那是關聯的極限，要認回同一個人得靠 vision/identity 的特徵庫。
        #
        # locked_id 為 None 時行為與加入 ByteTrack 之前完全相同：追畫面中最大的人。
        # ==========================================
        self.tracker_cfg = "bytetrack.yaml"
        self.locked_id = None
        self.lock_lost_frames = 0
        self.LOCK_LOST_LIMIT = 90   # 鎖定目標連續消失這麼多格 (約 3 秒) 就自動解除，避免無限懸停
        self.visible_ids = []       # 這一格看得到的 track id (已排序)，供 R 鍵循環

    def reset_target(self):
        """
        R 鍵：在「未鎖定 -> 鎖定第1人 -> 鎖定第2人 -> ... -> 未鎖定」之間循環。

        依 track id 由小到大排序，順序在人沒離開畫面的期間是穩定的。
        """
        ids = list(self.visible_ids)
        if not ids:
            self.locked_id = None
            print("[目標鎖定] 畫面中沒有可鎖定的人，維持未鎖定 (追最大的人)")
            return

        if self.locked_id is None or self.locked_id not in ids:
            nxt = ids[0]
        else:
            i = ids.index(self.locked_id) + 1
            nxt = ids[i] if i < len(ids) else None

        self.locked_id = nxt
        self.lock_lost_frames = 0
        if nxt is None:
            print(f"[目標鎖定] 解除鎖定 (回到追最大的人)")
        else:
            print(f"[目標鎖定] 已鎖定 ID {nxt}  (畫面中可選: {ids})")

    def reset(self):
        """切換飛行模式時清掉鎖定，避免切回來還咬著一個早就不在的 ID"""
        self.locked_id = None
        self.lock_lost_frames = 0

    def toggle_tracking_mode(self):
        self.tracking_mode = "hand" if self.tracking_mode == "chest" else "chest"
        print(f"[視覺系統] 追蹤模式已切換為: {'手部指揮' if self.tracking_mode == 'hand' else '胸腔鎖定'}")

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.target = None

        # persist=True 讓 ByteTrack 跨格保留軌跡狀態 (不加的話每格都會重新編號)
        results = self.model.track(frame, persist=True, tracker=self.tracker_cfg, verbose=False)
        annotated_frame = results[0].plot()

        target_list = []

        if results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes
            # ByteTrack 的 id。偵測剛出現、還沒被確認成軌跡時會是 None，此時退回沒有 id 的行為。
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            keypoints = results[0].keypoints.data 

            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.conf_threshold or int(box.cls[0]) != self.target_class_id:
                    continue

                kpts = keypoints[i].tolist()

                if len(kpts) > 10:
                    nose = kpts[0]
                    left_shoulder = kpts[5]
                    right_shoulder = kpts[6]
                    left_elbow = kpts[7]
                    right_elbow = kpts[8]
                    left_wrist = kpts[9]
                    right_wrist = kpts[10]

                    body_scale = 0
                    face_yaw_error = 0 

                    if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                        mid_sh_x = (left_shoulder[0] + right_shoulder[0]) / 2
                        mid_sh_y = (left_shoulder[1] + right_shoulder[1]) / 2

                        shoulder_width = math.hypot(left_shoulder[0] - right_shoulder[0], 
                                                    left_shoulder[1] - right_shoulder[1])
                        body_scale = max(body_scale, shoulder_width)
                        
                        # ==========================================
                        # 主動正臉對齊演算法
                        # 利用鼻子到左右肩膀的 2D 投影距離差，計算轉身角度
                        # ==========================================
                        if nose[2] > 0.5:
                            # 鏡頭中：使用者的左肩(5)在畫面右側，右肩(6)在畫面左側
                            dist_L = left_shoulder[0] - nose[0] 
                            dist_R = nose[0] - right_shoulder[0]
                            
                            # 正規化誤差 (約為 -1.0 到 1.0)，並放大為 0~100 的整數區間供 PID 使用
                            pixel_w = max(1.0, abs(left_shoulder[0] - right_shoulder[0]))
                            face_yaw_error = ((dist_L - dist_R) / pixel_w) * 100

                        if len(kpts) > 12:
                            left_hip, right_hip = kpts[11], kpts[12]
                            if left_hip[2] > 0.5 and right_hip[2] > 0.5:
                                mid_hip_x = (left_hip[0] + right_hip[0]) / 2
                                mid_hip_y = (left_hip[1] + right_hip[1]) / 2
                                torso_height = math.hypot(mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y)
                                body_scale = max(body_scale, torso_height * 0.8)

                        if nose[2] > 0.5:
                            neck_height = math.hypot(nose[0] - mid_sh_x, nose[1] - mid_sh_y)
                            body_scale = max(body_scale, neck_height * 1.5)

                    # 前臂降級 (略...)
                    l_forearm = 0
                    if left_elbow[2] > 0.5 and left_wrist[2] > 0.5:
                        l_forearm = math.hypot(left_elbow[0] - left_wrist[0], left_elbow[1] - left_wrist[1])
                    r_forearm = 0
                    if right_elbow[2] > 0.5 and right_wrist[2] > 0.5:
                        r_forearm = math.hypot(right_elbow[0] - right_wrist[0], right_elbow[1] - right_wrist[1])

                    max_forearm = max(l_forearm, r_forearm)
                    if max_forearm > 0:
                        body_scale = max(body_scale, max_forearm * 1.1)

                    if body_scale == 0:
                        continue

                    track_cx, track_cy = 0, 0
                    valid_wrists = []
                    if left_wrist[2] > 0.5: valid_wrists.append(left_wrist)
                    if right_wrist[2] > 0.5: valid_wrists.append(right_wrist)

                    if self.tracking_mode == "hand" and valid_wrists:
                        highest_wrist = min(valid_wrists, key=lambda w: w[1])
                        track_cx, track_cy = int(highest_wrist[0]), int(highest_wrist[1])
                    else:
                        if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                            track_cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
                            track_cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
                        elif valid_wrists:
                            highest_wrist = min(valid_wrists, key=lambda w: w[1])
                            track_cx, track_cy = int(highest_wrist[0]), int(highest_wrist[1])
                        else:
                            continue

                    target_list.append({
                        "cx": track_cx,
                        "cy": track_cy,
                        "body_scale": body_scale,
                        "face_yaw_error": face_yaw_error, # 輸出給大腦
                        "keypoints": kpts,
                        "track_id": ids[i] if i < len(ids) else None
                    })

        # 這一格看得到的 id，供 R 鍵循環用 (排序後順序穩定)
        self.visible_ids = sorted(t['track_id'] for t in target_list if t['track_id'] is not None)

        best_target = None
        if self.locked_id is not None:
            # 鎖定中：只認那個 id，找不到就懸停等它回來 (超過上限才自動解除)
            best_target = next((t for t in target_list if t['track_id'] == self.locked_id), None)
            if best_target is None:
                self.lock_lost_frames += 1
                if self.lock_lost_frames > self.LOCK_LOST_LIMIT:
                    print(f"[目標鎖定] ID {self.locked_id} 消失超過 {self.LOCK_LOST_LIMIT} 格 -> 自動解除鎖定")
                    self.locked_id = None
                    self.lock_lost_frames = 0
            else:
                self.lock_lost_frames = 0

        # 未鎖定 (或剛自動解除)：維持原本的行為，追畫面中最大的人
        if best_target is None and self.locked_id is None and target_list:
            best_target = max(target_list, key=lambda x: x['body_scale'])

        if best_target:
            data.target = best_target
            data.is_detected = True

            tx, ty = best_target['cx'], best_target['cy']
            scale_val = int(best_target['body_scale'])
            yaw_err = int(best_target['face_yaw_error'])

            color = (0, 165, 255) if self.tracking_mode == "hand" else (0, 255, 0) 
            marker = cv2.MARKER_SQUARE if self.tracking_mode == "hand" else cv2.MARKER_CROSS

            cv2.drawMarker(annotated_frame, (tx, ty), color, markerType=marker, markerSize=20, thickness=3)
            cv2.circle(annotated_frame, (tx, ty), scale_val//2, color, 2)

            mode_str = "HAND" if self.tracking_mode == "hand" else "CHEST"
            tid = best_target.get('track_id')
            tag = f"LOCK#{tid}" if self.locked_id is not None else (f"ID{tid}" if tid is not None else "ID-")
            cv2.putText(annotated_frame, f"[{mode_str}] {tag} SCALE: {scale_val} YAW: {yaw_err}", (tx - 80, ty - scale_val//2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 鎖定狀態列：讓飛手知道 R 鍵會切到哪些人
        if self.locked_id is not None:
            lost = f" LOST {self.lock_lost_frames}" if self.lock_lost_frames else ""
            cv2.putText(annotated_frame, f"LOCKED ID {self.locked_id}{lost}  (R: next)", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.visible_ids:
            cv2.putText(annotated_frame, f"IDs {self.visible_ids}  (R: lock)", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        data.annotated_frame = annotated_frame
        return data