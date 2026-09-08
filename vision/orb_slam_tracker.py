import cv2
import time
import math
import numpy as np

import orb_slam3 
from vision.base import VisionProcessor, VisionData

class OrbSlamTracker(VisionProcessor):
    def __init__(self, vocab_path="Vocabulary/ORBvoc.txt", settings_path="tello.yaml",
                 process_every_n_frames=2, enable_viewer=True):
        print("========================================")
        print("[系統訊息] 啟動 ORB-SLAM3...")
        print("========================================")

        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self.frame_index = 0
        self.last_pose = None
        self.last_tracking_state = 0

        # Pangolin 視窗會額外消耗資源；需要地圖視覺化時再傳入 True。
        self.slam = orb_slam3.ORB_SLAM3(
            vocab_path, settings_path, "MONOCULAR", enable_viewer
        )
        self.start_time = time.time()

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.x, data.y, data.z, data.yaw = self.last_pose or (0.0, 0.0, 0.0, 0.0)
        data.tracking_state = self.last_tracking_state

        annotated_frame = frame.copy()

        should_process = self.frame_index % self.process_every_n_frames == 0
        self.frame_index += 1

        if should_process:
            timestamp = time.time() - self.start_time
            try:
                pose = self.slam.TrackMonocular(frame, timestamp)

                if pose is not None and len(pose) > 0:
                    pose_matrix = np.asarray(pose).reshape(4, 4)
                    R = pose_matrix[0:3, 0:3]
                    yaw_rad = math.atan2(R[2, 0], R[2, 2])
                    self.last_pose = (
                        float(pose_matrix[0, 3]),
                        float(pose_matrix[1, 3]),
                        float(pose_matrix[2, 3]),
                        math.degrees(yaw_rad),
                    )
                    self.last_tracking_state = 1
                else:
                    self.last_tracking_state = 0
            except Exception:
                self.last_tracking_state = 0

        data.x, data.y, data.z, data.yaw = self.last_pose or (0.0, 0.0, 0.0, 0.0)
        data.is_detected = self.last_tracking_state == 1
        data.tracking_state = self.last_tracking_state

        if data.is_detected:
            cv2.putText(annotated_frame, f"SLAM [TRACKING] X:{data.x:.2f} Z:{data.z:.2f}",
                        (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "SLAM [LOST / INIT]", (40, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        data.annotated_frame = annotated_frame
        return data

    def shutdown(self):
        print("儲存地圖並安全關閉 SLAM...")
        self.slam.Shutdown()