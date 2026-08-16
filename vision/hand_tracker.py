import cv2
import mediapipe as mp
from vision.base import VisionProcessor, VisionData

class HandTracker(VisionProcessor):
    def __init__(self, max_hands=1, detection_con=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands, 
                                         min_detection_confidence=detection_con)
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame) -> VisionData:
        # 預設回傳「未偵測到」的資料
        data = VisionData(is_detected=False, annotated_frame=frame)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            # 畫上骨架
            self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            
            # 獲取尺寸與座標
            h, w, c = frame.shape
            x_list = [int(lm.x * w) for lm in hand_lms.landmark]
            y_list = [int(lm.y * h) for lm in hand_lms.landmark]
            
            data.is_detected = True
            data.cx = (min(x_list) + max(x_list)) // 2
            data.cy = (min(y_list) + max(y_list)) // 2
            data.w = max(x_list) - min(x_list)
            data.h = max(y_list) - min(y_list)
            data.annotated_frame = frame
            
        return data