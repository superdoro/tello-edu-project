import cv2
import numpy as np
from vision.base import VisionProcessor, VisionData

class ColorDetector(VisionProcessor):
    def __init__(self):
        # 定義目標 (例如綠色氣球) 的 HSV 範圍
        self.target_lower = np.array([35, 100, 100])
        self.target_upper = np.array([85, 255, 255])
        
        # 定義障礙物 (例如紅色氣球) 的 HSV 範圍
        # (紅色在 HSV 空間中橫跨 0 與 180 兩端，為簡化先用一端)
        self.obs_lower = np.array([0, 30, 70])
        self.obs_upper = np.array([50, 255, 255])

    def process_frame(self, frame):
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.target = None
        data.obstacle = None
        
        # 轉換為 HSV 色彩空間 (對光線變化較有容忍度)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 尋找目標與障礙物的遮罩 (Mask)
        mask_target = cv2.inRange(hsv, self.target_lower, self.target_upper)
        mask_obs = cv2.inRange(hsv, self.obs_lower, self.obs_upper)
        
        # 提取最大輪廓的函數
        def get_largest_object(mask, color, name):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 500: # 面積過濾，忽略雜訊
                    x, y, w, h = cv2.boundingRect(largest)
                    cx, cy = x + w//2, y + h//2
                    # 在畫面上標記
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    return {"cx": cx, "cy": cy, "w": w, "h": h, "area": area}
            return None

        # 取得畫面中的目標與障礙物資訊
        data.target = get_largest_object(mask_target, (0, 255, 0), "TARGET")
        data.obstacle = get_largest_object(mask_obs, (0, 0, 255), "OBSTACLE")
        
        if data.target or data.obstacle:
            data.is_detected = True
            
        data.annotated_frame = frame
        return data

# import cv2
# import numpy as np
# from vision.base import VisionProcessor, VisionData

# class ColorDetector(VisionProcessor):
#     def __init__(self):
#         # 【放寬綠色範圍】調低 Saturation 和 Value 的下限 (100 -> 50)
#         # 這樣即使在較暗的房間或是氣球顏色較淡，也能「大致」抓到綠色
#         self.target_lower = np.array([35, 50, 50])
#         self.target_upper = np.array([85, 255, 255])
        
#         # 【修復紅色範圍】紅色在 OpenCV 的 HSV 中是斷開的 (0~10 與 170~180)
#         # 我們定義兩組範圍，涵蓋所有的紅色系，並降低 S, V 下限增加靈敏度
#         self.obs_lower1 = np.array([0, 70, 50])
#         self.obs_upper1 = np.array([10, 255, 255])
        
#         self.obs_lower2 = np.array([170, 70, 50])
#         self.obs_upper2 = np.array([180, 255, 255])

#     def process_frame(self, frame):
#         data = VisionData(is_detected=False, annotated_frame=frame)
#         data.target = None
#         data.obstacle = None
        
#         # 1. 稍微模糊化影像，消除雜訊 (讓顏色色塊更平滑、更集中)
#         blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
#         # 2. 轉換為 HSV 色彩空間 
#         hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        
#         # 3. 尋找目標 (綠色) 的遮罩
#         mask_target = cv2.inRange(hsv, self.target_lower, self.target_upper)
        
#         # 4. 尋找障礙物 (紅色) 的遮罩 (將兩段紅色區間合併)
#         mask_obs1 = cv2.inRange(hsv, self.obs_lower1, self.obs_upper1)
#         mask_obs2 = cv2.inRange(hsv, self.obs_lower2, self.obs_upper2)
#         mask_obs = cv2.bitwise_or(mask_obs1, mask_obs2) # 把兩段紅色加起來
        
#         # 提取最大輪廓的函數
#         def get_largest_object(mask, color, name):
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if contours:
#                 largest = max(contours, key=cv2.contourArea)
#                 area = cv2.contourArea(largest)
#                 # 【提高距離靈敏度】將面積門檻從 500 降到 300，可以抓到更遠、更小的氣球
#                 if area > 300: 
#                     x, y, w, h = cv2.boundingRect(largest)
#                     cx, cy = x + w//2, y + h//2
#                     # 在畫面上標記
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#                     cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                     return {"cx": cx, "cy": cy, "w": w, "h": h, "area": area}
#             return None

#         # 取得畫面中的目標與障礙物資訊
#         data.target = get_largest_object(mask_target, (0, 255, 0), "TARGET")
#         data.obstacle = get_largest_object(mask_obs, (0, 0, 255), "OBSTACLE")
        
#         if data.target or data.obstacle:
#             data.is_detected = True
            
#         data.annotated_frame = frame
#         return data