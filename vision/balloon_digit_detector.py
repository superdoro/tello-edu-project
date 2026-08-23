# import cv2
# from ultralytics import YOLO
# from vision.base import VisionProcessor, VisionData

# class BalloonDigitDetector(VisionProcessor):
#     def __init__(self, balloon_model_path="model/yolo26/runs/detect/yolo_ballon_collect/best.pt", digit_model_path="model/yolo26/runs/detect/yolo_digit_collect/best.pt", balloon_class_id=0):
#         """
#         :param balloon_model_path: 負責找氣球的偵測模型 (Object Detection)
#         :param digit_model_path: 負責看數字的分類模型 (Image Classification)
#         :param balloon_class_id: 氣球在 balloon_model 中的類別 ID
#         """
#         print("========================================")
#         print("🎈 [系統訊息] 啟動純 AI 氣球獵手視覺模組 (偵測 + 分類 雙階段架構)...")
#         print("========================================")
        
#         # 1. 載入尋找氣球的偵測模型 (.boxes)
#         self.yolo_balloon = YOLO(balloon_model_path)
#         self.balloon_class_id = balloon_class_id
        
#         # 2. 載入數字的分類模型 (.probs)
#         self.yolo_digit = YOLO(digit_model_path)

#     def process_frame(self, frame) -> VisionData:
#         data = VisionData(is_detected=False, annotated_frame=frame)
#         data.balloon = None
#         data.marker_id = None
        
#         annotated_frame = frame.copy()
#         h_img, w_img = frame.shape[:2] # 取得原圖長寬，用於防呆

#         # ==========================================
#         # 步驟 A: 第一階段 - 尋找氣球 (Object Detection)
#         # ==========================================
#         results_balloon = self.yolo_balloon(frame, verbose=False)
#         balloons = []
        
#         # 確認模型成功回傳偵測框
#         if results_balloon and results_balloon[0].boxes is not None:
#             for box in results_balloon[0].boxes:
#                 conf = float(box.conf[0])
#                 if conf > 0.5 and int(box.cls[0]) == self.balloon_class_id:
#                     cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
#                     balloons.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})

#         # ==========================================
#         # 步驟 B: 第二階段 - 影像裁切與分類 (Classification)
#         # ==========================================
#         if balloons:
#             # 優先處理畫面中面積最大的氣球
#             best_balloon = max(balloons, key=lambda x: x['area'])
#             data.balloon = best_balloon
#             data.is_detected = True
            
#             # 1. 計算邊界，並確保座標沒有超出圖片範圍 (防止裁切時程式崩潰)
#             bx = max(0, best_balloon['cx'] - best_balloon['w'] // 2)
#             by = max(0, best_balloon['cy'] - best_balloon['h'] // 2)
#             bx_end = min(w_img, bx + best_balloon['w'])
#             by_end = min(h_img, by + best_balloon['h'])
            
#             # 畫出氣球的綠色 YOLO 框
#             cv2.rectangle(annotated_frame, (bx, by), (bx_end, by_end), (0, 255, 0), 2)
            
#             # 2. 裁切氣球區域 (ROI)
#             if (bx_end > bx) and (by_end > by):
#                 balloon_crop = frame[by:by_end, bx:bx_end]
                
#                 # 3. 將裁切後的小圖片，傳入分類模型進行推論
#                 results_digit = self.yolo_digit(balloon_crop, verbose=False)
                
#                 # 4. 讀取分類結果 (分類模型的結果會存放在 .probs 屬性中)
#                 if results_digit and results_digit[0].probs is not None:
#                     probs = results_digit[0].probs
#                     top1_id = probs.top1            # 取得信心度最高的類別 ID (即 0~9)
#                     top1_conf = float(probs.top1conf) # 該類別的信心度
                    
#                     # 門檻設定：分類信心度大於 0.6 才認定讀取成功
#                     if top1_conf > 0.6:
#                         data.marker_id = top1_id
#                         # 在畫面上方標示 AI 辨識出的數字與信心度
#                         cv2.putText(annotated_frame, f"ID: {data.marker_id} ({top1_conf:.2f})", 
#                                     (bx, by-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
#                 else:
#                     # 防呆提示：如果放錯模型，在畫面上印出警告
#                     cv2.putText(annotated_frame, "ERR: NOT CLS MODEL", (10, 50), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         data.annotated_frame = annotated_frame
#         return data

import cv2
import numpy as np
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class BalloonDigitDetector(VisionProcessor):
    def __init__(self, balloon_model_path="model/yolo26/runs/detect/yolo_ballon_collect/best.pt", digit_model_path="model/yolo26/runs/detect/yolo_digit_collect/best.pt", balloon_class_id=0):
        print("========================================")
        print("🎈 [系統訊息] 啟動純 AI 氣球獵手視覺模組 (含空白氣球抗噪機制)...")
        print("========================================")
        
        self.yolo_balloon = YOLO(balloon_model_path)
        self.balloon_class_id = balloon_class_id
        self.yolo_digit = YOLO(digit_model_path)

    def _is_blank_balloon(self, crop_img) -> bool:
        """
        利用 OpenCV 檢查裁切下來的氣球影像是否為「空白」
        原理：數字會產生強烈的對比邊緣。如果整張圖的邊緣像素極少，代表它是空白氣球。
        """
        # 轉為灰階
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # 使用 Canny 演算法尋找輪廓邊緣
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        
        # 計算邊緣像素佔整張圖片的比例
        total_pixels = crop_img.shape[0] * crop_img.shape[1]
        edge_pixels = cv2.countNonZero(edges)
        
        # 避免除以零
        if total_pixels == 0: return True
        
        edge_ratio = edge_pixels / total_pixels
        
        # 如果邊緣像素小於 1.5% (這個閾值你可以根據實測反光程度微調)，視為空白
        if edge_ratio < 0.015:
            return True
        return False

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.balloon = None
        data.marker_id = None
        
        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2] 

        # ==========================================
        # 步驟 A: 第一階段 - 尋找氣球 (Object Detection)
        # ==========================================
        results_balloon = self.yolo_balloon(frame, verbose=False)
        balloons = []
        
        if results_balloon and results_balloon[0].boxes is not None:
            for box in results_balloon[0].boxes:
                conf = float(box.conf[0])
                if conf > 0.5 and int(box.cls[0]) == self.balloon_class_id:
                    cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
                    balloons.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})

        # ==========================================
        # 步驟 B: 第二階段 - 影像裁切與分類 (Classification)
        # ==========================================
        if balloons:
            best_balloon = max(balloons, key=lambda x: x['area'])
            data.balloon = best_balloon
            data.is_detected = True
            
            bx = max(0, best_balloon['cx'] - best_balloon['w'] // 2)
            by = max(0, best_balloon['cy'] - best_balloon['h'] // 2)
            bx_end = min(w_img, bx + best_balloon['w'])
            by_end = min(h_img, by + best_balloon['h'])
            
            cv2.rectangle(annotated_frame, (bx, by), (bx_end, by_end), (0, 255, 0), 2)
            
            if (bx_end > bx) and (by_end > by):
                balloon_crop = frame[by:by_end, bx:bx_end]
                
                # 🔥 新增：在送給 AI 前，先用 OpenCV 檢查是不是空白氣球
                if self._is_blank_balloon(balloon_crop):
                    cv2.putText(annotated_frame, "BLANK", (bx, by-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                else:
                    # 不是空白的，才讓分類模型上場
                    results_digit = self.yolo_digit(balloon_crop, verbose=False)
                    
                    if results_digit and results_digit[0].probs is not None:
                        probs = results_digit[0].probs
                        top1_id = probs.top1            
                        top1_conf = float(probs.top1conf) 
                        
                        # 🔥 新增：因為 MNIST 很容易瞎猜，所以我們把門檻從 0.6 拉高到 0.8
                        if top1_conf > 0.8:
                            data.marker_id = top1_id
                            cv2.putText(annotated_frame, f"ID: {data.marker_id} ({top1_conf:.2f})", 
                                        (bx, by-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    else:
                        cv2.putText(annotated_frame, "ERR: NOT CLS MODEL", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        data.annotated_frame = annotated_frame
        return data