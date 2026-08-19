# import cv2
# from ultralytics import YOLO
# from vision.base import VisionProcessor, VisionData

# class HandTracker(VisionProcessor):
#     def __init__(self, target_class_id=0, obstacle_class_id=-1, conf_threshold=0.65, yolo_path="prototype\model\yolo26\runs\detect\yolo26_train7_hand\weights\best.pt"):
#         """
#         :param target_class_id: 目標物的 YOLO 類別 ID (預設 67: 手機)
#         :param obstacle_class_id: 障礙物的 YOLO 類別 ID (預設 64: 滑鼠)
#         :param conf_threshold: 信心度門檻
#         """
#         print("[系統訊息] 正在載入 YOLO 預訓練模型...")
#         self.model = YOLO(yolo_path) 
        
#         self.target_class_id = target_class_id
#         self.obstacle_class_id = obstacle_class_id
#         self.conf_threshold = conf_threshold

#     def process_frame(self, frame) -> VisionData:
#         data = VisionData(is_detected=False, annotated_frame=frame)
#         data.target = None
#         data.obstacle = None
        
#         # 執行推論 (verbose=False 避免終端機被洗版)
#         results = self.model(frame, verbose=False)
        
#         # 使用 Ultralytics 內建的標註功能，直接在畫面上畫出 Bounding Box
#         annotated_frame = results[0].plot()
        
#         target_list = []
#         obstacle_list = []
        
#         # 遍歷所有偵測到的物件
#         boxes = results[0].boxes
#         for box in boxes:
#             conf = float(box.conf[0])
#             if conf < self.conf_threshold:
#                 continue
                
#             cls_id = int(box.cls[0])
#             # YOLO xywh 格式為 (中心X, 中心Y, 寬度, 高度)
#             cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
#             area = w * h
            
#             obj_info = {"cx": cx, "cy": cy, "w": w, "h": h, "area": area}
            
#             # 分類打包
#             if cls_id == self.target_class_id:
#                 target_list.append(obj_info)
#             elif cls_id == self.obstacle_class_id:
#                 obstacle_list.append(obj_info)
                
#         # 距離判斷：若有多個同類物件，選取在畫面中面積最大 (距離最近) 的作為主要判斷依據
#         if target_list:
#             data.target = max(target_list, key=lambda x: x['area'])
#         if obstacle_list:
#             data.obstacle = max(obstacle_list, key=lambda x: x['area'])
            
#         if data.target or data.obstacle:
#             data.is_detected = True
            
#         data.annotated_frame = annotated_frame
#         return data

import cv2
import math
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class HandTracker(VisionProcessor):
    def __init__(self, target_class_id=0, obstacle_class_id=-1, conf_threshold=0.5, yolo_path=r"model/yolo26/runs/detect/yolo26_train_hand/weights/best.pt"):
        """
        :param target_class_id: 目標物的 YOLO 類別 ID 
        :param obstacle_class_id: 障礙物的 YOLO 類別 ID 
        :param conf_threshold: 信心度門檻
        """
        print("[系統訊息] 正在載入 YOLO 預訓練模型 (具備目標鎖定功能)...")
        self.model = YOLO(yolo_path) 
        
        self.target_class_id = target_class_id
        self.obstacle_class_id = obstacle_class_id
        self.conf_threshold = conf_threshold
        
        # ==========================================
        # 🔥 目標鎖定專用變數
        # ==========================================
        self.use_lock = False        # 鎖定開關 (預設為關閉)
        self.locked_target = None    # 記錄當前鎖定目標的座標與尺寸

    def toggle_lock(self):
        """切換目標鎖定開關"""
        self.use_lock = not self.use_lock
        if not self.use_lock:
            self.locked_target = None # 關閉時清空記憶
        print(f"🔒 [視覺系統] 目標鎖定模式: {'開啟' if self.use_lock else '關閉'}")

    def reset_target(self):
        """放棄當前鎖定的目標，下一幀將重新鎖定畫面中最大的目標"""
        self.locked_target = None
        print("🔄 [視覺系統] 已重置鎖定目標，將重新搜尋最大目標！")

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.target = None
        data.obstacle = None
        
        # 執行推論 (verbose=False 避免終端機被洗版)
        results = self.model(frame, verbose=False)
        
        # 使用 Ultralytics 內建的標註功能畫出所有 Bounding Box
        annotated_frame = results[0].plot()
        
        target_list = []
        obstacle_list = []
        
        # 遍歷所有偵測到的物件
        boxes = results[0].boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
                
            cls_id = int(box.cls[0])
            cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
            area = w * h
            
            obj_info = {"cx": cx, "cy": cy, "w": w, "h": h, "area": area}
            
            if cls_id == self.target_class_id:
                target_list.append(obj_info)
            elif cls_id == self.obstacle_class_id:
                obstacle_list.append(obj_info)
                
        # ==========================================
        # 🔥 目標選擇邏輯 (鎖定機制)
        # ==========================================
        if target_list:
            if self.use_lock:
                if self.locked_target is None:
                    # 【狀態 1】鎖定模式開啟，但還沒鎖定任何人：抓取畫面中最大的目標
                    best_target = max(target_list, key=lambda x: x['area'])
                else:
                    # 【狀態 2】已經有鎖定目標：找出距離「上一幀位置」最近的目標 (歐式距離)
                    last_cx, last_cy = self.locked_target['cx'], self.locked_target['cy']
                    best_target = min(target_list, key=lambda x: math.hypot(x['cx'] - last_cx, x['cy'] - last_cy))
                
                # 更新記憶，並將其設為最終目標
                self.locked_target = best_target
                data.target = best_target
            else:
                # 【狀態 3】鎖定模式關閉：永遠只抓畫面中最大的目標 (舊有邏輯)
                self.locked_target = None
                data.target = max(target_list, key=lambda x: x['area'])
                
        else:
            # 如果畫面上沒有任何目標，保留上一幀的記憶，等待它再次出現
            pass 

        # ==========================================
        # 障礙物處理 (保持原樣，永遠看最大的)
        # ==========================================
        # if obstacle_list:
        #     data.obstacle = max(obstacle_list, key=lambda x: x['area'])
            
        if data.target or data.obstacle:
            data.is_detected = True
            
        # ==========================================
        # UI 繪製：在鎖定的目標上畫出「追蹤準星」
        # ==========================================
        if data.target and self.use_lock:
            tx, ty = data.target['cx'], data.target['cy']
            tw, th = data.target['w'], data.target['h']
            
            # 畫出顯眼的紅色準星與鎖定文字
            cv2.circle(annotated_frame, (tx, ty), 5, (0, 0, 255), -1)
            cv2.rectangle(annotated_frame, (tx - tw//2, ty - th//2), (tx + tw//2, ty + th//2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, "LOCKED", (tx - tw//2, ty - th//2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
        # 在畫面左上角顯示當前鎖定狀態
        status_text = "LOCK: ON" if self.use_lock else "LOCK: OFF"
        color = (0, 255, 0) if self.use_lock else (150, 150, 150)
        cv2.putText(annotated_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        data.annotated_frame = annotated_frame
        return data