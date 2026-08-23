import cv2
import numpy as np
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class BalloonDetector(VisionProcessor):
    def __init__(self, yolo_path="model/yolo26/runs/detect/yolo26_train4/weights/best.pt", balloon_class_id=0):
        """
        :param yolo_path: YOLO 模型路徑 (建議用你自己訓練的氣球模型)
        :param balloon_class_id: 氣球在 YOLO 裡的類別 ID
        """
        print("========================================")
        print("[系統訊息] 啟動氣球獵手視覺模組 (YOLO + ArUco)...")
        print("========================================")
        
        # 1. 載入 YOLO (負責找氣球的外輪廓與中心點)
        self.model = YOLO(yolo_path)
        self.balloon_class_id = balloon_class_id

        # 2. 初始化 ArUco 字典 (負責讀取精準 ID)
        # 使用 4X4_50 字典 (方塊最大，在遠處最容易被看清楚，支援 0~49 號)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # OpenCV 版本相容性處理
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            self.aruco_detector = None # 舊版 OpenCV 降級處理

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.balloon = None
        data.marker_id = None
        
        annotated_frame = frame.copy()

        # ==========================================
        # 步驟 A: 執行 ArUco 標記偵測 (極速)
        # ==========================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # 在畫面上畫出所有找到的 ArUco 邊框 (方便除錯)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)

        # ==========================================
        # 步驟 B: 執行 YOLO 氣球偵測
        # ==========================================
        results = self.model(frame, verbose=False)
        balloons = []
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.5 and int(box.cls[0]) == self.balloon_class_id:
                cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
                balloons.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})

        # ==========================================
        # 步驟 C: 空間關聯與資訊打包
        # ==========================================
        if balloons:
            # 優先處理畫面中最大的氣球 (距離最近的)
            best_balloon = max(balloons, key=lambda x: x['area'])
            data.balloon = best_balloon
            data.is_detected = True
            
            bx = best_balloon['cx'] - best_balloon['w']//2
            by = best_balloon['cy'] - best_balloon['h']//2
            
            # 畫出氣球的綠色 YOLO 框
            cv2.rectangle(annotated_frame, (bx, by), (bx+best_balloon['w'], by+best_balloon['h']), (0, 255, 0), 2)
            
            # 檢查是否有任何 ArUco 標籤位於這個氣球的框框「內部」
            if ids is not None:
                for i, corner in enumerate(corners):
                    # 計算 ArUco 標籤的中心點
                    c = corner[0]
                    marker_cx = int((c[0][0] + c[2][0]) / 2)
                    marker_cy = int((c[0][1] + c[2][1]) / 2)
                    
                    # 碰撞測試：標籤中心點是否在氣球框內？
                    if (bx < marker_cx < bx + best_balloon['w']) and (by < marker_cy < by + best_balloon['h']):
                        data.marker_id = int(ids[i][0])
                        # 在畫面上顯示大大的紅色 ID
                        cv2.putText(annotated_frame, f"ID: {data.marker_id}", (bx, by-15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        break # 找到了就跳出迴圈

        data.annotated_frame = annotated_frame
        return data