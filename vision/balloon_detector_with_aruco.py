import cv2
import numpy as np
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class BalloonDetector(VisionProcessor):
    def __init__(self, 
                 yolo_path="model/yolo26/runs/detect/yolo_ballon_collect/best.pt", 
                 depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt",
                 balloon_class_id=0):
        print("========================================")
        print("[系統訊息] 啟動氣球獵手視覺 (含 ArUco 補償 & YOLO Depth 空間感知)...")
        print("========================================")
        
        self.model = YOLO(yolo_path)
        self.balloon_class_id = balloon_class_id

        self.depth_model = YOLO(depth_model_path) if depth_model_path else None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            self.aruco_detector = None 

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.balloon = None
        data.marker_id = None
        
        # 預設深度為 999 (代表無限遠/安全)
        data.center_depth = 999.0
        data.depth_L = 999.0
        data.depth_R = 999.0

        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]

        # ==========================================
        # 🔥 步驟 0: YOLO Depth 深度提取 (修復維度錯誤)
        # ==========================================
        depth_map = None
        if self.depth_model is not None:
            results_depth = self.depth_model(frame, verbose=False)
            res = results_depth[0]
            
            try:
                # 嘗試從官方的 depth 屬性提取 (針對特製的 YOLO Depth 模型)
                if hasattr(res, 'depth') and res.depth is not None:
                    depth_map = res.depth.data.cpu().numpy().squeeze()
                # 嘗試從 masks 屬性提取 (許多魔改深度模型會將深度存在 Mask 通道)
                elif hasattr(res, 'masks') and res.masks is not None:
                    depth_map = res.masks.data[0].cpu().numpy().squeeze()
            except Exception as e:
                print(f"[警告] 深度圖提取失敗: {e}")
                depth_map = None

            # 🔥 雙重防呆：確保 depth_map 絕對是 Numpy 陣列，且至少是二維 (2D)
            if depth_map is not None and isinstance(depth_map, np.ndarray) and len(depth_map.shape) >= 2:
                # 確保 depth_map 尺寸與原圖一致
                if depth_map.shape[:2] != (h_img, w_img):
                    depth_map = cv2.resize(depth_map, (w_img, h_img))
                
                # 取 1/4 到 3/4 高度的水平帶，切分成左中右
                strip_h_start, strip_h_end = h_img // 4, 3 * h_img // 4
                
                roi_L = depth_map[strip_h_start:strip_h_end, 0:w_img//3]
                roi_C = depth_map[strip_h_start:strip_h_end, w_img//3:2*w_img//3]
                roi_R = depth_map[strip_h_start:strip_h_end, 2*w_img//3:w_img]
                
                data.depth_L = float(np.mean(roi_L))
                data.center_depth = float(np.mean(roi_C))
                data.depth_R = float(np.mean(roi_R))
                
                # 在畫面上方印出左中右深度
                cv2.putText(annotated_frame, f"L:{int(data.depth_L)} C:{int(data.center_depth)} R:{int(data.depth_R)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # 如果模型回傳的格式不符，強制清空，避免影響後續切片
                depth_map = None

        # ==========================================
        # 步驟 A: ArUco 標記偵測
        # ==========================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)

        # ==========================================
        # 步驟 B: YOLO 氣球偵測
        # ==========================================
        results = self.model(frame, verbose=False)
        balloons = []
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.5 and int(box.cls[0]) == self.balloon_class_id:
                cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
                balloons.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w*h, "marker_id": None, "is_virtual": False})

        # ==========================================
        # 步驟 C: 感測器融合與盲區補償
        # ==========================================
        if ids is not None:
            flat_ids = ids.flatten() 
            
            for i, corner in enumerate(corners):
                c = corner[0]
                xs, ys = [pt[0] for pt in c], [pt[1] for pt in c]
                marker_cx, marker_cy = int(sum(xs) / 4), int(sum(ys) / 4)
                marker_w, marker_h = max(xs) - min(xs), max(ys) - min(ys)
                
                m_id = int(flat_ids[i])
                matched = False
                
                for b in balloons:
                    if (b['cx'] - b['w']//2) < marker_cx < (b['cx'] + b['w']//2) and \
                       (b['cy'] - b['h']//2) < marker_cy < (b['cy'] + b['h']//2):
                        b['marker_id'] = m_id
                        matched = True
                        break
                
                if not matched:
                    est_w, est_h = int(marker_w * 2.8), int(marker_h * 3.2)
                    balloons.append({
                        "cx": marker_cx, "cy": marker_cy, 
                        "w": est_w, "h": est_h, "area": est_w * est_h,
                        "marker_id": m_id, "is_virtual": True 
                    })

        # ==========================================
        # 步驟 D: 最佳目標選擇與深度整合
        # ==========================================
        if balloons:
            best_balloon = max(balloons, key=lambda x: x['area'])
            
            bx, by = best_balloon['cx'] - best_balloon['w']//2, best_balloon['cy'] - best_balloon['h']//2
            bx_end, by_end = bx + best_balloon['w'], by + best_balloon['h']
            
            best_balloon['depth'] = 999.0
            if depth_map is not None:
                # 確保邊界不會超出畫面
                crop_y1, crop_y2 = max(0, by), min(h_img, by_end)
                crop_x1, crop_x2 = max(0, bx), min(w_img, bx_end)
                
                if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                    balloon_depth_roi = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
                    best_balloon['depth'] = float(np.mean(balloon_depth_roi))

            data.balloon = best_balloon
            data.is_detected = True
            data.marker_id = best_balloon['marker_id']
            
            # --- UI 繪製 ---
            if best_balloon['is_virtual']:
                cv2.rectangle(annotated_frame, (bx, by), (bx_end, by_end), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "VIRT", (bx, by-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (bx, by), (bx_end, by_end), (0, 255, 0), 2)
            
            if data.marker_id is not None:
                color = (0, 255, 255) if best_balloon['is_virtual'] else (0, 0, 255)
                cv2.putText(annotated_frame, f"ID: {data.marker_id}", (bx + 60, by-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                            
            if depth_map is not None:
                cv2.putText(annotated_frame, f"D: {int(best_balloon['depth'])}", (bx, by-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        data.annotated_frame = annotated_frame
        return data