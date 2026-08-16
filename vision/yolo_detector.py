import cv2
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData

class YoloDetector(VisionProcessor):
    def __init__(self, target_class_id=67, obstacle_class_id=64, conf_threshold=0.5):
        """
        :param target_class_id: 目標物的 YOLO 類別 ID (預設 67: 手機)
        :param obstacle_class_id: 障礙物的 YOLO 類別 ID (預設 64: 滑鼠)
        :param conf_threshold: 信心度門檻
        """
        print("[系統訊息] 正在載入 YOLO 預訓練模型...")
        # 預設載入 yolov8n.pt (Nano 模型，最適合 CPU 即時運算)
        self.model = YOLO("yolov8n.pt") 
        
        self.target_class_id = target_class_id
        self.obstacle_class_id = obstacle_class_id
        self.conf_threshold = conf_threshold

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.target = None
        data.obstacle = None
        
        # 執行推論 (verbose=False 避免終端機被洗版)
        results = self.model(frame, verbose=False)
        
        # 使用 Ultralytics 內建的標註功能，直接在畫面上畫出 Bounding Box
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
            # YOLO xywh 格式為 (中心X, 中心Y, 寬度, 高度)
            cx, cy, w, h = [int(val) for val in box.xywh[0].tolist()]
            area = w * h
            
            obj_info = {"cx": cx, "cy": cy, "w": w, "h": h, "area": area}
            
            # 分類打包
            if cls_id == self.target_class_id:
                target_list.append(obj_info)
            elif cls_id == self.obstacle_class_id:
                obstacle_list.append(obj_info)
                
        # 距離判斷：若有多個同類物件，選取在畫面中面積最大 (距離最近) 的作為主要判斷依據
        if target_list:
            data.target = max(target_list, key=lambda x: x['area'])
        if obstacle_list:
            data.obstacle = max(obstacle_list, key=lambda x: x['area'])
            
        if data.target or data.obstacle:
            data.is_detected = True
            
        data.annotated_frame = annotated_frame
        return data