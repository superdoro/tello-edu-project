import cv2
import numpy as np
import torch  # 新增 PyTorch 模組
from ultralytics import YOLO
from vision.base import VisionProcessor, VisionData
from utils.dark_region import DarkRegionAnalyzer

class DepthExplorerVision(VisionProcessor):
    def __init__(self, depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt"):
        print("========================================")
        print("[系統訊息] 啟動深度視覺...")
        
        # 偵測並啟用 NVIDIA GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU 加速啟動] YOLO 模型已載入至 {gpu_name}！")
        else:
            print("[警告] 未偵測到 CUDA，YOLO 仍將使用 CPU 運算。請檢查 PyTorch 安裝！")
        print("========================================")
        
        # 載入模型並強制搬移到指定的硬體 (GPU/CPU)
        self.depth_model = YOLO(depth_model_path)
        self.depth_model.to(self.device)

        # 黑區分析器：量化畫面中的純黑比例，用來判斷深度值可不可信。
        # 純黑牆面缺乏紋理，單目深度模型常常把它推論成「很遠」，
        # 飛行策略若照單全收就會直接撞上去。
        self.dark_analyzer = DarkRegionAnalyzer()

    def reset(self):
        """重置黑區平滑狀態 (切換模式或重新起飛時呼叫)"""
        self.dark_analyzer.reset()

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=True, annotated_frame=frame)
        data.depth_L = 999.0
        data.depth_C = 999.0
        data.depth_R = 999.0

        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]

        # ==========================================
        # 黑區分析 (必須在深度判讀之前做完，讓後續決策知道深度可不可信)
        #
        # 註：analyze() 內部走的是 BGR->HSV。Tello 的畫面實際上是 RGB，
        #     但這裡只取用 V (明度) 與 S (飽和度) 兩個通道，兩者都不受
        #     R/B 通道對調影響，因此不需要額外轉換。
        # ==========================================
        dark_info = self.dark_analyzer.analyze(frame)
        data.dark_ratio = dark_info["ratio"]
        data.dark_L = dark_info["L"]
        data.dark_C = dark_info["C"]
        data.dark_R = dark_info["R"]
        data.depth_unreliable = dark_info["is_dark"]

        # 先畫黑區遮罩，讓後面的深度取樣框疊在上層不被蓋住。
        # 警告文字往下挪到 y=100，避開 TelloApp 畫在 (10, 30) 的模式名稱。
        self.dark_analyzer.draw(annotated_frame, dark_info, warn_org=(10, 100))

        # 🔥 核心修改：在推理時明確指定 device
        results = self.depth_model(frame, verbose=False, device=self.device)
        res = results[0]
        depth_map = None
        
        # ... 以下保留你原本的 depth_map 解析與繪圖邏輯 ...
        try:
            if hasattr(res, 'depth') and res.depth is not None:
                depth_map = res.depth.data.cpu().numpy().squeeze() * 100.0
            elif hasattr(res, 'masks') and res.masks is not None:
                depth_map = res.masks.data[0].cpu().numpy().squeeze() * 100.0
        except Exception:
            pass

        if depth_map is not None and isinstance(depth_map, np.ndarray) and len(depth_map.shape) >= 2:
            if depth_map.shape[:2] != (h_img, w_img):
                depth_map = cv2.resize(depth_map, (w_img, h_img))

            # 修正取樣區域：只取畫面高度的 55% ~ 65%
            roi_y1 = int(h_img * 0.35)
            roi_y2 = int(h_img * 0.45)
            w3 = w_img // 3

            # 計算左、中、右三個狹長區塊的平均值
            data.depth_L = float(np.mean(depth_map[roi_y1:roi_y2, 0:w3]))
            data.depth_C = float(np.mean(depth_map[roi_y1:roi_y2, w3:2*w3]))
            data.depth_R = float(np.mean(depth_map[roi_y1:roi_y2, 2*w3:w_img]))

            # --- UI 繪製 ---
            cv2.rectangle(annotated_frame, (0, roi_y1), (w3, roi_y2), (255, 200, 0), 1)
            cv2.rectangle(annotated_frame, (w3, roi_y1), (2*w3, roi_y2), (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (2*w3, roi_y1), (w_img, roi_y2), (255, 200, 0), 1)

            cv2.putText(annotated_frame, f"L: {int(data.depth_L)}cm", (10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            color_c = (0, 255, 0) if data.depth_C > 150 else (0, 0, 255)
            cv2.putText(annotated_frame, f"C: {int(data.depth_C)}cm", (w3 + 10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_c, 2)
            cv2.putText(annotated_frame, f"R: {int(data.depth_R)}cm", (2*w3 + 10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        data.annotated_frame = annotated_frame
        return data