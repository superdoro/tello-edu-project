import cv2
import numpy as np
from vision.base import VisionProcessor, VisionData

class FarnebackDetector(VisionProcessor):
    # 👇 這裡新增了 use_roi 開關，預設為 False (不裁切)
    def __init__(self, resize_dim=(128, 128), use_roi=False):
        self.resize_dim = resize_dim
        self.prev_gray = None
        self.use_cuda = False
        self.use_roi = use_roi 
        
        print("========================================")
        print(f"⚙️ [系統訊息] 正在載入傳統光流算法 (Farneback) | ROI 裁切: {'開啟' if self.use_roi else '關閉'}")
        
        # 自動偵測是否具備 OpenCV CUDA 支援
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_cuda = True
                self.optflow = cv2.cuda_FarnebackOpticalFlow.create(
                    numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                    numIters=3, polyN=5, polySigma=1.2, flags=0
                )
                print("✅ OpenCV CUDA 支援已啟用！將使用 GPU 加速傳統光流。")
        except AttributeError:
            print("💡 [提示] 目前安裝的 OpenCV 未包含 CUDA 模組。")
            print("   將自動使用 CPU 高度優化的 C++ 底層運算 (在 128x128 下極快！)")
            
        print("========================================")

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        
        # 1. 影像預處理：傳統算法只需要「灰階」影像，運算量極低
        frame_resized = cv2.resize(frame, self.resize_dim)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return data
            
        # 2. 執行光流計算
        if self.use_cuda:
            # GPU 運算路線
            gpu_prev = cv2.cuda_GpuMat(self.prev_gray)
            gpu_curr = cv2.cuda_GpuMat(gray)
            gpu_flow = self.optflow.calc(gpu_prev, gpu_curr, None)
            flow = gpu_flow.download()
        else:
            # CPU 運算路線 (使用經典推薦參數)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
        # 3. 取得位移向量並計算強度
        u = flow[..., 0]
        v = flow[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        # 4. 繪製彩色熱力圖 (將移動角度與強度轉為 HSV 色彩)
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # 防止出現除以零的錯誤，加上極小值
        _, ang = cv2.cartToPolar(u, v + 1e-5) 
        
        hsv = np.zeros((self.resize_dim[1], self.resize_dim[0], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = mag_norm
        flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 放大回原始尺寸以供顯示
        fh, fw = frame.shape[:2]
        flow_map_resized = cv2.resize(flow_map, (fw, fh))

        # ==========================================
        # 5. 擷取區域：根據開關決定是否只看中間
        # ==========================================
        h, w = magnitude.shape
        
        if self.use_roi:
            h_start, h_end = int(h * 0.25), int(h * 0.75)
        else:
            h_start, h_end = 0, h  # 關閉時，掃描整張畫面 (0 到 100%)
            
        w_third = w // 3
        
        roi_magnitude = magnitude[h_start:h_end, :]
        
        left_mag = np.mean(roi_magnitude[:, :w_third])
        center_mag = np.mean(roi_magnitude[:, w_third:2*w_third])
        right_mag = np.mean(roi_magnitude[:, 2*w_third:])
        
        data.flow_regions = {
            "left": left_mag,
            "center": center_mag,
            "right": right_mag
        }
        data.is_detected = True

        # ==========================================
        # 6. 繪製 UI 數據與警戒線
        # ==========================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(flow_map_resized, f"L: {left_mag:.1f}", (20, 40), font, 1, (255,255,255), 2)
        cv2.putText(flow_map_resized, f"C: {center_mag:.1f}", (fw//2 - 40, 40), font, 1, (255,255,255), 2)
        cv2.putText(flow_map_resized, f"R: {right_mag:.1f}", (fw - 120, 40), font, 1, (255,255,255), 2)
        
        # 只有在開啟 ROI 時，才畫出綠色的水平警戒線
        if self.use_roi:
            draw_y_start, draw_y_end = int(fh * 0.25), int(fh * 0.75)
            cv2.line(flow_map_resized, (0, draw_y_start), (fw, draw_y_start), (0, 255, 0), 2)
            cv2.line(flow_map_resized, (0, draw_y_end), (fw, draw_y_end), (0, 255, 0), 2)
        
        data.annotated_frame = flow_map_resized
        self.prev_gray = gray
        
        return data