import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from vision.base import VisionProcessor, VisionData
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class RaftDetector(VisionProcessor):
    def __init__(self, resize_dim=(256, 256)):
        """
        :param resize_dim: 為了確保即時性，將影像縮小處理。數值越小越快，但細節越少。
        """
        self.resize_dim = resize_dim

        try:
            # 自動偵測是否可以使用 GPU 加速
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[系統訊息] 正在載入 RAFT 光流模型 (執行環境: {self.device})...")
            if self.device.type == "cpu":
                print("⚠️ [警告] 尚未偵測到 GPU。在 CPU 上執行光流計算可能會有較高延遲！")

            # 載入內建的輕量級 RAFT 模型與預訓練權重
            # self.weights = Raft_Large_Weights.DEFAULT
            # self.model = raft_large(weights=self.weights, progress=False).to(self.device)
            self.weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=self.weights, progress=False).to(self.device)
            print(f"[系統訊息] 完成載入 RAFT 光流模型 (執行環境: {self.device})...")
            self.model.eval() # 設為評估模式
            
        except Exception as e:
            print(f"❌ 發生 raft 錯誤: {e}")
            import traceback
            traceback.print_exc()

        # 使用官方定義的前處理工具
        self.transforms = self.weights.transforms()
        
        # 用來儲存前一個 frame
        self.prev_tensor = None

    # def process_frame(self, frame) -> VisionData:
    #     data = VisionData(is_detected=False, annotated_frame=frame)
        
    #     # 1. 影像預處理：BGR 轉 RGB -> 縮放 -> 轉 Tensor -> 批次化(增加一個維度)
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame_resized = cv2.resize(frame_rgb, self.resize_dim)
        
    #     # 轉換形狀為 (C, H, W)，值域轉為 0~1 的浮點數
    #     curr_tensor = torch.from_numpy(frame_resized.transpose((2, 0, 1))).float() / 255.0
    #     curr_tensor = curr_tensor.unsqueeze(0).to(self.device) # Shape: (1, 3, H, W)
        
    #     # 2. 如果沒有上一幀，先記錄下來並直接回傳原圖
    #     if self.prev_tensor is None:
    #         self.prev_tensor = curr_tensor
    #         return data

    #     # 3. 執行 RAFT 推論
    #     with torch.no_grad(): 
    #         img1, img2 = self.transforms(self.prev_tensor, curr_tensor)
    #         list_of_flows = self.model(img1, img2)
    #         predicted_flow = list_of_flows[-1][0] 

    #     # 取得 CPU 上的光流向量陣列
    #     flow_uv = predicted_flow.cpu().numpy()
        
    #     # 1. 產生彩色熱力圖
    #     flow_map = self._flow_to_bgr(flow_uv)
    #     flow_map_resized = cv2.resize(flow_map, (frame.shape[1], frame.shape[0]))

    #     # ==========================================
    #     # 🔥 新增：光流數值量化與區域切割
    #     # ==========================================
    #     # 計算每個像素的移動強度 (向量長度：√(x^2 + y^2))
    #     u = flow_uv[0, :, :]
    #     v = flow_uv[1, :, :]
    #     magnitude = np.sqrt(u**2 + v**2)
        
    #     # 取得寬度並切分成左、中、右三等份
    #     h, w = magnitude.shape
    #     w_third = w // 3
        
    #     # 計算這三個區域的平均移動強度
    #     # (數值越大 = 該區域物體越靠近 或 相對速度越快)
    #     left_mag = np.mean(magnitude[:, :w_third])
    #     center_mag = np.mean(magnitude[:, w_third:2*w_third])
    #     right_mag = np.mean(magnitude[:, 2*w_third:])
        
    #     # 將數值打包進 VisionData
    #     data.flow_regions = {
    #         "left": left_mag,
    #         "center": center_mag,
    #         "right": right_mag
    #     }
    #     data.is_detected = True

    #     # 在畫面上印出這三個數值，方便你等一下試飛時抓「危險門檻值」
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     fw = frame.shape[1]
    #     cv2.putText(flow_map_resized, f"L: {left_mag:.1f}", (20, 40), font, 1, (255,255,255), 2)
    #     cv2.putText(flow_map_resized, f"C: {center_mag:.1f}", (fw//2 - 40, 40), font, 1, (255,255,255), 2)
    #     cv2.putText(flow_map_resized, f"R: {right_mag:.1f}", (fw - 120, 40), font, 1, (255,255,255), 2)
        
    #     # 將彩色光流圖作為結果畫面
    #     data.annotated_frame = flow_map_resized
    #     self.prev_tensor = curr_tensor
        
    #     return data

    def process_frame(self, frame) -> VisionData:
        data = VisionData(is_detected=False, annotated_frame=frame)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.resize_dim)
        
        curr_tensor = torch.from_numpy(frame_resized.transpose((2, 0, 1))).float() / 255.0
        curr_tensor = curr_tensor.unsqueeze(0).to(self.device) 
        
        if self.prev_tensor is None:
            self.prev_tensor = curr_tensor
            return data

        with torch.no_grad(): 
            img1, img2 = self.transforms(self.prev_tensor, curr_tensor)
            list_of_flows = self.model(img1, img2)
            predicted_flow = list_of_flows[-1][0] 

        flow_uv = predicted_flow.cpu().numpy()
        flow_map = self._flow_to_bgr(flow_uv)
        
        # 取得原圖長寬
        fh, fw = frame.shape[:2]
        flow_map_resized = cv2.resize(flow_map, (fw, fh))

        # ==========================================
        # 🔥 修改：擷取中間的水平帶狀區域 (Region of Interest)
        # ==========================================
        u = flow_uv[0, :, :]
        v = flow_uv[1, :, :]
        magnitude = np.sqrt(u**2 + v**2)
        
        h, w = magnitude.shape
        
        # 設定垂直裁切範圍 (例如：只取畫面中間 50% 的高度，忽略上下各 25%)
        # 你可以根據 Tello 實際的視角稍微上下調整這些比例
        h_start = int(h * 0.25)
        h_end = int(h * 0.75)
        
        w_third = w // 3
        
        # 只取我們感興趣的這個「水平帶狀區域」的陣列
        roi_magnitude = magnitude[h_start:h_end, :]
        
        # 在這個被裁切過的帶狀區域中，計算左、中、右的平均值
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
        # UI 繪製：在畫面上畫出雷射警戒線
        # ==========================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(flow_map_resized, f"L: {left_mag:.1f}", (20, 40), font, 1, (255,255,255), 2)
        cv2.putText(flow_map_resized, f"C: {center_mag:.1f}", (fw//2 - 40, 40), font, 1, (255,255,255), 2)
        cv2.putText(flow_map_resized, f"R: {right_mag:.1f}", (fw - 120, 40), font, 1, (255,255,255), 2)
        
        # 畫出上下兩條綠色警戒線，框出系統實際在看的範圍 (Y 座標對應回原圖)
        draw_y_start = int(fh * 0.25)
        draw_y_end = int(fh * 0.75)
        cv2.line(flow_map_resized, (0, draw_y_start), (fw, draw_y_start), (0, 255, 0), 2)
        cv2.line(flow_map_resized, (0, draw_y_end), (fw, draw_y_end), (0, 255, 0), 2)
        
        data.annotated_frame = flow_map_resized
        self.prev_tensor = curr_tensor
        
        return data

    def _flow_to_bgr(self, flow_uv):
        """將 (2, H, W) 的光流矩陣轉換為視覺化的 OpenCV BGR 影像"""
        u = flow_uv[0, :, :] # X 軸位移
        v = flow_uv[1, :, :] # Y 軸位移
        
        # 將笛卡爾座標 (X, Y) 轉換為極座標 (強度, 角度)
        mag, ang = cv2.cartToPolar(u, v)
        
        # 建立一個空白的 HSV 影像矩陣
        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        
        # 角度對應 H (色相)，強度對應 V (明度)，S (飽和度) 設為最大
        hsv[..., 0] = ang * 180 / np.pi / 2  
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # 轉回 BGR 以供 OpenCV 顯示
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr