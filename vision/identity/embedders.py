"""
特徵抽取器：把對齊好的人臉與裁切好的人體轉成單位長度的特徵向量。

這是整個身分辨識裡唯一需要 torch 與 GPU 的模組，因此也是唯一需要處理
VRAM 不足與權重快取的地方 —— 把這些防禦程式碼集中在一個檔案比散落各處好維護。

兩個重要的安全考量：
1. 絕不在飛行中下載權重。飛行時連的是 Tello 熱點、沒有外網，一旦發起下載就會卡住
   整個同步主迴圈，Tello 超過約 15 秒收不到指令會自動降落。所以載入前先檢查快取檔。
2. VRAM 不足不能讓飛控崩潰。OOM 時自動退回 CPU 繼續運作，其他例外則停用身分辨識，
   讓無人機退回「跟隨最靠近的人」的行為。
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from vision.identity.preprocess import hsv_stripe_hist

# CNN 特徵與色彩直方圖的相對權重。
# ImageNet 特徵是通用紋理特徵，同人與異人的 cosine 全擠在高位的窄帶裡；
# 色彩直方圖對「今天穿什麼」極度敏感，兩者合併才有足夠的鑑別力。
W_CNN, W_HIST = 0.80, 0.60

FACE_DIM = 512
BODY_CNN_DIM = 576
BODY_HIST_DIM = 144
BODY_DIM = BODY_CNN_DIM + BODY_HIST_DIM


def _pick_device(device):
    if device is not None:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class _BaseEmbedder:
    NAME = "embedder"

    def __init__(self, device=None, fp16=True):
        self.device = _pick_device(device)
        self.fp16 = fp16 and self.device.startswith('cuda')
        self.model = None
        self.failed = False
        self._load()

    def available(self):
        return self.model is not None and not self.failed

    def _load(self):
        path = self.weight_path()
        if path and not os.path.exists(path):
            # 權重不在快取裡：這時候載入會觸發下載，飛行中等同讓主迴圈卡死
            print(f"[身分辨識] 找不到 {self.NAME} 的權重快取：{path}")
            print(f"[身分辨識] 請先在有網路的環境執行：python utils/identity_warmup.py")
            self.failed = True
            return
        try:
            model = self.build()
            model.eval().to(self.device)
            if self.fp16:
                model.half()
            self.model = model
            self.warmup()
        except Exception as e:
            print(f"[身分辨識] {self.NAME} 載入失敗：{e}")
            self.failed = True

    def warmup(self):
        """先跑一次空推論。CUDA 第一次呼叫要花 100~200ms 建立 kernel，
        這段延遲如果發生在飛行中，就是一次無謂的控制指令空窗。"""
        try:
            h, w = self.warmup_shape()
            dummy = np.zeros((1, h, w, 3), dtype=np.uint8)
            self.embed_batch([dummy[0]])
        except Exception:
            pass

    def _forward(self, x):
        try:
            with torch.inference_mode():
                return self.model(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.device = 'cpu'
            self.fp16 = False
            self.model.float().to('cpu')
            print(f"[身分辨識] VRAM 不足，{self.NAME} 已永久改用 CPU 運算（會變慢）")
            with torch.inference_mode():
                return self.model(x.float().to('cpu'))

    def _to_tensor(self, batch_np):
        x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous()
        x = x.to(self.device)
        return x.half() if self.fp16 else x.float()


class FaceEmbedder(_BaseEmbedder):
    NAME = "人臉模型 (facenet vggface2)"
    DIM = FACE_DIM

    def warmup_shape(self):
        return 160, 160

    def weight_path(self):
        try:
            from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
            return os.path.join(get_torch_home(), 'checkpoints', '20180402-114759-vggface2.pt')
        except Exception:
            return None

    def build(self):
        from facenet_pytorch import InceptionResnetV1
        return InceptionResnetV1(pretrained='vggface2')

    def embed_batch(self, chips):
        """chips: list of 160x160x3 RGB uint8 -> (N, 512) 單位向量"""
        if not chips or not self.available():
            return None
        try:
            arr = np.stack(chips).astype(np.float32)
            arr = (arr - 127.5) / 128.0          # facenet 的 fixed_image_standardization
            x = self._to_tensor(arr)
            out = self._forward(x)
            return F.normalize(out.float(), p=2, dim=1).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[身分辨識] 人臉特徵抽取失敗：{e}")
            self.failed = True
            return None


class BodyEmbedder(_BaseEmbedder):
    NAME = "體態模型 (mobilenet_v3_small)"
    DIM = BODY_DIM

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def warmup_shape(self):
        from vision.identity.preprocess import BODY_OUT_WH
        return BODY_OUT_WH[1], BODY_OUT_WH[0]

    def weight_path(self):
        try:
            from torchvision.models import MobileNet_V3_Small_Weights
            url = MobileNet_V3_Small_Weights.IMAGENET1K_V1.url
            return os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(url))
        except Exception:
            return None

    def build(self):
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # 只取卷積主幹，不要 classifier：1000 類的 ImageNet 分數對「是不是同一個人」幾乎沒鑑別力
        return torch.nn.Sequential(net.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())

    def embed_batch(self, crops):
        """crops: list of HxWx3 RGB uint8 (尺寸相同) -> (N, 720) 單位向量"""
        if not crops or not self.available():
            return None
        try:
            arr = np.stack(crops).astype(np.float32) / 255.0
            arr = (arr - self.IMAGENET_MEAN) / self.IMAGENET_STD
            x = self._to_tensor(arr)
            out = self._forward(x)
            cnn = F.normalize(out.float(), p=2, dim=1).cpu().numpy().astype(np.float32)

            hist = np.stack([hsv_stripe_hist(c) for c in crops]).astype(np.float32)

            fused = np.concatenate([W_CNN * cnn, W_HIST * hist], axis=1)
            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            return (fused / np.maximum(norms, 1e-8)).astype(np.float32)
        except Exception as e:
            print(f"[身分辨識] 體態特徵抽取失敗：{e}")
            self.failed = True
            return None
