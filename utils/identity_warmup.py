"""
身分辨識模型的預熱腳本。

飛行時無人機連的是 Tello 熱點、沒有外網，而 torch 第一次使用預訓練模型會去下載權重，
在同步主迴圈裡下載等同讓飛控卡死。所以務必先在有網路的環境執行這支腳本一次：

    python utils/identity_warmup.py

之後權重會留在 ~/.cache/torch，飛行時就不需要網路了。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    print("=" * 50)
    print(f"torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}")
    print("=" * 50)

    from vision.identity.embedders import FaceEmbedder, BodyEmbedder

    print("\n[1/2] 下載並載入人臉模型 (約 107MB)...")
    from facenet_pytorch import InceptionResnetV1
    InceptionResnetV1(pretrained='vggface2').eval()

    print("[2/2] 下載並載入體態模型 (約 10MB)...")
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()

    print("\n驗證實際的抽取器...")
    import numpy as np
    face, body = FaceEmbedder(), BodyEmbedder()
    if not (face.available() and body.available()):
        print("預熱失敗：模型無法載入")
        return 1

    fv = face.embed_batch([np.zeros((160, 160, 3), np.uint8)])
    bv = body.embed_batch([np.zeros((160, 128, 3), np.uint8)])
    print(f"  人臉向量 {fv.shape}，長度 {np.linalg.norm(fv[0]):.4f}")
    print(f"  體態向量 {bv.shape}，長度 {np.linalg.norm(bv[0]):.4f}")
    print(f"  裝置：{face.device} (fp16={face.fp16})")
    print("\n預熱完成，飛行時不再需要網路。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
