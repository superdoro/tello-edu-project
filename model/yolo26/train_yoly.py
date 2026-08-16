import os
import torch
from tqdm import tqdm
from ultralytics import YOLO

def setup_epoch_progressbar(model, total_epochs):
    """
    透過 Ultralytics Callback 註冊 tqdm 進度條，顯示整體 Epoch 進度。
    """
    pbar = tqdm(total=total_epochs, desc="🚀 總訓練進度 (Epochs)", unit="epoch")

    def on_train_epoch_end(trainer):
        # 取得當前 epoch 的損失值資訊顯示在進度條上
        loss_info = {}
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            loss_info = {"loss": f"{trainer.loss_items.sum().item():.4f}"}
        pbar.set_postfix(loss_info)
        pbar.update(1)

    def on_train_end(trainer):
        pbar.close()

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

def main():
    # ---------------- 1. 下載資料集 ----------------
    # print("📥 正在透過 Roboflow 下載資料集...")
    # rf = Roboflow(api_key="5DgZZHpEhMCIIE7kQ0QQ")
    # project = rf.workspace("yolo-jqawq").project("ballon-q4wj1")
    # version = project.version(1)
    
    # 格式使用 "yolov8" 即可產出相容 Ultralytics 的 data.yaml
    # dataset = version.download("yolov8")
    # data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    # if not os.path.exists(data_yaml_path):
    #     print(f"❌ 找不到資料集設定檔: {data_yaml_path}")
    #     return

    # print(f"✅ 資料集準備完成，路徑: {data_yaml_path}")

    # ---------------- 2. 載入模型 ----------------
    model_name = "yolo26n.pt"  # 若本地無此特製權重，可改用 "yolov8n.pt" 或 "yolo11n.pt"
    print(f"🤖 正在載入模型: {model_name}...")
    model = YOLO(model_name)

    # ---------------- 3. 設定訓練參數與進度條 ----------------
    EPOCHS = 100
    setup_epoch_progressbar(model, total_epochs=EPOCHS)

    # 自動檢查是否具備 CUDA GPU
    device_to_use = 0 if torch.cuda.is_available() else "cpu"
    print(f"💻 訓練裝置: {'GPU (CUDA:0)' if device_to_use == 0 else 'CPU'}")

    # ---------------- 4. 開始訓練 ----------------
    print("🚀 開始訓練模型...")
    results = model.train(
        data="ballon-1/data.yaml",
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        patience=20,
        name="yolo26_train",
        device=device_to_use,
        verbose=True,  # 啟用內建的 Batch 等級詳細進度條與指標
        workers = 0,
    )
    
    print("\n🎉 訓練完成！")
    print("📁 模型權重已儲存於: runs/detect/yolo26_train/weights/best.pt")

if __name__ == "__main__":
    main()