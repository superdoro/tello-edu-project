"""
身分辨識子系統。

以「人臉 + 體態」兩種外觀特徵建立目標的特徵庫 (gallery)，
每一格影像即時抽取特徵 (probe) 與特徵庫比對 (query)，判斷畫面中哪一個人是鎖定目標。

模組分工：
    preprocess.py  純 numpy/cv2：人臉對齊、人體裁切、品質量測 (不需要 torch)
    embedders.py   唯一需要 torch/GPU 的地方：人臉與體態的特徵抽取器
    gallery.py     純 numpy：樣本、分箱配額、汰換、持久化
    enroller.py    純 numpy：按一次 R 之後的自動登錄狀態機
    tracker.py     門面：抽特徵 -> 比對決策 -> 自動登錄 -> 自動存檔

注意：本檔案只做延遲轉接，不在 import 時載入 torch 或 cv2，
      讓 gallery / enroller 的邏輯可以在沒有 GPU 的環境單獨測試。
"""

from vision.identity.tracker import IdentityTracker

__all__ = ["IdentityTracker"]
