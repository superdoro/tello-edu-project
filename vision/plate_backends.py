"""
PLATE READER 的可替換後端：車牌偵測器與 OCR。

真正的車牌偵測模型與 OCR 之後才會提供，所以飛行邏輯 (behaviors/plate_read.py)
與視覺管線 (vision/plate_detector.py) 只依賴這裡的兩個介面。
換成提供的模型時，實作 detect() / read() 之後在 core/tello_app.py 換掉那一行就好。

在那之前用 ArUco 標籤當假車牌 (ArucoPlateDetector + ArucoIdOCR)，
接近、停穩、讀取整個流程現在就能實飛測試。

⚠️ 色彩順序：djitellopy 2.5 給的畫面其實是 RGB (np.array(frame.to_image()))，
   而 OpenCV、OCR 套件、ultralytics 都假設 BGR。PlateVision 預設會先轉成 BGR 再交給後端，
   所以後端一律當成一般的 cv2 BGR 影像處理即可 (見 PlateVision 的 to_bgr)。
"""

import os
from abc import ABC, abstractmethod

import cv2


class PlateDetectorBackend(ABC):
    """
    車牌偵測器介面。

    plate_width_cm / plate_height_cm 是車牌的實際尺寸，PlateVision 用它從框的大小估距離。
    預設是台灣小客車牌 (32 x 15 cm)；機車牌、縮小列印的測試車牌要改成實際量到的值。
    """
    plate_width_cm = 32.0
    plate_height_cm = 15.0
    available = True          # False = 模型沒載入 (畫面顯示 NO PLATE MODEL，只會漫遊)

    @abstractmethod
    def detect(self, frame) -> list:
        """
        :param frame: BGR 影像 (通常是 Tello 原生 960x720)
        :return: [{"box": (x1, y1, x2, y2), "conf": float}, ...]，座標是 frame 的像素座標。
                 沒有偵測到就回空 list。這個函式在主迴圈裡呼叫，要快 (< 30ms)。
        """


class PlateOCRBackend(ABC):
    """OCR 介面。在 PlateReader 的背景執行緒裡呼叫，可以慢 (幾百 ms)，不會卡住飛控。"""

    @abstractmethod
    def read(self, crop) -> tuple:
        """
        :param crop: BGR 車牌截圖 (框外多留了一點邊界)
        :return: (號碼字串, 信心 0~1)；讀不出來回 (None, 0.0)

        ⚠️ 不可以連網下載模型：飛行時電腦連的是 Tello 熱點，沒有外網。
           需要權重的 OCR 套件請在有網路時先下載好，建構時就載入。
        """


# ==========================================
# 測試替身：ArUco 標籤當假車牌
# ==========================================
class ArucoPlateDetector(PlateDetectorBackend):
    """
    把 ArUco 標籤 (DICT_4X4_50，與 BALLOON HUNT / utils/generate_aruco.py 相同) 當成車牌。

    :param plate_size_cm: 標籤黑框的實際邊長。A4 滿版印 utils/all_markers_A4.png 時約 8cm，
                          請實際量過再填 —— 估距 (= 停在多遠) 完全靠這個值。
                          8cm 的標籤在合成畫面上約 2.5 公尺外就偵測不到 (實飛會更近)，想讓漫遊找得到請單張印大一點。
    """

    def __init__(self, plate_size_cm=8.0):
        self.plate_width_cm = plate_size_cm
        self.plate_height_cm = plate_size_cm
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())
        print(f"[車牌辨識] 偵測器：ArUco 替身 (DICT_4X4_50，邊長 {plate_size_cm:.0f}cm)")

    def detect(self, frame) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            out.append({"box": (float(x1), float(y1), float(x2), float(y2)), "conf": 1.0,
                        "marker_id": int(i)})
        return out


class ArucoIdOCR(PlateOCRBackend):
    """
    ArUco 替身的「OCR」：在截圖上重新偵測一次標籤，回傳 "ID<編號>"。
    不是直接沿用偵測器的編號 —— 轉機頭的動態模糊 (約 15px 以上) 或切掉一截就讀不到，
    跟真的 OCR 一樣會考驗截圖品質 (輕微的糊 ArUco 還讀得出來，比 OCR 寬鬆)。
    """

    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())

    def read(self, crop) -> tuple:
        if crop is None or crop.size == 0:
            return None, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None, 0.0
        return f"ID{int(ids.flatten()[0])}", 1.0


# ==========================================
# 提供的模型如果是 ultralytics 的 .pt，可以直接用這個
# ==========================================
class YoloPlateDetector(PlateDetectorBackend):
    """
    ultralytics YOLO 車牌偵測。

    :param plate_class_ids: 要收下的類別編號 (int 或一組)，None = 所有類別都是車牌
    """

    def __init__(self, yolo_path, conf_threshold=0.5, plate_class_ids=None, imgsz=960,
                 plate_width_cm=32.0, plate_height_cm=15.0):
        import torch
        from ultralytics import YOLO

        self.plate_width_cm = plate_width_cm
        self.plate_height_cm = plate_height_cm
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        if plate_class_ids is None or isinstance(plate_class_ids, (set, list, tuple)):
            self.class_ids = set(plate_class_ids) if plate_class_ids is not None else None
        else:
            self.class_ids = {plate_class_ids}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 先確認檔案存在再交給 YOLO()：不存在的路徑 ultralytics 可能會當成模型名稱跑去網路上下載，
        # 飛行時沒有外網，那個同步下載會卡住主迴圈 (同 TissueDetector)
        self.model = None
        if os.path.exists(yolo_path):
            self.model = YOLO(yolo_path)
            self.model.to(self.device)
            print(f"[車牌辨識] 偵測器：{yolo_path}")
        else:
            print(f"[警告] 找不到車牌模型 {yolo_path} -> 本模式只會漫遊避障，不會偵測車牌。")

    @property
    def available(self):
        return self.model is not None

    def detect(self, frame) -> list:
        if self.model is None:
            return []
        res = self.model(frame, imgsz=self.imgsz, verbose=False, device=self.device)
        out = []
        if res and res[0].boxes is not None:
            for box in res[0].boxes:
                if self.class_ids is not None and int(box.cls[0]) not in self.class_ids:
                    continue
                out.append({"box": tuple(float(v) for v in box.xyxy[0].tolist()),
                            "conf": float(box.conf[0])})
        return out


def sharpness(crop, width=200):
    """
    截圖的清晰度：縮到固定寬度後的 Laplacian 變異數 (越大越清晰)。
    先縮到同一個寬度，遠近不同的車牌才比得起來。
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if w <= 0 or h <= 0:
        return 0.0
    gray = cv2.resize(gray, (width, max(1, int(round(h * width / w)))),
                      interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
