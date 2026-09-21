import cv2
import numpy as np


class DarkRegionAnalyzer:
    """
    黑區分析器。

    YOLO depth 這類單張影像深度模型在面對純黑 (低反射、無紋理) 的牆面時，
    因為畫面缺乏明暗與紋理線索，推論出來的距離往往是錯的 (常常誤判成很遠)。
    這個分析器負責量化「畫面中有多少比例是黑的」，讓飛行策略能在深度資料
    不可信時改用其他脫困動作 (旋轉、上升)，而不是直接撞上去。
    """

    def __init__(self, value_thresh: int = 50, sat_thresh: int = 90,
                 roi_top: float = 0.20, roi_bottom: float = 0.80,
                 ema_alpha: float = 0.4, unreliable_ratio: float = 0.35):
        """
        :param value_thresh: HSV 明度低於此值視為黑色像素
        :param sat_thresh: 飽和度上限，避免把「暗但有顏色」的物體 (深藍布幕等) 也算進來
        :param roi_top / roi_bottom: 取樣區域的上下邊界 (畫面高度比例)
        :param ema_alpha: 指數平滑係數，抑制單張影像的雜訊造成狀態抖動
        :param unreliable_ratio: 平滑後的黑區比例超過此值就判定深度不可信
        """
        self.value_thresh = value_thresh
        self.sat_thresh = sat_thresh
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.ema_alpha = ema_alpha
        self.unreliable_ratio = unreliable_ratio

        # 平滑後的黑區比例 (0.0 ~ 1.0)
        self.dark_ratio = 0.0

    def reset(self):
        """重置平滑狀態 (切換模式或重新起飛時呼叫)"""
        self.dark_ratio = 0.0

    def analyze(self, frame):
        """
        分析單張影像的黑區分佈。

        :return: dict 內容為
                 ratio    : 平滑後的整體黑區比例
                 raw      : 這張影像的原始黑區比例
                 L / C / R: 左中右三個區塊各自的黑區比例
                 is_dark  : 深度資料是否已因黑區過多而不可信
                 mask     : ROI 內的黑色像素遮罩 (供 UI 疊圖使用)
                 roi      : (y1, y2) 取樣區域範圍
        """
        h_img, w_img = frame.shape[:2]
        y1 = int(h_img * self.roi_top)
        y2 = int(h_img * self.roi_bottom)
        w3 = w_img // 3

        hsv = cv2.cvtColor(frame[y1:y2], cv2.COLOR_BGR2HSV)
        v_ch = hsv[:, :, 2]
        s_ch = hsv[:, :, 1]
        mask = (v_ch < self.value_thresh) & (s_ch < self.sat_thresh)

        raw_ratio = float(mask.mean()) if mask.size else 0.0
        a = self.ema_alpha
        self.dark_ratio = a * raw_ratio + (1.0 - a) * self.dark_ratio

        return {
            "ratio": self.dark_ratio,
            "raw": raw_ratio,
            "L": float(mask[:, 0:w3].mean()) if mask.size else 0.0,
            "C": float(mask[:, w3:2 * w3].mean()) if mask.size else 0.0,
            "R": float(mask[:, 2 * w3:w_img].mean()) if mask.size else 0.0,
            "is_dark": self.dark_ratio >= self.unreliable_ratio,
            "mask": mask,
            "roi": (y1, y2),
        }

    @staticmethod
    def draw(annotated_frame, info, show_mask=True, warn_org=(10, 30)):
        """
        把黑區資訊畫到影像上 (紅色色調代表被判定為黑色的像素)

        :param warn_org: 「深度不可信」警告文字的座標。呼叫端如果已經在
                         畫面左上角畫了其他資訊 (例如 TelloApp 的模式名稱)，
                         要往下挪開避免互相覆蓋。
        """
        y1, y2 = info["roi"]

        if show_mask and info["mask"].size:
            region = annotated_frame[y1:y2]
            tint = np.zeros_like(region)
            tint[:, :, 2] = 255  # BGR -> 紅色
            mask3 = info["mask"][:, :, None]
            blended = cv2.addWeighted(region, 0.6, tint, 0.4, 0)
            annotated_frame[y1:y2] = np.where(mask3, blended, region)

        h_img = annotated_frame.shape[0]
        color = (0, 0, 255) if info["is_dark"] else (200, 200, 200)
        cv2.putText(annotated_frame,
                    f"DARK: {info['ratio'] * 100:.0f}%",
                    (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if info["is_dark"]:
            cv2.putText(annotated_frame, "DEPTH UNRELIABLE (BLACK SURFACE)",
                        warn_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_frame
