from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing

@dataclass
class VisionData:
    """統一的視覺辨識結果資料結構"""
    is_detected: bool = False # 是否有偵測到目標
    cx: int = 0               # 目標中心 X 座標
    cy: int = 0               # 目標中心 Y 座標
    w: int = 0                # 目標寬度
    h: int = 0                # 目標高度
    annotated_frame: typing.Any = None # 畫好標記的影像

    '''新增的可選用類別(使用 typing.Optional[dict] 
    代表預設可以是 None, 有找到時再放入字典)'''
    target: typing.Optional[dict] = None   # 目標物資訊 {"cx", "cy", "w", "h", "area"}
    obstacle: typing.Optional[dict] = None # 障礙物資訊 {"cx", "cy", "w", "h", "area"}

class VisionProcessor(ABC):
    """視覺處理器的抽象基底類別"""
    @abstractmethod
    def process_frame(self, frame) -> VisionData:
        pass