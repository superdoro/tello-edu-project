from abc import ABC, abstractmethod

class FlightBehavior(ABC):
    """
    飛行策略的抽象基底類別 (Interface)。
    所有新的飛行模式 (手動、跟追、語音) 都必須繼承此類別並實作 calculate_command。
    """
    
    @abstractmethod
    def calculate_command(self, user_input, vision_data=None) -> tuple:
        """
        計算並回傳飛行指令。
        :param user_input: 來自 UIController 的 UserInput 物件
        :param vision_data: 來自視覺辨識的結果 (手動模式下通常為 None)
        :return: (lr, fb, ud, yv) 四個軸向的速度數值
        """
        pass

    def get_mode(self) -> str:
        """
        回傳模式資訊:return: string
        """
        pass