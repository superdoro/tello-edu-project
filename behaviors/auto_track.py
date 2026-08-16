from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController
'''
手部追蹤
'''
class AutoTrackControl(FlightBehavior):
    def __init__(self):
        # 初始化 PID 控制器 (P參數可根據實測調整)
        self.pid_yv = PIDController(kp=0.15, ki=0, kd=0, limit=50) # 左右旋轉
        self.pid_ud = PIDController(kp=0.15, ki=0, kd=0, limit=50) # 上下
        self.pid_fb = PIDController(kp=0.30, ki=0, kd=0, limit=50) # 前後
        
        # 預設畫面中心 (以 720x480 解析度為準) 與目標距離大小
        self.target_cx = 360
        self.target_cy = 240
        self.target_w = 100

    def calculate_command(self, user_input, vision_data) -> tuple:
        # 如果沒看到目標，就懸停不動
        if not vision_data or not vision_data.is_detected:
            return (0, 0, 0, 0)
            
        # 1. 計算各軸誤差
        error_x = vision_data.cx - self.target_cx
        error_y = self.target_cy - vision_data.cy
        error_w = self.target_w - vision_data.w
        
        # 2. 透過 PID 轉為速度
        yv = self.pid_yv.compute(error_x)
        ud = self.pid_ud.compute(error_y)
        fb = self.pid_fb.compute(error_w)
        
        # 3. 設定緩衝區：誤差很小時懸停，避免無人機瘋狂微調抽搐
        if abs(error_x) < 20: yv = 0
        if abs(error_y) < 20: ud = 0
        if abs(error_w) < 15: fb = 0
        
        return (0, fb, ud, yv) # 回傳 (lr, fb, ud, yv)