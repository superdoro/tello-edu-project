from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController
'''
氣球追蹤(人工勢場法 (Artificial Potential Field))
'''

class AutoNavControl(FlightBehavior):
    def __init__(self):
        # 用於追蹤目標的 PID 控制器
        self.pid_yv = PIDController(kp=0.2, ki=0, kd=0.1, limit=50) # 左右旋轉
        self.pid_ud = PIDController(kp=0.2, ki=0, kd=0.1, limit=50) # 上下
        
        self.target_cx = 360 # 畫面中心 X
        self.target_cy = 240 # 畫面中心 Y
        
        # 定義危險閾值 (障礙物在畫面中的面積，面積越大代表越近)
        self.DANGER_AREA = 15000 

    def calculate_command(self, user_input, vision_data):
        # 預設指令：懸停
        lr, fb, ud, yv = 0, 0, 0, 0
        
        if not vision_data or not vision_data.is_detected:
            return (lr, fb, ud, yv)
            
        target = getattr(vision_data, 'target', None)
        obstacle = getattr(vision_data, 'obstacle', None)

        # 優先權 1：障礙物迴避 (Repulsion)
        if obstacle and obstacle['area'] > self.DANGER_AREA:
            # print("⚠️ 警告：偵測到障礙物！執行迴避！")
            # 如果障礙物在畫面右邊，無人機就往左邊平移迴避；反之亦然
            if obstacle['cx'] > self.target_cx:
                lr = -40 # 向左閃
                yv = 40
            else:
                lr = 40  # 向右閃
                yv = -40
                
            # 同時稍微向後退以保持安全距離
            fb = -20
            return (lr, fb, ud, yv)

        # 優先權 2：向目標導航 (Attraction)
        if target:
            # PID 鎖定目標中心
            error_x = target['cx'] - self.target_cx
            error_y = self.target_cy - target['cy']
            
            yv = self.pid_yv.compute(error_x)
            ud = self.pid_ud.compute(error_y)
            
            # 設定前進速度 (當目標在視野中心附近時，穩定向前推進)
            if abs(error_x) < 50 and abs(error_y) < 50:
                fb = 30 # 穩定向前走
            else:
                fb = 0  # 先對準目標再走
                
        return (lr, fb, ud, yv)