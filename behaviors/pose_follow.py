from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class BodyFollowControl(FlightBehavior):
    def __init__(self):
        self.pid_yv = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        self.pid_ud = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        self.pid_fb = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=70)
        self.pid_lr = PIDController(kp=0.5, ki=0.0, kd=0.2, limit=50)

        self.target_cx = 360 
        self.target_cy = 200 
        self.TARGET_SHOULDER_WIDTH = 120

    def calculate_command(self, user_input, vision_data):
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        target = getattr(vision_data, 'target', None) if vision_data else None
        lr, fb, ud, yv = 0, 0, 0, 0

        if target:
            # A. 旋轉控制 (對齊畫面中心)
            error_x = target['cx'] - self.target_cx
            yv = self.pid_yv.compute(error_x)

            # B. 升降控制 
            error_y = self.target_cy - target['cy']
            ud = self.pid_ud.compute(error_y)

            # C. 前後距離控制 
            error_width = target['body_scale'] - self.TARGET_SHOULDER_WIDTH
            fb = -self.pid_fb.compute(error_width)
            
            # D. 橫移控制 (對齊正臉)
            # 讀取視覺模組算出來的轉身誤差
            yaw_error = target.get('face_yaw_error', 0)
            lr = -self.pid_lr.compute(yaw_error)

            # Deadzone (死區)：避免無人機在完美對齊時還神經質地發抖
            if abs(error_x) < 30: yv = 0
            if abs(error_y) < 30: ud = 0
            if abs(error_width) < 10: fb = 0
            if abs(yaw_error) < 20: lr = 0 # 轉身角度誤差不大於一定值時不啟動環繞

        return (int(lr), int(fb), int(ud), int(yv))