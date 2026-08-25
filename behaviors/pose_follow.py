# from behaviors.base import FlightBehavior
# from utils.pid_controller import PIDController

# class BodyFollowControl(FlightBehavior):
#     def __init__(self):
#         # 1. 旋轉 PID (對準目標 X 軸) -> 極速解鎖至 100
#         self.pid_yv = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)

#         # 2. 升降 PID (對準目標 Y 軸) -> 極速解鎖至 100
#         self.pid_ud = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        
#         # 3. 前後 PID (維持雙肩寬度) -> 極速限制為 50
#         self.pid_fb = PIDController(kp=0.4, ki=0.0, kd=0.15, limit=70)

#         self.target_cx = 360 # 畫面 X 中心
#         self.target_cy = 200 # 畫面 Y 中心 (稍微偏上，讓頭部不會被切出畫面)
        
#         # 距離定義：期望的雙肩像素寬度
#         self.TARGET_SHOULDER_WIDTH = 120
    
#     def calculate_command(self, user_input, vision_data):
#         # 人工接管優先
#         if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
#             return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

#         target = getattr(vision_data, 'target', None) if vision_data else None
#         lr, fb, ud, yv = 0, 0, 0, 0

#         if target:
#             # A. 旋轉控制 (YAW) - 🔥 改用通用的 cx
#             error_x = target['cx'] - self.target_cx
#             yv = self.pid_yv.compute(error_x)
            
#             # B. 升降控制 (UP/DOWN) - 🔥 改用通用的 cy
#             error_y = self.target_cy - target['cy']
#             ud = self.pid_ud.compute(error_y)
            
#             # C. 前後距離控制 (FORWARD/BACKWARD) - 維持使用肩膀寬度
#             error_width = target['body_scale'] - self.TARGET_SHOULDER_WIDTH
#             fb = -self.pid_fb.compute(error_width)
            
#             # Deadzone (死區)：當誤差很小時，不再微調輸出 0，避免無人機瘋狂抖動
#             if abs(error_x) < 30: yv = 0
#             if abs(error_y) < 30: ud = 0
#             if abs(error_width) < 10: fb = 0

#         return (int(lr), int(fb), int(ud), int(yv))

from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class BodyFollowControl(FlightBehavior):
    def __init__(self):
        self.pid_yv = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        self.pid_ud = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        self.pid_fb = PIDController(kp=0.4, ki=0.0, kd=0.15, limit=70)
        
        # 🔥 新增：側移 PID (專門負責把無人機推向你的正前方)
        self.pid_lr = PIDController(kp=0.4, ki=0.0, kd=0.1, limit=50)

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
            
            # D. 🔥 橫移控制 (對齊正臉)
            # 讀取視覺模組算出來的轉身誤差
            yaw_error = target.get('face_yaw_error', 0)
            lr = -self.pid_lr.compute(yaw_error)

            # Deadzone (死區)：避免無人機在完美對齊時還神經質地發抖
            if abs(error_x) < 30: yv = 0
            if abs(error_y) < 30: ud = 0
            if abs(error_width) < 10: fb = 0
            if abs(yaw_error) < 15: lr = 0 # 轉身角度誤差不大於一定值時不啟動環繞

        return (int(lr), int(fb), int(ud), int(yv))