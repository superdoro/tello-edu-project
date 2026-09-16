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
        self.TARGET_SHOULDER_WIDTH = 250

        # ==========================================
        # 環繞模式參數 (按 O 鍵開關)
        # ==========================================
        self.orbit_enabled = False
        self.ORBIT_LR_SPEED = 40  # 橫移速度 (正值：往右飛 = 俯視逆時針繞人)
        self.ORBIT_YAW_FF = 20    # 旋轉前饋：橫移時預先往反方向轉頭，減少目標滑出畫面中心的延遲

    def toggle_orbit(self):
        self.orbit_enabled = not self.orbit_enabled
        print(f"[環繞模式] {'開啟' if self.orbit_enabled else '關閉'}")

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

            # E. 環繞模式：固定速度橫移 + 旋轉鎖定目標 + 維持距離 = 以人為圓心繞圈
            # 往右橫移時，目標會在畫面中往左跑，因此需要往左轉 (yv 為負) 才能保持面向目標
            if self.orbit_enabled:
                lr = self.ORBIT_LR_SPEED
                yaw_ff = -self.ORBIT_YAW_FF if self.ORBIT_LR_SPEED > 0 else self.ORBIT_YAW_FF
                yv = max(-100, min(100, yv + yaw_ff))

        # 目標丟失時全部歸零 (懸停)，環繞模式也會停止移動等待重新偵測
        return (int(lr), int(fb), int(ud), int(yv))
