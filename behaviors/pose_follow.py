from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

# 環繞方向 (皆為「俯視」視角)
#   CCW 逆時針 = 無人機往右橫移
#   CW  順時針 = 無人機往左橫移
ORBIT_OFF, ORBIT_CCW, ORBIT_CW = "OFF", "CCW", "CW"

class BodyFollowControl(FlightBehavior):
    def __init__(self):
        self.pid_yv = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        self.pid_ud = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=100)
        self.pid_fb = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=70)
        self.pid_lr = PIDController(kp=0.5, ki=0.0, kd=0.2, limit=50)

        self.target_cx = 360
        self.target_cy = 200

        # ==========================================
        # 追蹤距離設定
        # 單位是 body_scale (像素)：人在畫面中佔得越大 = 無人機離人越近，
        # 所以「數值越大 = 飛得越近」，與直覺相反，調參時要特別注意。
        # ==========================================
        self.FOLLOW_SHOULDER_WIDTH = 120  # 平時跟隨：距離較遠，視野完整、安全餘裕大
        self.ORBIT_SHOULDER_WIDTH = 250   # 環繞時：距離較近，繞出來的圓較小，人不易滑出畫面

        # 切換環繞時目標距離會跳變，直接套用會讓無人機突然衝刺，
        # 而且 PID 的微分項會收到一個並非來自「人移動」的假誤差尖峰。
        # 因此改用限速漸變，讓設定點平滑地移動到位。
        self.SCALE_RAMP_RATE = 3.0        # 每格最多改變的目標值 (約 30fps 下 0.8 秒完成切換)
        self.target_scale = self.FOLLOW_SHOULDER_WIDTH

        # ==========================================
        # 環繞模式參數 (按 O 鍵開關)
        # ==========================================
        self.orbit_mode = ORBIT_OFF
        self.ORBIT_LR_SPEED = 40  # 橫移速度「大小」，方向由 orbit_mode 決定
        self.ORBIT_YAW_FF = 20    # 旋轉前饋：橫移時預先往反方向轉頭，減少目標滑出畫面中心的延遲
        # 前饋量與環繞半徑有關：維持面向目標所需的角速度 w = v / r，
        # 而 body_scale 與距離成反比 (越近越大)，所以前饋量正比於 body_scale。
        # 上面那個 20 是在 REF_SCALE 這個距離下調出來的，飛近之後必須等比放大，
        # 否則轉頭跟不上橫移，目標會持續往畫面外側滑。
        self.ORBIT_YAW_FF_REF_SCALE = 250

    @property
    def orbit_enabled(self) -> bool:
        """是否正在環繞。同時決定追蹤距離要用近的 (環繞) 還是遠的 (平時跟隨)。"""
        return self.orbit_mode != ORBIT_OFF

    def toggle_orbit(self):
        """O 鍵：OFF -> 逆時針 -> 順時針 -> OFF 循環"""
        self.orbit_mode = {ORBIT_OFF: ORBIT_CCW,
                           ORBIT_CCW: ORBIT_CW,
                           ORBIT_CW: ORBIT_OFF}[self.orbit_mode]
        goal = self.ORBIT_SHOULDER_WIDTH if self.orbit_enabled else self.FOLLOW_SHOULDER_WIDTH
        desc = {ORBIT_OFF: "關閉 -> 退遠",
                ORBIT_CCW: "逆時針 (俯視) -> 往右橫移、拉近",
                ORBIT_CW:  "順時針 (俯視) -> 往左橫移、拉近"}[self.orbit_mode]
        print(f"[環繞模式] {desc} (目標 body_scale {self.target_scale:.0f} -> {goal:.0f})")

    def stop_orbit(self):
        """直接關閉環繞。切換飛行模式時用這個，不要讓它跟著 O 鍵的循環往下走。"""
        if self.orbit_mode != ORBIT_OFF:
            self.orbit_mode = ORBIT_OFF
            print("[環繞模式] 關閉 (切換飛行模式)")

    def orbit_status(self):
        """畫面上要顯示的環繞狀態文字；沒在環繞時回傳 None"""
        return None if self.orbit_mode == ORBIT_OFF else f"ORBIT: {self.orbit_mode}"

    def calculate_command(self, user_input, vision_data):
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        # 依目前是否環繞決定目標距離，並限速漸變到位 (避免切換瞬間衝刺)。
        # 放在偵測判斷之前：即使目標暫時丟失，設定點也會持續移動到定位，
        # 這樣重新找到人的時候不會再補一次突兀的加速。
        desired_scale = self.ORBIT_SHOULDER_WIDTH if self.orbit_enabled else self.FOLLOW_SHOULDER_WIDTH
        gap = desired_scale - self.target_scale
        if abs(gap) <= self.SCALE_RAMP_RATE:
            self.target_scale = desired_scale
        else:
            self.target_scale += self.SCALE_RAMP_RATE if gap > 0 else -self.SCALE_RAMP_RATE

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
            error_width = target['body_scale'] - self.target_scale
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
            # 逆時針 (往右橫移，lr 正) 時目標會在畫面中往左跑，必須往左轉 (yv 負) 才能保持面向目標；
            # 順時針則橫移與轉向同時反號。
            if self.orbit_enabled:
                lr = self.ORBIT_LR_SPEED if self.orbit_mode == ORBIT_CCW else -self.ORBIT_LR_SPEED
                # 前饋量依實際環繞半徑等比縮放 (距離越近 -> body_scale 越大 -> 需要轉得越快)。
                # 用漸變中的 target_scale 而非最終值，讓拉近過程中的轉速同步跟上。
                ff_mag = self.ORBIT_YAW_FF * (self.target_scale / self.ORBIT_YAW_FF_REF_SCALE)
                yaw_ff = -ff_mag if lr > 0 else ff_mag
                yv = max(-100, min(100, yv + yaw_ff))

        # 目標丟失時全部歸零 (懸停)，環繞模式也會停止移動等待重新偵測
        return (int(lr), int(fb), int(ud), int(yv))
