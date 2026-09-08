import time
from behaviors.base import FlightBehavior

class SlamControl(FlightBehavior):
    def __init__(self):
        self.state = "WAITING"
        self.state_start_time = time.time()

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            print(f"🔄 [SLAM 行為模組] 狀態切換 -> {self.state}")

    def calculate_command(self, user_input, vision_data, real_altitude_cm=100):
        # 防呆機制：如果使用者撥動搖桿介入，無條件切回手動模式並歸零狀態
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("WAITING")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = time.time()

        # 🎯 條件 1：SLAM 已經初始化成功
        if getattr(vision_data, 'tracking_state', 0) == 1:
            if self.state != "TRACKING":
                self.change_state("TRACKING")
                print("✅ SLAM 視差初始化成功！Map Initialized！")
            
            # 在 TRACKING 狀態下，暫時懸停。
            # 你可以在這裡加入自動導航前往特定座標的邏輯
            return (0, 0, 0, 0)

        # 🎯 條件 2：SLAM 尚未初始化，啟動自動初始化之舞
        if self.state == "WAITING":
            print("🚀 開始執行自動視差初始化之舞...")
            self.change_state("INIT_LEFT")

        time_in_state = now - self.state_start_time

        # 反覆左右平移，創造特徵點視差 (速度設為 25 避免畫面模糊)
        if self.state == "INIT_LEFT":
            lr = -25
            if time_in_state > 2.5:
                self.change_state("INIT_RIGHT")
                
        elif self.state == "INIT_RIGHT":
            lr = 25
            if time_in_state > 2.5:
                self.change_state("INIT_LEFT")

        return (int(lr), int(fb), int(ud), int(yv))