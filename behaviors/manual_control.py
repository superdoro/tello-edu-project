from behaviors.base import FlightBehavior

class ManualControl(FlightBehavior):
    def __init__(self):
        # [新增] 語音控制的暫存速度與計時器
        self.v_lr, self.v_fb, self.v_ud, self.v_yv = 0, 0, 0, 0
        self.voice_timer = 0
        self.voice_duration = 150 # 大約維持 150 個 frame (約 5 秒鐘)

    def calculate_command(self, user_input, vision_data=None) -> tuple:
        # [新增] 檢查是否有新的方向性語音指令
        cmd = user_input.voice_command
        if cmd:
            # 清空速度
            self.v_lr, self.v_fb, self.v_ud, self.v_yv = 0, 0, 0, 0
            self.voice_timer = self.voice_duration # 啟動計時器
            
            if "前進" in cmd: self.v_fb = 50
            elif "後退" in cmd: self.v_fb = -50
            elif "向左" in cmd: self.v_lr = -50
            elif "向右" in cmd: self.v_lr = 50
            elif "上升" in cmd: self.v_ud = 50
            elif "下降" in cmd: self.v_ud = -50
            elif "轉向" in cmd: self.v_yv = 50 # 簡單旋轉測試
        
        # 1. 優先使用鍵盤 (只要有按鍵，立刻蓋過語音)
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.voice_timer = 0 # 打斷語音飛行
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)
            
        # 2. 若沒按鍵盤，檢查是否在語音飛行期間
        if self.voice_timer > 0:
            self.voice_timer -= 1
            return (self.v_lr, self.v_fb, self.v_ud, self.v_yv)
            
        # 3. 都沒有指令，懸停不動
        return (0, 0, 0, 0)