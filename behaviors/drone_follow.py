import time
from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class DroneFollowControl(FlightBehavior):
    def __init__(self):
        self.state = "SEARCH"
        
        # 3D 空間追蹤 PID 控制器
        # 1. 旋轉 (對齊 X 軸)
        self.pid_yv = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=80)
        # 2. 升降 (對齊 Y 軸)
        self.pid_ud = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=80)
        # 3. 前後 (維持距離)
        self.pid_fb = PIDController(kp=0.5, ki=0.0, kd=0.15, limit=60)

        # 畫面中心點
        self.target_cx = 360 
        self.target_cy = 240 
        
        # 追蹤距離設定 (利用 Bounding Box 面積來推算距離)
        # 數值需要依據你期望的安全跟車距離來實測微調
        self.TARGET_AREA = 36000 

        # ==========================================
        # 視覺記憶參數 (Vision Memory)
        # ==========================================
        self.last_drone = None
        self.last_drone_time = 0.0
        self.MEMORY_DURATION = 1.0
        self.state_start_time = time.time()

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            print(f"[狀態切換] 空戰追蹤模式: {self.state}")
            

    def calculate_command(self, user_input, vision_data):
        # 1. 人工接管優先
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("SEARCH") 
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = time.time()

        # 取得當前幀的真實視覺資料
        current_drone = getattr(vision_data, 'drone', None) 

        # ==========================================
        # 視覺記憶補償邏輯
        # ==========================================
        if current_drone is not None:
            # YOLO 看到了！更新記憶庫
            self.last_drone = current_drone
            self.last_drone_time = now
            target = current_drone
        else:
            # YOLO 沒看到，檢查記憶是否還在有效期限內
            if now - self.last_drone_time <= self.MEMORY_DURATION and self.last_drone is not None:
                target = self.last_drone
                # 默默使用上一幀的資料繼續追蹤，填補閃爍空窗期
            else:
                # 記憶過期，徹底宣告目標丟失
                target = None
                self.last_drone = None

        # ==========================================
        # 追蹤狀態機 (Air-to-Air FSM)
        # ==========================================
        if self.state == "SEARCH":
            # 原地緩慢旋轉尋找目標
            yv = 40
            if target:
                print("[鎖定] 發現目標無人機，開始攔截！")
                self.change_state("TRACK")

        elif self.state == "TRACK":
            if not target:
                print("[丟失] 目標無人機脫離視野！")
                self.change_state("SEARCH")
            else:
                # A. 升降控制 (UP/DOWN) - 咬住敵機高度
                error_y = self.target_cy - target['cy']
                ud = self.pid_ud.compute(error_y)

                # B. 旋轉控制 (YAW) - 咬住敵機方位
                error_x = target['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)

                # C. 前後距離控制 (FORWARD/BACKWARD) - 維持編隊距離
                # 將面積誤差縮小，避免 PID 數值爆掉
                error_area = (self.TARGET_AREA - target['area']) / 100.0
                fb = self.pid_fb.compute(error_area)

                # Deadzone (死區)：目標完美在準心時，微調動力歸零，避免兩架無人機產生共振抽搐
                if abs(error_x) < 30: yv = 0
                if abs(error_y) < 30: ud = 0
                if abs(error_area) < 30: fb = 0

        return (int(lr), int(fb), int(ud), int(yv))