import time
from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class BalloonHuntControl(FlightBehavior):
    def __init__(self):
        # 狀態機初始化
        self.state = "SEARCH"
        self.current_target_id = 1  # 目前要撞擊的目標順序 (從 1 開始)
        
        # 控制器設定
        self.pid_yv = PIDController(kp=0.3, ki=0, kd=0.1, limit=60)
        self.target_cx = 360
        
        # 距離/面積定義
        self.ORBIT_AREA = 80000   # 適合開始環繞的距離 (面積)
        self.ATTACK_AREA = 150000 # 撞擊判定面積
        
        self.state_start_time = time.time()

    def change_state(self, new_state):
        """切換狀態並印出提示"""
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            print(f"🔄 [狀態切換] 進入狀態: {self.state} (尋找目標: {self.current_target_id}號)")

    def calculate_command(self, user_input, vision_data):
        # 1. 人工接管防護
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("SEARCH") # 被打斷後重新開始
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        
        # 假設 vision_data 會提供這兩個資訊：
        balloon = getattr(vision_data, 'balloon', None) # 最大的氣球物件
        marker_id = getattr(vision_data, 'marker_id', None) # 氣球上的數字 (若看不到則為 None)

        # ==========================================
        # 🧠 有限狀態機 (FSM) 決策邏輯
        # ==========================================
        
        if self.state == "SEARCH":
            # [行為] 原地緩慢旋轉，尋找任何氣球
            # yv = 30
            if balloon:
                self.change_state("APPROACH")

        elif self.state == "APPROACH":
            # [行為] 看到氣球了，把機頭對準它並飛過去，直到達到「環繞距離」
            if not balloon:
                self.change_state("SEARCH")
            else:
                error_x = balloon['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)
                
                if balloon['area'] < self.ORBIT_AREA:
                    fb = 40 # 還太遠，繼續往前
                else:
                    # 距離夠近了，開始找數字
                    if marker_id is not None:
                        # 看見數字了，判斷該怎麼做
                        self._decide_action_by_id(marker_id)
                    else:
                        # 沒看見數字(可能在背面)，開始環繞
                        self.change_state("ORBIT")

        elif self.state == "ORBIT":
            # [行為] 保持機頭對準氣球，同時向側邊平移 (形成圓形軌跡繞背)
            if not balloon:
                self.change_state("SEARCH")
            else:
                error_x = balloon['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)
                lr = 25 # 向右橫移，配合 PID 轉頭，就會形成繞著氣球飛的效果
                
                # 在環繞過程中如果看到數字了
                if marker_id is not None:
                    self._decide_action_by_id(marker_id)

        elif self.state == "ATTACK":
            # [行為] 全速衝撞！
            if not balloon:
                # 撞破了，或氣球飛走消失了
                print(f"💥 {self.current_target_id} 號氣球已擊破！")
                self.current_target_id += 1 # 準備找下一個
                self.change_state("SEARCH")
            else:
                error_x = balloon['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)
                fb = 60 # 高速前進衝撞
                
                # 如果面積超大，代表撞到了
                if balloon['area'] > self.ATTACK_AREA:
                    print(f"💥 物理接觸確認！")
                    self.current_target_id += 1
                    # 撞完後後退一點，然後重新搜尋
                    fb = -40
                    self.change_state("SEARCH")

        elif self.state == "AVOID":
            # [行為] 這是 0 號氣球，後退並遠離
            if balloon:
                fb = -30
                lr = -40 # 往反方向躲開
                if balloon['area'] < self.ORBIT_AREA * 0.5:
                    # 退得夠遠了，重新搜尋
                    self.change_state("SEARCH")
            else:
                self.change_state("SEARCH")

        return (int(lr), int(fb), int(ud), int(yv))

    def _decide_action_by_id(self, marker_id):
        """根據讀取到的 ID 決定下一個狀態"""
        if marker_id == self.current_target_id:
            self.change_state("ATTACK")
        elif marker_id == 0:
            self.change_state("AVOID")
        else:
            # 是數字，但不是現在要找的順序 (例如現在要找 1，但看到 2)
            # 先退後遠離它，並恢復搜尋模式去找正確的氣球
            print(f"👁️ 這是 {marker_id} 號，但目前需要找 {self.current_target_id} 號。跳過！")
            self.change_state("SEARCH")