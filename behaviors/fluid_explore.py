import time
import random
from behaviors.base import FlightBehavior

class FluidExploreControl(FlightBehavior):
    def __init__(self):
        self.state = "FORWARD"
        self.state_start_time = time.time()
        
        # 距離參數設定 (公分)
        self.SAFE_DIST = 300    # 觸發避障的安全距離
        self.CLEAR_DIST = 250   # 遲滯區間 : 判定前方已經開闊的距離
        self.CRITICAL_DIST = 150 # 極限危險距離 : 低於此距離代表弧線轉不過去，必須煞車
        
        self.turn_speed = 0
        self.exit_curve_duration = 0.0 # 用來儲存隨機決定的出彎延遲時間

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            print(f"[狀態切換] {self.state}")

    def calculate_command(self, user_input, vision_data):
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("FORWARD")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = time.time()
        time_in_state = now - self.state_start_time
        
        depth_L = getattr(vision_data, 'depth_L', 999.0)
        depth_C = getattr(vision_data, 'depth_C', 999.0)
        depth_R = getattr(vision_data, 'depth_R', 999.0)

        # ==========================================
        # 流暢弧線避障狀態機 (Smooth Arcing Walk)
        # ==========================================
        if self.state == "FORWARD":
            fb = 60 # 安全推進
            
            if depth_C <= self.SAFE_DIST:
                if depth_L > depth_R:
                    self.turn_speed = -75 # 向左轉 (速度稍微調柔和，配合前進畫出漂亮弧線)
                    print(f"⚠️ 遭遇障礙物 ({int(depth_C)}cm) -> 判定左側空曠，進入左弧線")
                else:
                    self.turn_speed = 75  # 向右轉
                    print(f"⚠️ 遭遇障礙物 ({int(depth_C)}cm) -> 判定右側空曠，進入右弧線")
                    
                self.change_state("CURVING")

        elif self.state == "CURVING":
            # 🔥 弧線過彎邏輯：同時包含前進 (fb) 與 旋轉 (yv)
            if depth_C < self.CRITICAL_DIST:
                # 如果轉彎的弧度不夠，快要擦撞牆壁了，就取消前進改為微微後退
                fb = -60
            else:
                fb = 40 # 維持一定的前進速度，畫出弧線
                
            yv = self.turn_speed
            
            # 當前方確認開闊，不馬上切回直線，而是進入「出彎延續」狀態
            if depth_C > self.CLEAR_DIST:
                self.exit_curve_duration = random.uniform(0.25, 0.75)
                print(f"前方開闊 ({int(depth_C)}cm) -> 延續弧線 {self.exit_curve_duration:.2f} 秒創造隨機軌跡")
                self.change_state("EXIT_CURVE")
                
        elif self.state == "EXIT_CURVE":
            # 延續上一狀態的弧線飛行
            fb = 40 
            yv = self.turn_speed
            
            # 防呆機制：如果在延續弧線的過程中，又掃到新的障礙物，立刻切回答避障模式
            if depth_C <= self.SAFE_DIST:
                print(f"⚠️ 出彎時遭遇新障礙 ({int(depth_C)}cm) -> 切回過彎模式")
                self.change_state("CURVING")
            # 隨機延遲時間結束，完美出彎，恢復直線探索
            elif time_in_state > self.exit_curve_duration:
                self.change_state("FORWARD")

        return (int(lr), int(fb), int(ud), int(yv))