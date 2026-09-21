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

        # ==========================================
        # 黑區脫困參數
        # 純黑牆面沒有紋理，單目深度模型往往誤判成「很遠」，
        # 此時 depth_C 完全不能用，必須改用不依賴深度的脫困動作。
        # ==========================================
        self.DARK_CLEAR_RATIO = 0.25  # 遲滯區間：黑區比例降到此值以下才算脫困 (進入門檻在分析器內為 0.35)
        self.DARK_MIN_DURATION = 1.0  # 最短脫困時間 (秒)，避免在門檻附近反覆進出狀態
        self.DARK_YAW_SPEED = 60      # 脫困旋轉速度
        self.DARK_CLIMB = 25          # 脫困上升速度：拉高視角讓天花板與燈光入鏡，深度模型才有紋理線索

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
        
        depth_L = getattr(vision_data, 'depth_L', 999.0)
        depth_C = getattr(vision_data, 'depth_C', 999.0)
        depth_R = getattr(vision_data, 'depth_R', 999.0)

        # 黑區資訊 (視覺模組沒掛黑區分析器時，預設為「畫面不黑、深度可信」)
        depth_unreliable = getattr(vision_data, 'depth_unreliable', False)
        dark_ratio = getattr(vision_data, 'dark_ratio', 0.0)
        dark_L = getattr(vision_data, 'dark_L', 0.0)
        dark_R = getattr(vision_data, 'dark_R', 0.0)

        # ==========================================
        # 黑區覆寫：優先於所有狀態
        # depth_C 在黑牆前會誤報成很遠，FORWARD 會毫無所覺地全速撞上去，
        # 所以一旦判定深度不可信就強制中斷當前狀態改為脫困。
        # ==========================================
        if depth_unreliable and self.state != "DARK_ESCAPE":
            # 轉向「比較不黑」的那一側：黑區少代表那邊有紋理，深度才重新可用
            self.turn_speed = -self.DARK_YAW_SPEED if dark_L < dark_R else self.DARK_YAW_SPEED
            side = "左" if self.turn_speed < 0 else "右"
            print(f"[黑區警告] 畫面 {dark_ratio*100:.0f}% 為純黑，深度不可信 -> 停止前進，往{side}轉並上升脫困")
            self.change_state("DARK_ESCAPE")

        time_in_state = now - self.state_start_time

        # ==========================================
        # 流暢弧線避障狀態機 (Smooth Arcing Walk)
        # ==========================================
        if self.state == "DARK_ESCAPE":
            # 深度不可信，絕不前進：只做原地旋轉 + 緩慢上升這兩個不依賴深度的動作
            fb = 0
            yv = self.turn_speed
            ud = self.DARK_CLIMB

            # 遲滯 + 最短持續時間，雙重防止在門檻附近抖動
            if time_in_state > self.DARK_MIN_DURATION and dark_ratio < self.DARK_CLEAR_RATIO:
                print(f"[黑區解除] 黑區降至 {dark_ratio*100:.0f}% -> 深度恢復可信，繼續探索")
                self.change_state("FORWARD")

        elif self.state == "FORWARD":
            fb = 50
            
            if depth_C <= self.SAFE_DIST:
                if depth_L > depth_R:
                    self.turn_speed = -75 # 向左轉 (速度稍微調柔和，配合前進畫出漂亮弧線)
                    print(f"遭遇障礙物 ({int(depth_C)}cm) -> 判定左側空曠，進入左弧線")
                else:
                    self.turn_speed = 75  # 向右轉
                    print(f"遭遇障礙物 ({int(depth_C)}cm) -> 判定右側空曠，進入右弧線")
                    
                self.change_state("CURVING")

        elif self.state == "CURVING":
            # 弧線過彎邏輯：同時包含前進 (fb) 與 旋轉 (yv)
            if depth_C < self.CRITICAL_DIST:
                # 如果轉彎的弧度不夠，就取消前進改為後退
                fb = -60
            else:
                fb = 30 # 維持一定的前進速度，畫出弧線
                
            yv = self.turn_speed
            
            # 當前方確認開闊，不馬上切回直線，而是進入「出彎延續」狀態
            if depth_C > self.CLEAR_DIST:
                self.exit_curve_duration = random.uniform(0.25, 0.75)
                print(f"前方開闊 ({int(depth_C)}cm) -> 延續弧線 {self.exit_curve_duration:.2f} 秒創造隨機軌跡")
                self.change_state("EXIT_CURVE")
                
        elif self.state == "EXIT_CURVE":
            # 延續上一狀態的弧線飛行
            fb = 30 
            yv = self.turn_speed
            
            # 防呆機制：如果在延續弧線的過程中，又掃到新的障礙物，立刻切回答避障模式
            if depth_C <= self.SAFE_DIST:
                print(f"出彎時遭遇新障礙 ({int(depth_C)}cm) -> 切回過彎模式")
                self.change_state("CURVING")
            # 隨機延遲時間結束，完美出彎，恢復直線探索
            elif time_in_state > self.exit_curve_duration:
                self.change_state("FORWARD")

        return (int(lr), int(fb), int(ud), int(yv))