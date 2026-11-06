import time
import random
from behaviors.base import FlightBehavior

class FluidExploreControl(FlightBehavior):
    """
    漫遊避障：直線前進，遇到障礙就往比較空曠的一側轉一個隨機角度，再繼續前進。

    不後退。舊版前方太近時會以 fb=-60 一邊轉一邊倒退，動作劇烈，
    而 Tello 背後沒有任何感測 (10/19 20:46 錄影漫遊時一直在快速後退)。
    改成：還有空間就一邊慢慢前進一邊轉 (畫弧線)，太近就原地轉，任何情況都不往後退。
    轉的角度隨機，軌跡才不會每次都一樣，能漫遊到房間各處。

    狀態
        FORWARD      直線前進
        TURN         往比較空的一側轉隨機角度；轉完前方仍擋著就同方向再轉一次
        DARK_ESCAPE  畫面大片純黑、深度不可信 -> 原地旋轉 + 緩慢上升
    """

    def __init__(self, now_fn=time.time, rng=None):
        """
        :param now_fn: 取得目前時間的函式 (離線測試時傳入假時鐘)
        :param rng: random.Random 實例 (測試時可固定種子)，None 就用新的
        """
        self._now = now_fn
        self._rng = rng if rng is not None else random.Random()
        self.state = "FORWARD"
        self.state_start_time = self._now()

        # 距離參數設定 (公分)
        self.SAFE_DIST = 300    # 觸發避障的安全距離
        self.CLEAR_DIST = 250   # 遲滯區間 : 判定前方已經開闊的距離
        self.CRITICAL_DIST = 150 # 低於此距離不再前進，改為原地轉 (不後退)

        self.FORWARD_SPEED = 50
        self.TURN_SPEED = 60          # 轉向速度
        self.TURN_ARC_FB = 20         # 前方還有空間時，轉彎的同時慢慢前進 (畫弧線)
        self.TURN_DEG_PER_SEC = 60.0  # rc yaw 60 約每秒 60 度 (依電量略有差異，角度只是大概)
        self.TURN_MIN_DEG = 45
        self.TURN_MAX_DEG = 120
        self.SIDE_RATIO = 1.2         # 左右深度差超過這個倍數才算「有一邊比較空」，否則方向隨機
        self.turn_speed = 0
        self.turn_sec = 0.0

        # 偏好方向 (prefer_side)：例如紙條模式剛避開一條有色紙條，
        # 接下來一段時間轉彎時優先往遠離它的方向，不要又轉回去
        self.bias_dir = 0
        self.bias_until = 0.0

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
            self.state_start_time = self._now()
            print(f"[狀態切換] {self.state}")

    def prefer_side(self, direction, seconds):
        """接下來 seconds 秒內，轉彎時優先往 direction (+1 右 / -1 左)"""
        self.bias_dir = 1 if direction > 0 else -1
        self.bias_until = self._now() + seconds

    def _start_turn(self, depth_L, depth_R, depth_C, keep_dir=False):
        """
        決定轉向方向與隨機角度，進入 TURN。
        方向：keep_dir 時沿用上一次；否則偏好方向還有效就用它；
              再來是比較空的一側；兩邊差不多就隨機。
        """
        if not (keep_dir and self.turn_speed):
            hi, lo = max(depth_L, depth_R), max(min(depth_L, depth_R), 1.0)
            if self._now() < self.bias_until:
                direction = self.bias_dir
            elif hi / lo >= self.SIDE_RATIO:
                direction = -1 if depth_L > depth_R else 1
            else:
                direction = self._rng.choice((-1, 1))
            self.turn_speed = direction * self.TURN_SPEED
        angle = self._rng.uniform(self.TURN_MIN_DEG, self.TURN_MAX_DEG)
        self.turn_sec = angle / self.TURN_DEG_PER_SEC
        side = "左" if self.turn_speed < 0 else "右"
        print(f"遭遇障礙物 ({int(depth_C)}cm) -> 往{side}轉約 {angle:.0f} 度")
        self.change_state("TURN")
        self.state_start_time = self._now()    # 「再轉一次」時狀態沒變，也要重新計時

    def calculate_command(self, user_input, vision_data):
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("FORWARD")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = self._now()

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
        # 隨機轉向避障狀態機 (不後退)
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
            fb = self.FORWARD_SPEED
            if depth_C <= self.SAFE_DIST:
                self._start_turn(depth_L, depth_R, depth_C)
                time_in_state = 0.0

        if self.state == "TURN":
            # 前方還有空間就邊走邊轉 (弧線)，太近就原地轉 —— 不後退
            fb = self.TURN_ARC_FB if depth_C >= self.CRITICAL_DIST else 0
            yv = self.turn_speed
            if time_in_state >= self.turn_sec:
                if depth_C > self.CLEAR_DIST:
                    print(f"前方開闊 ({int(depth_C)}cm) -> 繼續前進")
                    self.change_state("FORWARD")
                    fb, yv = self.FORWARD_SPEED, 0
                else:
                    # 還是擋著：同方向再轉一個隨機角度 (不左右來回)
                    self._start_turn(depth_L, depth_R, depth_C, keep_dir=True)

        return (int(lr), int(fb), int(ud), int(yv))
