import time

from behaviors.base import FlightBehavior


class ManualControl(FlightBehavior):
    """
    手動控制：鍵盤即時操作，外加語音指令的定時移動。

    語音指令不是「講一次動一下」，而是「講一次持續飛 VOICE_DURATION_SEC 秒」，
    因為語音有辨識延遲，做成點動的話根本來不及反應。期間按任何方向鍵都會立刻接管。
    """

    # 語音指令的持續時間（秒）與速度。
    #
    # 原本是用「影格數」計時 (170 格)，註解寫約 5 秒 —— 那是假設 34fps。
    # 但主迴圈的速度隨模式差很多：實測 MANUAL 模式沒有掛視覺模組，一格只要 0.7ms，
    # 170 格其實只有 0.1 秒，根本看不出無人機有動；HAND TRACKER 是 0.9 秒、
    # TISSUE CHARGE 是 2.7 秒，同一個指令在不同模式下長度差 27 倍。
    #
    # 改用真實時間計時之後，不論哪個模式、電腦快慢，持續時間都一樣。
    # 速度 70 時大約是 70cm/s，所以 1.5 秒約會移動 1 公尺 —— 室內要留意空間，
    # 想看得更清楚就把秒數加大，或把速度調小讓它慢慢飛。
    VOICE_DURATION_SEC = 1.5
    VOICE_SPEED = 70

    def __init__(self):
        self.v_lr, self.v_fb, self.v_ud, self.v_yv = 0, 0, 0, 0
        self.voice_until = 0.0        # 語音飛行的結束時刻 (time.time())
        self.voice_label = ""         # 目前執行中的語音指令，供畫面顯示

    def calculate_command(self, user_input, vision_data=None) -> tuple:
        now = time.time()

        # 收到新的語音指令：重設速度並重新計時
        cmd = user_input.voice_command
        if cmd:
            self.v_lr, self.v_fb, self.v_ud, self.v_yv = 0, 0, 0, 0
            s = self.VOICE_SPEED
            if "前進" in cmd: self.v_fb = s
            elif "後退" in cmd: self.v_fb = -s
            elif "向左" in cmd: self.v_lr = -s
            elif "向右" in cmd: self.v_lr = s
            elif "上升" in cmd: self.v_ud = s
            elif "下降" in cmd: self.v_ud = -s
            elif "轉向" in cmd: self.v_yv = s
            # 「停止」等其他指令不設速度，於是下面會以全 0 懸停整段時間

            self.voice_until = now + self.VOICE_DURATION_SEC
            self.voice_label = cmd
            print(f"[手動控制] 語音「{cmd}」-> 持續 {self.VOICE_DURATION_SEC:.1f} 秒 "
                  f"(lr={self.v_lr} fb={self.v_fb} ud={self.v_ud} yv={self.v_yv})")

        # 1. 鍵盤優先：只要有按鍵就立刻接管並中斷語音飛行
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.voice_until = 0.0
            self.voice_label = ""
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        # 2. 語音飛行期間
        if now < self.voice_until:
            return (self.v_lr, self.v_fb, self.v_ud, self.v_yv)

        # 3. 都沒有指令，懸停不動
        self.voice_label = ""
        return (0, 0, 0, 0)

    def voice_status(self):
        """畫面上要顯示的語音飛行狀態；沒有在執行時回傳 None"""
        remain = self.voice_until - time.time()
        if remain <= 0 or not self.voice_label:
            return None
        return f"VOICE: {self.voice_label} {remain:.1f}s"

    def get_mode(self) -> str:
        return "MANUAL CONTROL"
