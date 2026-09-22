import time
from behaviors.base import FlightBehavior
from behaviors.fluid_explore import FluidExploreControl
from utils.pid_controller import PIDController


class TissueChargeControl(FlightBehavior):
    """
    衛生紙條獵手：漫遊找紙條 -> 對準 -> 衝撞 -> 繼續漫遊。

    狀態機
        SEARCH  把指令整段委派給 FluidExploreControl，在場地裡漫遊避障
        ALIGN   YOLO 找到紙條 -> 水平轉正、爬升到紙條下端橫過前進路線 (鎖定)
        CHARGE  鎖定後直線全速往前衝固定秒數，中途不做任何檢查

    為什麼搜尋要用 FluidExploreControl 而不是原地旋轉：
    原地轉只能看到同一個位置的視野，紙條掛在房間各處，轉完一圈看不到就卡死了。
    漫遊會帶著無人機換位置，而且那套弧線避障與黑區脫困已經調過、實飛驗證過，
    直接重用遠比在這裡重寫一份可靠。委派是整段委派 —— SEARCH 期間的
    lr/fb/ud/yv 全部由它決定，這裡不做任何加工，行為才會與 FLUID EXPLORER 完全一致。

    CHARGE 中途不看偵測也不看深度：衝到最後紙條會脹滿畫面或離開視野，
    視覺回授在那時候沒有意義；深度量到的是紙條背後的牆，
    拿來當中止條件會讓無人機在碰到紙條之前就先停下來。
    """

    def __init__(self):
        self.state = "SEARCH"
        self.state_start_time = time.time()

        # 搜尋階段的漫遊引擎。它需要的 depth_L/C/R 與 dark_* 由 TissueDetector 提供，
        # 欄位名稱與單位刻意和 fluid_explorer_vision.py 一致，可以直接餵進去。
        self.explorer = FluidExploreControl()

        # 畫面中心 (720x480)
        self.target_cx = 360
        self.target_cy = 240

        self.pid_yv = PIDController(kp=0.4, ki=0.0, kd=0.15, limit=60)
        self.pid_ud = PIDController(kp=0.4, ki=0.0, kd=0.15, limit=50)

        # --- 鎖定條件 ---
        # 對準 = 水平轉正 + 紙條下端橫過前進路線。一對準就衝，不看距離、不看深度。
        self.ALIGN_TOL_X = 40      # 水平對準容許誤差 (像素)

        # --- 垂直：讓紙條「橫過」前進路線，而不是瞄準紙條上的某一點 ---
        # 紙條是一條垂直的線，無人機水平往前飛，只要紙條橫過機身高度就一定撞得到，
        # 不需要對準紙條的中心或任何特定位置。
        #
        # 判斷依據是紙條的「下端」(自由垂下的那一端)：
        #   下端在畫面中心下方 -> 紙條延伸到機身高度以下 -> 往前衝一定撞到
        #   下端在畫面中心上方 -> 紙條整條在頭頂上 -> 往前飛會從上面穿過去，必須先爬升
        # 而且這個判斷與距離無關：下端在中心下方時，越靠近它只會越往畫面下方跑。
        #
        # 為什麼不能再用「瞄準點 + 天花板保護」：實測紙條比畫面長，上緣永遠被切在 y=0，
        # 保護永遠啟動、永遠不准爬，瞄準點就永遠對不上 —— 錄影裡 770 格 fb 全是 0。
        # 改用下端就沒有這個問題：下端是固定的實體位置，往它爬升會收斂，
        # 不會像追外框中心那樣越爬越高 (外框上緣被切掉時，中心會跟著無人機往上跑)。
        self.PATH_MARGIN = 30      # 紙條下端至少要低於畫面中心這麼多像素，確保撞到機身而不是擦過
        self.CLIMB_MAX = 30        # 爬升速度上限：要爬也爬得慢，萬一誤判還有時間接手

        # --- 硬性高度上限 (公分，相對起飛點) ---
        # 視覺判斷再小心，誤判一個天花板上的燈就可能一路爬上去 —— 之前已經撞過一次。
        # 只有真實高度擋得住。⚠️ 請依場地設定：天花板高度再扣掉至少 50cm
        # (太靠近天花板時螺旋槳氣流會把機身往上吸)。讀不到高度時此保護不生效。
        # 只在對準階段的爬升時作用，不影響衝撞。設為 None 可完全停用。
        self.MAX_HEIGHT = 200
        self.UNREACHABLE_SEC = 2.0 # 已到高度上限、紙條仍在頭頂上超過這麼久 -> 放棄這條
        self.capped_since = None   # 從哪一刻開始卡在高度上限

        # --- 衝撞：鎖定後直線全速往前，固定時間，中途不做任何檢查 ---
        self.CHARGE_FB = 80        # 衝撞速度
        self.CHARGE_TIME = 3.0     # 衝撞持續秒數

        # --- 放棄後的冷卻 ---
        # 高度上限搆不到的紙條會一直留在畫面裡，直接回到搜尋的話下一格又會被鎖定、
        # 又卡在上限，無限循環。放棄後這段時間內忽略偵測，讓漫遊把無人機帶到別處。
        self.IGNORE_SEC = 3.0
        self.ignore_until = 0.0

        # --- 視覺記憶：擋住偵測的單格閃爍 ---
        self.last_tissue = None
        self.last_tissue_time = 0.0
        self.MEMORY_DURATION = 0.5

        self.hit_count = 0         # 累計衝撞次數，只作顯示用

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            # 回到搜尋時把漫遊引擎重設成直線前進，
            # 不然它會沿用上次離開時的過彎狀態，一接手就莫名其妙開始繞圈。
            if new_state == "SEARCH":
                self.explorer.change_state("FORWARD")
            print(f"[紙條獵手] {self.state}")

    def calculate_command(self, user_input, vision_data):
        # 1. 人工接管優先，並把狀態重置回搜尋
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("SEARCH")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = time.time()
        time_in_state = now - self.state_start_time

        # 2. 視覺記憶：偵測不到時沿用上一格，避免閃爍害狀態機來回跳
        current = getattr(vision_data, 'tissue', None) if vision_data else None
        if current is not None:
            self.last_tissue = current
            self.last_tissue_time = now
            tissue = current
        elif now - self.last_tissue_time <= self.MEMORY_DURATION and self.last_tissue is not None:
            tissue = self.last_tissue
        else:
            tissue = None
            self.last_tissue = None

        # 3. 真實高度，供爬升時的高度上限使用
        height = getattr(vision_data, 'height', None) if vision_data else None

        # ==========================================
        # 狀態機
        # ==========================================
        if self.state == "SEARCH":
            if tissue and now >= self.ignore_until:
                self.change_state("ALIGN")
            else:
                # 整段委派給漫遊引擎：避障、過彎、黑區脫困全部由它負責
                return self.explorer.calculate_command(user_input, vision_data)

        if self.state == "ALIGN":
            if not tissue:
                self.change_state("SEARCH")
            else:
                top_y = tissue['cy'] - tissue['h'] / 2.0
                bottom_y = tissue['cy'] + tissue['h'] / 2.0

                # 水平：把紙條轉到畫面正中
                error_x = tissue['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)
                if abs(error_x) < 20: yv = 0

                # 垂直：只要紙條橫過前進路線就好 (理由見 __init__ 的 PATH_MARGIN)
                path_top = self.target_cy - self.PATH_MARGIN
                path_bot = self.target_cy + self.PATH_MARGIN
                if bottom_y < path_bot:
                    error_y = path_bot - bottom_y      # 紙條整條在頭頂上 -> 爬升
                elif top_y > path_top:
                    error_y = path_top - top_y         # 紙條整條在腳下 -> 下降
                else:
                    error_y = 0.0                      # 已經橫過前進路線，高度不用動
                ud = self.pid_ud.compute(error_y)
                if error_y == 0.0:
                    ud = 0
                ud = min(ud, self.CLIMB_MAX)

                # 硬性高度上限：到頂了就不准再爬，不管視覺怎麼說
                capped = (self.MAX_HEIGHT is not None and height is not None
                          and height >= self.MAX_HEIGHT and error_y > 0)
                if not capped:
                    self.capped_since = None
                else:
                    ud = min(ud, 0)
                    if self.capped_since is None:
                        self.capped_since = now
                    if now - self.capped_since > self.UNREACHABLE_SEC:
                        self.capped_since = None
                        print(f"[紙條獵手] 已達高度上限 {self.MAX_HEIGHT}cm，紙條仍在頭頂上 -> 放棄這條，"
                              f"{self.IGNORE_SEC:.0f} 秒內不鎖定")
                        self.ignore_until = now + self.IGNORE_SEC
                        self.change_state("SEARCH")
                        return (0, 0, 0, 0)

                # 「橫過前進路線」只要下端過了畫面中心就算數 —— 紙條已經垂到機身高度，往前衝就撞得到。
                # 爬升目標則刻意設在中心再往下 PATH_MARGIN，兩者不能共用同一個門檻：
                # P 控制越接近目標爬得越慢，誤差剩 1~2 像素時算出來的 ud 會被截成 0，
                # 單一門檻的話會永遠卡在差一點點的地方 (實測停在 268，到不了 270)。
                on_path = top_y <= self.target_cy <= bottom_y
                aligned = abs(error_x) < self.ALIGN_TOL_X and on_path

                if aligned:
                    print(f"[紙條獵手] 鎖定目標 -> 直線往前衝 {self.CHARGE_TIME:.0f} 秒")
                    self.change_state("CHARGE")
                # 還沒對準就只轉向與調整高度，不前進

        elif self.state == "CHARGE":
            # 固定時間直線全速前進，中途不看偵測、不看深度、不修正方向
            fb = self.CHARGE_FB
            if time_in_state > self.CHARGE_TIME:
                self.hit_count += 1
                print(f"[紙條獵手] 衝撞完成 (累計 {self.hit_count} 次) -> 繼續搜尋")
                self.change_state("SEARCH")

        return (int(lr), int(fb), int(ud), int(yv))

    def get_mode(self) -> str:
        if self.state == "SEARCH":
            return f"TISSUE: ROAMING ({self.explorer.state})"
        return f"TISSUE: {self.state}"
