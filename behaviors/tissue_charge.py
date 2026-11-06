import random
import time

from behaviors.base import FlightBehavior
from behaviors.fluid_explore import FluidExploreControl
from utils.pid_controller import PIDController


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class TissueChargeControl(FlightBehavior):
    """
    衛生紙條獵手：漫遊找紙條 -> 持續校正逼近 -> 剪斷 -> 退開 -> 繼續漫遊。

    狀態機
        SEARCH   把指令整段委派給 FluidExploreControl，在場地裡漫遊避障
        TRACK    每一格同時修正轉向、高度與前進速度，一邊校正一邊逼近。
                 不論顏色先接近；紙條面積夠大才停住判斷顏色，是目標才衝 (見 JUDGE_*)
        COAST    目標貼到眼前看不見了 -> 以追蹤速度再滑行一小段補完最後幾公分
        BACKOFF  剪完之後先退開，再交還給漫遊
        DODGE    正前方 (機身會掃到) 有不該撞的紙條 -> 往空曠的一側平移繞過去，再交還漫遊 (不後退)；
                 不在中間的不理。兩側都很近、或平移太久繞不開才改原地轉 (見 _start_dodge)
        PROBE / DETOUR  非白紙條擋在通往白紙條的路線上 -> 平移看視差分前後，必要時繞過去

    ==========================================================
    為什麼廢掉「盲衝」
    ==========================================================
    舊版是鎖定後以 fb=100 盲衝 0.6 秒 (約 50 公分)，問題是精準度不好又容易撞牆 ——
    起衝距離每次不同，紙條後面是什麼也完全不看。而且實測錄影顯示那個狀態
    從來沒有真的執行過 (18 段 TRACK 全部以「跟丟」收場，盲衝一次都沒觸發)。

    改成：全程閉迴路，前進速度隨對準程度連續變化，最後只留一小段滑行 (COAST_SEC)
    補完鏡頭看不到的那幾公分。滑行速度是追蹤速度 (40) 不是衝刺速度 (100)。

    ==========================================================
    為什麼搜尋要用 FluidExploreControl 而不是原地旋轉
    ==========================================================
    原地轉只能看到同一個位置的視野，紙條掛在房間各處，轉完一圈看不到就卡死了。
    漫遊會帶著無人機換位置，而且那套弧線避障與黑區脫困已經調過、實飛驗證過。
    委派是整段委派 —— SEARCH 期間的 lr/fb/yv 由它決定；高度鎖定 (見 _lock_altitude)。

    ==========================================================
    瞄準點在畫面 75% 高度，不是正中央
    ==========================================================
    Tello 的鏡頭略朝下，往前飛時機身前傾又讓它更朝下，「無人機真正往前飛過去的那一列」
    在畫面中心上方。17:25 那趟往前飛時，遠處牆上與無人機同高的位置在 y≈77；
    17:40 那趟 9 段 TRACK 裡紙條下緣全部在畫面中心上方 (62~232)，
    「紙條橫過前進路線」用 y=240 判斷永遠不成立，480 格裡只有 6 格有往前飛。
    所以瞄準點 (畫面上的紅點) 放在 AIM_Y_RATIO，前進判斷以這一列為準：
    紙條有橫過這一列才往前衝。高度鎖定 (見 __init__)，不會為了對齊而升降。

    ==========================================================
    誠實的限制：最後約 4 公分是看不見的
    ==========================================================
    「持續校正到碰到為止」物理上做不到。紙條逼近到一定程度就會脹滿畫面、
    上下緣被切掉，框的形狀崩潰，YOLO 也早就離開訓練分布了。
    TissueDetector 對已鎖定的目標放寬形狀門檻，把閉迴路的極限從約 9.6cm
    延伸到約 4.3cm；剩下那幾公分由 COAST 補完。
    重點是「沒有定時盲衝」，不是「完全沒有盲區」。
    """

    # ==========================================
    # 幾何常數
    # ==========================================
    # 紙條實寬與相機水平焦距 (720x480 影像)。由框寬估距離：dist = STRIP_W * F / w
    STRIP_WIDTH_CM = 3.0
    FOCAL_PX = 461.0
    DIST_K = STRIP_WIDTH_CM * FOCAL_PX           # 1383

    # 瞄準點 (畫面上的紅點)，以畫面寬高的比例表示，由上往下量。
    # 0.25 = 畫面高度 75% 的位置 (720x480 -> y=120)。理由見類別說明。
    AIM_X_RATIO = 0.5
    AIM_Y_RATIO = 0.25
    FRAME_W, FRAME_H = 720, 480

    def __init__(self, now_fn=time.time):
        """
        :param now_fn: 取得目前時間的函式。離線用錄影重播狀態機時傳入假時鐘，
                       所有計時 (記憶衰減、滑行、放棄) 才會與實飛一致。
        """
        self._now = now_fn
        self.state = "SEARCH"
        self.state_start_time = self._now()

        # 搜尋階段的漫遊引擎。它需要的 depth_L/C/R 與 dark_* 由 TissueDetector 提供，
        # 欄位名稱與單位刻意和 fluid_explorer_vision.py 一致，可以直接餵進去。
        self.explorer = FluidExploreControl(now_fn=now_fn)
        self._rng = random.Random()

        # 瞄準點 (720x480)：紙條的十字要對齊這裡
        self.target_cx = int(self.FRAME_W * self.AIM_X_RATIO)     # 360
        self.target_cy = int(self.FRAME_H * self.AIM_Y_RATIO)     # 120

        self.pid_yv = PIDController(kp=0.4, ki=0.0, kd=0.15, limit=60)

        # --- 水平對準 ---
        # 容差不能是固定像素值：越靠近，同樣的側偏在畫面上就越大。
        # 距離 30cm 時 1 公分側偏就是 14 像素，固定 40 像素等於要求 3 公分內的精度，
        # 而那個距離下轉向幾乎修不動側偏 —— 實測會在最後 30cm 前卡死。
        # 改成隨紙條在畫面中的寬度放大：偏差只要小於紙條本身的寬度就撞得到。
        self.ALIGN_TOL_X = 40          # 遠距離的水平對準容許誤差 (像素)
        self.WIDTH_TOL_FACTOR = 1.5    # 近距離時容差 = 紙條寬度 x 這個倍數
        self.YAW_DEADBAND_PX = 12      # 偏航死區。前進不再卡在二元對準旗標之後，
                                       # 所以可以收得比舊版的 20 像素緊

        # --- 高度鎖定 (見 _lock_altitude) ---
        #
        # 這個模式不會自動改變高度，只有飛手按 W/S 可以。
        # 之前試過的幾種自動升降都出過事：
        #   視覺追十字往上爬     -> 撞天花板 (15:42)
        #   照高度讀值定高       -> Tello 讀值會隨起飛點、下方桌櫃改變 (同一個天花板讀過 130 也讀過 230)，
        #                          被騙著一路往上撞天花板 (19:54)
        #   衝撞前的垂直修正     -> 要撞的時候往下掉，沒撞到 (19:34)
        # 所以高度由飛手決定：先手動飛到紙條尖端附近的高度，再切進這個模式。
        # 追紙條時只追「紙條有橫過紅點那一列 (前進路線)」的；橫不過就 UNREACHABLE_SEC 後放棄換下一條。
        self.UNREACHABLE_SEC = 1.0     # 紙條沒有橫過前進路線超過這麼久 -> 放棄
        self.off_path_since = None

        # --- 前進 ---
        # 舊版是二元開關 (對準了就 50、沒對準就 0)，實測 TRACK 期間只有 12% 的格數
        # 在前進，最長連續前進才 1.3 秒。改成隨對準程度連續變化，邊修正邊逼近。
        # 這也才是「依辨識的物件中心持續校正飛行姿勢」的字面意思。
        self.FB_FAR = 50               # 遠距離的巡航速度
        self.FB_NEAR = 30              # 近距離放慢，PID 才來得及把偏差修回來
        self.FAR_CM = 120              # 超過這個距離算遠
        self.TRACK_MAX_SEC = 15.0      # 追這麼久還碰不到就放棄，避免卡在同一條上

        # --- 不論顏色先接近，夠近才判斷 (TissueDetector.JUDGE_DIST_CM) ---
        # 遠處的紙條顏色常判錯，偵測器不在遠處判斷，而是把它當成「還沒判斷」的目標交過來
        # (tissue["judged"] = False；tissue["far"] = 還太遠)。這裡的做法：
        #   far              照常對準、往前飛 (估距 JUDGE_SLOW_CM 以內放慢，到判斷距離才不會衝過頭)
        #   不是 far         停住 (只轉機頭對準)，等偵測器投票
        #   判定是目標       judged 變 True，照常衝
        #   判定不是目標     它出現在閃避清單上、就在正前方 -> 繞開 (DODGE)
        #   看了 JUDGE_MAX_SEC 還沒結果、或判斷中跟丟超過 JUDGE_LOST_SEC -> 放棄 (在正前方就繞開)
        # 還沒判斷的紙條絕不滑行 (COAST)：跟丟可能正是因為它被判定為不是目標。
        # 所有估距 (tissue["dist_cm"]) 都是偵測器用紙條厚度量的，不是框寬 (見 TissueDetector)。
        self.JUDGE_SLOW_CM = 130
        self.JUDGE_MAX_SEC = 2.5
        self.JUDGE_LOST_SEC = 0.5
        self.APPROACH_IGNORE_SEC = 3.0 # 轉開之後這段時間不接近還沒判斷的紙條，讓漫遊把無人機帶開
        self.approach_ignore_until = 0.0
        self.judge_since = None        # 開始近距離判斷的時間
        self.judge_last = None         # 判斷中那條最後一次被看到的樣子 (跟丟時決定往哪邊轉開)
        self.track_judged = True       # 正在追的這條判斷過了沒

        # --- 深度避牆 ---
        # 紙條吊在離牆兩公尺的空間時，depth_ahead 量到的是後面的牆，逼近全程不會被擋。
        # 紙條就貼著牆時，避牆會在接觸前停住 —— 這是刻意的取捨，寧可剪不到也不撞牆。
        #
        # 用 depth_ahead 而不是 depth_C：深度取樣帶正好在畫面中央，對準好的紙條
        # 就落在裡面，近距離時 depth_C 量到的是紙條自己，會在每次逼近時誤觸發。
        # TissueDetector 會把目標框的欄位挖掉再取平均。
        #
        # ⚠️ 深度模型在 60cm 以下的行為沒有驗證過，這個值是保守起手值。
        # depth_ahead 有寫進 CSV，第一趟飛完再依實際數據校正。
        self.APPROACH_STOP_CM = 60

        # --- 閃避不該撞的紙條 (見 _path_strips) ---
        # TissueDetector 把顏色不符的紙條 (白色模式下的彩色紙條、彩色模式下的白紙條)
        # 放在 vision_data.avoid。深度模型幾乎看不到細細的紙條，所以要另外處理。
        # 只閃「在中間」的：紙條要橫過前進路線 (紅點那一列，上下各留 AVOID_MARGIN_PX ——
        # 機身有厚度，紙條尖端只比路線高一點點一樣會掃到)，而且水平方向在機身會掃到的範圍內
        # (_in_corridor)。不在中間的一律不理 —— 不轉、也不影響漫遊的轉彎方向。
        self.AVOID_MARGIN_PX = 40
        self.BODY_HALF_CM = 12         # 機身 (含螺旋槳護罩) 半寬，判斷「擋不擋在前進路線上」用

        # --- 正前方有不該撞的紙條：往空曠的一側平移繞過去 (DODGE 狀態，見 _start_dodge) ---
        # 10/19 20:46 錄影：90 秒裡「搖頭拒絕」11 次共 26 秒 -> 改成只理會正前方的、不搖頭不後退。
        # 之後的版本是原地轉 35~75 度，但那會把漫遊方向整個改掉，等於被紙條趕走。
        # 現在改成平移 (機頭不動)：紙條一離開機身會掃到的範圍 (再多 DODGE_HYST_PX) 就交還漫遊，
        # 漫遊照原本的方向往前 —— 從紙條旁邊繞過去。
        #   往哪邊   左右深度差超過 explorer.SIDE_RATIO -> 往比較空的一側；
        #            差不多 -> 往離紙條比較遠的一側 (要平移的距離比較短)
        #   改原地轉 兩側深度都低於 DODGE_OPEN_MIN_CM (平移過去也是撞牆)、
        #            或平移 DODGE_SLIDE_MAX_SEC 還沒繞開 -> 往同一側轉 DODGE_MIN~MAX_DEG
        # 畫面上的大橫幅 (TissueDetector) 照樣顯示，作為回饋。
        self.DODGE_DIST = 250
        self.DODGE_LR = 25             # 平移速度 (rc 25 約每秒 25 公分)
        self.DODGE_HYST_PX = 8         # 要比機身會掃到的範圍再多分開這麼多才算繞開 (遲滯，不會一放手又掃到)
        # 另外至少要多留這麼多公分 (換算成紙條那個深度的像素)：像素的餘裕在近距離換算成公分很小
        # (60cm 時 18px 只有約 2cm)，Tello 懸停會飄，近的時候要用公分保證
        self.DODGE_MARGIN_CM = 8
        self.DODGE_TRACK_GATE_PX = 60  # 平移時跨格追同一條紙條的水平距離
        self.DODGE_SLIDE_MAX_SEC = 2.5 # 平移這麼久 (約 60 公分) 還繞不開就改原地轉
        self.DODGE_LOST_SEC = 0.3      # 平移中紙條連續看不到這麼久才算看不到
        self.DODGE_BLIND_SEC = 1.0     # 紙條看不到時至少平移這麼久 (約 25 公分 > 機身半寬 + 餘裕)
        self.DODGE_OPEN_MIN_CM = 150   # 兩側深度都低於這個 -> 不平移，改原地轉 (= 漫遊的 CRITICAL_DIST)
        self.DODGE_YV = 60
        self.DODGE_DEG_PER_SEC = 60.0  # rc yaw 60 約每秒 60 度
        self.DODGE_MIN_DEG = 35
        self.DODGE_MAX_DEG = 75
        self.DODGE_BIAS_SEC = 5.0      # 原地轉之後這段時間內，漫遊轉彎優先往同一側
        self.DODGE_COOLDOWN = 2.0      # 原地轉之後這段時間內漫遊不再觸發 (平移靠遲滯，不需要冷卻)
        self.DODGE_NEAR_CM = 80        # 比這個還近就不等投票結果，直接避開
        self.dodge_dir = 1
        self.dodge_mode = "slide"      # "slide" 平移 / "turn" 原地轉
        self.dodge_sec = 0.0
        self.dodge_dist = 0.0
        self.dodge_kind = ""
        self.dodge_cx = 0.0            # 平移時追的那條紙條目前在畫面上的位置
        self.dodge_lost_since = None
        self.dodge_cooldown_until = 0.0

        # --- 運動視差：非白紙條在白紙條前面還是後面？(PROBE / DETOUR 狀態) ---
        # 舊做法用「紙條寬度推算的距離」比前後，但紙條會扭轉、捲曲，寬度很不可靠，
        # 後面的非白紙條常被當成比較近 -> 直接搖頭放棄白紙條。
        # 改用視差：往旁邊平移一小段，近的東西在畫面上移動得多、遠的移動得少。
        # 比的是兩條紙條移動量的「差」—— 平移時機頭若偏了一點，兩條會一起移動同樣的量，
        # 相減就抵消了；而且完全不需要知道紙條實際多寬。
        #
        #   非白紙條在後面、離白紙條夠遠   -> 直接衝白紙條
        #   在前面 / 在後面但太近 / 看不出來 -> 繼續往同一側平移 (繞路)，同時轉機頭對準白紙條，
        #                                     兩條在畫面上分得夠開 (機身不會掃到) 就衝
        #   繞了 DETOUR_MAX_SEC 還分不開      -> 才搖頭拒絕、放棄這條白紙條
        self.PROBE_LR = 20             # 探測時的平移速度
        self.PROBE_SEC = 0.8           # 探測多久 (約 16 公分)
        self.PARALLAX_MIN_PX = 8       # 兩條紙條移動量差超過這麼多像素才下結論
        self.BEHIND_GAP_MIN_CM = 70    # 非白紙條在後面、而且離白紙條至少這麼遠才直接衝 (滑行 40cm + 餘裕)
        self.DETOUR_LR = 20
        # 最多繞這麼久 (約 80 公分)。兩條紙條前後只差 40cm 又排成一直線時，
        # 要橫移約 50 公分畫面上才分得夠開 (模擬實測)，2.5 秒不夠，會白白放棄白紙條
        self.DETOUR_MAX_SEC = 4.0
        self.DETOUR_MARGIN_PX = 10
        # 繞路要多分開這麼多才結束 (遲滯)。沒有的話剛好分開就回去衝，
        # 機頭一修正又「擋路」，在繞路與探測之間來回 (模擬實測)
        self.DETOUR_HYST_PX = 8
        self.PARALLAX_MEMORY_SEC = 3.0 # 判斷結果記住多久，同一組紙條不重複探測
        self.PENDING_WAIT_SEC = 0.6    # 漫遊時有白紙條正在投票，先停住等這麼久再決定要不要避開
        self.probe_dir = 1
        self.probe_samples = []
        self.probe_avoid_cx = 0.0
        self.probe_target_dist = 0.0
        self.parallax_memory = None    # {"decision", "offset", "until", "dir"}
        self.parallax_text = ""        # 最近一次視差判斷 (log / 畫面顯示用)
        self.detour_dir = 1
        self.detour_avoid_cx = 0.0
        self.detour_lost_since = None
        self.pending_since = None

        # --- 最後的滑行 ---
        # 紙條從眼前消失代表已經貼到很近 (脹滿畫面或離開視野)，這時候偵測沒有意義。
        # 但這不是盲衝 —— 速度是追蹤速度 (40) 而不是舊版的衝刺速度 (100)，
        # 而且要「剛剛確實在前進」且「估距夠近」才會滑，遠距離跟丟就直接回去重找。
        #
        # 原本設 0.35 秒 (依 rc 40 約 14~40cm/s 推算是 5~14 公分)，之後依實飛調成 1 秒
        # (推算約 14~40 公分)。滑過頭的部分會被 BACKOFF 退回來。
        self.COAST_FB = 40
        self.COAST_SEC = 1
        self.COAST_MAX_DIST_CM = 60
        # 鎖定的目標被偵測器撤銷 (靠近後看到彩色條紋) 時，閃避清單上那條要在畫面上離目標最後的位置
        # 這麼近才算「同一條」—— 撤銷的就是它，不要滑行過去 (見 _auto_command)
        self.REVOKE_GATE_PX = 60

        # --- 剪完之後先退開 ---
        # 剪完時無人機離紙條後面的東西只有幾公分，而漫遊引擎一接手就會送 fb=50，
        # 此時深度模型正處在最不可信的近距離。先退開才能把深度模型拉回有效範圍，
        # 也把剛剪過的紙條推出取得門檻。
        # 0.8 秒 x 40 約 30 公分，小於剛剛前進的距離，不會倒回原處。
        self.BACKOFF_FB = 40
        self.BACKOFF_SEC = 0.8

        # --- 放棄後的冷卻 ---
        # 搆不到的紙條會一直留在畫面裡，直接回到搜尋的話下一格又會被鎖定、又卡住，
        # 無限循環。放棄後這段時間內忽略偵測，讓漫遊把無人機帶到別處。
        self.IGNORE_SEC = 3.0
        self.ignore_until = 0.0

        # --- 視覺記憶：擋住偵測的單格閃爍 ---
        # 舊版設 0.5 秒，配上偵測端 15 格的鎖定保留，最壞情況會用一筆已經舊了
        # 0.3 秒的觀測開迴路飛 0.82 秒 —— 而且是每一格都拿同一個凍結的誤差重跑 PID，
        # D 項歸零、輸出鎖死在 kp*error，等於連續半秒送出固定的最大偏航
        # (約 30 度沒有回授的旋轉)。實測偏航有 35% 的時間頂在 ±60 就是這樣來的。
        #
        # 現在縮到 0.25 秒 (偵測端也從 15 格縮到 5 格 = 0.1 秒)，
        # 而且記憶期間改成讓上一個指令衰減，完全不呼叫 PID ——
        # prev_error 保持乾淨，真觀測回來時 D 項才有意義。
        self.MEMORY_DURATION = 0.25
        self.last_tissue = None
        self.last_tissue_time = 0.0
        self.last_fb = self.last_ud = self.last_yv = 0
        self.last_dist_cm = 999.0

        self.cut_count = 0             # 累計剪斷次數，只作顯示用
        self._log = {}                 # 給 VideoRecorder 的診斷欄位

    # ==========================================
    def reset(self):
        """切換到這個模式時 (tello_app.toggle_mode) 呼叫"""
        self.change_state("SEARCH")
        self.explorer.change_state("FORWARD")

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = self._now()
            # 回到搜尋時把漫遊引擎重設成直線前進，
            # 不然它會沿用上次離開時的過彎狀態，一接手就莫名其妙開始繞圈。
            if new_state == "SEARCH":
                self.explorer.change_state("FORWARD")
            print(f"[紙條獵手] {self.state}")

    def _give_up(self, now, why):
        print(f"[紙條獵手] {why} -> 放棄這條，{self.IGNORE_SEC:.0f} 秒內不鎖定")
        self.ignore_until = now + self.IGNORE_SEC
        self.off_path_since = None
        self.change_state("SEARCH")

    def _decayed(self, now):
        """
        記憶期間的輸出：讓上一個指令線性衰減到 0，而不是重跑 PID。

        重跑 PID 會用同一個凍結的誤差算出固定的最大輸出 (見 MEMORY_DURATION 的說明)。
        衰減的話最壞情況是 0.25 秒內平滑歸零，而且 PID 的 prev_error 不會被污染。
        """
        k = max(0.0, 1.0 - (now - self.last_tissue_time) / self.MEMORY_DURATION)
        return (0, int(self.last_fb * k), int(self.last_ud * k), int(self.last_yv * k))

    @staticmethod
    def _blank_str(vd):
        if vd is None or not hasattr(vd, 'blank_L'):
            return ""
        return f"{vd.blank_L:.2f}/{vd.blank_C:.2f}/{vd.blank_R:.2f}" + \
            ("<L" if vd.blank_near_L else "") + ("R>" if vd.blank_near_R else "") + \
            ("!C" if vd.blank_near_C else "")

    def _path_strips(self, vision_data):
        """vision_data.avoid 裡會擋到前進路線 (紅點那一列，含上下餘裕) 的紙條"""
        strips = getattr(vision_data, 'avoid', None) if vision_data else None
        if not strips:
            return []
        m = self.AVOID_MARGIN_PX
        return [a for a in strips if a["top_y"] - m <= self.target_cy <= a["bottom_y"] + m]

    def _in_corridor(self, a):
        """這條紙條在正前方、機身直直往前會掃到 (側邊不擋路的不算)"""
        return abs(a["cx"] - self.target_cx) < self._clear_px(a)

    @staticmethod
    def _strip_px(a):
        """
        紙條在畫面上的厚度 (px)，也就是那個深度「3 公分 = 幾像素」的比例尺。
        偵測器有量 (strip_px) 就用它；框寬在紙條斜掛、晃動時會寬好幾倍，只當退路。
        """
        return max(1.0, a.get("strip_px") or a["w"])

    def _clear_px(self, a):
        """
        非白紙條要跟白紙條在畫面上分開多少像素，對準白紙條中心衝過去時機身才不會掃到它
        (= 非白紙條半寬 + 機身半寬 + 餘裕，都投影到非白紙條那個深度)。
        機身半寬的投影 = BODY_HALF_CM x FOCAL / Z = (BODY_HALF_CM / 紙條實寬) x 紙條厚度，與距離無關。
        """
        s = self._strip_px(a)
        return s / 2.0 + self.BODY_HALF_CM / self.STRIP_WIDTH_CM * s + self.DETOUR_MARGIN_PX

    def _conflict_strip(self, strips, tissue):
        """擋在通往白紙條路線上的非白紙條 (畫面上離白紙條太近)；有好幾條就取最靠近白紙條的"""
        cands = [a for a in strips if abs(a["cx"] - tissue["cx"]) < self._clear_px(a)]
        return min(cands, key=lambda a: abs(a["cx"] - tissue["cx"])) if cands else None

    @staticmethod
    def _nearest(strips, cx, gate):
        """畫面上離 cx 最近、而且在 gate 像素內的紙條 (跨格追同一條用)"""
        best = None
        for a in strips:
            d = abs(a["cx"] - cx)
            if d <= gate and (best is None or d < abs(best["cx"] - cx)):
                best = a
        return best

    def _remembered(self, conflict, tissue, now):
        """之前對「同一組紙條」(相對位置差不多) 的視差判斷 (整筆記憶)，過期或對不上就回 None"""
        m = self.parallax_memory
        if not m or now > m["until"]:
            return None
        if abs((conflict["cx"] - tissue["cx"]) - m["offset"]) <= 80:
            return m
        return None

    def _start_probe(self, conflict, tissue, now):
        """往遠離非白紙條的那一側平移，量兩條紙條的視差"""
        self.probe_dir = 1 if conflict["cx"] <= tissue["cx"] else -1     # 非白紙條在左 (或正前) -> 往右
        self.probe_avoid_cx = conflict["cx"]
        self.probe_samples = []
        self.probe_target_dist = tissue.get("dist_cm", self.DIST_K / self._strip_px(tissue))
        print(f"[紙條獵手] 非白紙條擋在白紙條的路線上 -> 往{'右' if self.probe_dir > 0 else '左'}平移，"
              f"用視差判斷誰在前面")
        self.change_state("PROBE")

    def _parallax_decision(self):
        """
        :return: (結論, 移動量差 px, 估計前後間距 cm)
                 結論是 "front" / "behind" / "behind-close" / "unsure" / "no-data"
        """
        early = [s_ for s_ in self.probe_samples if s_[0] <= 0.35 * self.PROBE_SEC and s_[2] is not None]
        late = [s_ for s_ in self.probe_samples if s_[0] >= 0.65 * self.PROBE_SEC and s_[2] is not None]
        if len(early) < 2 or len(late) < 2:
            return "no-data", 0.0, 0.0
        mean = lambda xs: sum(xs) / len(xs)
        shift_t = mean([s_[1] for s_ in late]) - mean([s_[1] for s_ in early])
        shift_a = mean([s_[2] for s_ in late]) - mean([s_[2] for s_ in early])
        # 平移時所有東西往反方向移動，越近移得越多。乘上 -probe_dir 之後 > 0 = 非白紙條比較近
        diff = (shift_a - shift_t) * (-self.probe_dir)
        if diff >= self.PARALLAX_MIN_PX:
            return "front", diff, 0.0
        if diff <= -self.PARALLAX_MIN_PX:
            # 在後面。位移與距離成反比：Z_非白 / Z_白 ≈ |位移_白| / |位移_非白|
            ratio = abs(shift_t) / max(abs(shift_a), 1.0)
            gap = self.probe_target_dist * (ratio - 1.0)
            return ("behind" if gap >= self.BEHIND_GAP_MIN_CM else "behind-close"), diff, gap
        return "unsure", diff, 0.0

    def _probe_cmd(self, now, current, strips):
        """PROBE：側向平移 PROBE_SEC 秒 (機頭不動)，同時記錄兩條紙條的位置，結束時下結論"""
        t = now - self.state_start_time
        if current is not None:
            a = self._nearest(strips, self.probe_avoid_cx, 60)
            if a is not None:
                self.probe_avoid_cx = a["cx"]
            self.probe_samples.append((t, current["cx"], a["cx"] if a is not None else None))
        if t < self.PROBE_SEC:
            return (self.probe_dir * self.PROBE_LR, 0, 0, 0)

        decision, diff, gap = self._parallax_decision()
        self.parallax_text = f"{decision} {diff:+.0f}px"
        self._log["parallax"] = self.parallax_text
        last = next((s_ for s_ in reversed(self.probe_samples) if s_[2] is not None), None)
        offset = (last[2] - last[1]) if last else 0.0
        if decision == "no-data" and not any(s_[1] is not None for s_ in self.probe_samples):
            print("[紙條獵手] 視差探測時白紙條跟丟了 -> 回去漫遊")
            self.change_state("SEARCH")
            return (0, 0, 0, 0)
        if decision == "behind":
            print(f"[紙條獵手] 視差：非白紙條在白紙條後面約 {gap:.0f}cm (差 {diff:+.0f}px) -> 直接衝白紙條")
            self.parallax_memory = {"decision": "behind", "offset": offset,
                                    "until": now + self.PARALLAX_MEMORY_SEC, "dir": self.probe_dir}
            self.change_state("TRACK")
            return (0, 0, 0, 0)
        why = {"front": f"在前面 (差 {diff:+.0f}px)",
               "behind-close": f"在後面但只隔約 {gap:.0f}cm，衝過頭會撞到",
               "unsure": f"看不出前後 (差 {diff:+.0f}px)",
               "no-data": "非白紙條沒看清楚"}[decision]
        print(f"[紙條獵手] 視差：非白紙條{why} -> 繞路")
        self.parallax_memory = {"decision": "front", "offset": offset,
                                "until": now + self.PARALLAX_MEMORY_SEC, "dir": self.probe_dir}
        conflict = {"cx": self.probe_avoid_cx}
        self._start_detour(conflict, current, now, self.probe_dir)
        return (self.detour_dir * self.DETOUR_LR, 0, 0, 0)

    def _detour_cmd(self, now, current, strips, vision_data=None):
        """
        DETOUR：繼續往旁邊平移，同時轉機頭對準白紙條，直到非白紙條離開機身會掃到的範圍。
        繞太久還分不開才搖頭拒絕。
        """
        t = now - self.state_start_time
        if current is None:
            if now - self.last_tissue_time > 1.0:
                print("[紙條獵手] 繞路時白紙條跟丟了 -> 回去漫遊")
                self.change_state("SEARCH")
                return (0, 0, 0, 0)
            return (self.detour_dir * self.DETOUR_LR, 0, 0, 0)

        error_x = current["cx"] - self.target_cx
        yv = self.pid_yv.compute(error_x)
        if abs(error_x) < self.YAW_DEADBAND_PX:
            yv = 0

        a = self._nearest(strips, self.detour_avoid_cx, 80)
        if a is not None:
            self.detour_avoid_cx = a["cx"]
            self.detour_lost_since = None
            if abs(a["cx"] - current["cx"]) >= self._clear_px(a) + self.DETOUR_HYST_PX:
                print("[紙條獵手] 繞開了，非白紙條不在路線上 -> 衝白紙條")
                self._remember_detour(a["cx"] - current["cx"], now)
                self.change_state("TRACK")
                return (0, 0, 0, int(yv))
        else:
            # 非白紙條看不到了 (移出畫面) 撐 0.5 秒確認，不是單格漏偵測
            if self.detour_lost_since is None:
                self.detour_lost_since = now
            elif now - self.detour_lost_since > 0.5:
                print("[紙條獵手] 非白紙條已經離開畫面 -> 衝白紙條")
                self._remember_detour(self.detour_avoid_cx - current["cx"], now)
                self.change_state("TRACK")
                return (0, 0, 0, int(yv))

        if t > self.DETOUR_MAX_SEC:
            self.ignore_until = now + self.IGNORE_SEC
            strip = a if a is not None else {"cx": self.detour_avoid_cx, "w": 10,
                                             "dist_cm": self.probe_target_dist}
            return self._avoid_or_ignore(strip, now, vision_data,
                                         "繞路太久還分不開，放棄這條白紙條")
        return (self.detour_dir * self.DETOUR_LR, 0, 0, int(yv))

    def _remember_detour(self, offset, now):
        """
        繞開之後記住「這一條在前面、往哪邊繞」。衝的途中又擋到 (機頭修正、飄移) 時，
        沿同一個方向繼續繞 —— 重新探測的話，方向會依「非白紙條在哪一邊」決定，
        而繞過去之後它已經跑到另一邊了，會往回走、把剛剛繞的路退回去。
        """
        self.parallax_memory = {"decision": "front", "offset": offset,
                                "until": now + self.PARALLAX_MEMORY_SEC, "dir": self.detour_dir}

    def _start_detour(self, conflict, tissue, now, direction):
        self.detour_dir = direction
        self.detour_avoid_cx = conflict["cx"]
        self.detour_lost_since = None
        self.change_state("DETOUR")

    def _start_dodge(self, strip, now, vision_data, why=None):
        """
        正前方 (機身會掃到) 有不該撞的紙條：往空曠的一側平移繞過去 (理由見 __init__ 的 DODGE_*)。
        why：終端機上的理由，沒給就是「正前方有有色 / 白色紙條」。
        """
        depth_L = getattr(vision_data, 'depth_L', 999.0) if vision_data else 999.0
        depth_R = getattr(vision_data, 'depth_R', 999.0) if vision_data else 999.0
        hi, lo = max(depth_L, depth_R), max(min(depth_L, depth_R), 1.0)
        if hi / lo >= self.explorer.SIDE_RATIO:
            self.dodge_dir = 1 if depth_R > depth_L else -1                 # 往比較空的一側
            side_why = "比較空"
        else:
            self.dodge_dir = 1 if strip["cx"] < self.target_cx else -1      # 紙條在左 -> 往右
            side_why = "離紙條比較遠"
        self.dodge_dist = strip.get("dist_cm", 0.0)
        if strip.get("judged") is False:
            self.dodge_kind = "還沒判斷顏色的"
        else:
            self.dodge_kind = "白色 (不是目標)" if strip.get("target") == "COLOR" else "有色"
        if why is None:
            why = f"正前方有{self.dodge_kind}紙條 ({self.dodge_dist:.0f}cm)"
        self.dodge_cx = strip["cx"]
        self.dodge_lost_since = None
        self.judge_since = None
        self.track_judged = True
        self.change_state("DODGE")
        if hi < self.DODGE_OPEN_MIN_CM:
            print(f"[紙條獵手] {why} -> 兩側都很近 (L{depth_L:.0f} R{depth_R:.0f}cm)，不平移")
            self._start_dodge_turn(now)
            return
        self.dodge_mode = "slide"
        self.state_start_time = now
        print(f"[紙條獵手] {why} -> 往{'右' if self.dodge_dir > 0 else '左'}平移繞過去 ({side_why})")

    def _start_dodge_turn(self, now):
        """改成原地往同一側轉一個隨機角度 (兩側都很近、或平移繞不開)"""
        angle = self._rng.uniform(self.DODGE_MIN_DEG, self.DODGE_MAX_DEG)
        self.dodge_mode = "turn"
        self.dodge_sec = angle / self.DODGE_DEG_PER_SEC
        self.explorer.prefer_side(self.dodge_dir, self.DODGE_BIAS_SEC)
        print(f"[紙條獵手] 改成原地往{'右' if self.dodge_dir > 0 else '左'}轉約 {angle:.0f} 度")
        self.state_start_time = now

    def _dodge_cmd(self, now, vision_data):
        """
        DODGE 的每一格，不前進、不後退。
          平移：每格找同一條紙條，離開機身會掃到的範圍 (再多 DODGE_HYST_PX) 就交還漫遊；
                連續看不到 DODGE_LOST_SEC、而且已經平移 DODGE_BLIND_SEC 以上也算繞開；
                平移 DODGE_SLIDE_MAX_SEC 還繞不開 -> 改原地轉
          原地轉：轉完就交還漫遊
        """
        t = now - self.state_start_time
        if self.dodge_mode == "turn":
            if t < self.dodge_sec:
                return (0, 0, 0, self.dodge_dir * self.DODGE_YV)
            self.dodge_cooldown_until = now + self.DODGE_COOLDOWN
            return self._end_dodge(now)

        strips = list(getattr(vision_data, 'avoid', None) or []) if vision_data else []
        tissue = getattr(vision_data, 'tissue', None) if vision_data else None
        if tissue is not None and tissue.get("judged") is False:
            strips.append(tissue)            # 判斷不出來而繞開的那條不在閃避清單上
        a = self._nearest(strips, self.dodge_cx, self.DODGE_TRACK_GATE_PX)
        if a is not None:
            self.dodge_cx = a["cx"]
            self.dodge_lost_since = None
            if abs(a["cx"] - self.target_cx) >= self._dodge_clear_px(a):
                print("[紙條獵手] 繞開了 (紙條不在機身會掃到的範圍) -> 繼續漫遊")
                return self._end_dodge(now)
        else:
            if self.dodge_lost_since is None:
                self.dodge_lost_since = now
            if now - self.dodge_lost_since >= self.DODGE_LOST_SEC and t >= self.DODGE_BLIND_SEC:
                print("[紙條獵手] 紙條看不到了、已經平移夠遠 -> 繼續漫遊")
                return self._end_dodge(now)
        if t > self.DODGE_SLIDE_MAX_SEC:
            print(f"[紙條獵手] 平移 {self.DODGE_SLIDE_MAX_SEC:.1f} 秒還繞不開")
            self._start_dodge_turn(now)
            return (0, 0, 0, self.dodge_dir * self.DODGE_YV)
        return (self.dodge_dir * self.DODGE_LR, 0, 0, 0)

    def _dodge_clear_px(self, a):
        """
        平移要讓紙條離畫面中央多遠才算繞開：機身會掃到的範圍再加遲滯，
        而且至少要多出 DODGE_MARGIN_CM (紙條那個深度 1 公分 = 紙條厚度 / 紙條實寬 像素)
        """
        cm = self.BODY_HALF_CM + self.STRIP_WIDTH_CM / 2.0 + self.DODGE_MARGIN_CM
        return max(self._clear_px(a) + self.DODGE_HYST_PX, cm * self._strip_px(a) / self.STRIP_WIDTH_CM)

    def _end_dodge(self, now):
        """交還漫遊。接下來一段時間不接近還沒判斷的紙條 (判斷不出來而繞開的那條還在畫面裡)"""
        self.approach_ignore_until = now + self.APPROACH_IGNORE_SEC
        self.change_state("SEARCH")
        return (0, 0, 0, 0)

    def _avoid_or_ignore(self, strip, now, vision_data, why):
        """放棄的紙條：在正前方 (機身會掃到) 就繞過去 (DODGE)；不在中間就不理，直接回漫遊"""
        if self._in_corridor(strip):
            self._start_dodge(strip, now, vision_data, why=why)
            return self._dodge_cmd(now, vision_data)
        print(f"[紙條獵手] {why} -> 不在正前方，不用閃，繼續漫遊")
        self.judge_since = None
        self.track_judged = True
        return self._end_dodge(now)

    def _blocking_strips(self, path_strips):
        """正前方、DODGE_DIST 以內、機身會掃到的不該撞紙條"""
        return [a for a in path_strips
                if a["dist_cm"] <= self.DODGE_DIST and self._in_corridor(a)]

    def alert_text(self):
        """畫面中央的大字提示 (tello_app 會畫出來)；沒有要提示時回傳 None"""
        if self.state == "DODGE":
            return "AVOID NON-TARGET"
        if self.state == "TRACK" and self.judge_since is not None:
            return "CHECKING TARGET"
        if self.state == "PROBE":
            return "PARALLAX CHECK"
        if self.state == "DETOUR":
            return "DETOUR AROUND NON-TARGET"
        return None

    @staticmethod
    def _lock_altitude(cmd):
        """高度鎖定：所有自動狀態的 ud 一律歸 0 (理由見 __init__ 的「高度鎖定」)"""
        lr, fb, _, yv = cmd
        return (lr, fb, 0, yv)

    # ==========================================
    def calculate_command(self, user_input, vision_data):
        # 人工接管優先，並把狀態重置回搜尋 —— 只有飛手可以改變高度
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("SEARCH")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        cmd = self._auto_command(user_input, vision_data)
        return self._lock_altitude(tuple(int(c) for c in cmd))

    def _auto_command(self, user_input, vision_data):
        lr = fb = ud = yv = 0
        now = self._now()
        time_in_state = now - self.state_start_time

        # 2. 視覺資料
        current = getattr(vision_data, 'tissue', None) if vision_data else None
        depth_ahead = getattr(vision_data, 'depth_ahead', 999.0) if vision_data else 999.0
        depth_unreliable = (getattr(vision_data, 'depth_unreliable', False)
                            if vision_data else False)

        using_memory = False
        if current is not None:
            self.last_tissue = current
            self.last_tissue_time = now
            tissue = current
            if self.state == "TRACK":
                if current.get("judged", True) and not self.track_judged:
                    print("[紙條獵手] 近距離看清楚了：是目標 -> 衝")
                    self.state_start_time = now          # 衝刺另外計時 (TRACK_MAX_SEC)
                    time_in_state = 0.0
                self.track_judged = current.get("judged", True)
                if self.track_judged or current.get("far", False):
                    self.judge_since = None
                elif self.judge_since is None:
                    self.judge_since = now
                self.judge_last = current
        elif (self.last_tissue is not None
              and now - self.last_tissue_time <= self.MEMORY_DURATION):
            tissue = self.last_tissue
            using_memory = True
        else:
            tissue = None
            self.last_tissue = None

        self._log = {
            "depth_ahead": round(depth_ahead, 1),
            "n_conf": getattr(vision_data, 'n_confirmed', "") if vision_data else "",
            "n_rej": getattr(vision_data, 'n_rejected', "") if vision_data else "",
            # 白色無紋理欄比例 L/C/R，大寫 = 判定為障礙 (見 utils/blank_region.py)
            "blank": self._blank_str(vision_data),
        }

        path_strips = self._path_strips(vision_data)
        self._log["avoid"] = (round(min(a["dist_cm"] for a in path_strips))
                              if path_strips else "")
        if self.state == "PROBE":
            self._log["parallax"] = "probing"
        elif self.state == "DETOUR":
            self._log["parallax"] = self.parallax_text

        # ==========================================
        # 狀態機
        # ==========================================
        if self.state == "SEARCH":
            near = self._blocking_strips(path_strips)
            pending = getattr(vision_data, 'n_pending', 0) if vision_data else 0
            approach_ok = tissue is not None and (tissue.get("judged", True)
                                                  or now >= self.approach_ignore_until)
            if tissue and not using_memory and now >= self.ignore_until and approach_ok:
                self.pending_since = None
                self.track_judged = tissue.get("judged", True)
                self.judge_since = (now if not self.track_judged and not tissue.get("far", False)
                                    else None)
                self.judge_last = tissue
                self.change_state("TRACK")
            elif near and now >= self.dodge_cooldown_until:
                nearest = min(near, key=lambda a: a["dist_cm"])
                # 畫面裡還有紙條正在投票 (可能是白紙條)，非白紙條又不是近在眼前 ->
                # 先停住等投票結果，不要急著拒絕 (提高剪白紙條的機會)
                if pending and nearest["dist_cm"] >= self.DODGE_NEAR_CM:
                    if self.pending_since is None:
                        self.pending_since = now
                    if now - self.pending_since < self.PENDING_WAIT_SEC:
                        return (0, 0, 0, 0)
                self.pending_since = None
                self._start_dodge(nearest, now, vision_data)
                return self._dodge_cmd(now, vision_data)
            else:
                self.pending_since = None
                # 整段委派給漫遊引擎：避障、過彎、黑區脫困全部由它負責
                # (白牆已在 TissueDetector 覆寫成 depth_L/C/R 的近距離；
                #  不在中間的不該撞紙條不理，見 AVOID_MARGIN_PX 的說明)
                return self.explorer.calculate_command(user_input, vision_data)

        # 上面可能剛從 SEARCH 切到 TRACK，計時要重新取 ——
        # 否則這一格用的還是「漫遊了多久」，只要漫遊超過 TRACK_MAX_SEC，
        # 追蹤的第一格就會直接觸發「追太久」而放棄。紙條稀疏的場地漫遊超過
        # 15 秒是常態，那等於整個模式永遠追不到東西。
        time_in_state = now - self.state_start_time

        if self.state == "TRACK" and not self.track_judged:
            # 還沒判斷的紙條：近距離看清楚的結果
            block = self._blocking_strips(path_strips)
            if block:
                # 正前方出現不該撞的紙條 —— 多半就是剛剛在看的這條被判定不是目標
                a = min(block, key=lambda a: a["dist_cm"])
                kind = "白色 (不是目標)" if a.get("target") == "COLOR" else "有色"
                self._start_dodge(a, now, vision_data,
                                  why=f"近距離看清楚了：{kind}紙條 ({a['dist_cm']:.0f}cm)")
                return self._dodge_cmd(now, vision_data)
            if self.judge_since is not None:
                stuck = now - self.judge_since > self.JUDGE_MAX_SEC
                lost = current is None and now - self.last_tissue_time > self.JUDGE_LOST_SEC
                if stuck or lost:
                    # 放棄這條：在正前方就繞過去，不在中間就直接回漫遊
                    last = self.judge_last or {"cx": self.target_cx, "w": 10, "dist_cm": 0.0}
                    why = (f"看了 {self.JUDGE_MAX_SEC:.1f} 秒還判斷不出來" if stuck
                           else "近距離判斷中跟丟了")
                    return self._avoid_or_ignore(last, now, vision_data, why)
                if current is None:
                    return (0, 0, 0, 0)                  # 判斷中短暫跟丟：停住等它回來

        if (self.state == "TRACK" and self.track_judged and current is None
                and self.last_tissue is not None):
            # 鎖定的目標不見了，同一個位置卻出現撤銷掉的紙條 —— 偵測器靠近之後看到彩色條紋，
            # 把它撤銷了 (TissueDetector._colour_watch)。不可以當成「貼到眼前」去滑行 (COAST)：
            # 10/20 22:47 錄影就這樣滑進條紋紙條。在正前方就繞開，不在中間就回漫遊
            revoked = [a for a in path_strips if a.get("revoked")]
            rev = self._nearest(revoked, self.last_tissue["cx"], self.REVOKE_GATE_PX)
            if rev is not None:
                return self._avoid_or_ignore(rev, now, vision_data,
                                             "鎖定的紙條靠近後看到彩色條紋，不是目標")

        if self.state == "TRACK":
            if using_memory:
                # 單格漏偵測：衰減上一個指令，不做任何狀態判斷也不碰 PID
                return self._decayed(now)

            if not tissue:
                # 記憶也過期了。分兩種情況：
                #   剛剛確實在前進、而且估距夠近 -> 已經貼到眼前，滑行補完最後幾公分
                #   還很遠或根本沒在前進         -> 真的跟丟了，回去漫遊重找
                # 還沒判斷的紙條絕不滑行 (見 JUDGE_* 的說明)
                if (self.track_judged and self.last_fb > 0
                        and self.last_dist_cm <= self.COAST_MAX_DIST_CM):
                    print(f"[紙條獵手] 紙條貼到眼前 (估距 {self.last_dist_cm:.0f}cm) -> 滑行補完")
                    self.change_state("COAST")
                    fb = self.COAST_FB          # 這一格就開始滑，不要空一拍
                else:
                    self.change_state("SEARCH")
                    return (0, 0, 0, 0)
            else:
                if time_in_state > self.TRACK_MAX_SEC:
                    self._give_up(now, f"追了 {self.TRACK_MAX_SEC:.0f} 秒還碰不到")
                    return (0, 0, 0, 0)

                w = max(1, tissue['w'])
                strip_px = self._strip_px(tissue)
                top_y = tissue.get('top_y', tissue['cy'] - tissue['h'] / 2.0)
                bottom_y = tissue.get('bottom_y', tissue['cy'] + tissue['h'] / 2.0)
                dist_cm = tissue.get('dist_cm', self.DIST_K / strip_px)

                # ---- 水平：把紙條轉到瞄準點正下方/正上方 (畫面水平正中) ----
                error_x = tissue['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)
                if abs(error_x) < self.YAW_DEADBAND_PX:
                    yv = 0

                # ---- 垂直：高度鎖定，不動 ----
                # 十字與紅點的垂直差只記進 log 供對照
                cross_y = tissue['cy']
                error_y = self.target_cy - cross_y        # >0 -> 十字在紅點上方
                ud = 0

                # 紙條有沒有橫過前進路線 (紅點那一列)
                crosses_path = top_y <= self.target_cy <= bottom_y

                # ---- 紙條沒有橫過前進路線 -> 高度不能動，撞不到 ----
                # 撐 UNREACHABLE_SEC 是為了容忍框的單格跳動
                if crosses_path:
                    self.off_path_since = None
                else:
                    if self.off_path_since is None:
                        self.off_path_since = now
                    if now - self.off_path_since > self.UNREACHABLE_SEC:
                        where = "上方" if bottom_y < self.target_cy else "下方"
                        self._give_up(now, f"紙條整條在前進路線{where} (高度鎖定)")
                        return (0, 0, 0, 0)

                # ---- 前進：隨對準程度連續變化 ----
                # crosses_path 是硬性否決 —— 紙條沒有橫過前進路線 (紅點那一列) 就絕不前進，
                # 否則會從上面或下面穿過去。
                blocked = (not depth_unreliable) and depth_ahead < self.APPROACH_STOP_CM

                # ---- 非白紙條擋在通往白紙條的路線上 -> 用視差判斷前後，必要時繞路 ----
                # (不再用紙條寬度推算的距離比前後 —— 紙條會扭轉，寬度不可靠)
                # (還沒判斷的紙條不做：要不要撞都還不知道，擋路的在上面已經轉開了)
                conflict = self._conflict_strip(path_strips, tissue) if self.track_judged else None
                if conflict is not None:
                    mem = self._remembered(conflict, tissue, now)
                    decision = mem["decision"] if mem else None
                    if decision == "front":
                        self._start_detour(conflict, tissue, now, mem["dir"])
                        return self._detour_cmd(now, current, path_strips, vision_data)
                    if decision != "behind":
                        self._start_probe(conflict, tissue, now)
                        return self._probe_cmd(now, current, path_strips)

                tol_x = max(self.ALIGN_TOL_X, strip_px * self.WIDTH_TOL_FACTOR)
                ax = _clamp(1.0 - abs(error_x) / (2.0 * tol_x), 0.0, 1.0)
                fb_cruise = self.FB_FAR if dist_cm > self.FAR_CM else self.FB_NEAR
                if not self.track_judged:
                    if not tissue.get("far", False):
                        fb_cruise = 0                    # 夠近了：停住看清楚
                    elif dist_cm <= self.JUDGE_SLOW_CM:
                        fb_cruise = self.FB_NEAR         # 快到判斷距離：放慢
                    else:
                        fb_cruise = self.FB_FAR          # 還很遠：照巡航速度接近
                fb = int(fb_cruise * ax) if (crosses_path and not blocked) else 0

                self.last_fb, self.last_ud, self.last_yv = fb, ud, yv
                self.last_dist_cm = dist_cm
                self._log.update({
                    "tw": w, "bottom_y": round(bottom_y, 1), "dist": round(dist_cm, 1),
                    "cy": round(cross_y, 1), "err_x": int(error_x), "err_y": round(error_y, 1),
                    "ax": round(ax, 2), "blocked": int(blocked),
                    "blob_aspect": tissue.get("blob_aspect", ""),
                    "white": tissue.get("white", ""),
                    "target": tissue.get("target", ""),     # 撞白色 (WHITE) 或彩色 (COLOR)
                    "judged": int(self.track_judged),         # 0 = 還沒判斷 (接近 / 判斷中)
                    "sw": round(strip_px, 1),                 # 紙條厚度 (px)，dist 就是用它換算的
                    "cls": tissue.get("cls", ""),             # 模型類別 (paper_plain / paper_red_marked)
                    "vote": "/".join(str(x) for x in tissue["votes"]) if "votes" in tissue else "",
                })

        elif self.state == "COAST":
            # 最後一小段滑行。這裡刻意不看偵測結果 —— 紙條已經近到看不見了，
            # 此時任何「偵測不到」的判斷都只會讓它在最後關頭煞車。
            # 但偏航歸零：在這個距離「直直往前」才是正確的先驗，
            # 沿用已經過期的偏航指令只會把機頭甩開。
            fb = self.COAST_FB
            if current is not None and current.get("judged", True):
                # 目標又出現了，代表剛剛只是短暫遮蔽，不是貼到眼前。
                # 這一格先把滑行走完，下一格交還給 TRACK 做完整的閉迴路控制。
                # 還沒判斷的紙條 (後面遠處的別條) 不算 —— 不論顏色先接近之後畫面裡常有，
                # 不能讓它打斷最後一小段滑行
                self.change_state("TRACK")
            elif time_in_state > self.COAST_SEC:
                self.cut_count += 1
                print(f"[紙條獵手] 已通過紙條 (累計 {self.cut_count} 次) -> 退開")
                self.change_state("BACKOFF")

        elif self.state == "DODGE":
            return self._dodge_cmd(now, vision_data)

        elif self.state == "PROBE":
            return self._probe_cmd(now, current, path_strips)

        elif self.state == "DETOUR":
            return self._detour_cmd(now, current, path_strips, vision_data)

        elif self.state == "BACKOFF":
            fb = -self.BACKOFF_FB
            if time_in_state > self.BACKOFF_SEC:
                self.ignore_until = now + self.IGNORE_SEC
                self.change_state("SEARCH")

        return (int(lr), int(fb), int(ud), int(yv))

    # ==========================================
    def aim_point(self):
        """畫面上紅點 (瞄準點) 的位置，tello_app 會畫在這裡"""
        return (self.target_cx, self.target_cy)

    def log_fields(self):
        """交給 VideoRecorder 的診斷欄位 (見 utils/video_recorder.py 的 LOG_FIELDS)"""
        return self._log

    def get_mode(self) -> str:
        if self.state == "SEARCH":
            return f"TISSUE: ROAMING ({self.explorer.state})"
        if self.state == "TRACK":
            d = self._log.get("dist")
            ax = self._log.get("ax")
            extra = f" d{d:.0f}cm ax{ax:.1f}" if d is not None and ax is not None else ""
            if self._log.get("blocked"):
                extra += " [WALL]"
            if not self.track_judged:
                return f"TISSUE: {'JUDGING' if self.judge_since is not None else 'APPROACH'}{extra}"
            return f"TISSUE: TRACK{extra}"
        if self.state == "COAST":
            return f"TISSUE: COAST {self._now() - self.state_start_time:.1f}s"
        if self.state == "DODGE":
            side = "R" if self.dodge_dir > 0 else "L"
            return f"TISSUE: DODGE {self.dodge_mode} {side} {self.dodge_kind} {self.dodge_dist:.0f}cm"
        if self.state in ("PROBE", "DETOUR"):
            return f"TISSUE: {self.state} {self.parallax_text}"
        return f"TISSUE: {self.state}"
