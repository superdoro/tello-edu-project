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
        TRACK    每一格同時修正轉向、高度與前進速度，一邊校正一邊逼近
        COAST    目標貼到眼前看不見了 -> 以追蹤速度再滑行一小段補完最後幾公分
        BACKOFF  剪完之後先退開，再交還給漫遊

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
        self.explorer = FluidExploreControl()

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

        # ==========================================
        # 狀態機
        # ==========================================
        if self.state == "SEARCH":
            if tissue and not using_memory and now >= self.ignore_until:
                self.change_state("TRACK")
            else:
                # 整段委派給漫遊引擎：避障、過彎、黑區脫困全部由它負責
                # (白牆已在 TissueDetector 覆寫成 depth_L/C/R 的近距離)
                return self.explorer.calculate_command(user_input, vision_data)

        # 上面可能剛從 SEARCH 切到 TRACK，計時要重新取 ——
        # 否則這一格用的還是「漫遊了多久」，只要漫遊超過 TRACK_MAX_SEC，
        # 追蹤的第一格就會直接觸發「追太久」而放棄。紙條稀疏的場地漫遊超過
        # 15 秒是常態，那等於整個模式永遠追不到東西。
        time_in_state = now - self.state_start_time

        if self.state == "TRACK":
            if using_memory:
                # 單格漏偵測：衰減上一個指令，不做任何狀態判斷也不碰 PID
                return self._decayed(now)

            if not tissue:
                # 記憶也過期了。分兩種情況：
                #   剛剛確實在前進、而且估距夠近 -> 已經貼到眼前，滑行補完最後幾公分
                #   還很遠或根本沒在前進         -> 真的跟丟了，回去漫遊重找
                if self.last_fb > 0 and self.last_dist_cm <= self.COAST_MAX_DIST_CM:
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
                top_y = tissue.get('top_y', tissue['cy'] - tissue['h'] / 2.0)
                bottom_y = tissue.get('bottom_y', tissue['cy'] + tissue['h'] / 2.0)
                dist_cm = tissue.get('dist_cm', self.DIST_K / w)

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
                tol_x = max(self.ALIGN_TOL_X, w * self.WIDTH_TOL_FACTOR)
                ax = _clamp(1.0 - abs(error_x) / (2.0 * tol_x), 0.0, 1.0)
                fb_cruise = self.FB_FAR if dist_cm > self.FAR_CM else self.FB_NEAR
                fb = int(fb_cruise * ax) if (crosses_path and not blocked) else 0

                self.last_fb, self.last_ud, self.last_yv = fb, ud, yv
                self.last_dist_cm = dist_cm
                self._log.update({
                    "tw": w, "bottom_y": round(bottom_y, 1), "dist": round(dist_cm, 1),
                    "cy": round(cross_y, 1), "err_x": int(error_x), "err_y": round(error_y, 1),
                    "ax": round(ax, 2), "blocked": int(blocked),
                    "blob_aspect": tissue.get("blob_aspect", ""),
                    "white": tissue.get("white", ""),
                    "vote": "/".join(str(x) for x in tissue["votes"]) if "votes" in tissue else "",
                })

        elif self.state == "COAST":
            # 最後一小段滑行。這裡刻意不看偵測結果 —— 紙條已經近到看不見了，
            # 此時任何「偵測不到」的判斷都只會讓它在最後關頭煞車。
            # 但偏航歸零：在這個距離「直直往前」才是正確的先驗，
            # 沿用已經過期的偏航指令只會把機頭甩開。
            fb = self.COAST_FB
            if current is not None:
                # 目標又出現了，代表剛剛只是短暫遮蔽，不是貼到眼前。
                # 這一格先把滑行走完，下一格交還給 TRACK 做完整的閉迴路控制。
                self.change_state("TRACK")
            elif time_in_state > self.COAST_SEC:
                self.cut_count += 1
                print(f"[紙條獵手] 已通過紙條 (累計 {self.cut_count} 次) -> 退開")
                self.change_state("BACKOFF")

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
            return f"TISSUE: TRACK{extra}"
        if self.state == "COAST":
            return f"TISSUE: COAST {self._now() - self.state_start_time:.1f}s"
        return f"TISSUE: {self.state}"
