import time
from collections import deque

from behaviors.base import FlightBehavior
from behaviors.fluid_explore import FluidExploreControl
from utils.pid_controller import PIDController


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class PlateReadControl(FlightBehavior):
    """
    車牌辨識：漫遊找車牌 -> 靠近並對準 -> 停穩 -> OCR 讀號碼 -> 懸停顯示結果。

    狀態機
        SEARCH    把指令整段委派給 FluidExploreControl 漫遊避障 (高度鎖定，見 _auto_command)
        APPROACH  轉機頭對準水平、升降對準垂直、依估距前進 / 後退到 TARGET_DIST_CM
        HOLD      已經在容差內，計時「停穩了多久」；連續 HOLD_SEC 才開始讀
        READ      繼續定點，把最清晰的截圖交給 PlateReader (背景 OCR)，同一個號碼 READ_AGREE 票就結束
        DONE      懸停顯示結果，不會自己飛走。R = 重讀；按方向鍵 = 飛手接手

    ==========================================================
    為什麼要「停穩」才讀
    ==========================================================
    無人機在動的時候截圖會糊 (尤其是轉機頭)，OCR 很容易把 8 讀成 B、把 1 讀成 7。
    停穩的定義 (_in_tol + _steady)：車牌在畫面中央、距離在目標附近，
    而且最近 JITTER_WINDOW_SEC 秒框的中心沒有在晃。連續 HOLD_SEC 秒都成立才讀。
    進入用嚴格容差、離開用寬鬆容差 (遲滯)，不會在邊界上進進出出。
    讀的時候也不是每格都送：收集 COLLECT_SEC 秒，挑最清晰的那張送。

    ==========================================================
    高度
    ==========================================================
    車牌通常很低 (離地 40~70cm)，近距離高度差太多車牌會跑出畫面，所以鎖定之後會自動上下對準：
    限速 UD_LIMIT、死區 UD_DEADBAND_PX，而且只用當格真的看到的車牌算 (記憶期間只衰減)。
    漫遊時高度鎖定 —— 跟 TISSUE CHARGE 一樣擋掉 FluidExploreControl 黑區脫困的爬升
    (高度讀值不可靠，自動爬升撞過天花板)。請先手動飛到大約車牌的高度再切進這個模式。

    ==========================================================
    誠實的限制
    ==========================================================
    只轉機頭、不橫移：斜著接近時會停在斜角讀。框的長寬比 (aspect) 有記進 log，之後要做正對再加。
    自動下降沒有地面保護 (高度讀值不可靠)，靠限速、只用即時偵測、飛手隨時可以接手。
    """

    FRAME_W, FRAME_H = 720, 480

    def __init__(self, reader=None, now_fn=time.time):
        """
        :param reader: PlateReader (vision/plate_reader.py)，None = 不讀 (只會接近、停穩)
        :param now_fn: 取得目前時間的函式 (離線測試時傳入假時鐘)
        """
        self._now = now_fn
        self.reader = reader
        self.state = "SEARCH"
        self.state_start_time = self._now()

        # 搜尋階段的漫遊引擎。它需要的 depth_L/C/R 與 dark_* 由 PlateVision 提供 (DepthExplorerVision 的欄位)
        self.explorer = FluidExploreControl(now_fn=now_fn)

        # 瞄準點 = 畫面正中央。讀車牌是懸停時做的，不像紙條要往前衝 (所以不用 75% 高度)
        self.target_cx = self.FRAME_W // 2
        self.target_cy = self.FRAME_H // 2

        self.pid_yv = PIDController(kp=0.4, ki=0.0, kd=0.15, limit=50)
        self.YAW_DEADBAND_PX = 12
        self.UD_LIMIT = 25             # 自動升降的速度上限 (約每秒 25 公分)
        self.pid_ud = PIDController(kp=0.3, ki=0.0, kd=0.1, limit=self.UD_LIMIT)
        self.UD_DEADBAND_PX = 20

        # --- 距離 (PlateVision 的 dist_cm，用車牌實際尺寸換算) ---
        self.TARGET_DIST_CM = 70       # 停在車牌前這麼遠讀 (32cm 的車牌約佔畫面 1/3 寬)
        # 估距小於這個一律不前進。正常情況下用不到 (目標距離比它遠)，
        # 是擋 TARGET_DIST_CM 被調得太近、或車牌尺寸設錯時一路撞上去
        self.MIN_DIST_CM = 40
        self.FAR_CM = 150
        self.FB_FAR = 40               # 遠距離的最快前進速度
        self.FB_NEAR = 20              # FAR_CM 以內的最快前進速度
        self.FB_BACK = 15              # 太近時最快的後退速度
        self.KP_FB = 0.6               # 前後速度 = KP_FB x 距離誤差 (cm)，再受上面的上限限制
        self.DIST_DEADBAND = 0.08      # 距離誤差在目標的 8% 以內就不前後動
        self.ALIGN_TOL_PX = 40         # 偏離中央這麼多時前進速度打 5 折，2 倍時不前進

        # --- 停穩 ---
        self.STABLE_X_PX = 30          # 進入 HOLD / 計時的容差
        self.STABLE_Y_PX = 30
        self.STABLE_DIST_TOL = 0.15
        self.EXIT_X_PX = 50            # 超出這個才退回 APPROACH (遲滯)
        self.EXIT_Y_PX = 50
        self.EXIT_DIST_TOL = 0.25
        self.JITTER_WINDOW_SEC = 0.5
        self.JITTER_PX = 12            # 框中心在視窗內晃動 (max - min) 不超過這個才算沒在晃
        self.HOLD_SEC = 1.0
        self.hold_since = None
        self._track = deque(maxlen=120)  # (時間, cx, cy)，算晃動用

        # --- 鎖定 / 跟丟 ---
        self.MIN_HITS = 3              # 偵測器連續看到這麼多格才去接近 (擋單格誤判)
        # 單格漏偵測時讓上一個指令衰減 (同 TissueChargeControl.MEMORY_DURATION 的理由：
        # 重跑 PID 會用凍結的誤差送出固定的最大輸出)
        self.MEMORY_DURATION = 0.25
        self.LOST_SEC = 1.0            # 看不到超過這麼久 -> 回去漫遊
        self.APPROACH_MAX_SEC = 20.0   # 這麼久還停不穩就放棄這個車牌
        self.IGNORE_SEC = 3.0          # 放棄 / 飛手接手後這段時間不鎖定，讓漫遊或飛手把無人機帶開
        self.ignore_until = 0.0
        self.engage_since = 0.0
        self.last_plate = None
        self.last_plate_time = -1e9
        self.last_cmd = (0, 0, 0, 0)

        # --- 讀取 ---
        self.READ_AGREE = 2            # 同一個號碼這麼多票就結束
        self.READ_MAX_SEC = 5.0        # 從開始讀算起最多這麼久：有票取票數最多的 (標 ?)，沒票 = 失敗
        self.COLLECT_SEC = 0.3         # 每次送 OCR 前收集這麼久，挑最清晰的那張
        self.read_since = None
        self.collect_since = 0.0
        self.best_crop = None
        self.best_sharp = -1.0
        self.result_text = None        # DONE 時顯示的號碼 (None = 讀不出來)
        self.result_uncertain = False

        self._log = {}

    # ==========================================
    def reset(self):
        """切換到這個模式時 (tello_app.toggle_mode) 呼叫"""
        self.ignore_until = 0.0
        self.change_state("SEARCH")
        self.explorer.change_state("FORWARD")

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = self._now()
            if new_state == "SEARCH":
                # 回到搜尋就把這個車牌的一切清掉；漫遊引擎從直線前進開始，不沿用上次的過彎狀態
                self.explorer.change_state("FORWARD")
                self._track.clear()
                self.hold_since = None
                self._clear_read()
                self.result_text = None
            print(f"[車牌辨識] {self.state}")

    def _clear_read(self):
        self.read_since = None
        self.best_crop = None
        self.best_sharp = -1.0
        if self.reader is not None:
            self.reader.clear()

    def reset_target(self):
        """R 鍵：DONE 時重新讀取；接近 / 停穩 / 讀取中則放棄這個車牌"""
        now = self._now()
        if self.state == "DONE":
            print("[車牌辨識] 重新讀取")
            self._clear_read()
            self.result_text = None
            self.hold_since = None
            self.engage_since = now            # 停不穩的放棄計時重新算
            self.change_state("HOLD")
        elif self.state != "SEARCH":
            self._give_up(now, "飛手放棄")

    def _give_up(self, now, why):
        print(f"[車牌辨識] {why} -> 放棄這個車牌，{self.IGNORE_SEC:.0f} 秒內不鎖定")
        self.ignore_until = now + self.IGNORE_SEC
        self.change_state("SEARCH")

    def _decayed(self, now):
        """記憶期間的輸出：上一個指令線性衰減到 0，不重跑 PID (見 MEMORY_DURATION)"""
        k = max(0.0, 1.0 - (now - self.last_plate_time) / self.MEMORY_DURATION)
        _, fb, ud, yv = self.last_cmd
        return (0, int(fb * k), int(ud * k), int(yv * k))

    # ==========================================
    # 控制律與停穩判定
    # ==========================================
    def _servo(self, plate):
        """定點控制：轉機頭對 x、升降對 y、前後對距離。APPROACH 到 DONE 都用這一套"""
        err_x = plate["cx"] - self.target_cx
        err_y = self.target_cy - plate["cy"]           # > 0 = 車牌在中央上方 -> 上升
        yv = self.pid_yv.compute(err_x)
        if abs(err_x) < self.YAW_DEADBAND_PX:
            yv = 0
        ud = self.pid_ud.compute(err_y)
        if abs(err_y) < self.UD_DEADBAND_PX:
            ud = 0

        dist = plate["dist_cm"]
        err_d = dist - self.TARGET_DIST_CM
        if abs(err_d) <= self.TARGET_DIST_CM * self.DIST_DEADBAND:
            fb = 0
        elif err_d > 0:
            # 還沒對準就少往前：車牌偏太多時直直往前，它會從畫面邊緣跑出去
            a = _clamp(1.0 - max(abs(err_x), abs(err_y)) / (2.0 * self.ALIGN_TOL_PX), 0.0, 1.0)
            cruise = self.FB_FAR if dist > self.FAR_CM else self.FB_NEAR
            fb = int(min(self.KP_FB * err_d, cruise) * a)
        else:
            fb = -int(min(self.KP_FB * -err_d, self.FB_BACK))
        if dist < self.MIN_DIST_CM:
            fb = min(fb, 0)
        return (0, fb, int(ud), int(yv)), err_x, err_y

    def _in_tol(self, plate, err_x, err_y, loose):
        tx, ty, td = ((self.EXIT_X_PX, self.EXIT_Y_PX, self.EXIT_DIST_TOL) if loose
                      else (self.STABLE_X_PX, self.STABLE_Y_PX, self.STABLE_DIST_TOL))
        return (abs(err_x) <= tx and abs(err_y) <= ty
                and abs(plate["dist_cm"] - self.TARGET_DIST_CM) <= td * self.TARGET_DIST_CM)

    def _steady(self, now):
        """最近 JITTER_WINDOW_SEC 秒，框的中心有沒有在晃"""
        pts = [(x, y) for t, x, y in self._track if now - t <= self.JITTER_WINDOW_SEC]
        if len(pts) < 3:
            return False
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return max(xs) - min(xs) <= self.JITTER_PX and max(ys) - min(ys) <= self.JITTER_PX

    def _start_read(self, now):
        if self.read_since is None:
            self.read_since = now
        self.collect_since = now
        self.best_crop = None
        self.best_sharp = -1.0
        print("[車牌辨識] 停穩了 -> 開始讀車牌")
        self.change_state("READ")

    def _collect(self, plate, now):
        """READ：收集最清晰的截圖，PlateReader 閒著、而且收集滿 COLLECT_SEC 才送出"""
        crop = plate.get("crop")
        if crop is not None and plate.get("sharp", 0.0) > self.best_sharp:
            self.best_crop, self.best_sharp = crop, plate.get("sharp", 0.0)
        if (self.reader is not None and self.best_crop is not None
                and now - self.collect_since >= self.COLLECT_SEC and not self.reader.busy):
            if self.reader.submit(self.best_crop):
                self.best_crop, self.best_sharp = None, -1.0
                self.collect_since = now

    def _finish(self, now):
        text, n = self.reader.leader() if self.reader is not None else (None, 0)
        self.result_text = text
        self.result_uncertain = text is not None and n < self.READ_AGREE
        if text is None:
            print(f"[車牌辨識] {self.READ_MAX_SEC:.0f} 秒內讀不出來 (按 R 重讀)")
        else:
            note = f" (只有 {n} 票，不確定)" if self.result_uncertain else ""
            print(f"[車牌辨識] 車牌號碼：{text}{note}")
            self.reader.save(text, self.result_uncertain)
        self.change_state("DONE")

    # ==========================================
    def calculate_command(self, user_input, vision_data):
        # 人工接管優先。讀完之後飛手接手，多半是要飛去下一個車牌 -> 這段時間不要又鎖回剛剛那個
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            if self.state == "DONE":
                self.ignore_until = self._now() + self.IGNORE_SEC
            self.change_state("SEARCH")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        return tuple(int(c) for c in self._auto_command(user_input, vision_data))

    def _auto_command(self, user_input, vision_data):
        now = self._now()
        current = getattr(vision_data, 'plate', None) if vision_data else None
        if current is not None:
            self.last_plate = current
            self.last_plate_time = now
            self._track.append((now, current["cx"], current["cy"]))
        self._log = {}

        if self.state == "SEARCH":
            if (current is not None and current.get("hits", 0) >= self.MIN_HITS
                    and now >= self.ignore_until):
                print(f"[車牌辨識] 發現車牌 (約 {current['dist_cm']:.0f}cm) -> 接近")
                self.engage_since = now
                self.change_state("APPROACH")
            else:
                # 整段委派給漫遊引擎，只有高度鎖定 (擋掉黑區脫困的爬升，見類別說明)
                lr, fb, _, yv = self.explorer.calculate_command(user_input, vision_data)
                return (lr, fb, 0, yv)

        # 背景 OCR 的結果每格都收 (漂離回 HOLD 期間跑完的也算)
        if self.reader is not None and self.state in ("HOLD", "READ"):
            self.reader.poll()
        if self.read_since is not None and self.state in ("HOLD", "READ"):
            _, n = self.reader.leader() if self.reader is not None else (None, 0)
            if n >= self.READ_AGREE or now - self.read_since > self.READ_MAX_SEC:
                self._finish(now)

        in_memory = (current is None and self.last_plate is not None
                     and now - self.last_plate_time <= self.MEMORY_DURATION)
        if current is None:
            if self.state == "DONE":
                return self._decayed(now) if in_memory else (0, 0, 0, 0)
            if now - self.last_plate_time > self.LOST_SEC:
                print("[車牌辨識] 車牌跟丟了 -> 回去漫遊")
                self.change_state("SEARCH")
                return (0, 0, 0, 0)
            self.hold_since = None               # 沒看到就不能說停穩了
            return self._decayed(now) if in_memory else (0, 0, 0, 0)

        cmd, err_x, err_y = self._servo(current)
        self.last_cmd = cmd
        strict = self._in_tol(current, err_x, err_y, loose=False)
        loose = self._in_tol(current, err_x, err_y, loose=True)

        if self.state == "APPROACH":
            if now - self.engage_since > self.APPROACH_MAX_SEC:
                self._give_up(now, f"{self.APPROACH_MAX_SEC:.0f} 秒還停不穩")
                return (0, 0, 0, 0)
            if strict:
                self.hold_since = None
                self.change_state("HOLD")

        if self.state == "HOLD":
            if self.read_since is None and now - self.engage_since > self.APPROACH_MAX_SEC:
                self._give_up(now, f"{self.APPROACH_MAX_SEC:.0f} 秒還停不穩")
                return (0, 0, 0, 0)
            if not loose:
                self.change_state("APPROACH")
            elif strict and self._steady(now):
                if self.hold_since is None:
                    self.hold_since = now
                if now - self.hold_since >= self.HOLD_SEC:
                    self._start_read(now)
            else:
                self.hold_since = None

        elif self.state == "READ":
            if not loose:
                print("[車牌辨識] 讀取中漂離 -> 重新停穩 (已讀到的票保留)")
                self.hold_since = None
                self.change_state("HOLD")
            else:
                self._collect(current, now)

        self._log = {
            "plate_w": int(current["w"]),
            "dist": round(current["dist_cm"], 1),
            "err_x": int(err_x), "err_y": int(err_y),
            "hold": round(now - self.hold_since, 2) if self.hold_since is not None else "",
            "sharp": int(current.get("sharp", 0)),
            "aspect": round(current.get("aspect", 0.0), 2),
            "ocr": self.reader.last_text if self.reader is not None else "",
            "ocr_votes": self.reader.votes_text() if self.reader is not None else "",
            "plate": self.result_text or "",
        }
        return cmd

    # ==========================================
    def alert_text(self):
        """畫面中央的大字提示 (tello_app 會畫出來)；沒有要提示時回傳 None"""
        if self.state == "HOLD":
            if self.hold_since is None:
                return "ALIGNING"
            return f"HOLD {self._now() - self.hold_since:.1f}s"
        if self.state == "READ":
            n = self.reader.leader()[1] if self.reader is not None else 0
            return f"READING {n}/{self.READ_AGREE}"
        if self.state == "DONE":
            if self.result_text is None:
                return "READ FAILED - PRESS R"
            return f"PLATE {self.result_text}" + (" ?" if self.result_uncertain else "")
        return None

    def aim_point(self):
        """畫面上紅點 (瞄準點) 的位置，tello_app 會畫在這裡"""
        return (self.target_cx, self.target_cy)

    def log_fields(self):
        """交給 VideoRecorder 的診斷欄位 (見 utils/video_recorder.py 的 LOG_FIELDS)"""
        return self._log

    def get_mode(self) -> str:
        if self.state == "SEARCH":
            return f"PLATE: ROAMING ({self.explorer.state})"
        d = self._log.get("dist")
        extra = f" d{d:.0f}cm" if d is not None else ""
        if self.state == "DONE":
            return f"PLATE: DONE {self.result_text or 'FAILED'}"
        return f"PLATE: {self.state}{extra}"
