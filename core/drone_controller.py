from djitellopy import tello
import logging
import threading
import time

class DroneController:
    """
    負責與 Tello 無人機進行硬體通訊、傳送飛行指令與獲取影像。
    """
    # 影像串流設定 (見 _configure_video)。預設「不改」，沿用 Tello 的 720P / 30fps / 自動位元率。
    # 10/19 20:21 與 20:27 兩趟設成 5Mbps 之後，畫面有 35% 的時間凍結 (每次 0.4~1.2 秒)、
    # 出現大片色塊，兩趟都在凍結期間盲飛撞上東西。Wi-Fi 品質好的場地才考慮打開。
    CONFIGURE_VIDEO = False
    VIDEO_BITRATE = tello.Tello.BITRATE_5MBPS
    def __init__(self):
        # 建立 Tello 物件
        self.drone = tello.Tello()
        self.is_connected = False

        # 預設隱藏 Tello 的日誌(隱藏所有 INFO 訊息)
        logging.getLogger("djitellopy").setLevel(logging.WARNING)

        # 電量快取 (見 get_battery 的說明)
        self._battery = None
        self._battery_ts = 0.0
        self.BATTERY_POLL_SEC = 1.0   # 電量變化很慢，一秒讀一次就夠

        # 高度快取 (見 get_height 的說明)
        self._height = None
        self._height_ts = 0.0
        self.HEIGHT_POLL_SEC = 0.1    # 高度變化快，要讀得比電量勤

        # --- 指令看門狗 (見 _watchdog) ---
        # Tello 會一直執行最後收到的 rc 指令，直到收到新的為止。
        # 2026-10-05 實測：主迴圈卡住 2.56 秒，最後一個指令是 fb=50，
        # 無人機就這樣直直飛了 2.5 秒撞上牆。主迴圈卡住的原因不只一種
        # (模型推論、系統排程、I/O)，所以在這一層統一擋：超過 WATCHDOG_SEC
        # 沒收到新指令，就由背景執行緒送一次 rc 0 0 0 0 讓它懸停。
        self.WATCHDOG_SEC = 0.3
        self._rc_lock = threading.Lock()
        self._last_rc = (0, 0, 0, 0)
        self._last_rc_time = time.time()
        self._watchdog_thread = None

        # 畫面新鮮度 (見 frame_age)：djitellopy 每解碼出一張新畫面就換一個新的陣列物件，
        # 串流卡住時 frame_read.frame 會一直回傳同一個物件
        self._last_frame_obj = None
        self._last_frame_time = time.time()

    def connect(self):
        """
        建立連線、獲取電量並開啟影像串流
        """
        try:
            self.drone.connect()
            self.is_connected = True
            
            # 讀取當前電量
            battery = self.drone.get_battery()
            print(f"[系統訊息] Tello 連線成功！當前電量: {battery}%")
            
            # 影像串流設定要在 streamon 之前送
            if self.CONFIGURE_VIDEO:
                self._configure_video()

            # 開啟影像串流
            self.drone.streamon()
            print("[系統訊息] 影像串流已啟動。")

            self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
            self._watchdog_thread.start()
            
        except Exception as e:
            print(f"[錯誤] 連線失敗，請檢查 Wi-Fi 是否確實連上 TELLO-XXXXXX。詳細錯誤: {e}")
            self.is_connected = False

    def _configure_video(self):
        """
        影像串流設為 720P (960x720)、30fps、位元率 VIDEO_BITRATE (只有 CONFIGURE_VIDEO = True 才會呼叫)。

        預設位元率是「自動」，Wi-Fi 一忙就降畫質，細細的紙條最先被壓縮雜訊吃掉
        (邊緣糊掉、出現方塊)，YOLO 就偵測不到。5Mbps 需要比較好的 Wi-Fi 品質，
        如果畫面開始卡頓、掉格，把 VIDEO_BITRATE 改回 BITRATE_AUTO 或 3MBPS。

        這三個指令需要 Tello EDU (SDK 3.0 以上的韌體)；不支援就印警告，照常使用預設值。
        """
        T = tello.Tello
        for name, setter, value in (
                ("解析度 720P", self.drone.set_video_resolution, T.RESOLUTION_720P),
                ("幀率 30fps", self.drone.set_video_fps, T.FPS_30),
                ("位元率 5Mbps", self.drone.set_video_bitrate, self.VIDEO_BITRATE)):
            try:
                setter(value)
                print(f"[系統訊息] 影像串流：{name}")
            except Exception as e:
                print(f"[警告] 影像串流設定「{name}」失敗 (韌體可能不支援)，沿用預設值：{e}")

    def takeoff(self):
        """控制無人機起飛"""
        if self.is_connected:
            self.drone.takeoff()

    def land(self):
        """控制無人機降落"""
        if self.is_connected:
            self.drone.land()

    def send_movement(self, lr: int, fb: int, ud: int, yv: int):
        """
        傳送移動指令 (RC Control)
        :param lr: 左右 (Left/Right) [-100~100]
        :param fb: 前後 (Forward/Backward) [-100~100]
        :param ud: 上下 (Up/Down) [-100~100]
        :param yv: 旋轉 (Yaw Velocity) [-100~100]
        """
        if self.is_connected:
            with self._rc_lock:
                self.drone.send_rc_control(lr, fb, ud, yv)
                self._last_rc = (lr, fb, ud, yv)
                self._last_rc_time = time.time()

    def _watchdog(self):
        """
        背景：主迴圈超過 WATCHDOG_SEC 沒送新指令、而最後一個指令不是靜止，就送一次懸停。

        只送一次 (送完把 _last_rc 記成靜止)，主迴圈恢復後下一格就會蓋掉，
        所以起飛、降落這類正常會阻塞的指令也不受影響。
        """
        while self.is_connected:
            time.sleep(0.05)
            with self._rc_lock:
                stale = time.time() - self._last_rc_time
                if stale <= self.WATCHDOG_SEC or self._last_rc == (0, 0, 0, 0):
                    continue
                try:
                    self.drone.send_rc_control(0, 0, 0, 0)
                except Exception:
                    pass
                print(f"[看門狗] 主迴圈 {stale:.2f} 秒沒有新指令 (最後是 {self._last_rc}) -> 懸停")
                self._last_rc = (0, 0, 0, 0)

    def get_battery(self):
        """
        讀取目前電量百分比 (0~100)，無法取得時回傳 None。

        djitellopy 的 get_battery() 是從背景執行緒收到的狀態封包 (UDP 8890) 讀快取，
        不是送指令等回應，所以不會阻塞主迴圈 —— 這點很重要，飛行主迴圈裡任何一個
        會等待回應的呼叫都可能讓 Tello 超過 15 秒收不到指令而自動降落。

        剛連上、還沒收到第一個狀態封包時會丟 TelloException，
        狀態封包偶爾遺失也一樣，因此包起來並沿用上一次的值。
        """
        if not self.is_connected:
            return None

        now = time.time()
        if now - self._battery_ts < self.BATTERY_POLL_SEC:
            return self._battery   # 還沒到下次讀取時間，回傳快取

        self._battery_ts = now
        try:
            self._battery = int(self.drone.get_battery())
        except Exception:
            pass   # 讀不到就維持上一次的值，不要讓顯示閃爍或中斷飛行
        return self._battery

    def get_height(self):
        """
        讀取目前高度 (公分，相對起飛點)，無法取得時回傳 None。

        與 get_battery 相同，djitellopy 是從背景狀態封包讀快取，不會阻塞主迴圈。
        用途是給需要爬升的模式當硬性高度上限 —— 視覺判斷再怎麼小心，
        誤判一個天花板上的燈就可能一路爬上去，只有真實高度擋得住。
        """
        if not self.is_connected:
            return None

        now = time.time()
        if now - self._height_ts < self.HEIGHT_POLL_SEC:
            return self._height

        self._height_ts = now
        try:
            self._height = int(self.drone.get_height())
        except Exception:
            pass   # 讀不到就沿用上一次的值
        return self._height

    def get_video_frame(self):
        """
        獲取最新的一張影像 (Frame)
        :return: OpenCV 格式的影像矩陣 (若無畫面則回傳 None)
        """
        if self.is_connected:
            frame_read = self.drone.get_frame_read()
            if frame_read is not None:
                frame = frame_read.frame
                if frame is not self._last_frame_obj:
                    self._last_frame_obj = frame
                    self._last_frame_time = time.time()
                return frame
        return None

    def frame_age(self):
        """
        最新畫面已經多久沒有更新 (秒)。

        影像串流會卡住 (Wi-Fi 塞住、封包遺失)，這時 get_video_frame 一直回傳同一張舊畫面，
        視覺模組照樣分析、飛行邏輯照樣下指令 —— 等於蒙著眼睛飛。
        10/19 20:21 那趟凍結 1.0~1.2 秒期間避障一直在 -60 倒退 + 轉彎，最後撞上。
        """
        return time.time() - self._last_frame_time

    def teardown(self):
        """
        關閉連線與串流，釋放資源
        """
        if self.is_connected:
            print("[系統訊息] 正在關閉無人機連線...")
            self.drone.streamoff()
            self.drone.end()
            self.is_connected = False