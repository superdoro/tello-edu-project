from djitellopy import tello
import logging
import time

class DroneController:
    """
    負責與 Tello 無人機進行硬體通訊、傳送飛行指令與獲取影像。
    """
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
            
            # 開啟影像串流
            self.drone.streamon()
            print("[系統訊息] 影像串流已啟動。")
            
        except Exception as e:
            print(f"[錯誤] 連線失敗，請檢查 Wi-Fi 是否確實連上 TELLO-XXXXXX。詳細錯誤: {e}")
            self.is_connected = False

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
            self.drone.send_rc_control(lr, fb, ud, yv)

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

    def get_video_frame(self):
        """
        獲取最新的一張影像 (Frame)
        :return: OpenCV 格式的影像矩陣 (若無畫面則回傳 None)
        """
        if self.is_connected:
            frame_read = self.drone.get_frame_read()
            if frame_read is not None:
                return frame_read.frame
        return None

    def teardown(self):
        """
        關閉連線與串流，釋放資源
        """
        if self.is_connected:
            print("[系統訊息] 正在關閉無人機連線...")
            self.drone.streamoff()
            self.drone.end()
            self.is_connected = False