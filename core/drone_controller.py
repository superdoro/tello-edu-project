from djitellopy import tello

class DroneController:
    """
    負責與 Tello 無人機進行硬體通訊、傳送飛行指令與獲取影像。
    """
    def __init__(self):
        # 建立 Tello 物件
        self.drone = tello.Tello()
        self.is_connected = False

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

    def get_video_frame(self):
        """
        獲取最新的一張影像畫格 (Frame)
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
            self.is_connected = False