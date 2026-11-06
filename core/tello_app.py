"""
應用程式主體，負責調度硬體、介面與飛行策略。
"""

import time

import cv2
from core.drone_controller import DroneController
from core.ui_controller import UIController
from behaviors.manual_control import ManualControl

from vision.pose_tracker import BodyPoseTracker
from behaviors.pose_follow import BodyFollowControl
from vision.fluid_explorer_vision import DepthExplorerVision
from behaviors.fluid_explore import FluidExploreControl
from vision.drone_detector import DroneDetector
from behaviors.drone_follow import DroneFollowControl
from behaviors.balloon_hunt import BalloonHuntControl
from vision.balloon_detector_with_aruco import BalloonDetector
from vision.tissue_detector import TissueDetector
from behaviors.tissue_charge import TissueChargeControl
from utils.video_recorder import VideoRecorder
# 畫面超過這麼久沒更新 (影像串流凍結) -> 自動模式改成懸停 (見 stale_video_guard)
FRAME_STALE_SEC = 0.3


def stale_video_guard(commands, user_input, frame_age, manual_mode, limit=FRAME_STALE_SEC):
    """
    影像串流凍結時，視覺模組分析的是同一張舊畫面，自動模式等於蒙著眼睛飛。
    10/19 20:21 / 20:27 兩趟都在凍結 (0.8~1.2 秒) 期間撞上東西。
    畫面超過 limit 秒沒更新 -> 自動模式的指令改成懸停；手動模式、或飛手正在按鍵時不攔。

    :return: (要送出的指令, 這一格是不是因為凍結而攔下)
    """
    if manual_mode or any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
        return commands, False
    if frame_age > limit:
        return (0, 0, 0, 0), True
    return commands, False


class TelloApp:
    SLOW_LOOP_SEC = 0.3    # 單一迴圈超過這麼久就印出各階段耗時 (與看門狗門檻一致)

    def __init__(self):
        # 初始化核心硬體與介面模組
        print("tello app -> 正在初始化硬體控制器...")
        self.drone = DroneController()

        print("tello app -> 正在初始化 UI 介面...")
        self.ui = UIController()

        # 實驗錄影 (V 鍵開關)，編碼與寫檔都在背景執行緒，不會拖慢飛控迴圈
        self.recorder = VideoRecorder()
        
        # ==============================================================
        # 定義所有可用的飛行模式清單
        # 未來新增模式時，只需要在此清單加入新的字典設定即可。
        # ==============================================================
        self.modes = [
            {
                "name": "MANUAL CONTROL",
                "behavior": ManualControl(),
                "vision": None  # 手動模式
            },
            {
                "name": "HAND TRACKER",
                "behavior": BodyFollowControl(),
                "vision": BodyPoseTracker() # 自動跟追模式(手掌、胸腔定位)
            },
#            {
#                 "name": "BALLOON HUNT",
#                 "behavior": BalloonHuntControl(),
#                 "vision": BalloonDetector()
#             },
#             {
#                 "name": "DRONE FOLLOW",
#                 "behavior": DroneFollowControl(),
#                 "vision": DroneDetector() # 空戰追蹤模式(咬住另一台無人機)
#             },
            {
                "name": "TISSUE CHARGE",
                "behavior": TissueChargeControl(),
                "vision": TissueDetector() # 衛生紙條獵手(HSV 白色 + 長條形狀，F 鍵可看遮罩)
            }
            # 未來擴充範例：
            # {"name": "VOICE CONTROL", "behavior": VoiceControlBehavior(), "vision": None}
        ]
        
        # 預設行為：清單中的第一個模式 (索引值 0 -> 手動控制)
        self.current_mode_index = 0
        self.is_running = True
        self._video_frozen = False

    @property
    def current_mode(self):
        """取得當前模式的字典設定"""
        return self.modes[self.current_mode_index]

    @property
    def behavior(self):
        """取得當前模式的飛行策略實例"""
        return self.current_mode["behavior"]

    @property
    def vision(self):
        """取得當前模式的視覺辨識實例"""
        return self.current_mode["vision"]

    def toggle_mode(self):
        """切換到清單中的下一個模式 (支援無限循環切換)"""
        # 離開模式前關閉環繞，避免切回來時無人機突然開始繞圈。
        # 用 stop_orbit 而非 toggle_orbit：O 鍵是三態循環，toggle 只會換到下一個方向。
        if hasattr(self.behavior, 'stop_orbit'):
            self.behavior.stop_orbit()
        self.current_mode_index = (self.current_mode_index + 1) % len(self.modes)
        # 清空新模式視覺模組的平滑/累積狀態 (例如黑區分析的 EMA)，
        # 避免沿用上次離開該模式時的舊值做出錯誤判斷。
        if self.vision and hasattr(self.vision, 'reset'):
            self.vision.reset()
        if hasattr(self.behavior, 'reset'):
            self.behavior.reset()
        print(f"[模式切換] 目前模式為: {self.current_mode['name']}")

    def toggle_tracking_mode(self):
        """切換當前模式的追蹤模式 (如果有支援的話)"""
        if self.vision and hasattr(self.vision, 'toggle_tracking_mode'):
            self.vision.toggle_tracking_mode()
            print("[追蹤模式切換] 目前追蹤模式已切換。")
        else:
            print("[追蹤模式切換] 當前模式不支援追蹤模式切換。")

    def reset_tracking_target(self):
        """重置 or 鎖定當前模式的追蹤目標 (如果有支援的話)"""
        if self.vision and hasattr(self.vision, 'reset_target'):
            self.vision.reset_target()
            print("[追蹤目標重置/鎖定] 追蹤目標已重置/鎖定。")
        else:
            print("[追蹤目標重置/鎖定] 當前模式不支援追蹤目標重置/鎖定。")

    def toggle_target_colour(self):
        """切換當前模式要追的目標顏色 (如果有支援的話)"""
        if self.vision and hasattr(self.vision, 'toggle_target'):
            self.vision.toggle_target()
        else:
            print("[目標顏色] 當前模式不支援切換目標顏色。")

    def toggle_orbit(self):
        """開關當前模式的環繞功能 (如果有支援的話)"""
        if hasattr(self.behavior, 'toggle_orbit'):
            self.behavior.toggle_orbit()
        else:
            print("[環繞模式] 當前模式不支援環繞功能。")

    @staticmethod
    def draw_battery(frame, battery):
        """
        在畫面右上角畫出電量。

        用 getTextSize 量出字寬再靠右對齊，換解析度時不會跑版。
        顏色分級對應 Tello 的實際行為：低於 20% 該準備降落，
        約 10% 以下它會自己強制降落。
        """
        if battery is None:
            text, color = "BAT --", (160, 160, 160)
        else:
            text = f"BAT {battery}%"
            if battery > 50:
                color = (0, 255, 0)      # 綠：充足
            elif battery >= 20:
                color = (0, 255, 255)    # 黃：注意
            else:
                color = (0, 0, 255)      # 紅：該降落了

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        w_img = frame.shape[1]
        cv2.putText(frame, text, (w_img - tw - 10, 30), font, scale, color, thick)

    def run(self):
        """啟動主迴圈"""
        # 1. 連線無人機
        print("[系統訊息] Tello 連線中...")
        self.drone.connect()
        
        while self.is_running:
            # 各階段計時：迴圈一旦卡住，Tello 會繼續執行上一個指令 (撞牆過一次)，
            # 看門狗擋得住後果，但要知道卡在哪才能根治
            t_loop = time.time()

            # 2. 獲取使用者輸入
            user_input = self.ui.get_input()
            t_input = time.time()
            
            # 3. 處理全域系統指令 (起飛、降落、退出、模式切換)
            if user_input.takeoff:
                self.drone.takeoff()
            elif user_input.land:
                self.drone.land()
            elif user_input.toggle_mode:  # 處理 Z 鍵切換
                self.toggle_mode()
            elif user_input.reserve_key_f:  # 處理 F 鍵切換 (保留給各視覺模式自行定義)
                self.toggle_tracking_mode() 
            elif user_input.reserve_key_r:  # 處理 R 鍵切換 (保留給各視覺模式自行定義)  
                self.reset_tracking_target()
            elif user_input.reserve_key_o:  # 處理 O 鍵切換 (開關環繞模式)
                self.toggle_orbit()
            elif user_input.reserve_key_c:  # 處理 C 鍵 (TISSUE CHARGE：撞白色 / 彩色紙條)
                self.toggle_target_colour()
            elif user_input.toggle_record:  # 處理 V 鍵 (開始/停止錄影)
                self.recorder.toggle(self.current_mode['name'])
            elif user_input.quit:
                self.shutdown()
                break # 退出迴圈
                
            # 4. 獲取影像並調整大小
            frame = self.drone.get_video_frame()
            vision_data = None # 預設視覺資料為空
            t_vision = 0.0
            
            raw_frame = None
            native_frame = None
            if frame is not None and frame.size > 0:
                # Tello 原生畫面 (960x720)。整條管線用縮小後的 720x480，
                # 但需要細節的偵測 (紙條) 可以直接拿原生畫面 (見 TissueDetector.set_native_frame)
                native = frame
                # INTER_AREA：縮小時取區域平均，細細的紙條比較不會被弄糊或一閃一閃
                frame = cv2.resize(native, (720, 480), interpolation=cv2.INTER_AREA)
                # 錄影用的乾淨畫面，必須在任何疊字之前複製 ——
                # 手動模式沒有 vision，下面的疊字會直接畫在 frame 上
                if self.recorder.recording:
                    raw_frame = frame.copy()
                    if native.shape[:2] != frame.shape[:2]:
                        native_frame = native.copy()
                
                # 如果當前模式有設定 vision 模組，才進行影像分析
                if self.vision:
                    if hasattr(self.vision, 'set_native_frame'):
                        self.vision.set_native_frame(native)
                    t_v0 = time.time()
                    vision_data = self.vision.process_frame(frame)
                    t_vision = time.time() - t_v0
                    # 取出畫上骨架/辨識框的影像
                    if vision_data and vision_data.annotated_frame is not None:
                        frame = vision_data.annotated_frame
                        # 畫上瞄準點，方便對齊目標。預設是畫面正中心；
                        # 模式可以用 aim_point() 指定別的位置 (紙條模式在 75% 高度)
                        aim = (self.behavior.aim_point()
                               if hasattr(self.behavior, 'aim_point') else (360, 240))
                        cv2.circle(frame, aim, 5, (255, 0, 0), cv2.FILLED)
                
                # 在畫面上標示目前的模式名稱 (使用綠色代表有掛載AI，紅色代表純手動)
                text_color = (0, 255, 0) if self.vision else (0, 0, 255)
                cv2.putText(frame, f"Mode: {self.current_mode['name']}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                # 由 behavior 自己決定要顯示什麼 (含環繞方向)，tello_app 不需要知道細節
                orbit_text = (self.behavior.orbit_status()
                              if hasattr(self.behavior, 'orbit_status') else None)
                if orbit_text:
                    cv2.putText(frame, orbit_text, (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 右上角電量 (讀的是背景狀態封包的快取，不會阻塞飛控)
                self.draw_battery(frame, self.drone.get_battery())

                # 左下角顯示最近聽到的話，讓飛手知道麥克風有在收音、聽成了什麼
                heard = self.ui.voice.recent_text()
                if heard:
                    cv2.putText(frame, f"[MIC] {heard}", (10, frame.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 語音飛行中就顯示指令與剩餘秒數 —— 不然動作太短根本看不出有沒有執行
                voice_st = (self.behavior.voice_status()
                            if hasattr(self.behavior, 'voice_status') else None)
                if voice_st:
                    cv2.putText(frame, voice_st, (10, frame.shape[0] - 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 模式要求的大字提示 (例如紙條模式看到不該撞的紙條、正在搖頭拒絕)，畫在畫面正中下方
                alert = (self.behavior.alert_text()
                         if hasattr(self.behavior, 'alert_text') else None)
                if alert:
                    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                    (tw, th), _ = cv2.getTextSize(alert, font, scale, thick)
                    x0 = (frame.shape[1] - tw) // 2
                    y0 = frame.shape[0] // 2 + 60
                    cv2.rectangle(frame, (x0 - 14, y0 - th - 14), (x0 + tw + 14, y0 + 14),
                                  (0, 0, 255), -1)
                    cv2.putText(frame, alert, (x0, y0), font, scale, (255, 255, 255), thick)
            
            # 高度讀值只寫進 log 供事後對照，飛行邏輯不使用：
            # Tello 的高度讀值會隨起飛點、下方的桌櫃改變 (同一個天花板讀過 130 也讀過 230)，
            # 拿來定高或當上限會被騙著撞天花板
            height = self.drone.get_height()

            # 5. 計算並發送飛行指令 
            # (統一將 user_input 與 vision_data 傳給當前的 behavior，由 behavior 決定如何使用)
            t_b0 = time.time()
            commands = self.behavior.calculate_command(user_input, vision_data)
            frame_age = self.drone.frame_age()
            commands, frozen = stale_video_guard(commands, user_input, frame_age,
                                                 isinstance(self.behavior, ManualControl))
            if frozen != self._video_frozen:
                print("[影像凍結] 畫面 %.1f 秒沒有更新 -> 自動模式懸停" % frame_age if frozen
                      else "[影像凍結] 畫面恢復更新")
                self._video_frozen = frozen
            if frozen and frame is not None and frame.size > 0:
                text = f"VIDEO FROZEN {frame_age:.1f}s - HOVER"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                x0, y0 = (frame.shape[1] - tw) // 2, frame.shape[0] // 2 - 40
                cv2.rectangle(frame, (x0 - 12, y0 - th - 12), (x0 + tw + 12, y0 + 12), (0, 0, 0), -1)
                cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            t_behavior = time.time() - t_b0
            
            # 使用 *commands 將 tuple (lr, fb, ud, yv) 解包傳入
            self.drone.send_movement(*commands)

            # 錄影：放在指令算出來之後，log 才記得到這一格實際送出去的指令
            if raw_frame is not None:
                lr, fb, ud, yv = commands
                row = {
                    "mode": self.current_mode['name'],
                    "state": getattr(self.behavior, 'state', ''),
                    "detected": int(bool(vision_data and vision_data.is_detected)),
                    "lr": lr, "fb": fb, "ud": ud, "yv": yv,
                    "battery": self.drone.get_battery(),
                    "height": height,
                    "frame_age": int(frame_age * 1000),    # 畫面多久沒更新 (毫秒)，> 300 會被攔成懸停
                }
                # 由 behavior 自己決定要多記哪些診斷欄位 —— 與 orbit_status() /
                # voice_status() 同一個模式，tello_app 不需要知道各模式的細節。
                # 多出來的欄位會被 DictWriter 的 extrasaction="ignore" 濾掉，
                # 所以新模式加欄位時記得同步加進 VideoRecorder.LOG_FIELDS。
                if hasattr(self.behavior, 'log_fields'):
                    row.update(self.behavior.log_fields())
                self.recorder.write(raw_frame, frame.copy(), row, native=native_frame)
                self.recorder.draw_indicator(frame)   # 只畫在螢幕上，不會進到錄影檔

            # 6. 顯示與刷新畫面
            self.ui.display_frame(frame)

            t_total = time.time() - t_loop
            if t_total > self.SLOW_LOOP_SEC:
                print(f"[迴圈過慢] {t_total:.2f}s  輸入/語音 {t_input - t_loop:.2f}  "
                      f"視覺 {t_vision:.2f}  決策 {t_behavior:.2f}  "
                      f"其他(取像/疊字/錄影/顯示) {t_total - (t_input - t_loop) - t_vision - t_behavior:.2f}")

    def shutdown(self):
        """關閉程序"""
        print("[系統訊息] 正在關閉程序...")
        self.is_running = False
        # 先停錄影：降落是阻塞呼叫，萬一中途出事，mp4 沒收尾就會整個檔案損毀
        self.recorder.stop()
        if self.vision and hasattr(self.vision, 'shutdown'):
            self.vision.shutdown()
        self.drone.land()      # 確保先降落
        self.drone.teardown()  # 關閉無人機連線與串流
        self.ui.teardown()     # 關閉視窗