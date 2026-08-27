import pygame
import cv2
from dataclasses import dataclass
from core.voice_controller import VoiceController

# ==========================================
# 定義輸入資料結構 (Data Transfer Object)
# ==========================================
@dataclass
class UserInput:
    """
    用來打包使用者輸入的指令狀態。
    包含了移動軸向數值 (lr, fb, ud, yv) 與系統功能布林值 (True/False)。
    """
    # 飛行方向數值 (預設為 0)
    lr: int = 0  # 左右平移
    fb: int = 0  # 前後平移
    ud: int = 0  # 上下升降
    yv: int = 0  # 左右旋轉
    
    # 系統指令狀態 (預設為 False)
    takeoff: bool = False     # 是否要求起飛
    land: bool = False        # 是否要求降落
    quit: bool = False        # 是否要求退出程式
    toggle_mode: bool = False # 是否要求切換模式 (如按 Z 鍵)

    # 聲音指令
    voice_command: str = ""


# ==========================================
# UI 控制器類別
# ==========================================
class UIController:
    """
    負責初始化與管理 Pygame 鍵盤監聽視窗，以及 OpenCV 影像顯示視窗。
    """
    def __init__(self):
        # 初始化 Pygame 控制面板
        pygame.init()
        self.win = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Tello 整合控制面板 (請點擊此視窗)")
        self.speed = 50 # 預設鍵盤控制速度
        self.voice = VoiceController() #聲控類別
        self.current_vision = None

    def get_input(self) -> UserInput:
        """
        掃描目前的鍵盤狀態，並轉換為 UserInput 物件回傳。
        """
        user_input = UserInput()
        
        # 1. 處理單次觸發事件 (例如按一下就切換狀態的按鍵)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    user_input.toggle_mode = True
                elif event.key == pygame.K_t:
                    user_input.takeoff = True
                elif event.key == pygame.K_l:
                    user_input.land = True
                elif event.key == pygame.K_q:
                    user_input.quit = True
                
        # 2. 處理持續按壓事件 (例如長按方向鍵來連續飛行)
        keys = pygame.key.get_pressed()
        
        # 水平控制 (左右方向鍵)
        if keys[pygame.K_LEFT]: user_input.lr = -self.speed
        elif keys[pygame.K_RIGHT]: user_input.lr = self.speed
        
        # 前後控制 (上下方向鍵)
        if keys[pygame.K_UP]: user_input.fb = self.speed
        elif keys[pygame.K_DOWN]: user_input.fb = -self.speed
        
        # 垂直控制 (W、S 鍵)
        if keys[pygame.K_w]: user_input.ud = self.speed
        elif keys[pygame.K_s]: user_input.ud = -self.speed
        
        # 旋轉控制 (A、D 鍵)
        if keys[pygame.K_a]: user_input.yv = -self.speed
        elif keys[pygame.K_d]: user_input.yv = self.speed

        # 假設 key 是 Pygame 讀取到的按鍵
        if self.current_vision and keys[pygame.K_f]:
            if hasattr(self.current_vision, 'toggle_tracking_mode'):
                self.current_vision.toggle_tracking_mode()

        if self.current_vision and keys[pygame.K_r]:
            if hasattr(self.current_vision, 'reset_target'):
                self.current_vision.reset_target()

        # cmd = self.voice.get_command()
        # if cmd:
        #     print(f"🎤 [語音辨識] 聽到指令: {cmd}")
        #     user_input.voice_command = cmd
            
            # 處理系統級別的語音指令
            # if "起飛" in cmd: user_input.takeoff = True
            # if "降落" in cmd: user_input.land = True
            # elif "切換" in cmd: user_input.toggle_mode = True
            
        return user_input

    def display_frame(self, frame):
        """
        顯示 OpenCV 影像並刷新 Pygame 控制面板，防止視窗當機。
        """
        if frame is not None and frame.size > 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cv2.imshow("Tello Live Video", frame)
            
        cv2.waitKey(1) # OpenCV 的必要刷新指令
        
        # 填滿深灰色背景並更新 Pygame 視窗
        self.win.fill((30, 30, 30))
        pygame.display.update()
        
    def teardown(self):
        """
        安全關閉所有視窗並結束引擎。
        """
        print("[系統訊息] 正在關閉介面與視窗...")
        pygame.quit()
        cv2.destroyAllWindows()