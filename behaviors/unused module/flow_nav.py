'''
測試用
'''

from behaviors.base import FlightBehavior

class FlowNavControl(FlightBehavior):
    def __init__(self):
        # 危險閾值 (Danger Threshold)：
        # 這個數字非常關鍵！你需要根據剛剛畫面上印出的 L/C/R 數值來調整。
        # 假設無人機向前飛時，離牆壁 1 公尺時中心數值飆到 5.0，那就可以設為 5.0。
        self.danger_threshold = 4.0 
        
        # 基礎前進速度 (必須保持前進才能產生光流)
        self.base_speed = 20 

    def calculate_command(self, user_input, vision_data):
        # 安全機制：只要使用者按鍵盤或用講話控制，立刻接管，打斷自動避障
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)
            
        if not getattr(vision_data, 'is_detected', False) or not hasattr(vision_data, 'flow_regions'):
            return (0, 0, 0, 0)

        # 取得左中右的壓迫感 (光流強度)
        regions = vision_data.flow_regions
        L = regions["left"]
        C = regions["center"]
        R = regions["right"]

        lr, fb, ud, yv = 0, 0, 0, 0

        # ==========================================
        # 光流避障決策樹
        # ==========================================
        if C > self.danger_threshold:
            print(f"⚠️ 正前方危險 (C={C:.1f})！緊急煞車並向空曠處平移")
            fb = -10 # 稍微後退
            # 比較左右哪邊比較空曠 (數值較小的方向)，就往哪邊躲
            lr = 30 if L < R else -30 
            
        elif L > self.danger_threshold:
            print(f"⚠️ 左側有障礙 (L={L:.1f})！向右閃避")
            lr = 35
            fb = 10 # 閃避時減慢前進速度
            
        elif R > self.danger_threshold:
            print(f"⚠️ 右側有障礙 (R={R:.1f})！向左閃避")
            lr = -35
            fb = 10
            
        else:
            # 視野開闊，持續向前探索 (產生光流)
            fb = self.base_speed
            
        return (lr, fb, ud, yv)