from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class RunningNavControl(FlightBehavior):
    def __init__(self, direction="cw"):
        """
        :param direction: "cw" (順時針, 牆壁在左側) 或 "ccw" (逆時針, 牆壁在右側)
        """
        self.direction = direction
        
        # 目標設定
        self.TARGET_WALL_DIST = 100
        self.CORNER_BRAKE_DIST = 120
        
        # 轉向 PID (負責維持側邊牆壁距離)
        self.pid_steer = PIDController(kp=0.6, ki=0.02, kd=0.2, limit=60)
        
    def calculate_command(self, user_input, vision_data):
        # 1. 人工接管優先
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.pid_steer.reset()
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        
        # 防呆：如果視覺沒有傳回深度，原地懸停
        if not hasattr(vision_data, 'center_depth'):
            return (0, 0, 0, 0)

        depth_C = vision_data.center_depth
        depth_L = vision_data.depth_L
        depth_R = vision_data.depth_R

        # ==========================================
        # 🧭 沿牆循跡演算法 (Wall-Following)
        # ==========================================
        
        # 基礎推進速度
        fb = 30 
        
        # 1. 側向牆壁跟蹤 (Steering)
        if self.direction == "cw":
            # 順時針飛行：牆壁在左邊，所以我們監控左側深度 (depth_L)
            # 如果左側太遠，誤差為正，PID 輸出正值，我們給予負的 yv (向左轉) 來靠近牆壁
            error = depth_L - self.TARGET_WALL_DIST
            yv = -self.pid_steer.compute(error)
        else:
            # 逆時針飛行：牆壁在右邊，監控右側深度 (depth_R)
            # 如果右側太遠，誤差為正，PID 輸出正值，我們給予正的 yv (向右轉) 來靠近牆壁
            error = depth_R - self.TARGET_WALL_DIST
            yv = self.pid_steer.compute(error)

        # 2. 轉角處理 (Corner Handling)
        # 當正前方遇到牆壁時，PID 還來不及把機頭拉過來，我們必須強力介入
        if depth_C < self.CORNER_BRAKE_DIST:
            # 動態煞車：離前方牆壁越近，往前飛的速度越慢
            fb = max(0, int((depth_C - 50) * 0.4))
            
            # 強力轉彎：覆寫 PID，直接給予強大的旋轉動力過彎
            turn_power = 60
            if self.direction == "cw":
                yv = turn_power  # 順時針遇到轉角，向右急轉
            else:
                yv = -turn_power # 逆時針遇到轉角，向左急轉
                
            print(f"⚠️ [過彎介入] 前方障礙 ({int(depth_C)}cm)，執行強勢轉向！")

        return (int(lr), int(fb), int(ud), int(yv))