# import time
# import random
# from behaviors.base import FlightBehavior
# from utils.pid_controller import PIDController

# class BalloonHuntControl(FlightBehavior):
#     def __init__(self):
#         self.state = "SCAN_ROOM"
#         self.current_target_id = 1
        
#         # PID 控制器設定
#         self.pid_yv = PIDController(kp=0.5, ki=0.01, kd=0.2, limit=60)
#         self.pid_fb = PIDController(kp=0.4, ki=0.0, kd=0.1, limit=50)
#         self.pid_ud = PIDController(kp=0.4, ki=0.0, kd=0.1, limit=50)
        
#         self.target_cx = 360
#         self.target_cy = 240 
        
#         self.ORBIT_AREA = 80000   
#         self.ATTACK_AREA = 250000 
        
#         # --- 狀態計時與探索參數 ---
#         self.state_start_time = time.time()
#         self.scan_duration = 80.0     # 360度環顧大約需要的時間(秒)
#         self.explore_duration = 4.0  # 每次往前推進換位子的時間(秒)
#         self.safe_depth = 0         # 牆壁安全距離(公分)

#     def change_state(self, new_state):
#         if self.state != new_state:
#             self.state = new_state
#             self.state_start_time = time.time()
#             print(f"🔄 [FSM 切換] {self.state} (當前目標: {self.current_target_id}號)")


#     def calculate_command(self, user_input, vision_data):
#         # 1. 人工接管優先，並重置狀態
#         if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
#             self.change_state("SCAN_ROOM") 
#             return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

#         lr, fb, ud, yv = 0, 0, 0, 0
#         now = time.time()
#         time_in_state = now - self.state_start_time

#         balloon = getattr(vision_data, 'balloon', None) 
#         marker_id = getattr(vision_data, 'marker_id', None)
        
#         # 取得深度資料
#         depth_C = getattr(vision_data, 'center_depth', 999)
#         depth_L = getattr(vision_data, 'depth_L', 999)
#         depth_R = getattr(vision_data, 'depth_R', 999)

#         # ==========================================
#         # 🧠 躍進掃描 FSM (Look-and-Leap Exploration)
#         # ==========================================
        
#         # --- 探索階段 ---
#         if self.state == "SCAN_ROOM":
#             yv = 40
#             if balloon:
#                 self.change_state("APPROACH")
#             elif time_in_state > self.scan_duration:
#                 self.change_state("EXPLORE_FORWARD")

#         elif self.state == "EXPLORE_FORWARD":
#             fb = 30
#             if balloon:
#                 self.change_state("APPROACH")
#             elif depth_C < self.safe_depth:
#                 self.change_state("AVOID_WALL")
#             elif time_in_state > self.explore_duration:
#                 self.change_state("SCAN_ROOM")

#         elif self.state == "AVOID_WALL":
#             fb = -20
#             if depth_L > depth_R:
#                 yv = -50
#             elif depth_R > depth_L:
#                 yv = 50
#             else:
#                 yv = random.choice([-50, 50])
                
#             if time_in_state > 1.5:
#                 self.change_state("EXPLORE_FORWARD")

#         # --- 接戰階段 ---
#         elif self.state == "APPROACH":
#             if not balloon:
#                 self.change_state("SCAN_ROOM")
#             else:
#                 error_y = self.target_cy - balloon['cy']
#                 ud = self.pid_ud.compute(error_y)
                
#                 error_x = balloon['cx'] - self.target_cx
#                 yv = self.pid_yv.compute(error_x)
                
#                 if balloon['area'] < self.ORBIT_AREA:
#                     fb = 20
#                 else:
#                     if marker_id is not None:
#                         self._decide_action(marker_id)
#                     else:
#                         self.change_state("ORBIT")

#         elif self.state == "ORBIT":
#             if not balloon:
#                 self.change_state("SCAN_ROOM")
#             else:
#                 # 🔥 環繞時同樣要進行上下鎖定
#                 error_y = self.target_cy - balloon['cy']
#                 ud = self.pid_ud.compute(error_y)
                
#                 error_x = balloon['cx'] - self.target_cx
#                 yv = self.pid_yv.compute(error_x)
                
#                 error_area = (self.ORBIT_AREA - balloon['area']) / 1000.0
#                 fb = self.pid_fb.compute(error_area)
#                 lr = 20
                
#                 if marker_id is not None:
#                     self._decide_action(marker_id)

#         elif self.state == "ATTACK":
#             if not balloon:
#                 print(f"💥 {self.current_target_id} 號氣球已擊破或丟失！")
#                 self.current_target_id += 1 
#                 self.change_state("SCAN_ROOM")
#             else:
#                 # 🔥 衝刺時也要維持高度對準，確保完美撞擊
#                 error_y = self.target_cy - balloon['cy']
#                 ud = self.pid_ud.compute(error_y)
                
#                 error_x = balloon['cx'] - self.target_cx
#                 yv = self.pid_yv.compute(error_x)
#                 fb = 70 
                
#                 if balloon['area'] > self.ATTACK_AREA:
#                     print(f"💥 物理接觸確認！(Area: {balloon['area']})")
#                     self.current_target_id += 1
#                     fb = -40 
#                     self.change_state("SCAN_ROOM")

#         elif self.state == "AVOID_BALLOON":
#             if balloon:
#                 fb = -60
#                 lr = -60
                
#                 # 退避時也稍微維持高度
#                 error_y = self.target_cy - balloon['cy']
#                 ud = self.pid_ud.compute(error_y)
                
#                 if balloon['area'] < self.ORBIT_AREA * 0.4:
#                     self.change_state("SCAN_ROOM")
#             else:
#                 self.change_state("SCAN_ROOM")

#         return (int(lr), int(fb), int(ud), int(yv))

#     def _decide_action(self, marker_id):
#         if marker_id == self.current_target_id:
#             self.change_state("ATTACK")
#         elif marker_id == 0:
#             self.change_state("AVOID_BALLOON")
#         else:
#             print(f"這是 {marker_id} 號，當前目標為 {self.current_target_id} 號。略過並繼續尋找。")
#             self.change_state("SCAN_ROOM")

import time
import random
from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class BalloonHuntControl(FlightBehavior):
    def __init__(self):
        self.state = "SCAN_ROOM"
        self.current_target_id = 1

        # PID 控制器設定
        self.pid_yv = PIDController(kp=0.5, ki=0.01, kd=0.2, limit=60)
        self.pid_fb = PIDController(kp=0.4, ki=0.0, kd=0.1, limit=50)
        self.pid_ud = PIDController(kp=0.4, ki=0.0, kd=0.1, limit=50)

        self.target_cx = 360
        self.target_cy = 240 

        self.ORBIT_AREA = 80000   
        self.ATTACK_AREA = 250000 

        # --- 狀態計時與探索參數 ---
        self.state_start_time = time.time()
        self.scan_duration = 80.0     # 360度環顧大約需要的時間(秒)
        self.explore_duration = 4.0  # 每次往前推進換位子的時間(秒)
        self.safe_depth = 0         # 牆壁安全距離(公分)

        # ==========================================
        # 🔥 新增：視覺記憶防閃爍參數
        # ==========================================
        self.last_balloon = None
        self.last_balloon_time = 0.0
        self.MEMORY_DURATION = 1.0  # 容許 YOLO 盲目的最長時間 (秒)

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            print(f"🔄 [FSM 切換] {self.state} (當前目標: {self.current_target_id}號)")

    def calculate_command(self, user_input, vision_data):
        # 1. 人工接管優先，並重置狀態
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("SCAN_ROOM") 
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = time.time()
        time_in_state = now - self.state_start_time

        # 取得當前幀的真實視覺資料
        current_balloon = getattr(vision_data, 'balloon', None) 
        marker_id = getattr(vision_data, 'marker_id', None)

        # ==========================================
        # 🔥 視覺記憶補償邏輯 (Vision Memory)
        # ==========================================
        if current_balloon is not None:
            # YOLO 看到了！更新記憶庫
            self.last_balloon = current_balloon
            self.last_balloon_time = now
            balloon = current_balloon
        else:
            # YOLO 沒看到，檢查記憶是否還在有效期限內
            if now - self.last_balloon_time <= self.MEMORY_DURATION and self.last_balloon is not None:
                balloon = self.last_balloon
                # (背景默默使用上一幀的資料繼續飛，不中斷 PID)
            else:
                # 記憶過期，徹底宣告目標丟失
                balloon = None
                self.last_balloon = None

        # 取得深度資料
        depth_C = getattr(vision_data, 'center_depth', 999)
        depth_L = getattr(vision_data, 'depth_L', 999)
        depth_R = getattr(vision_data, 'depth_R', 999)

        # ==========================================
        # 🧠 躍進掃描 FSM (Look-and-Leap Exploration)
        # ==========================================

        # --- 探索階段 ---
        if self.state == "SCAN_ROOM":
            yv = 40
            if balloon:
                self.change_state("APPROACH")
            elif time_in_state > self.scan_duration:
                self.change_state("EXPLORE_FORWARD")

        elif self.state == "EXPLORE_FORWARD":
            fb = 30
            if balloon:
                self.change_state("APPROACH")
            elif depth_C < self.safe_depth:
                self.change_state("AVOID_WALL")
            elif time_in_state > self.explore_duration:
                self.change_state("SCAN_ROOM")

        elif self.state == "AVOID_WALL":
            fb = -20
            if depth_L > depth_R:
                yv = -50
            elif depth_R > depth_L:
                yv = 50
            else:
                yv = random.choice([-50, 50])

            if time_in_state > 1.5:
                self.change_state("EXPLORE_FORWARD")

        # --- 接戰階段 ---
        elif self.state == "APPROACH":
            if not balloon:
                self.change_state("SCAN_ROOM")
            else:
                error_y = self.target_cy - balloon['cy']
                ud = self.pid_ud.compute(error_y)

                error_x = balloon['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)

                if balloon['area'] < self.ORBIT_AREA:
                    fb = 20
                else:
                    if marker_id is not None:
                        self._decide_action(marker_id)
                    else:
                        self.change_state("ORBIT")

        elif self.state == "ORBIT":
            if not balloon:
                self.change_state("SCAN_ROOM")
            else:
                error_y = self.target_cy - balloon['cy']
                ud = self.pid_ud.compute(error_y)

                error_x = balloon['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)

                error_area = (self.ORBIT_AREA - balloon['area']) / 1000.0
                fb = self.pid_fb.compute(error_area)
                lr = 20

                if marker_id is not None:
                    self._decide_action(marker_id)

        elif self.state == "ATTACK":
            if not balloon:
                # 這裡的消失可能是因為撞破了，或者是脫離畫面超過 1 秒
                print(f"💥 {self.current_target_id} 號氣球已擊破或丟失！")
                self.current_target_id += 1 
                self.change_state("SCAN_ROOM")
            else:
                error_y = self.target_cy - balloon['cy']
                ud = self.pid_ud.compute(error_y)

                error_x = balloon['cx'] - self.target_cx
                yv = self.pid_yv.compute(error_x)
                fb = 70 

                if balloon['area'] > self.ATTACK_AREA:
                    print(f"💥 物理接觸確認！(Area: {balloon['area']})")
                    self.current_target_id += 1
                    fb = -40 
                    self.change_state("SCAN_ROOM")

        elif self.state == "AVOID_BALLOON":
            if balloon:
                fb = -60
                lr = -60

                error_y = self.target_cy - balloon['cy']
                ud = self.pid_ud.compute(error_y)

                if balloon['area'] < self.ORBIT_AREA * 0.4:
                    self.change_state("SCAN_ROOM")
            else:
                self.change_state("SCAN_ROOM")

        return (int(lr), int(fb), int(ud), int(yv))

    def _decide_action(self, marker_id):
        if marker_id == self.current_target_id:
            self.change_state("ATTACK")
        elif marker_id == 0:
            self.change_state("AVOID_BALLOON")
        else:
            print(f"👁️ 這是 {marker_id} 號，當前目標為 {self.current_target_id} 號。略過並繼續尋找。")
            self.change_state("SCAN_ROOM")