import time
import math
from behaviors.base import FlightBehavior

class BuildingSurroundControl(FlightBehavior):
    def __init__(self):
        # 初始狀態改為 SLAM 初始化之舞
        self.state = "INIT_DANCE"
        self.state_start_time = time.time()
        
        # ==========================================
        # 距離與遲滯區間參數設定 (公分)
        # ==========================================
        self.SAFE_DIST = 300      # 觸發避障的安全距離 (前方遭遇內轉角或死胡同)
        self.CLEAR_DIST = 350     # 遲滯區間：判定前方已經開闊，可以結束轉彎
        self.CRITICAL_DIST = 200  # 極限危險距離：低於此距離代表弧線轉不過去，必須煞車或後退
        self.WALL_LOST_DIST = 400 # 判定右側牆壁消失的距離 (遭遇建築物外轉角)
        
        # 🛡️ 側向防撞 (壁讓) 參數
        self.SIDE_SAFE_DIST = 150 # 側邊安全距離，小於此距離觸發側向閃避
        self.SIDE_AVOID_SPEED = 25 # 側向微調的速度
        
        # ==========================================
        # SLAM 閉環記憶參數
        # ==========================================
        self.RETURN_RADIUS = 1.0   # 閉環半徑 (距離起點小於 1 公尺即算完成)
        self.MIN_FLIGHT_DIST = 4.0 # 防呆：總飛行距離必須大於 4 公尺才能觸發閉環
        self.start_pose = None
        self.last_pose = None
        self.total_dist = 0.0

        self.turn_speed = 0

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            # 狀態切換時，順便印出累積的飛行距離
            print(f"[狀態切換] {self.state} | 總距離: {self.total_dist:.1f}m")

    def calculate_command(self, user_input, vision_data):
        # 人工防呆介入：隨時可以推動搖桿接管
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            self.change_state("FORWARD")
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        now = time.time()
        time_in_state = now - self.state_start_time
        
        # 取得 YOLO 左中右深度分區資料
        depth_L = getattr(vision_data, 'depth_L', 999.0)
        depth_C = getattr(vision_data, 'depth_C', 999.0)
        depth_R = getattr(vision_data, 'depth_R', 999.0)
        
        # 取得 SLAM 狀態 (由 Vision 模組提供)
        slam_active = getattr(vision_data, 'tracking_state', 0) == 1

        # ==========================================
        # SLAM 軌跡紀錄與終極閉環判定
        # ==========================================
        if slam_active:
            curr_x, curr_z = vision_data.x, vision_data.z
            
            # 累加總飛行距離
            if self.last_pose is not None:
                dist_moved = math.sqrt((curr_x - self.last_pose[0])**2 + (curr_z - self.last_pose[1])**2)
                self.total_dist += dist_moved
            self.last_pose = (curr_x, curr_z)

            # 檢查是否完成環繞並回到起點附近
            if self.start_pose is not None and self.state not in ["MISSION_COMPLETE", "INIT_DANCE"]:
                dist_to_start = math.sqrt((curr_x - self.start_pose[0])**2 + (curr_z - self.start_pose[1])**2)
                if self.total_dist > self.MIN_FLIGHT_DIST and dist_to_start < self.RETURN_RADIUS:
                    print(f"🎉 任務達成！成功環繞建築物並回到起點 (誤差: {dist_to_start:.2f}m)")
                    self.change_state("MISSION_COMPLETE")

        # ==========================================
        # 右手法則流暢弧線大腦 (Right-Priority Smooth Arcing)
        # ==========================================
        
        # 階段 0：起飛後的視差初始化之舞
        if self.state == "INIT_DANCE":
            if slam_active:
                # SLAM 成功定位，鎖定起點座標
                self.start_pose = (vision_data.x, vision_data.z)
                self.total_dist = 0.0
                print("✅ SLAM 初始化成功！鎖定起點，開始順時針環繞建築物。")
                self.change_state("FORWARD")
            else:
                # 左右平移製造特徵點供 SLAM 抓取
                lr = -25 if (int(time_in_state / 2) % 2 == 0) else 25

        # 階段 1：沿牆直飛
        elif self.state == "FORWARD":
            fb = 40 # 穩定的基礎前進速度
            
            # 🛡️ 側向防撞邏輯 (壁讓)
            if depth_R < self.SIDE_SAFE_DIST and depth_L >= self.SIDE_SAFE_DIST:
                lr = -self.SIDE_AVOID_SPEED  # 右邊太近，往左閃
            elif depth_L < self.SIDE_SAFE_DIST and depth_R >= self.SIDE_SAFE_DIST:
                lr = self.SIDE_AVOID_SPEED   # 左邊太近，往右閃
            elif depth_L < self.SIDE_SAFE_DIST and depth_R < self.SIDE_SAFE_DIST:
                lr = self.SIDE_AVOID_SPEED if depth_L > depth_R else -self.SIDE_AVOID_SPEED
            
            # 狀況 A：前方撞牆 (遇到建築物內轉角或障礙) -> 優先進入左弧線
            if depth_C <= self.SAFE_DIST:
                self.turn_speed = -65 
                print(f"⚠️ 遭遇前方障礙 ({int(depth_C)}cm) -> 進入左弧線")
                self.change_state("CURVING_LEFT")
                
            # 狀況 B：前方暢通，但右側牆壁消失 (遇到建築物外轉角) -> 右弧線追蹤
            elif depth_R > self.WALL_LOST_DIST:
                self.turn_speed = 65  
                print(f"📐 右側牆壁消失 ({int(depth_R)}cm) -> 進入右弧線探索")
                self.change_state("CURVING_RIGHT")

        # 階段 2：左弧線過彎 (避開內轉角)
        elif self.state == "CURVING_LEFT":
            if depth_C < self.CRITICAL_DIST:
                fb = -30 # 轉彎弧度不夠快撞上時，取消前進改為倒車
            else:
                fb = 25  # 維持些微前進，畫出流暢弧線
                
            yv = self.turn_speed
            
            # 利用遲滯區間：前方確認足夠開闊時才切回直飛
            if depth_C > self.CLEAR_DIST:
                self.change_state("FORWARD")

        # 階段 3：右弧線過彎 (繞過外轉角)
        elif self.state == "CURVING_RIGHT":
            fb = 30 # 尋找牆壁時，前進速度稍微帶快一點，拉大探索範圍
            yv = self.turn_speed
            
            # 防呆 1：如果轉右彎過程中，前方突然掃到新障礙，立刻切回左轉避障
            if depth_C <= self.SAFE_DIST:
                print(f"🚨 右彎時遭遇新障礙 ({int(depth_C)}cm) -> 緊急切入左弧線")
                self.turn_speed = -65
                self.change_state("CURVING_LEFT")
                
            # 防呆 2：當右邊再次掃到牆壁 (加入 50cm 遲滯)，代表成功繞過轉角
            elif depth_R <= self.WALL_LOST_DIST - 50:
                print(f"🧱 重新捕捉到右側牆壁 ({int(depth_R)}cm) -> 切回直線沿牆飛行")
                self.change_state("FORWARD")

        # 階段 4：任務完成懸停
        elif self.state == "MISSION_COMPLETE":
            fb, lr, ud, yv = 0, 0, 0, 0

        return (int(lr), int(fb), int(ud), int(yv))