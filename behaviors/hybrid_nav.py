from behaviors.base import FlightBehavior
from utils.pid_controller import PIDController

class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = 0.0
        
    def update(self, new_val):
        self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

class HybridNavControl(FlightBehavior):
    def __init__(self):
        # 1. YOLO 追蹤設定
        self.pid_yv = PIDController(kp=0.2, ki=0, kd=0.1, limit=50) 
        self.target_cx = 360 

        # 2. RAFT 避障設定
        self.center_danger_threshold = 5.0
        self.side_danger_threshold = 2.0
        
        self.repulse_gain_fb = 8.0  
        self.repulse_gain_lr = 10.0 
        
        self.ema_L = EMAFilter(alpha=0.4)
        self.ema_C = EMAFilter(alpha=0.4)
        self.ema_R = EMAFilter(alpha=0.4)

        # ==========================================
        # 🔥 新增：目標短期記憶機制
        # ==========================================
        self.last_error_x = 0      # 記住目標最後在偏左還是偏右
        self.lost_frames = 999     # 記錄已經跟丟了幾個畫面
        self.memory_limit = 90     # 記憶維持時間

    def calculate_command(self, user_input, vision_data):
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        target = getattr(vision_data, 'target', None)
        regions = getattr(vision_data, 'flow_regions', None)

        force_fb, force_lr, force_yv = 0, 0, 0

        # ==========================================
        # 斥力計算 (保持不變)
        # ==========================================
        if regions:
            L = self.ema_L.update(regions["left"])
            C = self.ema_C.update(regions["center"])
            R = self.ema_R.update(regions["right"])

            if C > self.center_danger_threshold:
                overflow = C - self.center_danger_threshold
                force_fb -= min(70, overflow * self.repulse_gain_fb)
                
            if L > self.side_danger_threshold:
                overflow = L - self.side_danger_threshold
                force_lr += min(50, overflow * self.repulse_gain_lr)
                force_fb -= min(40, overflow * self.repulse_gain_fb)
                
            if R > self.side_danger_threshold:
                overflow = R - self.side_danger_threshold
                force_lr -= min(50, overflow * self.repulse_gain_lr) 
                force_fb -= min(40, overflow * self.repulse_gain_fb)

        # ==========================================
        # 引力計算與記憶尋回 (Memory Recovery)
        # ==========================================
        if target:
            # 💡 [狀態：看到目標]
            error_x = target['cx'] - self.target_cx
            force_yv = self.pid_yv.compute(error_x)

            # 更新短期記憶
            self.last_error_x = error_x
            self.lost_frames = 0
            
            if abs(error_x) < 40:
                force_fb += 40 
        else:
            # 💡 [狀態：失去目標]
            self.lost_frames += 1
            
            if self.lost_frames < self.memory_limit:
                # 剛跟丟！根據殘影記憶，強制轉頭找回來
                # 這在橫移閃避時，會自動形成完美的「補償轉向」
                if self.last_error_x > 0:
                    force_yv = 20  # 目標最後在右邊，機頭往右追
                else:
                    force_yv = -20 # 目標最後在左邊，機頭往左追
                    
                # 只有在前幾幀印出提示，避免洗版
                if self.lost_frames % 15 == 1:
                    direction = "右" if self.last_error_x > 0 else "左"
                    print(f"🔍 [記憶尋回] 閃避導致目標丟失，強制向{direction}轉頭鎖定！")
            else:
                # 徹底遺忘，恢復平靜搜尋模式
                if force_fb == 0 and force_lr == 0:
                    force_fb = 0
                    # force_yv = 20 # 讓它原地緩慢轉圈尋找 (可依需求解除註解)

        # ==========================================
        # 輸出限制與除錯
        # ==========================================
        final_fb = int(max(-50, min(60, force_fb)))
        final_lr = int(max(-50, min(50, force_lr)))
        final_yv = int(force_yv)

        if final_lr != 0 or force_fb < 0:
            print(f"🌊 [勢場狀態] FB: {final_fb} | LR: {final_lr} | YV: {final_yv}")

        return (final_lr, final_fb, 0, final_yv)