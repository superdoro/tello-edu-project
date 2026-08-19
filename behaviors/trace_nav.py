from behaviors.base import FlightBehavior

class TraceNavControl(FlightBehavior):
    def __init__(self):
        # 畫面中心點
        self.target_cx = 360 
        self.target_cy = 240 

        # 距離控制的面積閾值
        self.DANGER_AREA = 13800   # 太近：觸發後退
        self.TOO_FAR_AREA = 66400   # 剛好：懸停 (小於這個值代表太遠，觸發前進)

        # 對齊容許誤差 (Deadzone)：在這個像素範圍內視為「已對準」，輸出 0
        self.align_threshold = 30

        # ==========================================
        # 目標短期記憶機制
        # ==========================================
        self.last_error_x = 0
        self.last_error_y = 0
        self.lost_frames = 999
        self.memory_limit = 120 # 記憶維持的 frame 數
        
    def calculate_command(self, user_input, vision_data):
        # 1. 人工接管優先
        if any([user_input.lr, user_input.fb, user_input.ud, user_input.yv]):
            return (user_input.lr, user_input.fb, user_input.ud, user_input.yv)

        lr, fb, ud, yv = 0, 0, 0, 0
        target = getattr(vision_data, 'target', None) if vision_data else None

        if target:
            # 💡 【狀態 1：看到目標】
            self.lost_frames = 0
            error_x = target['cx'] - self.target_cx
            error_y = self.target_cy - target['cy'] # 注意 Tello 的 Y 軸方向：向上飛是正值
            
            # 更新殘影記憶
            self.last_error_x = error_x
            self.last_error_y = error_y

            # -------------------------
            # A. 對齊 (永遠 100, -100 或 0)
            # -------------------------
            if abs(error_x) > self.align_threshold:
                yv = 100 if error_x > 0 else -100
                
            if abs(error_y) > self.align_threshold:
                ud = 100 if error_y > 0 else -100

            # -------------------------
            # B. 距離控制 (利用面積大小)
            # -------------------------
            area = target['area']
            if area > self.DANGER_AREA:
                fb = -30  # 太近，向後退
            elif area < self.TOO_FAR_AREA:
                # 太遠，而且必須「機頭已經對準」的情況下才往前衝
                if yv == 0 and ud == 0:
                    fb = 30
            else:
                fb = 0    # 距離適中，維持不前後移動

        else:
            # 💡 【狀態 2：目標丟失 (記憶尋回)】
            self.lost_frames += 1
            
            if self.lost_frames < self.memory_limit:
                # 根據殘影，使用最大速度暴力轉頭/升降尋找目標
                if abs(self.last_error_x) > self.align_threshold:
                    yv = 100 if self.last_error_x > 0 else -100
                if abs(self.last_error_y) > self.align_threshold:
                    ud = 100 if self.last_error_y > 0 else -100
                    
                # 避免終端機瘋狂洗版，每半秒印一次提示
                if self.lost_frames % 15 == 1:
                    print(f"🔍 [記憶尋回] 目標丟失！全速找回中 (YV:{yv}, UD:{ud})")
            else:
                # 徹底遺忘，原地懸停發呆
                lr, fb, ud, yv = 0, 0, 0, 0

        # 保險機制：確保回傳的一定是整數
        return (int(lr), int(fb), int(ud), int(yv))