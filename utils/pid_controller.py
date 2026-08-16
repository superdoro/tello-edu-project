import numpy as np

class PIDController:
    """獨立的 PID 速度計算器"""
    def __init__(self, kp: float, ki: float, kd: float, limit: int = 60):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.prev_error = 0

    def compute(self, error: float) -> int:
        # 目前為簡化的 PD 控制
        output = self.kp * error + self.kd * (error - self.prev_error)
        self.prev_error = error
        
        # 限制最高速度以防暴衝
        return int(np.clip(output, -self.limit, self.limit))