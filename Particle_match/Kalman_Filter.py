import numpy as np
class Kalman_Filter(object):
    def __init__(self, dt, process_noise_std=10, measurement_noise_std=10):
        # 时间间隔
        self.dt = dt

        # 状态转移矩阵 F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # 观测矩阵 H
        self.H = np.eye(4)

        # 状态协方差矩阵 P
        self.P = np.eye(4) * 1.0
        pos_q = 5    # 每帧位置变化不确定性（像素）
        vel_q = 5    # 每帧速度变化不确定性（像素/帧）

        # 过程噪声协方差矩阵 Q
        q = process_noise_std ** 2

        self.Q = np.array([
            [pos_q, 0, 0, 0],
            [0, pos_q, 0, 0],
            [0, 0, pos_q, 0],
            [0, 0, 0, pos_q]
        ])

        # 观测噪声协方差矩阵 R
        r = measurement_noise_std ** 2
        self.R = np.eye(4) * r

        # 初始状态向量 x
        self.x = np.zeros((4, 1))

    def predict(self):
        # 预测下一状态
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # z: 观测值 [x, y, vx, vy]
        z = np.array(z).reshape((4, 1))

        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态估计和协方差
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x