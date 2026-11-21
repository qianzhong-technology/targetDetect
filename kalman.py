"""二维匀速卡尔曼滤波器。"""

import numpy as np

class TargetKalman:

    def __init__(self):
        self.initialized = False
        self.state = np.zeros(
            (4, 1),
            dtype=np.float64
        )
        self.P = np.eye(
            4,
            dtype=np.float64
        ) * 100.0
        self.R = np.eye(2, dtype=np.float64) * 9.0
        self.Q_strength = 35.0
        self.last_measurement = None
        self.last_update_time = None
        self.missed_frames = 0
        self.max_missed_frames = 12
        self.last_innovation = 0.0

    def reset(self):
        self.initialized = False
        self.state[:] = 0
        self.P = np.eye(
            4,
            dtype=np.float64
        ) * 100.0
        self.last_measurement = None
        self.last_update_time = None
        self.missed_frames = 0
        self.last_innovation = 0.0

    def predict(self, dt):
        if not self.initialized:
            return None

        dt = max(0.001, min(float(dt), 0.2))
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        Q = self.Q_strength * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2]
        ], dtype=np.float64)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.missed_frames += 1

        # 连续多帧没有测量时逐步降低速度可信度。
        if self.missed_frames >= 6:
            decay = max(0.05, 1.0 - 0.12 * (self.missed_frames - 5))
            self.state[2, 0] *= decay
            self.state[3, 0] *= decay

        if self.missed_frames > self.max_missed_frames:
            self.reset()
            return None

        return (
            float(self.state[0, 0]),
            float(self.state[1, 0])
        )

    def update(self, measurement, dt):
        mx, my = measurement
        dt = max(
            0.001,
            min(float(dt), 0.2)
        )
        if not self.initialized:
            self.state[:, 0] = [
                mx,
                my,
                0.0,
                0.0
            ]
            self.initialized = True
            self.last_measurement = np.array([mx, my], dtype=np.float64)
            self.last_update_time = dt
            self.missed_frames = 0
            return mx, my
        if self.last_measurement is not None:
            measurement_delta = np.array([mx, my], dtype=np.float64) - self.last_measurement
            measured_velocity = measurement_delta / dt
            measured_velocity = np.clip(measured_velocity, -500.0, 500.0)
            jump = float(np.linalg.norm(measurement_delta))
            if jump > 120.0:
                measured_velocity[:] = 0.0
            else:
                self.state[2, 0] = (
                    0.25 * self.state[2, 0]
                    + 0.75 * measured_velocity[0]
                )
                self.state[3, 0] = (
                    0.25 * self.state[3, 0]
                    + 0.75 * measured_velocity[1]
                )
        self.state[0, 0] = mx
        self.state[1, 0] = my
        self.last_measurement = np.array([mx, my], dtype=np.float64)
        self.missed_frames = 0
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0])
        )
