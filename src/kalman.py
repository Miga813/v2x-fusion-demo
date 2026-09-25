import numpy as np


class ConstantVelocityKalman:
    """二维常速度卡尔曼滤波器。

    状态向量：
        state = [x, y, vx, vy]

    分别表示：
        x、y：目标位置
        vx、vy：目标速度
    """

    def __init__(self, process_noise: float = 0.25):
        self.process_noise = process_noise

    def predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据上一时刻的状态预测当前状态。"""

        # 状态转移矩阵
        state_transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # 将加速度噪声映射到位置和速度
        noise_mapping = np.array(
            [
                [0.5 * dt**2, 0.0],
                [0.0, 0.5 * dt**2],
                [dt, 0.0],
                [0.0, dt],
            ]
        )

        acceleration_noise = (
            np.eye(2) * self.process_noise
        )

        process_covariance = (
            noise_mapping
            @ acceleration_noise
            @ noise_mapping.T
        )

        predicted_state = state_transition @ state

        predicted_covariance = (
            state_transition
            @ covariance
            @ state_transition.T
            + process_covariance
        )

        return predicted_state, predicted_covariance

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用位置测量值修正预测状态。"""

        # 测量只有 x、y，不直接测量 vx、vy
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        predicted_measurement = (
            measurement_matrix @ predicted_state
        )

        innovation = measurement - predicted_measurement

        innovation_covariance = (
            measurement_matrix
            @ predicted_covariance
            @ measurement_matrix.T
            + measurement_covariance
        )

        kalman_gain = (
            predicted_covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        updated_state = (
            predicted_state
            + kalman_gain @ innovation
        )

        identity_matrix = np.eye(4)

        updated_covariance = (
            identity_matrix
            - kalman_gain @ measurement_matrix
        ) @ predicted_covariance

        return updated_state, updated_covariance