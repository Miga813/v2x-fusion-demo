from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models import Detection


@dataclass
class FusionConfig:
    """融合算法参数。"""

    spatial_gate_m: float = 4.0
    stale_after_s: float = 0.8
    adaptive: bool = True
    fixed_vehicle_weight: float = 0.5


def associate(
    vehicle_detections: list[Detection],
    roadside_detections: list[Detection],
    spatial_gate_m: float,
) -> tuple[
    list[tuple[int, int, float]],
    list[int],
    list[int],
]:
    """关联车端目标与路侧目标。

    返回：
        matches:
            (车端下标, 路侧下标, 空间距离)
        unmatched_vehicle:
            未匹配的车端目标下标
        unmatched_roadside:
            未匹配的路侧目标下标
    """

    if not vehicle_detections or not roadside_detections:
        return (
            [],
            list(range(len(vehicle_detections))),
            list(range(len(roadside_detections))),
        )

    vehicle_count = len(vehicle_detections)
    roadside_count = len(roadside_detections)

    # 先将所有匹配代价设置成一个很大的数
    cost_matrix = np.full(
        (vehicle_count, roadside_count),
        1e6,
        dtype=float,
    )

    for vehicle_index, vehicle_target in enumerate(
        vehicle_detections
    ):
        for roadside_index, roadside_target in enumerate(
            roadside_detections
        ):
            # 类别不同，不允许关联
            if vehicle_target.category != roadside_target.category:
                continue

            distance = np.linalg.norm(
                vehicle_target.position
                - roadside_target.position
            )

            # 只有距离小于门限的目标才能进入匹配
            if distance <= spatial_gate_m:
                cost_matrix[
                    vehicle_index,
                    roadside_index,
                ] = distance

    # 匈牙利算法寻找全局总代价最小的匹配方案
    row_indices, column_indices = linear_sum_assignment(
        cost_matrix
    )

    matches = []

    for vehicle_index, roadside_index in zip(
        row_indices,
        column_indices,
    ):
        distance = cost_matrix[
            vehicle_index,
            roadside_index,
        ]

        # 排除类别不一致或超过门限的无效匹配
        if distance < 1e5:
            matches.append(
                (
                    int(vehicle_index),
                    int(roadside_index),
                    float(distance),
                )
            )

    matched_vehicle_indices = {
        vehicle_index
        for vehicle_index, _, _ in matches
    }

    matched_roadside_indices = {
        roadside_index
        for _, roadside_index, _ in matches
    }

    unmatched_vehicle = [
        index
        for index in range(vehicle_count)
        if index not in matched_vehicle_indices
    ]

    unmatched_roadside = [
        index
        for index in range(roadside_count)
        if index not in matched_roadside_indices
    ]

    return matches, unmatched_vehicle, unmatched_roadside
def sensor_quality(
    detection: Detection,
    delay_s: float,
    loss_rate: float,
) -> float:
    """计算单个目标消息的综合质量分数。"""

    distance = float(
        np.linalg.norm(detection.position)
    )

    uncertainty = float(
        np.trace(detection.covariance)
    )

    # 距离越远，感知质量越低
    distance_factor = np.exp(
        -distance / 150.0
    )

    # 时延越大，消息质量越低
    delay_factor = np.exp(
        -2.2 * delay_s
    )

    # 丢包率越高，链路可靠性越低
    loss_factor = max(
        0.05,
        1.0 - loss_rate,
    )

    # 协方差越大，状态估计越不确定
    uncertainty_factor = (
        1.0 / (1.0 + uncertainty)
    )

    quality = (
        detection.confidence
        * distance_factor
        * delay_factor
        * loss_factor
        * uncertainty_factor
    )

    # 防止质量分数为0
    return max(1e-6, quality)


def fuse_pair(
    vehicle_detection: Detection,
    roadside_detection: Detection,
    delay_s: float,
    loss_rate: float,
    config: FusionConfig,
) -> tuple[Detection, float, float]:
    """融合一对已经完成关联的车端和路侧目标。"""

    if config.adaptive:
        vehicle_quality = sensor_quality(
            detection=vehicle_detection,
            delay_s=0.0,
            loss_rate=0.0,
        )

        roadside_quality = sensor_quality(
            detection=roadside_detection,
            delay_s=delay_s,
            loss_rate=loss_rate,
        )

        total_quality = (
            vehicle_quality + roadside_quality
        )

        vehicle_weight = (
            vehicle_quality / total_quality
        )

    else:
        vehicle_weight = (
            config.fixed_vehicle_weight
        )

    roadside_weight = 1.0 - vehicle_weight

    fused_position = (
        vehicle_weight
        * vehicle_detection.position
        + roadside_weight
        * roadside_detection.position
    )

    fused_velocity = (
        vehicle_weight
        * vehicle_detection.velocity
        + roadside_weight
        * roadside_detection.velocity
    )

    fused_covariance = (
        vehicle_weight**2
        * vehicle_detection.covariance
        + roadside_weight**2
        * roadside_detection.covariance
    )

    fused_confidence = min(
        0.99,
        vehicle_weight
        * vehicle_detection.confidence
        + roadside_weight
        * roadside_detection.confidence
        + 0.04,
    )

    fused_detection = Detection(
        # 融合轨迹沿用车端目标ID
        object_id=vehicle_detection.object_id,
        category=vehicle_detection.category,
        position=fused_position,
        velocity=fused_velocity,
        confidence=fused_confidence,
        covariance=fused_covariance,
        source="fused",
        timestamp=max(
            vehicle_detection.timestamp,
            roadside_detection.timestamp,
        ),
        metadata={
            "vehicle_source_id":
                vehicle_detection.object_id,
            "roadside_source_id":
                roadside_detection.object_id,
            "vehicle_weight":
                vehicle_weight,
            "roadside_weight":
                roadside_weight,
        },
    )

    return (
        fused_detection,
        vehicle_weight,
        roadside_weight,
    )
def calculate_rmse(
    estimated_positions: np.ndarray,
    truth_positions: np.ndarray,
) -> float:
    """计算二维位置的均方根误差。"""

    position_errors = (
        estimated_positions
        - truth_positions
    )

    squared_distances = np.sum(
        position_errors**2,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(squared_distances)
        )
    )