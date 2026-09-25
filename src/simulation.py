from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.fusion import FusionConfig
from src.fusion import associate
from src.fusion import calculate_rmse
from src.fusion import fuse_pair
from src.kalman import ConstantVelocityKalman
from src.models import Detection


@dataclass
class SimulationResult:
    """保存一次完整仿真的结果。"""

    frames: pd.DataFrame
    events: pd.DataFrame
    metrics: dict


def create_measurement(
    rng: np.random.Generator,
    truth_position: np.ndarray,
    position_noise: float,
    confidence: float,
    source: str,
    timestamp: float,
    object_id: int,
) -> Detection:
    """根据真实位置生成带噪声的目标检测。"""

    position = (
        truth_position
        + rng.normal(
            loc=0.0,
            scale=position_noise,
            size=2,
        )
    )

    velocity = (
        np.array([7.2, 0.35])
        + rng.normal(
            loc=0.0,
            scale=position_noise * 0.08,
            size=2,
        )
    )

    return Detection(
        object_id=object_id,
        category="vehicle",
        position=position,
        velocity=velocity,
        confidence=confidence,
        covariance=(
            np.eye(2) * position_noise**2
        ),
        source=source,
        timestamp=timestamp,
    )


def truth_position_at(time_s: float) -> np.ndarray:
    """生成目标车辆的真实位置。"""

    x = -32.0 + 7.2 * time_s

    y = (
        -2.0
        + 0.35 * time_s
        + 0.35 * np.sin(time_s * 0.7)
    )

    return np.array([x, y])


def run_simulation(
    delay_s: float = 0.3,
    loss_rate: float = 0.1,
    occlusion: bool = True,
    adaptive: bool = True,
    seed: int = 42,
    duration_s: float = 12.0,
    dt: float = 0.1,
) -> SimulationResult:
    """运行一次车路协同感知离线仿真。"""

    seed_sequence = np.random.SeedSequence(seed)

    random_streams = seed_sequence.spawn(4)

    vehicle_dropout_rng = np.random.default_rng(
        random_streams[0]
)

    vehicle_noise_rng = np.random.default_rng(
        random_streams[1]
)

    roadside_loss_rng = np.random.default_rng(
        random_streams[2]
)

    roadside_noise_rng = np.random.default_rng(
        random_streams[3]
)

    times = np.arange(
        0.0,
        duration_s + 1e-9,
        dt,
    )

    kalman_filter = ConstantVelocityKalman(
        process_noise=0.16
    )

    fusion_config = FusionConfig(
        spatial_gate_m=4.0,
        adaptive=adaptive,
        fixed_vehicle_weight=0.5,
    )

    initial_truth = truth_position_at(0.0)

    # 单车基线和融合方案分别维护自己的状态
    vehicle_state = np.array(
        [
            initial_truth[0],
            initial_truth[1],
            7.2,
            0.35,
        ]
    )

    fused_state = vehicle_state.copy()

    vehicle_covariance = np.diag(
        [0.8, 0.8, 0.5, 0.5]
    )

    fused_covariance = vehicle_covariance.copy()

    frame_records = []
    event_records = []

    for frame_index, current_time in enumerate(times):
        truth_position = truth_position_at(
            current_time
        )

        if frame_index > 0:
            vehicle_state, vehicle_covariance = (
                kalman_filter.predict(
                    state=vehicle_state,
                    covariance=vehicle_covariance,
                    dt=dt,
                )
            )

            fused_state, fused_covariance = (
                kalman_filter.predict(
                    state=fused_state,
                    covariance=fused_covariance,
                    dt=dt,
                )
            )

        # 目标在4～7秒进入车端遮挡区域
        in_occlusion = (
            occlusion
            and 4.0 <= current_time <= 7.0
        )

        # 遮挡期间约78%的车端检测丢失
        if in_occlusion:
            vehicle_available = (
                vehicle_dropout_rng.random() > 0.78
            )
        else:
            vehicle_available = True

        # 按设置的丢包率决定路侧消息是否到达
        roadside_available = (
            roadside_loss_rng.random()
            >= loss_rate
        )

        vehicle_detection = None
        roadside_detection = None

        if vehicle_available:
            vehicle_detection = create_measurement(
                rng=vehicle_noise_rng,
                truth_position=truth_position,
                position_noise=0.14,
                confidence=0.90,
                source="vehicle",
                timestamp=current_time,
                object_id=101,
            )

            # 单车基线只使用车端检测
            vehicle_state, vehicle_covariance = (
                kalman_filter.update(
                    predicted_state=vehicle_state,
                    predicted_covariance=(
                        vehicle_covariance
                    ),
                    measurement=(
                        vehicle_detection.position
                    ),
                    measurement_covariance=(
                        vehicle_detection.covariance
                    ),
                )
            )

        # 路侧消息产生于过去时刻
        roadside_timestamp = max(
            0.0,
            current_time - delay_s,
        )

        if roadside_available:
            delayed_truth = truth_position_at(
                roadside_timestamp
            )

            roadside_detection = create_measurement(
                rng=roadside_noise_rng,
                truth_position=delayed_truth,
                position_noise=0.08,
                confidence=0.95,
                source="roadside",
                timestamp=roadside_timestamp,
                object_id=501,
            )

            # 将路侧历史消息补偿到当前时刻
            roadside_detection = (
                roadside_detection.predicted_to(
                    current_time
                )
            )

        vehicle_weight = 1.0
        roadside_weight = 0.0
        association_distance = np.nan
        fusion_measurement = None

        if (
            vehicle_detection is not None
            and roadside_detection is not None
        ):
            matches, _, _ = associate(
                vehicle_detections=[
                    vehicle_detection
                ],
                roadside_detections=[
                    roadside_detection
                ],
                spatial_gate_m=(
                    fusion_config.spatial_gate_m
                ),
            )

            if matches:
                association_distance = (
                    matches[0][2]
                )

                (
                    fusion_measurement,
                    vehicle_weight,
                    roadside_weight,
                ) = fuse_pair(
                    vehicle_detection=(
                        vehicle_detection
                    ),
                    roadside_detection=(
                        roadside_detection
                    ),
                    delay_s=delay_s,
                    loss_rate=loss_rate,
                    config=fusion_config,
                )

                event_text = (
                    "类别/空间门控 → "
                    "匈牙利关联 → "
                    + (
                        "自适应融合"
                        if adaptive
                        else "固定权重融合"
                    )
                )

        elif vehicle_detection is not None:
            fusion_measurement = (
                vehicle_detection
            )

            event_text = (
                "路侧消息丢失 → 单车更新"
            )

        elif roadside_detection is not None:
            fusion_measurement = (
                roadside_detection
            )

            vehicle_weight = 0.0
            roadside_weight = 1.0

            event_text = (
                "车端遮挡 → 路侧补盲"
            )

        else:
            event_text = (
                "车端遮挡且路侧丢包 → "
                "卡尔曼短时外推"
            )

        if fusion_measurement is not None:
            fused_state, fused_covariance = (
                kalman_filter.update(
                    predicted_state=fused_state,
                    predicted_covariance=(
                        fused_covariance
                    ),
                    measurement=(
                        fusion_measurement.position
                    ),
                    measurement_covariance=(
                        fusion_measurement.covariance
                    ),
                )
            )

        frame_records.append(
            {
                "frame": frame_index,
                "time": current_time,
                "truth_x": truth_position[0],
                "truth_y": truth_position[1],
                "vehicle_measurement_x": (
                    vehicle_detection.position[0]
                    if vehicle_detection is not None
                    else np.nan
                ),
                "vehicle_measurement_y": (
                    vehicle_detection.position[1]
                    if vehicle_detection is not None
                    else np.nan
                ),
                "roadside_measurement_x": (
                    roadside_detection.position[0]
                    if roadside_detection is not None
                    else np.nan
                ),
                "roadside_measurement_y": (
                    roadside_detection.position[1]
                    if roadside_detection is not None
                    else np.nan
                ),
                "vehicle_estimate_x":
                    vehicle_state[0],
                "vehicle_estimate_y":
                    vehicle_state[1],
                "fused_estimate_x":
                    fused_state[0],
                "fused_estimate_y":
                    fused_state[1],
                "vehicle_weight":
                    vehicle_weight,
                "roadside_weight":
                    roadside_weight,
                "association_distance":
                    association_distance,
                "occluded":
                    in_occlusion,
                "vehicle_available":
                    vehicle_available,
                "roadside_available":
                    roadside_available,
                "both_available": (
                    vehicle_detection is not None
                    and roadside_detection is not None
),
            }
        )

        event_records.append(
            {
                "frame": frame_index,
                "time": current_time,
                "event": event_text,
            }
        )

    frames = pd.DataFrame(frame_records)
    events = pd.DataFrame(event_records)

    truth_positions = frames[
        ["truth_x", "truth_y"]
    ].to_numpy()

    vehicle_positions = frames[
        [
            "vehicle_estimate_x",
            "vehicle_estimate_y",
        ]
    ].to_numpy()

    fused_positions = frames[
        [
            "fused_estimate_x",
            "fused_estimate_y",
        ]
    ].to_numpy()

    vehicle_rmse = calculate_rmse(
        vehicle_positions,
        truth_positions,
    )

    fused_rmse = calculate_rmse(
        fused_positions,
        truth_positions,
    )

    improvement_pct = (
        (vehicle_rmse - fused_rmse)
        / vehicle_rmse
        * 100.0
    )

    paired_roadside_weights = frames.loc[
        frames["both_available"],
        "roadside_weight",
    ]

    metrics = {
        "vehicle_rmse": vehicle_rmse,
        "fused_rmse": fused_rmse,
        "improvement_pct": improvement_pct,
        "vehicle_miss_rate": (
            1.0
            - frames["vehicle_available"].mean()
        ),
        "roadside_receive_rate": (
            frames["roadside_available"].mean()
        ),
        "mean_paired_roadside_weight": (
            paired_roadside_weights.mean()
            if not paired_roadside_weights.empty
            else 0.0
        ),
    }

    return SimulationResult(
        frames=frames,
        events=events,
        metrics=metrics,
    )