import numpy as np

from src.fusion import FusionConfig
from src.fusion import associate
from src.fusion import fuse_pair
from src.models import Detection
from src.simulation import run_simulation


def create_detection(
    object_id: int,
    category: str = "vehicle",
    x: float = 0.0,
    y: float = 0.0,
    source: str = "vehicle",
    confidence: float = 0.9,
    covariance_value: float = 0.1,
    timestamp: float = 1.0,
) -> Detection:
    """创建测试用目标消息。"""

    return Detection(
        object_id=object_id,
        category=category,
        position=np.array(
            [x, y],
            dtype=float,
        ),
        velocity=np.array(
            [4.0, 1.0],
            dtype=float,
        ),
        confidence=confidence,
        covariance=(
            np.eye(2) * covariance_value
        ),
        source=source,
        timestamp=timestamp,
    )


def test_time_compensation():
    """时间补偿后的位置应符合匀速运动模型。"""

    detection = create_detection(
        object_id=501,
        x=10.0,
        y=5.0,
        source="roadside",
        timestamp=1.0,
    )

    compensated = detection.predicted_to(
        target_time=1.3
    )

    expected_position = np.array(
        [11.2, 5.3]
    )

    assert np.allclose(
        compensated.position,
        expected_position,
    )

    assert np.isclose(
        compensated.timestamp,
        1.3,
    )

    assert (
        compensated.covariance[0, 0]
        > detection.covariance[0, 0]
    )


def test_association_respects_category_and_gate():
    """目标关联应同时满足类别和空间门限。"""

    vehicle_detections = [
        create_detection(
            object_id=101,
            category="vehicle",
            x=10.0,
            y=5.0,
        ),
        create_detection(
            object_id=102,
            category="pedestrian",
            x=20.0,
            y=4.0,
        ),
    ]

    roadside_detections = [
        create_detection(
            object_id=501,
            category="vehicle",
            x=10.4,
            y=5.3,
            source="roadside",
        ),
        create_detection(
            object_id=502,
            category="vehicle",
            x=20.1,
            y=4.1,
            source="roadside",
        ),
    ]

    (
        matches,
        unmatched_vehicle,
        unmatched_roadside,
    ) = associate(
        vehicle_detections=vehicle_detections,
        roadside_detections=roadside_detections,
        spatial_gate_m=4.0,
    )

    assert len(matches) == 1

    vehicle_index, roadside_index, distance = (
        matches[0]
    )

    assert vehicle_index == 0
    assert roadside_index == 0
    assert np.isclose(distance, 0.5)

    assert unmatched_vehicle == [1]
    assert unmatched_roadside == [1]


def test_degraded_link_reduces_roadside_weight():
    """高时延高丢包应降低路侧融合权重。"""

    vehicle_detection = create_detection(
        object_id=101,
        x=10.0,
        y=5.0,
        source="vehicle",
        confidence=0.90,
        covariance_value=0.10,
    )

    roadside_detection = create_detection(
        object_id=501,
        x=10.4,
        y=5.3,
        source="roadside",
        confidence=0.95,
        covariance_value=0.06,
    )

    config = FusionConfig(
        adaptive=True
    )

    (
        _,
        _,
        normal_roadside_weight,
    ) = fuse_pair(
        vehicle_detection=vehicle_detection,
        roadside_detection=roadside_detection,
        delay_s=0.0,
        loss_rate=0.0,
        config=config,
    )

    (
        _,
        _,
        degraded_roadside_weight,
    ) = fuse_pair(
        vehicle_detection=vehicle_detection,
        roadside_detection=roadside_detection,
        delay_s=0.5,
        loss_rate=0.3,
        config=config,
    )

    assert (
        degraded_roadside_weight
        < normal_roadside_weight
    )


def test_simulation_is_reproducible():
    """相同随机种子应产生完全相同的结果。"""

    first_result = run_simulation(
        delay_s=0.3,
        loss_rate=0.1,
        seed=42,
    )

    second_result = run_simulation(
        delay_s=0.3,
        loss_rate=0.1,
        seed=42,
    )

    assert np.allclose(
        first_result.frames[
            "fused_estimate_x"
        ],
        second_result.frames[
            "fused_estimate_x"
        ],
    )

    assert np.isclose(
        first_result.metrics[
            "fused_rmse"
        ],
        second_result.metrics[
            "fused_rmse"
        ],
    )


def test_communication_does_not_change_vehicle_baseline():
    """路侧通信参数不应影响单车基线。"""

    low_loss_result = run_simulation(
        delay_s=0.1,
        loss_rate=0.05,
        seed=42,
    )

    high_loss_result = run_simulation(
        delay_s=0.5,
        loss_rate=0.30,
        seed=42,
    )

    assert np.isclose(
        low_loss_result.metrics[
            "vehicle_rmse"
        ],
        high_loss_result.metrics[
            "vehicle_rmse"
        ],
    )


def test_fusion_improves_default_scenario():
    """默认场景下融合RMSE应低于单车RMSE。"""

    result = run_simulation(
        delay_s=0.3,
        loss_rate=0.1,
        occlusion=True,
        adaptive=True,
        seed=42,
    )

    assert (
        result.metrics["fused_rmse"]
        < result.metrics["vehicle_rmse"]
    )


def test_fixed_weight_is_half_on_paired_frames():
    """固定权重的双端关联帧应保持0.5。"""

    result = run_simulation(
        delay_s=0.5,
        loss_rate=0.3,
        occlusion=True,
        adaptive=False,
        seed=42,
    )

    mean_weight = result.metrics[
        "mean_paired_roadside_weight"
    ]

    assert np.isclose(
        mean_weight,
        0.5,
    )