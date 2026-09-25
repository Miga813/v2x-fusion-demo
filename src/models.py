from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Source = Literal["vehicle", "roadside", "fused", "truth"]


@dataclass
class Detection:
    """车端、路侧和融合模块共用的目标消息结构。"""

    object_id: int
    category: str
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    covariance: np.ndarray
    source: Source
    timestamp: float
    age: float = 0.0
    metadata: dict = field(default_factory=dict)

    def predicted_to(self, target_time: float) -> "Detection":
        """将历史目标按照当前速度外推到指定时刻。"""

        dt = max(0.0, target_time - self.timestamp)

        predicted_position = self.position + self.velocity * dt

        predicted_covariance = (
            self.covariance
            + np.eye(2) * (0.06 * dt + 0.02 * dt**2)
        )

        return Detection(
            object_id=self.object_id,
            category=self.category,
            position=predicted_position,
            velocity=self.velocity.copy(),
            confidence=self.confidence,
            covariance=predicted_covariance,
            source=self.source,
            timestamp=target_time,
            age=self.age + dt,
            metadata={
                **self.metadata,
                "compensated_dt": dt,
            },
        )