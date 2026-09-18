"""Shape types and classification result data structures."""

from enum import Enum
from dataclasses import dataclass


class ShapeType(Enum):
    BOX = 'BOX'
    CYLINDER = 'CYLINDER'
    SPHERE = 'SPHERE'
    UNKNOWN = 'UNKNOWN'


@dataclass
class ClassificationResult:
    """Result of shape classification."""
    shape_type: ShapeType
    position: tuple  # (x, y, z) center
    rotation: tuple  # (rx, ry, rz) euler angles
    dimensions: tuple  # (sx, sy, sz) scale/size
    confidence: float  # 0-1
    method: str  # which algorithm produced this
