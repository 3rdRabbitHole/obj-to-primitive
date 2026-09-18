"""Shape classification algorithms for Object to Primitive addon.

Public API
----------
- ``classify(obj, settings)`` — dispatch to the selected algorithm.
- ``classify_pca``, ``classify_ransac``, ``classify_hybrid`` — individual classifiers.
- ``ShapeType`` — enum of recognised shapes.
- ``ClassificationResult`` — dataclass returned by all classifiers.
"""

from .types import ShapeType, ClassificationResult
from .pipeline import classify, classify_pca, classify_ransac, classify_hybrid

__all__ = [
    'ShapeType',
    'ClassificationResult',
    'classify',
    'classify_pca',
    'classify_ransac',
    'classify_hybrid',
]
