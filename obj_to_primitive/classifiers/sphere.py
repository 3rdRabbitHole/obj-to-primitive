"""Sphere-specific classification helpers: RANSAC sphere fitting."""

import numpy as np


def _ransac_fit_sphere(points, n_iterations=100, threshold=None):
    """
    RANSAC sphere fitting.
    Returns (center, radius, inlier_ratio) or None.
    """
    best_inlier_ratio = 0
    best_center = None
    best_radius = 0
    n = len(points)

    if threshold is None:
        bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        threshold = bbox_diag * 0.05

    for _ in range(n_iterations):
        # Sample 4 points to define a sphere
        idx = np.random.choice(n, 4, replace=False)
        sample = points[idx]

        # Solve for sphere center using 4 points
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in range(3):
            A[i] = 2 * (sample[i + 1] - sample[0])
            b[i] = np.sum(sample[i + 1] ** 2 - sample[0] ** 2)

        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue

        radius = np.linalg.norm(sample[0] - center)
        if radius < 1e-10:
            continue

        dists = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inlier_ratio = np.sum(dists < threshold) / n

        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_center = center
            best_radius = radius

    if best_center is None:
        return None
    return best_center, best_radius, best_inlier_ratio
