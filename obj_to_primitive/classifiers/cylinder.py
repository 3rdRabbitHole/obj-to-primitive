"""Cylinder-specific classification helpers: normal-based axis finding and RANSAC fit."""

import numpy as np

from .geometry import _get_face_normals_world


def _find_cylinder_axis_from_normals(obj, normals=None, areas=None):
    """
    Find cylinder axis direction from face normals.
    The cylinder axis is perpendicular to side face normals.
    Uses area-weighted PCA on normals: smallest eigenvector = axis direction.
    Returns unit vector or None.
    """
    if normals is None or areas is None:
        normals, areas = _get_face_normals_world(obj)

    if normals is None or len(normals) < 6:
        return None

    # --- First pass: area-weighted PCA on ALL normals for rough axis ---
    weights = areas / areas.sum()
    weighted = normals * np.sqrt(weights[:, None])
    cov = weighted.T @ weighted

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    rough_axis = eigenvectors[:, 0].copy()

    # --- Second pass: filter out cap faces, redo PCA with side faces only ---
    dots_rough = np.abs(normals @ rough_axis)
    side_mask = dots_rough < 0.866  # faces > 30° from axis = side faces
    side_normals = normals[side_mask]
    side_areas = areas[side_mask]

    if len(side_normals) >= 3 and side_areas.sum() > 1e-10:
        sw = side_areas / side_areas.sum()
        side_weighted = side_normals * np.sqrt(sw[:, None])
        side_cov = side_weighted.T @ side_weighted
        side_evals, side_evecs = np.linalg.eigh(side_cov)
        axis = side_evecs[:, 0].copy()
    else:
        axis = rough_axis

    # Verify: most area should be perpendicular to this axis (side faces)
    dots = np.abs(normals @ axis)
    perp_area = np.sum((dots < 0.3) * areas) / areas.sum()

    if perp_area < 0.4:
        return None

    return axis


def _ransac_fit_cylinder(points, n_iterations=100, threshold=None):
    """
    RANSAC cylinder fitting.
    Returns (point_on_axis, axis_direction, radius, inlier_ratio) or None.
    """
    best_inlier_ratio = 0
    best_result = None
    n = len(points)

    if threshold is None:
        bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        threshold = bbox_diag * 0.05

    # Use PCA to get initial axis estimate
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Try main PCA axis and perturbations
    candidate_axes = [eigenvectors[:, 0]]
    for _ in range(n_iterations):
        # Pick random axis from PCA neighborhood
        ax_idx = np.random.randint(0, 3)
        axis = eigenvectors[:, ax_idx]
        noise = np.random.randn(3) * 0.1
        axis = axis + noise
        axis = axis / np.linalg.norm(axis)
        candidate_axes.append(axis)

    for axis in candidate_axes[:n_iterations]:
        # Project points onto plane perpendicular to axis
        projections = centered - np.outer(centered @ axis, axis)
        proj_dists = np.linalg.norm(projections, axis=1)

        radius = np.median(proj_dists)
        if radius < 1e-10:
            continue

        dists = np.abs(proj_dists - radius)
        inlier_ratio = np.sum(dists < threshold) / n

        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_result = (center, axis, radius, inlier_ratio)

    return best_result
