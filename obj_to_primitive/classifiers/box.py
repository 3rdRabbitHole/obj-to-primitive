"""Box-specific classification helpers: normal-based axis finding and RANSAC fit."""

import numpy as np
from itertools import combinations

from .geometry import (
    _get_face_normals_world,
    _pca,
    _refine_axes_from_normals,
    _pick_tighter_axes,
    _compute_obb_dimensions,
)


def _find_box_axes_from_normals(obj, normals=None, areas=None):
    """
    Find 3 orthogonal face normal directions for box rotation correction.
    Returns 3x3 axes array (rows are axes) or None if 3 orthogonal
    directions cannot be found (e.g. hexagonal prisms, cylinders).
    """
    if normals is None or areas is None:
        normals, areas = _get_face_normals_world(obj)

    if normals is None or len(normals) < 6:
        return None

    # Cluster normals by direction (treating ±n as same)
    used = np.zeros(len(normals), dtype=bool)
    clusters = []

    for _ in range(20):
        if used.all():
            break
        remaining_areas = areas.copy()
        remaining_areas[used] = 0
        if remaining_areas.max() < 1e-10:
            break

        seed_idx = np.argmax(remaining_areas)
        seed_n = normals[seed_idx]

        # Cluster: |dot| > cos(30deg) ~ 0.866
        dots = np.abs(normals @ seed_n)
        cluster_mask = (dots > 0.866) & (~used)
        if not cluster_mask.any():
            used[seed_idx] = True
            continue

        cluster_normals = normals[cluster_mask]
        cluster_areas = areas[cluster_mask]

        # Area-weighted average direction (align signs to seed)
        signs = np.sign(cluster_normals @ seed_n)
        signs[signs == 0] = 1
        aligned = cluster_normals * signs[:, None]
        avg_dir = (aligned * cluster_areas[:, None]).sum(axis=0)
        norm_val = np.linalg.norm(avg_dir)
        if norm_val > 1e-10:
            avg_dir /= norm_val

        clusters.append((avg_dir, cluster_areas.sum()))
        used |= cluster_mask

    if len(clusters) < 3:
        return None

    # Sort by area, try to find 3 orthogonal directions from top candidates
    clusters.sort(key=lambda x: -x[1])
    candidates = clusters[:min(8, len(clusters))]

    best_axes = None
    best_area = 0

    for combo in combinations(range(len(candidates)), 3):
        dirs = [candidates[i][0] for i in combo]
        total_area = sum(candidates[i][1] for i in combo)

        # Orthogonality: |dot| < 0.15 for all pairs
        d01 = abs(np.dot(dirs[0], dirs[1]))
        d02 = abs(np.dot(dirs[0], dirs[2]))
        d12 = abs(np.dot(dirs[1], dirs[2]))

        if d01 < 0.15 and d02 < 0.15 and d12 < 0.15:
            if total_area > best_area:
                best_area = total_area
                best_axes = np.array(dirs)

    if best_axes is None:
        return None

    # Gram-Schmidt orthogonalization
    best_axes[0] /= np.linalg.norm(best_axes[0])
    best_axes[1] -= np.dot(best_axes[1], best_axes[0]) * best_axes[0]
    n1 = np.linalg.norm(best_axes[1])
    if n1 < 1e-10:
        return None
    best_axes[1] /= n1
    best_axes[2] = np.cross(best_axes[0], best_axes[1])
    n2 = np.linalg.norm(best_axes[2])
    if n2 < 1e-10:
        return None
    best_axes[2] /= n2

    return best_axes


def _ransac_fit_box(points, obj=None, n_iterations=50, threshold=None):
    """
    Check how well points fit an oriented bounding box.
    Returns inlier ratio (points close to box surfaces).
    """
    center, pca_axes, eigenvalues = _pca(points)
    if obj is not None:
        refined = _refine_axes_from_normals(obj, pca_axes)
        axes = _pick_tighter_axes(refined, pca_axes, points, center)
    else:
        axes = pca_axes
    dims, obb_center = _compute_obb_dimensions(points, axes, center)

    if threshold is None:
        bbox_diag = np.linalg.norm(dims)
        threshold = bbox_diag * 0.05

    # Project to OBB local space
    centered = points - obb_center
    local = centered @ axes.T

    half = dims / 2.0

    # Distance to nearest box face — vectorized
    # For each point, distance to the closest of 6 faces is
    # min over axes of | |coord| - half_extent |
    face_dists = np.min(np.abs(np.abs(local) - half), axis=1)

    inlier_ratio = np.sum(face_dists < threshold) / len(points)
    return inlier_ratio
