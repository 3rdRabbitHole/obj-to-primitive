"""
Shape classification algorithms for Object to Primitive addon.
Supports PCA, RANSAC (experimental), and Hybrid (experimental) methods.

Performance notes
-----------------
- Vertex extraction uses ``foreach_get`` + numpy batch matrix multiply
  instead of per-vertex ``matrix_world @ v.co`` loops.
- Face normals are extracted once (``_get_face_normals_world``) and shared
  across all functions that need them, avoiding repeated polygon iteration.

Design notes
------------
- ``sample_count`` (default 2000) sub-samples vertices for PCA eigenvalue
  analysis only.  Volume ratio, face normals, and OBB computation always
  use the **full** mesh so that orientation and bounding box remain exact.
  This is intentional: PCA eigenvalues converge quickly with fewer samples,
  but OBB tightness and normal clustering degrade with sampling.
- RANSAC and Hybrid classifiers are experimental.  They do **not** include
  the face-normal rotation correction that the PCA path uses, so their
  output rotations may be less accurate on symmetric shapes.  Use PCA for
  production work.
"""

import numpy as np
from mathutils import Vector, Matrix, Quaternion
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from itertools import permutations as _perms


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


def _get_vertices_world(obj, sample_count=0):
    """Extract world-space vertex positions, optionally subsampled.

    Uses ``foreach_get`` for bulk extraction and numpy matrix multiply
    for the world-space transform — significantly faster than per-vertex
    ``matrix_world @ v.co`` on meshes with >100 vertices.
    """
    mesh = obj.data
    n = len(mesh.vertices)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Bulk extract local coords via foreach_get
    coords = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    local = coords.reshape(n, 3)

    # Batch world transform: v_world = v_local @ M^T + t
    mat = np.array(obj.matrix_world, dtype=np.float64)  # 4x4
    rot_scale = mat[:3, :3].T  # transpose for row-vector multiply
    translation = mat[:3, 3]
    verts = local @ rot_scale + translation

    if sample_count > 0 and len(verts) > sample_count:
        indices = np.random.choice(len(verts), sample_count, replace=False)
        verts = verts[indices]

    return verts


def _get_face_normals_world(obj):
    """Extract world-space face normals and areas for all polygons.

    Returns ``(normals, areas)`` where normals is (N, 3) unit vectors
    and areas is (N,) polygon areas, or ``(None, None)`` when the mesh
    has no polygons.  Both arrays exclude degenerate zero-area faces.

    Uses ``foreach_get`` for bulk extraction.
    """
    mesh = obj.data
    n_poly = len(mesh.polygons)
    if n_poly == 0:
        return None, None

    # Bulk extract local normals and areas
    raw_normals = np.empty(n_poly * 3, dtype=np.float64)
    mesh.polygons.foreach_get("normal", raw_normals)
    local_normals = raw_normals.reshape(n_poly, 3)

    raw_areas = np.empty(n_poly, dtype=np.float64)
    mesh.polygons.foreach_get("area", raw_areas)

    # Transform normals to world space: n_world = (M^-T) @ n_local
    mat3 = obj.matrix_world.to_3x3()
    normal_mat = np.array(mat3.inverted_safe().transposed(), dtype=np.float64)
    world_normals = local_normals @ normal_mat.T  # (N,3) @ (3,3)

    # Normalize
    norms = np.linalg.norm(world_normals, axis=1, keepdims=True)
    valid = (norms.ravel() > 1e-10)
    if not valid.any():
        return None, None

    world_normals = world_normals[valid]
    norms = norms[valid]
    world_normals /= norms
    areas = raw_areas[valid]

    return world_normals, areas


def _pca(vertices):
    """
    Perform PCA on vertices.
    Returns: center, axes (3x3 matrix, rows are principal axes), eigenvalues (sorted descending)
    """
    center = np.mean(vertices, axis=0)
    centered = vertices - center

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return center, eigenvectors.T, eigenvalues


def _cross_section_iso_ratio(vertices_2d):
    """
    Compute the isoperimetric ratio (4*pi*area / perimeter^2) of the
    convex hull of a 2D point cloud.  Returns 1.0 for a perfect circle,
    pi/4 ≈ 0.785 for a square, and intermediate values for polygons.

    Uses a manual Graham scan so no scipy dependency is needed.
    """
    pts = vertices_2d
    if len(pts) < 3:
        return 1.0

    # Graham scan — convex hull
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(pts.tolist(), key=lambda p: (p[0], p[1]))
    lower = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])

    if len(hull) < 3:
        return 1.0

    # Perimeter
    edges = np.diff(np.vstack([hull, hull[0:1]]), axis=0)
    perimeter = np.sum(np.linalg.norm(edges, axis=1))
    if perimeter < 1e-10:
        return 1.0

    # Area (shoelace)
    area = 0.5 * abs(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1)
                             - np.roll(hull[:, 0], -1) * hull[:, 1]))

    return 4.0 * np.pi * area / (perimeter ** 2)


def _compute_circularity(vertices_2d):
    """
    Compute circularity of 2D point cloud.
    Returns value 0-1 where 1 = perfect circle.

    Two-metric approach:
    1. Adaptive angular bin method — bin count adapts to vertex count so
       low-poly meshes (8-segment cylinders) are not penalized for sparse
       angular coverage.
    2. Min/max radius ratio of boundary points — simple, robust fallback
       that works regardless of angular distribution.

    Final score is the maximum of both metrics (either one detecting
    circularity is sufficient).
    """
    center = np.mean(vertices_2d, axis=0)
    dists = np.linalg.norm(vertices_2d - center, axis=1)

    if dists.max() < 1e-10:
        return 1.0

    angles = np.arctan2(vertices_2d[:, 1] - center[1], vertices_2d[:, 0] - center[0])

    # --- Metric 1: Adaptive angular-bin CV ---
    # Determine how many distinct angular positions exist
    n_unique = len(np.unique(np.round(angles, 2)))
    # Bin count adapts: at least 6, at most 36, roughly matching unique positions
    n_bins = max(6, min(36, n_unique))

    bin_max_dists = []
    for i in range(n_bins):
        a_min = -np.pi + i * 2 * np.pi / n_bins
        a_max = a_min + 2 * np.pi / n_bins
        mask = (angles >= a_min) & (angles < a_max)
        if mask.any():
            bin_max_dists.append(np.max(dists[mask]))

    circ_cv = 0.0
    if len(bin_max_dists) >= 4:
        bin_max_dists = np.array(bin_max_dists)
        mean_d = np.mean(bin_max_dists)
        if mean_d > 1e-10:
            cv = np.std(bin_max_dists) / mean_d
            circ_cv = max(0.0, 1.0 - cv)

            # Mild coverage penalty only when coverage is very low (<50%)
            # This catches genuinely partial shapes (side-view projections)
            # without penalizing low-poly circles
            coverage = len(bin_max_dists) / n_bins
            if coverage < 0.5:
                circ_cv *= (coverage / 0.5)

    # --- Metric 2: Boundary radius ratio ---
    # Use points in the outer 50% of distances (boundary extraction)
    # This avoids interior points (cap fills) distorting the metric
    d_threshold = dists.max() * 0.5
    boundary_mask = dists >= d_threshold
    if boundary_mask.sum() >= 3:
        boundary_dists = dists[boundary_mask]
        r_min = np.min(boundary_dists)
        r_max = np.max(boundary_dists)
        if r_max > 1e-10:
            circ_ratio = r_min / r_max
        else:
            circ_ratio = 0.0

        # Also check angular spread of boundary points
        boundary_angles = angles[boundary_mask]
        n_unique_boundary = len(np.unique(np.round(boundary_angles, 1)))
        # Need at least 4 distinct angular positions for a circle claim
        if n_unique_boundary < 4:
            circ_ratio *= 0.5
    else:
        circ_ratio = 0.0

    return max(circ_cv, circ_ratio)


def _compute_obb_dimensions(vertices, axes, center):
    """Compute oriented bounding box dimensions along given axes."""
    centered = vertices - center
    projected = centered @ axes.T  # project onto principal axes

    mins = np.min(projected, axis=0)
    maxs = np.max(projected, axis=0)
    dimensions = maxs - mins

    # Adjust center to OBB center
    obb_center_local = (mins + maxs) / 2.0
    obb_center_world = center + obb_center_local @ axes

    return dimensions, obb_center_world


def _pick_tighter_axes(candidate, fallback, verts, center):
    """Return whichever axes produce a smaller OBB (by volume).

    Used as a guard when face-normal-based refinement may produce
    worse axes than PCA (e.g. beveled/rounded edges).
    """
    c_dims, _ = _compute_obb_dimensions(verts, candidate, center)
    f_dims, _ = _compute_obb_dimensions(verts, fallback, center)
    c_vol = np.prod(c_dims)
    f_vol = np.prod(f_dims)
    # Relative tolerance: treat volumes within 1e-6 as equal (prefer candidate)
    if f_vol > 1e-20:
        return candidate if c_vol <= f_vol * (1 + 1e-6) else fallback
    return candidate if c_vol <= f_vol else fallback


def _refine_axes_from_normals(obj, pca_axes, normals=None, areas=None):
    """
    Refine PCA axes using mesh face normals.
    For each PCA axis, find the face normal most aligned with it.
    This fixes PCA instability on symmetric shapes (cubes, etc).

    If *normals* and *areas* are provided they are used directly
    (avoids re-extracting them).  Otherwise they are computed from *obj*.
    """
    if normals is None or areas is None:
        normals, areas = _get_face_normals_world(obj)

    if normals is None or len(normals) == 0:
        return pca_axes

    # For each PCA axis, find the normal most aligned with it
    # Weight by face area so large faces (main box faces) dominate over small bevel faces
    # After picking a normal for axis[i], exclude near-parallel normals from axis[i+1]
    # to prevent two axes from snapping to the same face normal direction.
    refined = np.zeros_like(pca_axes)
    excluded_mask = np.zeros(len(normals), dtype=bool)

    for i in range(3):
        pca_ax = pca_axes[i]
        # Area-weighted alignment score, with exclusion
        dots = np.abs(normals @ pca_ax) * areas
        dots[excluded_mask] = -1.0  # exclude already-used directions
        best_idx = np.argmax(dots)

        raw_dot = abs(np.dot(normals[best_idx], pca_ax))
        if raw_dot > 0.3:
            n = normals[best_idx].copy()
            if np.dot(n, pca_ax) < 0:
                n = -n
            refined[i] = n
            # Exclude normals near-parallel to the chosen one (same face direction)
            parallel = np.abs(normals @ normals[best_idx]) > 0.5
            excluded_mask |= parallel
        else:
            refined[i] = pca_ax

    # Orthogonalize using Gram-Schmidt
    refined[0] = refined[0] / np.linalg.norm(refined[0])
    refined[1] = refined[1] - np.dot(refined[1], refined[0]) * refined[0]
    norm1 = np.linalg.norm(refined[1])
    if norm1 > 1e-10:
        refined[1] = refined[1] / norm1
    else:
        refined[1] = pca_axes[1]

    refined[2] = np.cross(refined[0], refined[1])
    norm2 = np.linalg.norm(refined[2])
    if norm2 > 1e-10:
        refined[2] = refined[2] / norm2
    else:
        refined[2] = pca_axes[2]

    return refined


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

    from itertools import combinations  # stdlib — fast, kept local for clarity
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


def _compute_mesh_volume(obj):
    """
    Compute mesh volume using the convex hull of the mesh vertices.
    This is robust to inconsistent face normals (common in CAD imports)
    because it depends only on vertex positions, not on face winding.

    Uses Blender's bmesh convex_hull operator + recalc_face_normals
    to guarantee a watertight hull with consistent outward normals,
    then bmesh.calc_volume() for the actual volume.

    For convex shapes the result equals the true mesh volume.
    For concave shapes it overestimates (fills in concavities), which
    is intentional: the volume ratio is used to filter genuinely complex
    shapes, and a higher ratio means fewer false UNKNOWN results.
    """
    import bmesh
    from mathutils import Vector as _Vec

    mesh = obj.data
    n = len(mesh.vertices)
    # Bulk extract and transform (reuse the same approach as _get_vertices_world)
    coords = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    local = coords.reshape(n, 3)
    mat = np.array(obj.matrix_world, dtype=np.float64)
    world_verts = local @ mat[:3, :3].T + mat[:3, 3]

    bm = bmesh.new()
    for i in range(n):
        bm.verts.new(_Vec(world_verts[i]))
    bm.verts.ensure_lookup_table()

    bmesh.ops.convex_hull(bm, input=bm.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    vol = bm.calc_volume()
    bm.free()

    return abs(vol)


def _compute_volume_ratio(obj, axes, center, verts):
    """
    Compute mesh_volume / obb_volume ratio.
    Simple shapes (box, cylinder, sphere) have predictable ratios >= ~0.45.
    Complex shapes (Suzanne, gears, etc.) have much lower ratios.
    """
    mesh_vol = _compute_mesh_volume(obj)

    # OBB volume along refined axes
    centered = verts - center
    projected = centered @ axes.T
    dims = np.max(projected, axis=0) - np.min(projected, axis=0)
    obb_vol = float(np.prod(dims))

    if obb_vol < 1e-10:
        return 1.0

    return mesh_vol / obb_vol


def _check_sphere_surface(verts):
    """
    Check if vertices lie on a sphere surface.
    Returns (is_sphere, inlier_ratio) where inlier_ratio is the fraction
    of vertices within 5% of the fitted sphere radius.

    Guards against false positives:
    - Bounding box must be roughly cubic (aspect ratio > 0.6) — cylinders
      with height ≈ diameter fail this when their OBB is elongated.
    - Directional uniformity: split residuals by octant; a cylinder has
      low residuals in cross-section but high along its axis.
    """
    center = np.mean(verts, axis=0)
    dists = np.linalg.norm(verts - center, axis=1)
    radius = np.median(dists)  # median is robust to polar clustering

    if radius < 1e-10:
        return False, 0.0

    # Bbox aspect ratio guard — sphere bbox is roughly cubic
    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    bbox_dims = bbox_max - bbox_min
    bbox_dims_sorted = np.sort(bbox_dims)
    if bbox_dims_sorted[2] > 1e-10:
        bbox_aspect = bbox_dims_sorted[0] / bbox_dims_sorted[2]
        if bbox_aspect < 0.75:
            # Too elongated for a sphere
            return False, 0.0

    threshold = radius * 0.05
    inlier_ratio = np.sum(np.abs(dists - radius) < threshold) / len(verts)
    return inlier_ratio > 0.8, float(inlier_ratio)


def _face_normal_spread(obj, normals=None, areas=None):
    """
    Measure how spread out face normals are.
    Returns a value 0-1:
      ~1.0 = normals point in many different directions (sphere-like)
      ~0.0 = normals are clustered in few directions (box-like)
    Uses the area-weighted mean resultant length of normals.
    A sphere has normals pointing uniformly outward → low resultant.
    A box has 6 clusters of parallel normals → high resultant per cluster.

    Method: count how many dominant normal directions exist.
    Sphere: many directions, no dominant cluster.
    Box: ~6 dominant clusters (or 3 pairs of opposite faces).
    """
    if normals is None or areas is None:
        normals, areas = _get_face_normals_world(obj)

    if normals is None or len(normals) < 4:
        return 0.5

    total_area = areas.sum()
    if total_area < 1e-10:
        return 0.5

    # Cluster normals: count dominant directions using greedy clustering
    # (group normals within 30° of each other)
    weights = areas / total_area
    used = np.zeros(len(normals), dtype=bool)
    n_clusters = 0
    cluster_weights = []

    for _ in range(20):  # max 20 clusters
        if used.all():
            break
        # Find the unused normal with largest total area-weight
        remaining_weights = weights.copy()
        remaining_weights[used] = 0
        if remaining_weights.max() < 1e-10:
            break
        seed_idx = np.argmax(remaining_weights)
        seed_n = normals[seed_idx]

        # Cluster: normals within ~45° (|dot| > cos45 ≈ 0.707)
        # Use |dot| to treat opposite normals as same direction
        # 45° is wide enough that bevel faces cluster with their parent
        # face direction, preventing beveled boxes from fragmenting into
        # many clusters and being mistaken for spheres.
        dots = np.abs(normals @ seed_n)
        cluster_mask = (dots > 0.707) & (~used)
        cluster_weight = weights[cluster_mask].sum()
        cluster_weights.append(cluster_weight)
        used |= cluster_mask
        n_clusters += 1

    # Sphere: many small clusters (>8). Box: ~3-6 clusters.
    # Normalized: clamp to [0,1] range
    # 3 clusters → very boxy (0.0), 12+ clusters → very spherical (1.0)
    if n_clusters <= 3:
        return 0.0
    elif n_clusters >= 12:
        return 1.0
    else:
        return (n_clusters - 3) / 9.0


def _axes_to_euler(axes):
    """Convert 3x3 axis matrix to Euler angles."""
    # Build rotation matrix ensuring right-handedness
    x_axis = Vector(axes[0]).normalized()
    y_axis = Vector(axes[1]).normalized()
    z_axis = x_axis.cross(y_axis).normalized()
    y_axis = z_axis.cross(x_axis).normalized()

    rot_matrix = Matrix((
        (*x_axis, 0),
        (*y_axis, 0),
        (*z_axis, 0),
        (0, 0, 0, 1),
    )).transposed()

    return rot_matrix.to_euler()


def _pick_best_axes(axes_refined, dims_refined, pca_axes_raw,
                    verts, center, ref_matrix):
    """
    Try both refined and raw PCA axes through _minimize_rotation,
    return whichever produces rotation closest to ref_matrix.

    Refined axes help low-vertex shapes (greedy snap to face normals).
    Raw PCA axes are better when refinement is misled (e.g. bevel normals
    tilting the axis away from the true orientation).
    """
    ref_quat = ref_matrix.to_quaternion()

    # Refined path
    axes_r, dims_r = _minimize_rotation(axes_refined, dims_refined, ref_matrix)
    # Recompute dims for final axes (in-plane rotation may have changed them)
    dims_r, _ = _compute_obb_dimensions(verts, axes_r, center)
    euler_r = _axes_to_euler(axes_r)
    angle_r = ref_quat.rotation_difference(euler_r.to_quaternion()).angle

    # Raw PCA path
    dims_raw, _ = _compute_obb_dimensions(verts, pca_axes_raw, center)
    axes_p, dims_p = _minimize_rotation(pca_axes_raw, dims_raw, ref_matrix)
    # Recompute dims for final axes (in-plane rotation may have changed them)
    dims_p, _ = _compute_obb_dimensions(verts, axes_p, center)
    euler_p = _axes_to_euler(axes_p)
    angle_p = ref_quat.rotation_difference(euler_p.to_quaternion()).angle

    if angle_p < angle_r:
        return axes_p, dims_p
    return axes_r, dims_r


def _minimize_rotation(axes, dims, ref_matrix=None):
    """
    Choose the sign combination and degenerate-axis permutation that
    produces the OBB rotation closest to a reference orientation.

    PCA axes have two kinds of ambiguity on symmetric shapes:
      - Sign: any axis can be flipped 180° (must flip in pairs to
        keep right-handedness).
      - Permutation: when two OBB extents are nearly equal, the
        corresponding axes can be swapped.

    Parameters
    ----------
    axes : ndarray (3,3) — OBB axes
    dims : ndarray (3,)  — OBB extents
    ref_matrix : Matrix (3x3) or None
        The object's world rotation matrix.  When provided the function
        picks the combination closest to this orientation (measured as
        quaternion angle).  When None, minimises from identity.
    """

    # Right-handed sign combos: flip any two axes at once
    SIGN_COMBOS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]

    if ref_matrix is not None:
        ref_quat = ref_matrix.to_quaternion()
    else:
        ref_quat = Quaternion()  # identity

    # Permutations to try: identity + swaps of near-equal dim pairs
    # + cyclic permutations when all three dims are near-equal.
    # Two tiers:
    #   - "free" swaps (ratio > 0.95): always tried
    #   - "conditional" swaps (0.85 < ratio <= 0.95): only accepted
    #     when they reduce the rotation angle by > 30°
    FREE_THRESH = 0.95
    COND_THRESH = 0.85
    COND_MIN_IMPROVEMENT = 0.524  # ~30° in radians

    free_perms = [(0, 1, 2)]
    cond_perms = []
    near_equal_pairs = set()
    for i in range(3):
        for j in range(i + 1, 3):
            d_max = max(abs(dims[i]), abs(dims[j]))
            if d_max < 1e-10:
                continue
            ratio = min(abs(dims[i]), abs(dims[j])) / d_max
            if ratio > FREE_THRESH:
                near_equal_pairs.add((i, j))
                p = [0, 1, 2]
                p[i], p[j] = p[j], p[i]
                free_perms.append(tuple(p))
            elif ratio > COND_THRESH:
                p = [0, 1, 2]
                p[i], p[j] = p[j], p[i]
                cond_perms.append(tuple(p))
    # If all three pairs are near-equal, add cyclic permutations
    if len(near_equal_pairs) == 3:
        free_perms.append((1, 2, 0))
        free_perms.append((2, 0, 1))

    # --- Pass 1: free permutations (always accepted) ---
    best_angle = float('inf')
    best_axes = axes.copy()
    best_dims = dims.copy()

    for perm in free_perms:
        pa = axes[list(perm)]
        pd = dims[list(perm)]
        for signs in SIGN_COMBOS:
            sa = pa.copy()
            for k in range(3):
                sa[k] *= signs[k]
            euler = _axes_to_euler(sa)
            cand_quat = euler.to_quaternion()
            angle = ref_quat.rotation_difference(cand_quat).angle
            if angle < best_angle:
                best_angle = angle
                best_axes = sa.copy()
                best_dims = pd.copy()

    # --- Pass 2: conditional permutations (only if large improvement) ---
    for perm in cond_perms:
        pa = axes[list(perm)]
        pd = dims[list(perm)]
        for signs in SIGN_COMBOS:
            sa = pa.copy()
            for k in range(3):
                sa[k] *= signs[k]
            euler = _axes_to_euler(sa)
            cand_quat = euler.to_quaternion()
            angle = ref_quat.rotation_difference(cand_quat).angle
            if angle < best_angle and (best_angle - angle) > COND_MIN_IMPROVEMENT:
                best_angle = angle
                best_axes = sa.copy()
                best_dims = pd.copy()

    # --- Pass 3: in-plane rotation for degenerate axis pairs ---
    # When two dims are nearly equal, the corresponding PCA axes span
    # a 2D plane in which any rotation is equally valid.  Find the
    # in-plane angle that best aligns with the reference orientation.
    if ref_matrix is not None and near_equal_pairs:
        ref_np = np.array(ref_matrix)  # 3x3, columns are ref axes
        for (i, j) in near_equal_pairs:
            a_i = best_axes[i]
            a_j = best_axes[j]
            # Project ref axis i onto the plane spanned by a_i, a_j
            ref_axis = ref_np[:, i]  # column i of ref rotation
            ci = np.dot(ref_axis, a_i)
            cj = np.dot(ref_axis, a_j)
            if abs(ci) < 1e-10 and abs(cj) < 1e-10:
                continue
            theta = np.arctan2(cj, ci)
            # Rotate a_i and a_j in-plane by theta
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            new_i = cos_t * a_i + sin_t * a_j
            new_j = -sin_t * a_i + cos_t * a_j
            # Try all sign combos with the rotated axes
            for signs in SIGN_COMBOS:
                sa = best_axes.copy()
                sa[i] = new_i * signs[i]
                sa[j] = new_j * signs[j]
                # Fix third axis to maintain right-handedness
                k = 3 - i - j
                sa[k] = np.cross(sa[i], sa[j])
                n = np.linalg.norm(sa[k])
                if n < 1e-10:
                    continue
                sa[k] = sa[k] / n * signs[k]
                euler = _axes_to_euler(sa)
                cand_quat = euler.to_quaternion()
                angle = ref_quat.rotation_difference(cand_quat).angle
                if angle < best_angle:
                    best_angle = angle
                    best_axes = sa.copy()
                    # dims unchanged: degenerate pair has same extents

    return best_axes, best_dims


# ============================================================
# PCA Classifier
# ============================================================

def classify_pca(obj, settings):
    """
    PCA-based shape classification.
    1. PCA to get principal axes and eigenvalues
    2. Eigenvalue ratios to narrow candidates
    3. Circularity test on cross-sections to distinguish round vs rectangular
    """
    # Full vertex set for OBB / normals; sub-sampled set for PCA only.
    verts_full = _get_vertices_world(obj, sample_count=0)
    if settings.sample_count > 0 and len(verts_full) > settings.sample_count:
        idx = np.random.choice(len(verts_full), settings.sample_count, replace=False)
        verts_pca = verts_full[idx]
    else:
        verts_pca = verts_full

    if len(verts_full) < 4:
        return ClassificationResult(
            shape_type=ShapeType.UNKNOWN,
            position=tuple(obj.location),
            rotation=(0, 0, 0),
            dimensions=(1, 1, 1),
            confidence=0.0,
            method='PCA',
        )

    # Extract face normals once — shared by all downstream functions
    face_normals, face_areas = _get_face_normals_world(obj)

    center, pca_axes, eigenvalues = _pca(verts_pca)
    pca_axes_raw = pca_axes.copy()  # save before refinement

    # Use full verts for OBB and all spatial queries
    verts = verts_full

    # Refine axes from face normals — PCA axes can be diagonal
    # on low-vertex shapes (e.g. 8-vertex cubes), causing OBB misalignment.
    # Guard: keep refinement only when it produces a tighter OBB.
    refined_axes = _refine_axes_from_normals(obj, pca_axes,
                                              normals=face_normals, areas=face_areas)
    axes = _pick_tighter_axes(refined_axes, pca_axes, verts, center)

    # Volume ratio check for complex/organic shapes
    vol_ratio = _compute_volume_ratio(obj, axes, center, verts)
    if vol_ratio < settings.complexity_threshold:
        dims, obb_center = _compute_obb_dimensions(verts, axes, center)
        ref_rot = obj.matrix_world.to_3x3().normalized()
        axes, dims = _pick_best_axes(
            axes, dims, pca_axes_raw, verts, center, ref_rot)
        euler = _axes_to_euler(axes)
        return ClassificationResult(
            shape_type=ShapeType.UNKNOWN,
            position=tuple(obb_center),
            rotation=tuple(euler),
            dimensions=tuple(dims),
            confidence=float(round(vol_ratio, 3)),
            method='PCA',
        )

    # Prevent division by zero
    ev = eigenvalues.copy()
    ev[ev < 1e-10] = 1e-10
    ev_max = ev[0]

    # Inter-eigenvalue ratios (key insight: compare axes to EACH OTHER)
    ratio_01 = ev[1] / ev[0]  # how similar axis 1 is to axis 0
    ratio_12 = ev[2] / ev[1]  # how similar axis 2 is to axis 1
    ratio_02 = ev[2] / ev[0]  # how similar axis 2 is to axis 0

    # OBB dimensions using refined axes
    dims, obb_center = _compute_obb_dimensions(verts, axes, center)
    euler = _axes_to_euler(axes)

    # Project onto plane perpendicular to each axis and test circularity
    centered = verts - center
    projected_on_axes = centered @ axes.T

    circ_thresh = settings.circularity_threshold
    sphere_thresh = settings.sphere_ratio_threshold
    cyl_thresh = settings.cylinder_ratio_threshold

    shape = ShapeType.UNKNOWN
    confidence = 0.0
    cylinder_axis_idx = None  # PCA axis index of cylinder symmetry axis

    # Compute all three cross-section circularities upfront
    cross_section_0 = projected_on_axes[:, 1:3]  # perp to axis 0
    circ_0 = _compute_circularity(cross_section_0)
    cross_section_1 = projected_on_axes[:, [0, 2]]  # perp to axis 1
    circ_1 = _compute_circularity(cross_section_1)
    cross_section_2 = projected_on_axes[:, 0:2]  # perp to axis 2
    circ_2 = _compute_circularity(cross_section_2)

    # Sphere needs enough vertices for reliable circularity (< 20 verts → skip sphere)
    enough_for_sphere = len(verts) >= 10
    # Cylinder also needs enough verts — very low poly (< 16) gives
    # false circularity (e.g. 8-vert cube projects to a square whose
    # min/max radius ratio is 1.0)
    enough_for_cylinder = len(verts) >= 16

    # Sphere surface check — independent of PCA eigenvalue bias.
    # Used as a fallback when eigenvalue ratios are ambiguous.
    is_sphere_surface, sphere_inlier = _check_sphere_surface(verts)

    # Face normal spread — distinguishes sphere (many directions)
    # from rounded box (few dominant clusters)
    normal_spread = _face_normal_spread(obj, normals=face_normals, areas=face_areas)

    # Classification logic using inter-eigenvalue ratios
    # Sphere surface check is used ONLY as confirmation after circularity
    # has already suggested sphere, or as a last-resort fallback for UV spheres
    # with biased eigenvalues. It never overrides a circularity-based result.

    if ratio_01 >= sphere_thresh and ratio_02 >= sphere_thresh:
        # All three eigenvalues similar -> sphere or cube
        # Circularity alone is not enough — beveled boxes also score high.
        # Require sphere surface confirmation (inlier > 0.9) to avoid
        # false positives on heavily rounded boxes.
        if is_sphere_surface and sphere_inlier > 0.9 and vol_ratio < 0.65:
            shape = ShapeType.SPHERE
            confidence = sphere_inlier
        else:
            shape = ShapeType.BOX
            confidence = 0.7

    elif ratio_12 >= cyl_thresh and ratio_01 < sphere_thresh:
        # Axes 1,2 similar, different from axis 0 -> cylinder along axis 0
        # Do NOT attempt sphere override here — elongated eigenvalues
        # mean the shape is NOT a sphere regardless of circularity.
        if circ_0 > circ_thresh and enough_for_cylinder:
            # Guard: convex hull isoperimetric ratio of cross-section.
            # Beveled boxes pass circularity but have low iso_ratio.
            iso = _cross_section_iso_ratio(cross_section_0)
            if iso >= 0.93:
                shape = ShapeType.CYLINDER
                confidence = circ_0
                cylinder_axis_idx = 0
            else:
                shape = ShapeType.BOX
                confidence = 0.7
        else:
            shape = ShapeType.BOX
            confidence = 0.7

    elif ratio_01 >= cyl_thresh and ratio_12 < sphere_thresh:
        # Axes 0,1 similar, axis 2 different -> cylinder along axis 2
        if circ_2 > circ_thresh and enough_for_cylinder:
            iso = _cross_section_iso_ratio(cross_section_2)
            if iso >= 0.93:
                shape = ShapeType.CYLINDER
                confidence = circ_2
                cylinder_axis_idx = 2
            else:
                shape = ShapeType.BOX
                confidence = 0.7
        else:
            shape = ShapeType.BOX
            confidence = 0.7
    else:
        # All different or no clear pattern
        shape = ShapeType.BOX
        confidence = 0.7

    # Sphere fallback: PCA eigenvalues can be biased by non-uniform vertex
    # density (UV sphere poles vs equator), making eigenvalue ratios look
    # cylindrical or box-like. Override BOX or CYLINDER when evidence is
    # very strong: high sphere surface inlier AND nearly-cubic bounding box.
    if shape in (ShapeType.BOX, ShapeType.CYLINDER) and enough_for_sphere:
        if is_sphere_surface and sphere_inlier > 0.95 and vol_ratio < 0.65:
            # Compute bbox aspect for strict check
            bbox_min_fb = np.min(verts, axis=0)
            bbox_max_fb = np.max(verts, axis=0)
            bbox_dims_fb = np.sort(bbox_max_fb - bbox_min_fb)
            if bbox_dims_fb[2] > 1e-10 and bbox_dims_fb[0] / bbox_dims_fb[2] > 0.9:
                shape = ShapeType.SPHERE
                confidence = sphere_inlier

    # Cylinder fallback: when eigenvalue ratios led to BOX but
    # cross-section shape is clearly circular (high iso_ratio).
    # Catches hollow cylinders and non-uniform vertex distributions
    # where eigenvalue-based circularity is misleading.
    if shape == ShapeType.BOX and enough_for_cylinder:
        for ax_idx, cs in [(0, cross_section_0), (1, cross_section_1),
                           (2, cross_section_2)]:
            iso = _cross_section_iso_ratio(cs)
            if iso >= 0.95:
                shape = ShapeType.CYLINDER
                confidence = iso
                cylinder_axis_idx = ax_idx
                break

    # Low confidence -> unknown
    if confidence < 0.4:
        shape = ShapeType.UNKNOWN

    # Reorder axes so cylinder symmetry axis maps to Z (primitive depth axis).
    if shape == ShapeType.CYLINDER and cylinder_axis_idx is not None:
        if cylinder_axis_idx == 0:
            axes = axes[[1, 2, 0]]
            dims = dims[[1, 2, 0]]
        elif cylinder_axis_idx == 1:
            axes = axes[[0, 2, 1]]
            dims = dims[[0, 2, 1]]
        # cylinder_axis_idx == 2: already in correct position

    # --- Rotation correction using face normals ---
    # .normalized() strips non-uniform scale so ref_rot is a pure rotation.
    ref_rot = obj.matrix_world.to_3x3().normalized()

    if shape == ShapeType.BOX:
        # Face-normal-based axes: find 3 orthogonal normal pairs.
        # More accurate than PCA when chamfers/bevels bias vertex
        # distribution. Falls back to PCA when normals don't form
        # 3 orthogonal pairs (e.g. hexagonal prisms).
        box_axes = _find_box_axes_from_normals(obj, normals=face_normals, areas=face_areas)
        # Guard: reject normal-based axes when they produce a larger OBB
        # (beveled/rounded edges can mislead normal clustering).
        if box_axes is not None:
            if _pick_tighter_axes(box_axes, axes, verts, center) is not box_axes:
                box_axes = None
        if box_axes is not None:
            axes = box_axes
            dims, obb_center = _compute_obb_dimensions(verts, axes, center)
            # Full permutation + sign correction: box normal axes give
            # correct directions but in arbitrary order — permutation
            # re-orders them to match the object's local frame.
            if ref_rot is not None:
                ref_quat = ref_rot.to_quaternion()
            else:
                ref_quat = Quaternion()
            SIGN_COMBOS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
            best_angle = float('inf')
            best_axes = axes.copy()
            for perm in _perms(range(3)):
                for signs in SIGN_COMBOS:
                    sa = np.zeros_like(axes)
                    for k in range(3):
                        sa[k] = axes[perm[k]] * signs[k]
                    if np.linalg.det(sa) < 0:
                        sa[2] = -sa[2]
                    euler = _axes_to_euler(sa)
                    cand_quat = euler.to_quaternion()
                    angle = ref_quat.rotation_difference(cand_quat).angle
                    if angle < best_angle:
                        best_angle = angle
                        best_axes = sa.copy()
            axes = best_axes
            dims, obb_center = _compute_obb_dimensions(verts, axes, center)
        else:
            # Fallback: compare refined vs raw PCA axes
            pca_raw_for_min = pca_axes_raw.copy()
            axes, dims = _pick_best_axes(
                axes, dims, pca_raw_for_min, verts, center, ref_rot)

    elif shape == ShapeType.CYLINDER:
        # Face-normal-based axis direction for cylinder.
        # More accurate when vertex distribution is non-uniform.
        cyl_axis_dir = _find_cylinder_axis_from_normals(obj, normals=face_normals, areas=face_areas)
        if cyl_axis_dir is not None:
            ax2_candidate = cyl_axis_dir / np.linalg.norm(cyl_axis_dir)
            # Consistency check: normal-based axis must agree with
            # cross-section-based axis (axes[2] after reordering).
            # If they diverge (>45°), discard the normal result.
            agreement = abs(np.dot(ax2_candidate, axes[2]))
            if agreement >= 0.707:  # cos(45°)
                ax2 = ax2_candidate
                # Build cross-section axes from ref_rot, not from
                # PCA/refined axes which can be tilted by polygon
                # face normals.  The cross-section is circular, so
                # any in-plane orientation is geometrically equivalent;
                # aligning with the object's own rotation is cleanest.
                ref_np = np.array(ref_rot) if ref_rot is not None \
                    else np.eye(3)
                ax0 = ref_np[:, 0] - np.dot(ref_np[:, 0], ax2) * ax2
                n0 = np.linalg.norm(ax0)
                if n0 > 1e-10:
                    ax0 /= n0
                else:
                    # ref X is parallel to cylinder axis; use ref Y
                    ax0 = ref_np[:, 1] - np.dot(ref_np[:, 1], ax2) * ax2
                    n0 = np.linalg.norm(ax0)
                    if n0 > 1e-10:
                        ax0 /= n0
                    else:
                        ref_vec = np.array([1.0, 0, 0]) if abs(ax2[0]) < 0.9 \
                            else np.array([0, 1.0, 0])
                        ax0 = np.cross(ax2, ref_vec)
                        ax0 /= np.linalg.norm(ax0)
                ax1 = np.cross(ax2, ax0)
                ax1 /= np.linalg.norm(ax1)
                axes = np.array([ax0, ax1, ax2])
                dims, obb_center = _compute_obb_dimensions(verts, axes, center)
        # Sign-only correction for cylinder: flip axis pairs so each
        # axis points toward the reference direction.  No permutation —
        # axis[2] must stay the cylinder axis.
        if ref_rot is not None:
            ref_quat = ref_rot.to_quaternion()
        else:
            ref_quat = Quaternion()
        SIGN_COMBOS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        best_angle = float('inf')
        best_axes = axes.copy()
        for signs in SIGN_COMBOS:
            sa = axes.copy()
            for k in range(3):
                sa[k] *= signs[k]
            euler = _axes_to_euler(sa)
            cand_quat = euler.to_quaternion()
            angle = ref_quat.rotation_difference(cand_quat).angle
            if angle < best_angle:
                best_angle = angle
                best_axes = sa.copy()
        axes = best_axes
        dims, obb_center = _compute_obb_dimensions(verts, axes, center)

    else:
        # SPHERE, UNKNOWN: compare refined vs raw PCA axes
        pca_raw_for_min = pca_axes_raw.copy()
        axes, dims = _pick_best_axes(
            axes, dims, pca_raw_for_min, verts, center, ref_rot)

    euler = _axes_to_euler(axes)

    return ClassificationResult(
        shape_type=shape,
        position=tuple(obb_center),
        rotation=tuple(euler),
        dimensions=tuple(dims),
        confidence=float(round(confidence, 3)),
        method='PCA',
    )


# ============================================================
# RANSAC Classifier
# ============================================================

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


def classify_ransac(obj, settings):
    """
    RANSAC-based shape classification (experimental).
    Fits sphere, cylinder, and box, picks the best inlier ratio.

    NOTE: This classifier does **not** include the face-normal rotation
    correction that PCA uses, so output rotations may be less accurate
    on symmetric shapes.  Use PCA for production work.
    """
    verts = _get_vertices_world(obj, settings.sample_count)

    if len(verts) < 4:
        return ClassificationResult(
            shape_type=ShapeType.UNKNOWN,
            position=tuple(obj.location),
            rotation=(0, 0, 0),
            dimensions=(1, 1, 1),
            confidence=0.0,
            method='RANSAC',
        )

    face_normals, face_areas = _get_face_normals_world(obj)

    center, pca_axes, eigenvalues = _pca(verts)
    refined_axes = _refine_axes_from_normals(obj, pca_axes,
                                              normals=face_normals, areas=face_areas)
    axes = _pick_tighter_axes(refined_axes, pca_axes, verts, center)
    dims, obb_center = _compute_obb_dimensions(verts, axes, center)
    euler = _axes_to_euler(axes)

    # Volume ratio check for complex/organic shapes
    vol_ratio = _compute_volume_ratio(obj, axes, center, verts)
    if vol_ratio < settings.complexity_threshold:
        return ClassificationResult(
            shape_type=ShapeType.UNKNOWN,
            position=tuple(obb_center),
            rotation=tuple(euler),
            dimensions=tuple(dims),
            confidence=float(round(vol_ratio, 3)),
            method='RANSAC',
        )

    # Fit each primitive type
    sphere_result = _ransac_fit_sphere(verts)
    cylinder_result = _ransac_fit_cylinder(verts)
    box_inlier = _ransac_fit_box(verts, obj=obj)

    scores = {'BOX': box_inlier}

    if sphere_result:
        # Penalize sphere score when face normals indicate box-like geometry.
        # Rounded/beveled boxes have high sphere inlier ratio but few normal
        # directions; true spheres have normals pointing in many directions.
        normal_spread = _face_normal_spread(obj, normals=face_normals, areas=face_areas)
        sphere_score = sphere_result[2]
        if normal_spread < 0.4:
            # Box-like normal distribution → heavily penalize sphere fit
            sphere_score *= 0.5
        scores['SPHERE'] = sphere_score

    if cylinder_result:
        scores['CYLINDER'] = cylinder_result[3]

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Build result based on best fit
    if best_type == 'SPHERE' and sphere_result:
        s_center, s_radius, _ = sphere_result
        shape = ShapeType.SPHERE
        pos = tuple(s_center)
        dim = (s_radius * 2, s_radius * 2, s_radius * 2)
        rot = (0, 0, 0)
    elif best_type == 'CYLINDER' and cylinder_result:
        c_center, c_axis, c_radius, _ = cylinder_result
        shape = ShapeType.CYLINDER
        # Compute cylinder height from projection
        centered = verts - c_center
        proj_along = centered @ c_axis
        height = proj_along.max() - proj_along.min()
        pos = tuple(c_center + c_axis * (proj_along.max() + proj_along.min()) / 2)
        dim = (c_radius * 2, c_radius * 2, height)

        # Rotation: align Z to cylinder axis
        z = Vector(c_axis).normalized()
        if abs(z.dot(Vector((0, 0, 1)))) > 0.999:
            x = Vector((1, 0, 0))
        else:
            x = z.cross(Vector((0, 0, 1))).normalized()
        y = z.cross(x).normalized()
        rot_mat = Matrix((
            (*x, 0),
            (*y, 0),
            (*z, 0),
            (0, 0, 0, 1),
        )).transposed()
        rot = tuple(rot_mat.to_euler())
    else:
        shape = ShapeType.BOX
        pos = tuple(obb_center)
        dim = tuple(dims)
        rot = tuple(euler)

    if best_score < 0.3:
        shape = ShapeType.UNKNOWN

    return ClassificationResult(
        shape_type=shape,
        position=pos,
        rotation=rot,
        dimensions=dim,
        confidence=float(round(best_score, 3)),
        method='RANSAC',
    )


# ============================================================
# Hybrid Classifier
# ============================================================

def classify_hybrid(obj, settings):
    """
    Hybrid (experimental): PCA for quick pre-filter, RANSAC for verification.

    When PCA confidence is high (>0.8), uses PCA result directly and benefits
    from full rotation correction.  When confidence is low, falls back to
    RANSAC which lacks rotation correction — see ``classify_ransac`` note.
    """
    pca_result = classify_pca(obj, settings)

    # If PCA is very confident, trust it
    if pca_result.confidence > 0.8:
        pca_result.method = 'Hybrid(PCA)'
        return pca_result

    # Otherwise run RANSAC to verify
    ransac_result = classify_ransac(obj, settings)

    # Take the one with higher confidence
    if ransac_result.confidence > pca_result.confidence:
        ransac_result.method = 'Hybrid(RANSAC)'
        return ransac_result
    else:
        pca_result.method = 'Hybrid(PCA)'
        return pca_result


# ============================================================
# Dispatcher
# ============================================================

def classify(obj, settings):
    """Classify object shape using the selected algorithm."""
    algo = settings.algorithm

    if algo == 'PCA':
        return classify_pca(obj, settings)
    elif algo == 'RANSAC':
        return classify_ransac(obj, settings)
    elif algo == 'HYBRID':
        return classify_hybrid(obj, settings)
    else:
        return classify_pca(obj, settings)
