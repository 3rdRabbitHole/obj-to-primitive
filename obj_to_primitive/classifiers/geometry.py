"""Shared geometry helpers: vertex/normal extraction, PCA, OBB, circularity,
volume computation, and rotation utilities.

Performance notes
-----------------
- Vertex extraction uses ``foreach_get`` + numpy batch matrix multiply
  instead of per-vertex ``matrix_world @ v.co`` loops.
- Face normals are extracted once (``_get_face_normals_world``) and shared
  across all functions that need them, avoiding repeated polygon iteration.
"""

import numpy as np
from mathutils import Vector, Matrix, Quaternion
from itertools import permutations as _perms


# ============================================================
# Vertex / Normal Extraction
# ============================================================

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


# ============================================================
# PCA
# ============================================================

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


# ============================================================
# Cross-section Analysis
# ============================================================

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


# ============================================================
# OBB (Oriented Bounding Box)
# ============================================================

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


# ============================================================
# Axis Refinement from Face Normals
# ============================================================

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


# ============================================================
# Volume
# ============================================================

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


# ============================================================
# Sphere / Normal Spread Detection
# ============================================================

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


# ============================================================
# Rotation Utilities
# ============================================================

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
