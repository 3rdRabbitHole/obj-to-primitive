"""Classification pipelines: PCA, RANSAC (experimental), Hybrid (experimental).

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
from itertools import permutations as _perms

from .types import ShapeType, ClassificationResult
from .geometry import (
    _get_vertices_world,
    _get_face_normals_world,
    _pca,
    _cross_section_iso_ratio,
    _compute_circularity,
    _compute_obb_dimensions,
    _pick_tighter_axes,
    _refine_axes_from_normals,
    _compute_volume_ratio,
    _check_sphere_surface,
    _face_normal_spread,
    _axes_to_euler,
    _pick_best_axes,
    _minimize_rotation,
)
from .box import _find_box_axes_from_normals, _ransac_fit_box
from .cylinder import _find_cylinder_axis_from_normals, _ransac_fit_cylinder
from .sphere import _ransac_fit_sphere


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
