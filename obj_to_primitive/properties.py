import bpy
from bpy.props import EnumProperty, FloatProperty, BoolProperty, IntProperty, FloatVectorProperty

# Debounce timer handle — module-level so it persists across calls
_debounce_timer = None
_debounce_rebuild_types = set()


def _schedule_preview_rebuild(shape_types):
    """Schedule a debounced preview rebuild for the given shape type(s)."""
    global _debounce_timer, _debounce_rebuild_types

    _debounce_rebuild_types.update(shape_types)

    # Cancel existing timer
    if _debounce_timer is not None:
        try:
            bpy.app.timers.unregister(_debounce_timer)
        except ValueError:
            pass
        _debounce_timer = None

    def _do_rebuild():
        global _debounce_timer, _debounce_rebuild_types
        _debounce_timer = None
        types_to_rebuild = _debounce_rebuild_types.copy()
        _debounce_rebuild_types.clear()
        _rebuild_preview_primitives(types_to_rebuild)
        return None  # don't repeat

    _debounce_timer = _do_rebuild
    bpy.app.timers.register(_do_rebuild, first_interval=0.3)


def _rebuild_preview_primitives(shape_types):
    """Delete and recreate preview primitives for the given shape types."""
    from .classifiers import classify, ShapeType
    from .operators import (
        PREVIEW_PREFIX, _create_primitive_mesh,
        _create_convex_hull_mesh, _setup_preview_material,
    )
    from mathutils import Euler

    scene = bpy.context.scene
    settings = scene.obj2prim

    if not settings.is_previewing:
        return

    # Find preview objects matching the requested shape types
    to_rebuild = []
    for obj in list(bpy.data.objects):
        if not obj.name.startswith(PREVIEW_PREFIX):
            continue
        obj_shape = obj.get("_o2p_shape", "")
        if obj_shape in shape_types:
            original_name = obj.get("_o2p_original", "")
            to_rebuild.append((obj, original_name, obj_shape))

    if not to_rebuild:
        return

    # Delete old previews
    bpy.ops.object.select_all(action='DESELECT')
    for obj, _, _ in to_rebuild:
        mesh_data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh_data and mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)

    # Recreate
    for _, original_name, _ in to_rebuild:
        if original_name not in bpy.data.objects:
            continue

        orig_obj = bpy.data.objects[original_name]
        result = classify(orig_obj, settings)

        bpy.ops.object.select_all(action='DESELECT')

        if (result.shape_type == ShapeType.UNKNOWN
                and settings.unknown_mode == 'CONVEX_HULL'):
            prim_obj = _create_convex_hull_mesh(orig_obj)
        else:
            prim_obj = _create_primitive_mesh(
                result.shape_type, result.dimensions, settings)

        if prim_obj is None:
            continue

        is_convex_hull = (result.shape_type == ShapeType.UNKNOWN
                          and settings.unknown_mode == 'CONVEX_HULL')
        if not is_convex_hull:
            prim_obj.location = result.position
            prim_obj.rotation_euler = Euler(result.rotation)

        prim_obj.name = f"{PREVIEW_PREFIX}{original_name}"
        prim_obj["_o2p_original"] = original_name
        prim_obj["_o2p_shape"] = result.shape_type.value
        prim_obj["_o2p_confidence"] = float(result.confidence)
        prim_obj["_o2p_method"] = result.method

        _setup_preview_material(prim_obj, result.shape_type, settings.preview_opacity)


# --- Update callbacks ---

def _on_cylinder_update(self, context):
    _schedule_preview_rebuild({'CYLINDER'})


def _on_sphere_update(self, context):
    _schedule_preview_rebuild({'SPHERE'})


def _on_unknown_mode_update(self, context):
    _schedule_preview_rebuild({'UNKNOWN'})


def _on_opacity_update(self, context):
    """Update opacity on all existing preview materials without rebuilding."""
    if not self.is_previewing:
        return
    from .operators import PREVIEW_PREFIX, _setup_preview_material
    from .classifiers import ShapeType
    for obj in bpy.data.objects:
        if obj.name.startswith(PREVIEW_PREFIX):
            shape_str = obj.get("_o2p_shape", "UNKNOWN")
            try:
                shape_type = ShapeType(shape_str)
            except ValueError:
                shape_type = ShapeType.UNKNOWN
            _setup_preview_material(obj, shape_type, self.preview_opacity)


def _on_color_update(self, context):
    """Update preview material and object color when user changes a shape color."""
    if not self.is_previewing:
        return
    from .operators import PREVIEW_PREFIX, _setup_preview_material
    from .classifiers import ShapeType
    for obj in bpy.data.objects:
        if obj.name.startswith(PREVIEW_PREFIX):
            shape_str = obj.get("_o2p_shape", "UNKNOWN")
            try:
                shape_type = ShapeType(shape_str)
            except ValueError:
                shape_type = ShapeType.UNKNOWN
            _setup_preview_material(obj, shape_type, self.preview_opacity)


class Obj2PrimSettings(bpy.types.PropertyGroup):
    algorithm: EnumProperty(
        name="Algorithm",
        description="Shape classification algorithm",
        items=[
            ('PCA', "PCA", "Principal Component Analysis with circularity test"),
            ('RANSAC', "RANSAC (Experimental)", "RANSAC-based primitive fitting (experimental, less accurate)"),
            ('HYBRID', "Hybrid (Experimental)", "PCA pre-filter + RANSAC verification (experimental)"),
        ],
        default='PCA',
    )

    sample_count: IntProperty(
        name="Sample Vertices",
        description="Max vertices to sample for classification (0 = use all)",
        default=2000,
        min=0,
        max=50000,
    )

    circularity_threshold: FloatProperty(
        name="Circularity Threshold",
        description="Threshold to distinguish round vs rectangular cross-section (0-1)",
        default=0.85,
        min=0.5,
        max=1.0,
    )

    sphere_ratio_threshold: FloatProperty(
        name="Sphere Ratio",
        description="Eigenvalue ratio threshold for sphere detection",
        default=0.8,
        min=0.5,
        max=1.0,
    )

    cylinder_ratio_threshold: FloatProperty(
        name="Cylinder Ratio",
        description="Eigenvalue ratio for two similar axes (cylinder detection)",
        default=0.7,
        min=0.3,
        max=1.0,
    )

    complexity_threshold: FloatProperty(
        name="Complexity Threshold",
        description="Volume ratio below which a shape is classified as UNKNOWN (mesh_vol / obb_vol)",
        default=0.35,
        min=0.1,
        max=0.8,
    )

    preview_opacity: FloatProperty(
        name="Preview Opacity",
        description="Opacity of preview primitives",
        default=0.4,
        min=0.1,
        max=0.9,
        update=_on_opacity_update,
    )

    # --- Primitive shape settings ---

    cylinder_segments: IntProperty(
        name="Segments",
        description="Number of vertices around the cylinder circumference",
        default=32,
        min=6,
        max=128,
        update=_on_cylinder_update,
    )

    sphere_segments: IntProperty(
        name="Segments",
        description="Number of horizontal segments (longitude) for sphere",
        default=32,
        min=6,
        max=128,
        update=_on_sphere_update,
    )

    sphere_rings: IntProperty(
        name="Rings",
        description="Number of vertical rings (latitude) for sphere",
        default=16,
        min=3,
        max=64,
        update=_on_sphere_update,
    )

    unknown_mode: EnumProperty(
        name="Mode",
        description="What primitive to use for UNKNOWN shapes",
        items=[
            ('BOX', "Box", "Use oriented bounding box"),
            ('CONVEX_HULL', "Convex Hull", "Use convex hull of original mesh"),
        ],
        default='BOX',
        update=_on_unknown_mode_update,
    )

    # --- Primitive Colors ---

    color_box: FloatVectorProperty(
        name="Box",
        subtype='COLOR',
        default=(0.2, 0.5, 1.0),
        min=0.0, max=1.0,
        size=3,
        update=_on_color_update,
    )

    color_cylinder: FloatVectorProperty(
        name="Cylinder",
        subtype='COLOR',
        default=(0.2, 0.8, 0.3),
        min=0.0, max=1.0,
        size=3,
        update=_on_color_update,
    )

    color_sphere: FloatVectorProperty(
        name="Sphere",
        subtype='COLOR',
        default=(1.0, 0.3, 0.3),
        min=0.0, max=1.0,
        size=3,
        update=_on_color_update,
    )

    color_unknown: FloatVectorProperty(
        name="Unknown",
        subtype='COLOR',
        default=(0.6, 0.6, 0.6),
        min=0.0, max=1.0,
        size=3,
        update=_on_color_update,
    )

    # --- Archive setting ---

    archive_to_collection: BoolProperty(
        name="Move to Collection",
        description="Move originals to archive collection. If unchecked, originals are only hidden",
        default=True,
    )

    # --- UI fold state ---

    show_primitive_settings: BoolProperty(
        name="Primitive Settings",
        default=True,
    )

    show_parameters: BoolProperty(
        name="Parameters",
        default=False,
    )

    show_primitive_colors: BoolProperty(
        name="Primitive Color",
        default=False,
    )

    # State tracking
    is_previewing: BoolProperty(
        name="Is Previewing",
        default=False,
    )


# Default colors (fallback when settings are not available)
SHAPE_COLORS_DEFAULT = {
    'BOX': (0.2, 0.5, 1.0, 1.0),       # Blue
    'CYLINDER': (0.2, 0.8, 0.3, 1.0),   # Green
    'SPHERE': (1.0, 0.3, 0.3, 1.0),     # Red
    'UNKNOWN': (0.6, 0.6, 0.6, 1.0),    # Gray
}


def get_shape_color(shape_key):
    """Get shape color from scene settings, falling back to defaults."""
    try:
        settings = bpy.context.scene.obj2prim
        color_map = {
            'BOX': settings.color_box,
            'CYLINDER': settings.color_cylinder,
            'SPHERE': settings.color_sphere,
            'UNKNOWN': settings.color_unknown,
        }
        c = color_map.get(shape_key)
        if c is not None:
            return (c[0], c[1], c[2], 1.0)
    except (AttributeError, KeyError):
        pass
    return SHAPE_COLORS_DEFAULT.get(shape_key, (0.6, 0.6, 0.6, 1.0))

SHAPE_LABELS = {
    'BOX': "[BOX]",
    'CYLINDER': "[CYL]",
    'SPHERE': "[SPH]",
    'UNKNOWN': "[UNK]",
}


def register():
    bpy.utils.register_class(Obj2PrimSettings)
    bpy.types.Scene.obj2prim = bpy.props.PointerProperty(type=Obj2PrimSettings)


def unregister():
    global _debounce_timer
    if _debounce_timer is not None:
        try:
            bpy.app.timers.unregister(_debounce_timer)
        except ValueError:
            pass
        _debounce_timer = None
    del bpy.types.Scene.obj2prim
    bpy.utils.unregister_class(Obj2PrimSettings)
