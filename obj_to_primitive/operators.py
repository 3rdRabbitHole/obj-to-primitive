"""
Operators for Object to Primitive addon.
- Preview: classify + overlay transparent primitive
- Confirm: replace original with primitive, archive original
- Cancel: remove preview primitives
"""

import bpy
import bmesh
import math
from mathutils import Euler

from .classifiers import classify, ShapeType
from .properties import get_shape_color, SHAPE_LABELS


PREVIEW_PREFIX = "_O2P_PREVIEW_"
ARCHIVE_COLLECTION = "Obj2Prim_Original"


def _get_or_create_collection(name, hide=False):
    """Get or create a collection by name."""
    if name in bpy.data.collections:
        col = bpy.data.collections[name]
    else:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)

    if hide:
        # Hide in viewport
        layer_col = _find_layer_collection(bpy.context.view_layer.layer_collection, name)
        if layer_col:
            layer_col.hide_viewport = True
            layer_col.exclude = False

    return col


def _find_layer_collection(layer_col, name):
    """Recursively find a LayerCollection by name."""
    if layer_col.name == name:
        return layer_col
    for child in layer_col.children:
        found = _find_layer_collection(child, name)
        if found:
            return found
    return None


def _create_primitive_mesh(shape_type, dimensions, settings):
    """Create a primitive mesh data based on shape type."""
    # Dimensions from classifier are full extents (not half)
    dx, dy, dz = dimensions

    if shape_type == ShapeType.CYLINDER:
        # Create cylinder: radius = average of dx,dy / 2, depth = dz
        radius = (dx + dy) / 4.0
        depth = dz
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=settings.cylinder_segments,
            radius=radius,
            depth=depth,
            location=(0, 0, 0),
        )
    elif shape_type == ShapeType.SPHERE:
        # Create sphere: radius = average of all dims / 2
        radius = (dx + dy + dz) / 6.0
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=settings.sphere_segments,
            ring_count=settings.sphere_rings,
            radius=radius,
            location=(0, 0, 0),
        )
    else:
        # Box (including UNKNOWN)
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(0, 0, 0),
        )
        # Scale to match dimensions
        obj = bpy.context.active_object
        obj.scale = (dx, dy, dz)
        bpy.ops.object.transform_apply(scale=True)
        return obj

    return bpy.context.active_object


def _create_convex_hull_mesh(source_obj):
    """Create a convex hull mesh from the source object's vertices."""
    mat = source_obj.matrix_world

    bm = bmesh.new()
    bm.from_mesh(source_obj.data)
    bm.transform(mat)

    result = bmesh.ops.convex_hull(bm, input=bm.verts)

    # Remove interior geometry
    interior = result.get("geom_interior", [])
    unused = result.get("geom_unused", [])
    to_delete = set()
    for g in interior + unused:
        if isinstance(g, bmesh.types.BMVert):
            to_delete.add(g)
    if to_delete:
        bmesh.ops.delete(bm, geom=list(to_delete), context='VERTS')

    mesh_data = bpy.data.meshes.new(name="_o2p_convex_hull")
    bm.to_mesh(mesh_data)
    bm.free()

    hull_obj = bpy.data.objects.new(name="_o2p_convex_hull", object_data=mesh_data)
    bpy.context.collection.objects.link(hull_obj)
    bpy.context.view_layer.objects.active = hull_obj
    hull_obj.select_set(True)

    return hull_obj


def _setup_preview_material(obj, shape_type, opacity):
    """Set up transparent viewport display for preview."""
    color = get_shape_color(shape_type.value)

    # Viewport display settings
    obj.display_type = 'SOLID'
    obj.color = (*color[:3], opacity)
    obj.show_transparent = True

    # Also set viewport display color
    obj.data.materials.clear()
    mat_name = f"_O2P_{shape_type.value}_preview"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Alpha"].default_value = opacity
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'
        if hasattr(mat, 'surface_render_method'):
            mat.surface_render_method = 'BLENDED'

    obj.data.materials.append(mat)


def _create_temp_mesh_from_edit_selection(context):
    """Create a temporary mesh object from selected geometry in Edit Mode.

    Supports multi-object editing: selected vertices and faces from all
    objects in edit mode are merged into a single mesh in world space.
    Returns None if nothing is selected.
    """
    combined_bm = bmesh.new()

    for obj in context.objects_in_mode:
        bm = bmesh.from_edit_mesh(obj.data)
        mat = obj.matrix_world

        vert_map = {}
        for v in bm.verts:
            if v.select:
                new_v = combined_bm.verts.new(mat @ v.co)
                vert_map[v.index] = new_v

        combined_bm.verts.ensure_lookup_table()

        for f in bm.faces:
            if f.select:
                try:
                    new_verts = [vert_map[v.index] for v in f.verts]
                    combined_bm.faces.new(new_verts)
                except (KeyError, ValueError):
                    pass

    if not combined_bm.verts:
        combined_bm.free()
        return None

    mesh_data = bpy.data.meshes.new("_o2p_temp_edit")
    combined_bm.to_mesh(mesh_data)
    combined_bm.free()

    temp_obj = bpy.data.objects.new("_o2p_temp_edit", mesh_data)
    context.collection.objects.link(temp_obj)

    return temp_obj


class OBJ2PRIM_OT_preview(bpy.types.Operator):
    """Preview: classify selected objects and overlay transparent primitives"""
    bl_idname = "obj2prim.preview"
    bl_label = "Preview Classification"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.scene.obj2prim.is_previewing:
            return False
        if context.mode == 'EDIT_MESH':
            return bool(context.objects_in_mode)
        return (
            context.selected_objects
            and any(obj.type == 'MESH' for obj in context.selected_objects)
        )

    def execute(self, context):
        settings = context.scene.obj2prim

        if context.mode == 'EDIT_MESH':
            return self._execute_edit_mode(context, settings)

        return self._execute_object_mode(context, settings)

    def _execute_edit_mode(self, context, settings):
        """Preview from Edit Mode: classify selected geometry as one shape."""
        # Build temp mesh from selection (while still in Edit Mode)
        temp_obj = _create_temp_mesh_from_edit_selection(context)
        if temp_obj is None:
            self.report({'WARNING'}, "No vertices selected")
            return {'CANCELLED'}

        vert_count = len(temp_obj.data.vertices)
        if vert_count < 4:
            # Clean up temp mesh
            temp_mesh = temp_obj.data
            bpy.data.objects.remove(temp_obj, do_unlink=True)
            if temp_mesh and temp_mesh.users == 0:
                bpy.data.meshes.remove(temp_mesh)
            self.report({'WARNING'}, f"Need at least 4 vertices (selected {vert_count})")
            return {'CANCELLED'}

        # Classify the combined selection
        result = classify(temp_obj, settings)

        # Switch to Object Mode for primitive creation
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

        # Create primitive
        if (result.shape_type == ShapeType.UNKNOWN
                and settings.unknown_mode == 'CONVEX_HULL'):
            prim_obj = _create_convex_hull_mesh(temp_obj)
        else:
            prim_obj = _create_primitive_mesh(
                result.shape_type, result.dimensions, settings)

        # Clean up temp mesh
        temp_mesh = temp_obj.data
        bpy.data.objects.remove(temp_obj, do_unlink=True)
        if temp_mesh and temp_mesh.users == 0:
            bpy.data.meshes.remove(temp_mesh)

        if prim_obj is None:
            self.report({'WARNING'}, "Failed to create primitive")
            return {'CANCELLED'}

        # Position and rotate
        is_convex_hull = (result.shape_type == ShapeType.UNKNOWN
                          and settings.unknown_mode == 'CONVEX_HULL')
        if not is_convex_hull:
            prim_obj.location = result.position
            prim_obj.rotation_euler = Euler(result.rotation)

        prim_obj.name = f"{PREVIEW_PREFIX}EditSelection"

        # Store classification info (no original to reference)
        prim_obj["_o2p_original"] = ""
        prim_obj["_o2p_shape"] = result.shape_type.value
        prim_obj["_o2p_confidence"] = float(result.confidence)
        prim_obj["_o2p_method"] = result.method
        prim_obj["_o2p_editmode"] = True
        prim_obj["_o2p_position"] = list(result.position)
        prim_obj["_o2p_rotation"] = list(result.rotation)
        prim_obj["_o2p_dimensions"] = list(result.dimensions)

        _setup_preview_material(prim_obj, result.shape_type, settings.preview_opacity)

        self.report(
            {'INFO'},
            f"EditSelection → {result.shape_type.value} "
            f"(confidence: {result.confidence}, method: {result.method})"
        )

        settings.is_previewing = True
        return {'FINISHED'}

    def _execute_object_mode(self, context, settings):
        """Preview from Object Mode: classify each selected mesh object."""
        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if not mesh_objects:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        results = []

        for obj in mesh_objects:
            # Skip preview objects
            if obj.name.startswith(PREVIEW_PREFIX):
                continue

            result = classify(obj, settings)
            results.append((obj, result))

        if not results:
            self.report({'WARNING'}, "No objects to classify")
            return {'CANCELLED'}

        # Create preview primitives
        for obj, result in results:
            # Deselect all
            bpy.ops.object.select_all(action='DESELECT')

            # Create primitive (convex hull for UNKNOWN if enabled)
            if (result.shape_type == ShapeType.UNKNOWN
                    and settings.unknown_mode == 'CONVEX_HULL'):
                prim_obj = _create_convex_hull_mesh(obj)
            else:
                prim_obj = _create_primitive_mesh(result.shape_type, result.dimensions, settings)

            if prim_obj is None:
                continue

            # Position and rotate (convex hull is already in world space)
            is_convex_hull = (result.shape_type == ShapeType.UNKNOWN
                              and settings.unknown_mode == 'CONVEX_HULL')
            if not is_convex_hull:
                prim_obj.location = result.position
                prim_obj.rotation_euler = Euler(result.rotation)

            # Name with prefix for tracking
            prim_obj.name = f"{PREVIEW_PREFIX}{obj.name}"

            # Store reference to original object and classification info
            prim_obj["_o2p_original"] = obj.name
            prim_obj["_o2p_shape"] = result.shape_type.value
            prim_obj["_o2p_confidence"] = float(result.confidence)
            prim_obj["_o2p_method"] = result.method
            # Cache geometry for rebuild without re-classify
            prim_obj["_o2p_position"] = list(result.position)
            prim_obj["_o2p_rotation"] = list(result.rotation)
            prim_obj["_o2p_dimensions"] = list(result.dimensions)

            # Set up preview appearance
            _setup_preview_material(prim_obj, result.shape_type, settings.preview_opacity)

            self.report(
                {'INFO'},
                f"{obj.name} → {result.shape_type.value} "
                f"(confidence: {result.confidence}, method: {result.method})"
            )

        # Make originals unselectable during preview
        for obj, result in results:
            obj.hide_select = True

        settings.is_previewing = True
        return {'FINISHED'}


class OBJ2PRIM_OT_confirm(bpy.types.Operator):
    """Confirm: finalize replacement, archive originals"""
    bl_idname = "obj2prim.confirm"
    bl_label = "Confirm Replacement"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.obj2prim.is_previewing

    def execute(self, context):
        settings = context.scene.obj2prim

        # Find all preview objects
        preview_objects = [
            obj for obj in bpy.data.objects
            if obj.name.startswith(PREVIEW_PREFIX)
        ]

        # Collect ALL original names (including excluded ones) to restore later
        all_original_names = set()
        for obj in preview_objects:
            name = obj.get("_o2p_original", "")
            if name:
                all_original_names.add(name)
        # Also find excluded originals (they have no preview but have the flag)
        for obj in bpy.data.objects:
            if obj.get("obj2prim_exclude", False):
                all_original_names.add(obj.name)

        if not preview_objects:
            # No primitives left (all excluded) — just clean up
            self._restore_originals(all_original_names)
            settings.is_previewing = False
            self.report({'INFO'}, "No primitives to confirm (all excluded)")
            return {'CANCELLED'}

        # Archive setup
        use_collection = settings.archive_to_collection
        if use_collection:
            archive_col = _get_or_create_collection(ARCHIVE_COLLECTION, hide=True)

        replaced_count = 0

        for prim_obj in preview_objects:
            original_name = prim_obj.get("_o2p_original", "")
            shape_type = prim_obj.get("_o2p_shape", "UNKNOWN")
            is_editmode = prim_obj.get("_o2p_editmode", False)

            # Rename primitive: remove preview prefix, add shape label
            label = SHAPE_LABELS.get(shape_type, "[UNK]")
            if is_editmode:
                prim_obj.name = f"{label} EditSelection"
            else:
                if not original_name or original_name not in bpy.data.objects:
                    continue
                prim_obj.name = f"{label} {original_name}"

            # Make primitive fully opaque
            color = get_shape_color(shape_type)
            prim_obj.color = (*color[:3], 1.0)

            # Update material opacity
            if prim_obj.data.materials:
                mat = prim_obj.data.materials[0]
                if mat and mat.use_nodes:
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf:
                        bsdf.inputs["Alpha"].default_value = 1.0

            # Archive original (skip for Edit Mode previews)
            if not is_editmode and original_name in bpy.data.objects:
                orig_obj = bpy.data.objects[original_name]
                if use_collection:
                    for col in orig_obj.users_collection:
                        col.objects.unlink(orig_obj)
                    archive_col.objects.link(orig_obj)
                else:
                    orig_obj.hide_viewport = True

            # Clean up custom properties from primitive
            for key in list(prim_obj.keys()):
                if key.startswith("_o2p_"):
                    del prim_obj[key]

            replaced_count += 1

        # Restore all originals (hide_select, flags)
        self._restore_originals(all_original_names)

        settings.is_previewing = False
        self.report({'INFO'}, f"Replaced {replaced_count} object(s)")
        return {'FINISHED'}

    @staticmethod
    def _restore_originals(original_names):
        """Restore hide_select and clear exclude flags on originals."""
        for name in original_names:
            if name in bpy.data.objects:
                obj = bpy.data.objects[name]
                obj.hide_select = False
                if "obj2prim_exclude" in obj:
                    del obj["obj2prim_exclude"]


class OBJ2PRIM_OT_cancel(bpy.types.Operator):
    """Cancel: remove all preview primitives"""
    bl_idname = "obj2prim.cancel"
    bl_label = "Cancel Preview"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.obj2prim.is_previewing

    def execute(self, context):
        settings = context.scene.obj2prim

        # Find and delete all preview objects, collect original names
        preview_objects = [
            obj for obj in bpy.data.objects
            if obj.name.startswith(PREVIEW_PREFIX)
        ]

        original_names = set()
        for obj in preview_objects:
            name = obj.get("_o2p_original", "")
            if name:
                original_names.add(name)

        for obj in preview_objects:
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

        # Also find excluded originals (no preview but have the flag)
        for obj in bpy.data.objects:
            if obj.get("obj2prim_exclude", False):
                original_names.add(obj.name)

        # Restore hide_select and clear flags on all originals
        for name in original_names:
            if name in bpy.data.objects:
                obj = bpy.data.objects[name]
                obj.hide_select = False
                if "obj2prim_exclude" in obj:
                    del obj["obj2prim_exclude"]

        # Clean up preview materials
        for mat in list(bpy.data.materials):
            if mat.name.startswith("_O2P_") and mat.users == 0:
                bpy.data.materials.remove(mat)

        settings.is_previewing = False
        self.report({'INFO'}, "Preview cancelled")
        return {'FINISHED'}


class OBJ2PRIM_OT_toggle_exclude(bpy.types.Operator):
    """Exclude selected preview primitives from conversion"""
    bl_idname = "obj2prim.toggle_exclude"
    bl_label = "Exclude Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not context.scene.obj2prim.is_previewing:
            return False
        # At least one selected object must be a preview primitive
        return any(
            obj.name.startswith(PREVIEW_PREFIX)
            for obj in context.selected_objects
        )

    def execute(self, context):
        excluded_count = 0

        for obj in list(context.selected_objects):
            # Only operate on preview primitives — ignore everything else
            if not obj.name.startswith(PREVIEW_PREFIX):
                continue

            original_name = obj.get("_o2p_original", "")

            # Mark the original as excluded
            if original_name and original_name in bpy.data.objects:
                bpy.data.objects[original_name]["obj2prim_exclude"] = True

            # Delete the preview primitive
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

            excluded_count += 1

        if excluded_count:
            self.report({'INFO'}, f"Excluded {excluded_count} object(s)")
        else:
            self.report({'INFO'}, "No preview primitives selected")

        return {'FINISHED'}


classes = (
    OBJ2PRIM_OT_preview,
    OBJ2PRIM_OT_confirm,
    OBJ2PRIM_OT_cancel,
    OBJ2PRIM_OT_toggle_exclude,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
