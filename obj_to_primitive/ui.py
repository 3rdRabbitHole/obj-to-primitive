"""
UI Panel for Object to Primitive addon.
Located in View3D > Sidebar > Obj2Prim
"""

import bpy
from .properties import get_shape_color


class OBJ2PRIM_PT_main(bpy.types.Panel):
    bl_label = "Object to Primitive"
    bl_idname = "OBJ2PRIM_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Obj2Prim"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.obj2prim

        # Algorithm selector
        box = layout.box()
        box.label(text="Classification", icon='MODIFIER')
        box.prop(settings, "algorithm", text="Method")

        # --- Primitive Settings (collapsible, default expanded) ---
        box = layout.box()
        row = box.row()
        row.prop(
            settings, "show_primitive_settings",
            icon='TRIA_DOWN' if settings.show_primitive_settings else 'TRIA_RIGHT',
            text="Primitive Settings",
            emboss=False,
        )
        if settings.show_primitive_settings:
            col = box.column(align=True)

            col.label(text="Cylinder", icon='MESH_CYLINDER')
            col.prop(settings, "cylinder_segments")

            col.separator()
            col.label(text="Sphere", icon='MESH_UVSPHERE')
            col.prop(settings, "sphere_segments")
            col.prop(settings, "sphere_rings")

            col.separator()
            col.label(text="Unknown", icon='QUESTION')
            col.prop(settings, "unknown_mode")

            col.separator()
            col.prop(settings, "preview_opacity")

        # --- Parameters (collapsible, default collapsed) ---
        box = layout.box()
        row = box.row()
        row.prop(
            settings, "show_parameters",
            icon='TRIA_DOWN' if settings.show_parameters else 'TRIA_RIGHT',
            text="Parameters",
            emboss=False,
        )
        if settings.show_parameters:
            col = box.column(align=True)
            col.prop(settings, "sample_count")
            col.prop(settings, "circularity_threshold")
            col.prop(settings, "sphere_ratio_threshold")
            col.prop(settings, "cylinder_ratio_threshold")
            col.prop(settings, "complexity_threshold")

        # Actions
        box = layout.box()
        box.label(text="Actions", icon='PLAY')
        box.prop(settings, "archive_to_collection")

        if not settings.is_previewing:
            # Show selected object count
            mesh_count = sum(1 for obj in context.selected_objects if obj.type == 'MESH')
            excluded_count = sum(
                1 for obj in context.selected_objects
                if obj.type == 'MESH' and obj.get("obj2prim_exclude", False)
            )
            if mesh_count > 0:
                label = f"{mesh_count} mesh object(s) selected"
                if excluded_count > 0:
                    label += f" ({excluded_count} excluded)"
                box.label(text=label)
            else:
                box.label(text="Select mesh object(s)", icon='ERROR')

            row = box.row()
            row.scale_y = 1.5
            row.operator("obj2prim.preview", icon='HIDE_OFF')
        else:
            box.label(text="Preview active", icon='CHECKMARK')

            # Exclude toggle available during preview
            box.operator("obj2prim.toggle_exclude", icon='RESTRICT_SELECT_ON')

            row = box.row(align=True)
            row.scale_y = 1.5
            row.operator("obj2prim.confirm", icon='CHECKMARK')
            row.operator("obj2prim.cancel", icon='CANCEL')

        # Primitive Color (collapsible, default collapsed)
        box = layout.box()
        row = box.row()
        row.prop(
            settings, "show_primitive_colors",
            icon='TRIA_DOWN' if settings.show_primitive_colors else 'TRIA_RIGHT',
            text="Primitive Color",
            emboss=False,
        )
        if settings.show_primitive_colors:
            col = box.column(align=True)
            col.label(text="Viewport: Object Color mode", icon='INFO')
            col.separator()
            col.prop(settings, "color_box")
            col.prop(settings, "color_cylinder")
            col.prop(settings, "color_sphere")
            col.prop(settings, "color_unknown")

        # Classification result info (shown during preview)
        if settings.is_previewing:
            box = layout.box()
            box.label(text="Results", icon='INFO')
            for obj in bpy.data.objects:
                if obj.name.startswith("_O2P_PREVIEW_"):
                    original = obj.get("_o2p_original", "?")
                    shape = obj.get("_o2p_shape", "?")
                    conf = obj.get("_o2p_confidence", 0)
                    method = obj.get("_o2p_method", "?")
                    box.label(text=f"{original} → {shape} ({conf:.2f}) [{method}]")


classes = (
    OBJ2PRIM_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
