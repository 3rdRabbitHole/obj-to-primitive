bl_info = {
    "name": "Object to Primitive",
    "author": "3RH",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Obj2Prim",
    "description": "Replace mesh objects with simple geometric primitives",
    "category": "Object",
}

import bpy

from . import properties
from . import operators
from . import ui


def register():
    properties.register()
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
    properties.unregister()


if __name__ == "__main__":
    register()
