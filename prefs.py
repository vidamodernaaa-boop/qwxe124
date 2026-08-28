"""Add-on preferences."""

import os
import bpy

from . import paths


class MHFRT_Prefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    cage_blend_path: bpy.props.StringProperty(
        name="Cage .blend Override",
        description="Optional path to a custom cage .blend. Leave empty to use the "
                    "bundled cage",
        subtype='FILE_PATH',
        default="",
    )

    auto_character_files: bpy.props.BoolProperty(
        name="Character Files (.mhfrt)",
        description="Write a .mhfrt file for every character, created with its "
                    "row in the list and kept up to date as you work. It holds "
                    "the character's landmarks and settings AND the record of "
                    "what this add-on added to your scene - including your own "
                    "armature exactly as it was before a merge - which is what "
                    "Remove replays to hand your character back untouched. "
                    "Turning this off means a removal can only fall back to "
                    "what the .blend itself remembers",
        default=True,
    )

    lock_interface_for_render: bpy.props.BoolProperty(
        name="Lock Interface While Rendering",
        description="Switch Blender's own Lock Interface on for any scene that "
                    "holds a live rig. The rig is evaluated in Python: a render "
                    "writes its bones from the render thread, and without the "
                    "lock Blender's main thread keeps updating the same "
                    "depsgraph at the same time - which crashes a few frames "
                    "into an animation render. Turn this off only if you set "
                    "Lock Interface yourself (Output > Performance)",
        default=True,
    )

    character_dir: bpy.props.StringProperty(
        name="Character File Folder",
        description="Where .mhfrt character files are written. Leave empty to "
                    "keep them beside the .blend, in a folder named after it",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "cage_blend_path")
        resolved = get_cage_blend_path(context)
        ok = bool(resolved) and os.path.exists(resolved)
        if self.cage_blend_path:
            col.label(text=f"Using: {resolved}", icon='CHECKMARK' if ok else 'ERROR')
        else:
            col.label(
                text="Using bundled cage" if ok else "Bundled cage not found",
                icon='CHECKMARK' if ok else 'ERROR',
            )
        col.separator()
        col.prop(self, "lock_interface_for_render")
        col.separator()
        col.prop(self, "auto_character_files")
        row = col.row()
        row.enabled = self.auto_character_files
        row.prop(self, "character_dir")
        if self.auto_character_files:
            from .core import sidecar
            col.label(text=f"Writing to: {sidecar.auto_dir()}", icon='FILE_TICK')


def get_cage_blend_path(context):
    """Resolve the cage blend: preference override -> bundled fallback."""
    override = ""
    try:
        prefs = context.preferences.addons[__package__].preferences
        override = bpy.path.abspath(prefs.cage_blend_path) if prefs.cage_blend_path else ""
    except (KeyError, AttributeError):
        pass
    if override and os.path.exists(override):
        return override
    return paths.BUNDLED_BLEND


def register():
    bpy.utils.register_class(MHFRT_Prefs)


def unregister():
    bpy.utils.unregister_class(MHFRT_Prefs)
