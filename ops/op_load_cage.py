"""Operator: append the low-poly head cage from the bundled blend."""

import os
import bpy

from ..prefs import get_cage_blend_path

# Internal object name pattern inside the bundled .blend (not shown to the user).
SOURCE_TEMPLATE = "Ada_Head_head_lod{lod}_mesh"
# Friendly name the appended cage gets in the scene.
CAGE_NAME = "Head Cage"


def append_cage(context, lod=None):
    """Append a fresh cage of the given LOD and return it.

    Split out of the operator so a character restored from a .mhfrt file can
    rebuild its cage without a UI round trip: the cage is OUR asset, so a
    missing one is never something to bother the artist about.

    Raises RuntimeError with a readable reason.
    """
    blend = get_cage_blend_path(context)
    if lod is None:
        lod = context.scene.mhfrt.cage_lod
    source_name = SOURCE_TEMPLATE.format(lod=lod)
    if not blend or not os.path.exists(blend):
        raise RuntimeError(f"Cage blend not found: {blend!r}")

    # Append ONLY the cage object; we just need geometry + vertex groups.
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(blend, link=False) as (data_from, data_to):
        if source_name not in data_from.objects:
            raise RuntimeError(
                f"'{source_name}' not found inside {os.path.basename(blend)}")
        data_to.objects = [source_name]

    appended = [o for o in data_to.objects if o is not None]
    if not appended:
        raise RuntimeError("Append failed (no object returned)")

    cage = appended[0]
    context.scene.collection.objects.link(cage)
    # Drop the Armature modifier (the rig comes later, per character) and purge
    # whatever the load dependency-pulled along (the asset skeleton rides in on
    # that modifier).
    for mod in list(cage.modifiers):
        cage.modifiers.remove(mod)
    hitchhikers = set(bpy.data.objects) - before - {cage}
    datas = {o.data for o in hitchhikers if o.data is not None}
    for o in hitchhikers:
        bpy.data.objects.remove(o, do_unlink=True)
    for d in datas:
        if d.users == 0 and isinstance(d, bpy.types.Armature):
            bpy.data.armatures.remove(d)
    cage.name = CAGE_NAME
    cage.data.name = CAGE_NAME
    return cage


class MHFRT_OT_load_cage(bpy.types.Operator):
    bl_idname = "mhfrt.load_cage"
    bl_label = "Load Head Cage"
    bl_description = "Append the chosen head LOD as the wrap cage from the bundled blend"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            cage = append_cage(context)
        except RuntimeError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        # Place the cage where the artist's 3D cursor sits, so it lands next
        # to whatever they're currently working on instead of always at the
        # world origin.  The append itself gives an identity matrix; we just
        # translate its world matrix to the cursor position.
        from mathutils import Matrix
        cage.matrix_world = Matrix.Translation(
            context.scene.cursor.location.copy())

        for ob in context.selected_objects:
            ob.select_set(False)
        cage.select_set(True)
        context.view_layer.objects.active = cage

        # One cage, one character.  If the panel already holds a character
        # that has BOTH a cage and a head, this cage is plainly the start of
        # the next one - so a fresh record is opened for it.  Without this the
        # new cage was filed into the previous character's collection (the
        # head slot still named that character) and the two folded into a
        # single row, which is how characters "disappeared" from the list.
        from ..core import registry, provenance
        mh = context.scene.mhfrt
        record = registry.active(context.scene)
        occupied = (record is not None
                    and record.cage is not None
                    and record.target is not None)
        if occupied or record is None:
            record = registry.new(context.scene)
            registry.set_active(context.scene, record.uid)
            mh.switching_character = True
            try:
                mh.skeleton = None
                mh.target = None
            finally:
                mh.switching_character = False
            mh.active_cage = None
            mh.active_target = None

        # The cage is ours, and it is the first thing we put in the artist's
        # file.  Stamping it here is what lets removal delete it without
        # having to recognise it by name or by role later.
        provenance.mark(record, cage)
        provenance.mark(record, cage.data)

        # Assigning the slot files the character immediately - its own root
        # collection is created and the cage moved into '02 Head Cage', rather
        # than the cage sitting loose until a rig gets built.
        mh.cage = cage

        # studio look right away - via the toggle, so the user's previous
        # shading is snapshotted and one click brings it back
        from .op_shading import tint_objects
        mh = context.scene.mhfrt
        if mh.studio_shading:
            tint_objects(context)          # already styled; just tint the new cage
        else:
            mh.studio_shading = True

        self.report(
            {'INFO'},
            f"Loaded '{cage.name}' - {len(cage.data.vertices)} verts, "
            f"{len(cage.vertex_groups)} vertex groups",
        )
        return {'FINISHED'}


_classes = (MHFRT_OT_load_cage,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
