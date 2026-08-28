"""Guided eye-center and tongue-rest fine tuning.

These tools only edit existing rest bones. They never rename, recreate, or
reparent joints, and they commit through ``core.rest_tuning`` so RigLogic and
future Update Rig passes retain the artist's correction.
"""

import bpy

from ..core import organization
from ..core import rest_tuning
from ..props import RIG_ID_PROP
from . import op_rig
from . import op_skeleton


_SEP = "\x1f"
SESSION_ACTIVE_PROP = rest_tuning.TONGUE_SESSION_PROP
SESSION_CAGE_PROP = rest_tuning.TONGUE_SESSION_CAGE_PROP
SESSION_TARGET_PROP = rest_tuning.TONGUE_SESSION_TARGET_PROP
SESSION_CURSOR_PROP = "mhfrt_tongue_edit_cursor"
SESSION_PIVOT_PROP = "mhfrt_tongue_edit_pivot"
SESSION_DISPLAY_PROP = "mhfrt_tongue_edit_display"
SESSION_IN_FRONT_PROP = "mhfrt_tongue_edit_in_front"
SESSION_HIDE_VIEWPORT_PROP = "mhfrt_tongue_edit_hide_viewport"
SESSION_SHOW_BONES_PROP = "mhfrt_tongue_edit_show_bones"
SESSION_ORIENTATION_PROP = "mhfrt_tongue_edit_orientation"
SESSION_SOLOS_PROP = "mhfrt_tongue_edit_solos"
SESSION_TONGUE_VISIBLE_PROP = "mhfrt_tongue_edit_collection_visible"
SESSION_HIDDEN_BONES_PROP = "mhfrt_tongue_edit_hidden_bones"
SESSION_ACTIVE_OBJECT_PROP = "mhfrt_tongue_edit_active_object"
SESSION_SELECTED_PROP = "mhfrt_tongue_edit_selected"
SESSION_MODE_PROP = "mhfrt_tongue_edit_mode"
SESSION_SELECTED_BONES_PROP = "mhfrt_tongue_edit_selected_bones"
SESSION_SELECTED_HEADS_PROP = "mhfrt_tongue_edit_selected_heads"
SESSION_SELECTED_TAILS_PROP = "mhfrt_tongue_edit_selected_tails"
SESSION_ACTIVE_BONE_PROP = "mhfrt_tongue_edit_active_bone"
SESSION_ARMATURE_BONES_PROP = "mhfrt_tongue_edit_armature_bones"
SESSION_TONGUE_NAMES_PROP = "mhfrt_tongue_edit_tongue_bones"
SESSION_TONGUE_PARENTS_PROP = "mhfrt_tongue_edit_parents"
SESSION_TONGUE_CONNECTED_PROP = "mhfrt_tongue_edit_connected"
SESSION_TONGUE_DEFORM_PROP = "mhfrt_tongue_edit_deform"

_SESSION_PROPS = (
    SESSION_ACTIVE_PROP,
    SESSION_CAGE_PROP,
    SESSION_TARGET_PROP,
    SESSION_CURSOR_PROP,
    SESSION_PIVOT_PROP,
    SESSION_DISPLAY_PROP,
    SESSION_IN_FRONT_PROP,
    SESSION_HIDE_VIEWPORT_PROP,
    SESSION_SHOW_BONES_PROP,
    SESSION_ORIENTATION_PROP,
    SESSION_SOLOS_PROP,
    SESSION_TONGUE_VISIBLE_PROP,
    SESSION_HIDDEN_BONES_PROP,
    SESSION_ACTIVE_OBJECT_PROP,
    SESSION_SELECTED_PROP,
    SESSION_MODE_PROP,
    SESSION_SELECTED_BONES_PROP,
    SESSION_SELECTED_HEADS_PROP,
    SESSION_SELECTED_TAILS_PROP,
    SESSION_ACTIVE_BONE_PROP,
    SESSION_ARMATURE_BONES_PROP,
    SESSION_TONGUE_NAMES_PROP,
    SESSION_TONGUE_PARENTS_PROP,
    SESSION_TONGUE_CONNECTED_PROP,
    SESSION_TONGUE_DEFORM_PROP,
)


def _active_skeleton(context):
    mh = getattr(context.scene, "mhfrt", None)
    skel = mh.skeleton if mh else None
    if skel is None or skel.type != 'ARMATURE' or not skel.get(RIG_ID_PROP):
        return None
    return skel


def _mode_object(context):
    active = context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass


def _prepare_armature_edit(context, skel):
    _mode_object(context)
    for obj in list(context.selected_objects):
        obj.select_set(False)
    try:
        skel.hide_set(False)
    except RuntimeError:
        pass
    skel.hide_viewport = False
    skel.select_set(True)
    context.view_layer.objects.active = skel
    try:
        bpy.ops.object.mode_set(mode='EDIT')
    except RuntimeError:
        return False
    return True


def _capture_context(context, skel):
    active = context.view_layer.objects.active
    return {
        "active": active,
        "selected": list(context.selected_objects),
        "mode": active.mode if active is not None else 'OBJECT',
        "skel_hidden": skel.hide_get(),
        "skel_hide_viewport": bool(skel.hide_viewport),
    }


def _restore_context(context, state, skel):
    _mode_object(context)
    for obj in list(context.selected_objects):
        obj.select_set(False)
    for obj in state["selected"]:
        if context.view_layer.objects.get(obj.name) is not None:
            try:
                obj.select_set(True)
            except RuntimeError:
                pass
    active = state["active"]
    if active is not None and context.view_layer.objects.get(active.name) is not None:
        context.view_layer.objects.active = active
        mode = state["mode"]
        if mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode=mode)
            except RuntimeError:
                pass
    try:
        skel.hide_set(bool(state["skel_hidden"]))
    except RuntimeError:
        pass
    skel.hide_viewport = bool(state["skel_hide_viewport"])


def _refresh_live_rig(context, skel):
    try:
        mh = context.scene.mhfrt
        if not op_skeleton.ensure_rest_source_to_rig_basis(
                context, skel, mh.cage):
            return "Run Update Rig once to migrate this older merged rig"
        # setup_rig rebuilds the rig cache, and with it every per-bone RigLogic
        # constant - the tuned rest and length layers are picked up there.
        return op_rig.setup_rig(
            context, skel, cage=mh.cage, target=mh.target)
    except Exception as exc:
        op_rig.invalidate()
        return f"{type(exc).__name__}: {exc}"


def _bone_collection(skel, role):
    return next(
        (coll for coll in skel.data.collections_all
         if coll.get(organization.BONE_COLL_PROP) == role),
        None,
    )


def _join_names(names):
    return _SEP.join(name for name in names if name)


def _split_names(value):
    return [name for name in str(value or "").split(_SEP) if name]


def _delete_prop(owner, name):
    if name in owner:
        del owner[name]


def _tongue_bones(skel):
    """The tongue joints this skeleton actually has, base first.

    Every bone the region collection shows as purple, so what the tool selects
    is what the artist sees isolated - and what it snapshots, commits and
    restores is the same set again.  A joint the skeleton does not have is
    simply skipped instead of refusing the whole step: this used to be a hard
    error naming the first absent bone, which turned a rig missing one tip
    joint into a rig whose tongue could not be tuned at all.
    """
    bones = skel.data.bones
    return tuple(name for name in rest_tuning.TONGUE_BONES
                 if bones.get(name) is not None)


def _session_tongue_bones(skel):
    """The bones the OPEN session was started on.

    A session carries its own list rather than reading the module constant,
    so one opened before the constant changed still finishes on exactly the
    bones it snapshotted - the alternative is committing a correction for a
    bone that was never backed up.
    """
    stored = _split_names(skel.get(SESSION_TONGUE_NAMES_PROP))
    return tuple(stored) if stored else tuple(rest_tuning.TONGUE_BONES)


class MHFRT_OT_place_eye_center(bpy.types.Operator):
    bl_idname = "mhfrt.place_eye_center"
    bl_label = "Place Eye Center"
    bl_description = ("Move the selected character eye's three-bone chain so "
                      "its true rotation pivot lands exactly on the 3D Cursor")
    bl_options = {'REGISTER', 'UNDO'}

    side: bpy.props.EnumProperty(items=[
        ('LEFT', "Character Left", "Character's left eye (viewer's right)"),
        ('RIGHT', "Character Right", "Character's right eye (viewer's left)"),
    ])

    @classmethod
    def description(cls, context, properties):
        view_side = "right" if properties.side == 'LEFT' else "left"
        return (f"Place the character's {properties.side.lower()} eye pivot "
                f"(viewer's {view_side}) at the 3D Cursor. Put the cursor at "
                "the center of the eyeball sphere - the point the eye rotates "
                "around - not at the eye mesh's object origin, which the rig "
                "ignores")

    @classmethod
    def poll(cls, context):
        skel = _active_skeleton(context)
        return bool(skel and not skel.get(SESSION_ACTIVE_PROP))

    def execute(self, context):
        skel = _active_skeleton(context)
        target = context.scene.mhfrt.target
        if skel is None or target is None:
            self.report({'ERROR'}, "Build the rig and set the Head Target first")
            return {'CANCELLED'}

        names = rest_tuning.EYE_BONES[self.side]
        missing = [name for name in names if skel.data.bones.get(name) is None]
        if missing:
            self.report({'ERROR'}, f"Eye bone missing: {missing[0]}")
            return {'CANCELLED'}

        state = _capture_context(context, skel)
        rest_tuning.snapshot_bones(skel, names)
        committed = False
        try:
            if not _prepare_armature_edit(context, skel):
                self.report({'ERROR'}, "Could not enter armature Edit Mode")
                return {'CANCELLED'}
            edit_bones = skel.data.edit_bones
            root = edit_bones.get(names[0])
            cursor_local = skel.matrix_world.inverted() @ context.scene.cursor.location
            delta = cursor_local - root.head
            for name in names:
                bone = edit_bones.get(name)
                bone.head += delta
                bone.tail += delta
            bpy.ops.object.mode_set(mode='OBJECT')
            rest_tuning.commit_snapshots(skel, names)
            committed = True

            flag = (rest_tuning.EYE_LEFT_DONE_PROP
                    if self.side == 'LEFT'
                    else rest_tuning.EYE_RIGHT_DONE_PROP)
            skel[flag] = True
            error = _refresh_live_rig(context, skel)
            if error:
                self.report({'WARNING'}, f"Eye placed, rig refresh warning: {error}")
            else:
                self.report(
                    {'INFO'},
                    f"Character {self.side.lower()} eye pivot placed at the 3D Cursor",
                )
            return {'FINISHED'}
        finally:
            if not committed:
                if skel.mode == 'EDIT':
                    rest_tuning.restore_snapshots_edit_mode(skel, names)
                    bpy.ops.object.mode_set(mode='OBJECT')
                rest_tuning.clear_snapshots(skel, names)
            _restore_context(context, state, skel)
            # Cursor-to-Selected is commonly used for both eyes in sequence.
            # Clear the first eye selection so pressing L over the second eye
            # cannot accidentally average both centers together.
            if (context.view_layer.objects.active == target
                    and target.mode == 'EDIT'):
                try:
                    bpy.ops.mesh.select_all(action='DESELECT')
                except RuntimeError:
                    pass


def _save_tongue_session(context, skel, names):
    mh = context.scene.mhfrt
    active = context.view_layer.objects.active
    skel[SESSION_ACTIVE_PROP] = True
    skel[SESSION_TONGUE_NAMES_PROP] = _join_names(names)
    if mh.cage is not None:
        skel[SESSION_CAGE_PROP] = mh.cage
    if mh.target is not None:
        skel[SESSION_TARGET_PROP] = mh.target
    skel[SESSION_CURSOR_PROP] = list(context.scene.cursor.location)
    skel[SESSION_PIVOT_PROP] = context.scene.tool_settings.transform_pivot_point
    skel[SESSION_DISPLAY_PROP] = skel.data.display_type
    skel[SESSION_IN_FRONT_PROP] = bool(skel.show_in_front)
    skel[SESSION_HIDE_VIEWPORT_PROP] = bool(skel.hide_viewport)
    skel[SESSION_SHOW_BONES_PROP] = bool(mh.show_bones)
    try:
        skel[SESSION_ORIENTATION_PROP] = context.scene.transform_orientation_slots[0].type
    except (AttributeError, IndexError):
        skel[SESSION_ORIENTATION_PROP] = 'GLOBAL'
    skel[SESSION_SOLOS_PROP] = _join_names(
        coll.name for coll in skel.data.collections_all if coll.is_solo)
    tongue_coll = _bone_collection(skel, "region:tongue")
    skel[SESSION_TONGUE_VISIBLE_PROP] = bool(
        tongue_coll.is_visible if tongue_coll else True)
    skel[SESSION_HIDDEN_BONES_PROP] = _join_names(
        name for name in names
        if (bone := skel.data.bones.get(name)) is not None and bone.hide)
    skel[SESSION_ACTIVE_OBJECT_PROP] = active.name if active else ""
    skel[SESSION_SELECTED_PROP] = _join_names(
        obj.name for obj in context.selected_objects)
    skel[SESSION_MODE_PROP] = active.mode if active else 'OBJECT'

    source_bones = (skel.data.edit_bones
                    if active == skel and active.mode == 'EDIT'
                    else skel.data.bones)
    skel[SESSION_ARMATURE_BONES_PROP] = _join_names(
        bone.name for bone in source_bones)
    skel[SESSION_TONGUE_PARENTS_PROP] = _SEP.join(
        (source_bones.get(name).parent.name
         if source_bones.get(name) is not None
         and source_bones.get(name).parent is not None else "")
        for name in names
    )
    skel[SESSION_TONGUE_CONNECTED_PROP] = _join_names(
        name for name in names
        if source_bones.get(name) is not None
        and source_bones.get(name).use_connect)
    skel[SESSION_TONGUE_DEFORM_PROP] = _join_names(
        name for name in names
        if source_bones.get(name) is not None
        and source_bones.get(name).use_deform)

    selected_bones = []
    selected_heads = []
    selected_tails = []
    active_bone = ""
    if active == skel and active.mode == 'EDIT':
        selected_bones = [bone.name for bone in skel.data.edit_bones
                          if bone.select]
        selected_heads = [bone.name for bone in skel.data.edit_bones
                          if bone.select_head]
        selected_tails = [bone.name for bone in skel.data.edit_bones
                          if bone.select_tail]
        if skel.data.edit_bones.active is not None:
            active_bone = skel.data.edit_bones.active.name
    elif active == skel and active.mode == 'POSE':
        selected_bones = [bone.name for bone in skel.data.bones
                          if bone.select]
        if skel.data.bones.active is not None:
            active_bone = skel.data.bones.active.name
    skel[SESSION_SELECTED_BONES_PROP] = _join_names(selected_bones)
    skel[SESSION_SELECTED_HEADS_PROP] = _join_names(selected_heads)
    skel[SESSION_SELECTED_TAILS_PROP] = _join_names(selected_tails)
    skel[SESSION_ACTIVE_BONE_PROP] = active_bone


def _restore_tongue_session(context, skel):
    mh = context.scene.mhfrt
    _mode_object(context)

    saved_solos = set(_split_names(skel.get(SESSION_SOLOS_PROP)))
    for coll in skel.data.collections_all:
        coll.is_solo = coll.name in saved_solos
    tongue_coll = _bone_collection(skel, "region:tongue")
    if tongue_coll is not None:
        tongue_coll.is_visible = bool(
            skel.get(SESSION_TONGUE_VISIBLE_PROP, True))

    hidden = set(_split_names(skel.get(SESSION_HIDDEN_BONES_PROP)))
    for name in _session_tongue_bones(skel):
        bone = skel.data.bones.get(name)
        if bone is not None:
            bone.hide = name in hidden

    skel.data.display_type = str(
        skel.get(SESSION_DISPLAY_PROP, skel.data.display_type))
    skel.show_in_front = bool(skel.get(SESSION_IN_FRONT_PROP, True))
    skel.hide_viewport = bool(skel.get(SESSION_HIDE_VIEWPORT_PROP, False))
    cursor = skel.get(SESSION_CURSOR_PROP)
    if cursor is not None and len(cursor) == 3:
        context.scene.cursor.location = cursor
    pivot = str(skel.get(SESSION_PIVOT_PROP, 'MEDIAN_POINT'))
    try:
        context.scene.tool_settings.transform_pivot_point = pivot
    except (TypeError, ValueError):
        pass
    try:
        context.scene.transform_orientation_slots[0].type = str(
            skel.get(SESSION_ORIENTATION_PROP, 'GLOBAL'))
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    for obj in list(context.selected_objects):
        obj.select_set(False)
    for name in _split_names(skel.get(SESSION_SELECTED_PROP)):
        obj = context.view_layer.objects.get(name)
        if obj is not None:
            obj.select_set(True)
    active = context.view_layer.objects.get(
        str(skel.get(SESSION_ACTIVE_OBJECT_PROP, "")))
    restored_mode = 'OBJECT'
    if active is not None:
        context.view_layer.objects.active = active
        restored_mode = str(skel.get(SESSION_MODE_PROP, 'OBJECT'))
        if restored_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode=restored_mode)
            except RuntimeError:
                pass

    selected_bones = set(_split_names(skel.get(SESSION_SELECTED_BONES_PROP)))
    selected_heads = set(_split_names(skel.get(SESSION_SELECTED_HEADS_PROP)))
    selected_tails = set(_split_names(skel.get(SESSION_SELECTED_TAILS_PROP)))
    active_bone_name = str(skel.get(SESSION_ACTIVE_BONE_PROP, ""))
    if active == skel and restored_mode == 'EDIT' and skel.mode == 'EDIT':
        for bone in skel.data.edit_bones:
            bone.select = bone.name in selected_bones
            bone.select_head = bone.name in selected_heads
            bone.select_tail = bone.name in selected_tails
        skel.data.edit_bones.active = skel.data.edit_bones.get(active_bone_name)
    elif active == skel and restored_mode == 'POSE' and skel.mode == 'POSE':
        for bone in skel.data.bones:
            bone.select = bone.name in selected_bones
        skel.data.bones.active = skel.data.bones.get(active_bone_name)

    mh.show_bones = bool(skel.get(SESSION_SHOW_BONES_PROP, True))
    try:
        context.workspace.status_text_set(None)
    except AttributeError:
        pass
    for prop in _SESSION_PROPS:
        _delete_prop(skel, prop)


def _select_tongue_for_edit(context, skel):
    mh = context.scene.mhfrt
    mh.show_bones = True
    skel.show_in_front = True
    skel.data.display_type = 'OCTAHEDRAL'
    skel.data.show_bone_colors = True

    tongue_coll = _bone_collection(skel, "region:tongue")
    if tongue_coll is None:
        # the region collections are missing entirely - rebuild them
        organization.organize_skeleton(skel, force=True)
        tongue_coll = _bone_collection(skel, "region:tongue")
    for coll in skel.data.collections_all:
        coll.is_solo = False
    if tongue_coll is not None:
        tongue_coll.is_visible = True
        tongue_coll.is_solo = True

    if not _prepare_armature_edit(context, skel):
        return False
    edit_bones = skel.data.edit_bones
    for bone in edit_bones:
        bone.select = False
        bone.select_head = False
        bone.select_tail = False
    names = _session_tongue_bones(skel)
    for name in names:
        bone = edit_bones.get(name)
        if bone is None:
            continue
        bone.hide = False
        bone.select = True
        bone.select_head = True
        bone.select_tail = True
    root = edit_bones.get(names[0]) if names else None
    if root is None:
        return False
    edit_bones.active = root
    context.scene.cursor.location = skel.matrix_world @ root.head
    context.scene.tool_settings.transform_pivot_point = 'CURSOR'
    try:
        context.scene.transform_orientation_slots[0].type = 'GLOBAL'
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        context.workspace.status_text_set(
            "Tongue: S scale / R rotate / G move | Finish or Cancel in MH Transfer")
    except AttributeError:
        pass
    return True


def _tongue_lengths_valid(skel):
    bones = skel.data.edit_bones if skel.mode == 'EDIT' else skel.data.bones
    return all(
        (bone := bones.get(name)) is not None and bone.length >= 1e-5
        for name in _session_tongue_bones(skel)
    )


def _tongue_structure_valid(skel):
    bones = skel.data.edit_bones if skel.mode == 'EDIT' else skel.data.bones
    expected_names = set(_split_names(skel.get(SESSION_ARMATURE_BONES_PROP)))
    if not expected_names or {bone.name for bone in bones} != expected_names:
        return False

    names = _session_tongue_bones(skel)
    expected_parents = str(
        skel.get(SESSION_TONGUE_PARENTS_PROP, "")).split(_SEP)
    if len(expected_parents) != len(names):
        # A session opened by a version whose tongue list was a different
        # length. Its parent record cannot be lined up with these names, so
        # the check is unverifiable - and an unverifiable check must not be a
        # failing one, or the artist is left with a session that can neither
        # be finished nor cancelled.
        return True
    connected = set(_split_names(skel.get(SESSION_TONGUE_CONNECTED_PROP)))
    deform = set(_split_names(skel.get(SESSION_TONGUE_DEFORM_PROP)))
    for index, name in enumerate(names):
        bone = bones.get(name)
        data_bone = skel.data.bones.get(name)
        if (bone is None or data_bone is None
                or rest_tuning.BACKUP_MATRIX_PROP not in data_bone):
            return False
        parent_name = bone.parent.name if bone.parent is not None else ""
        if parent_name != expected_parents[index]:
            return False
        if bool(bone.use_connect) != (name in connected):
            return False
        if bool(bone.use_deform) != (name in deform):
            return False
    return True


class MHFRT_OT_tune_tongue(bpy.types.Operator):
    bl_idname = "mhfrt.tune_tongue"
    bl_label = "Tune Tongue"
    bl_description = ("Isolate every purple tongue bone, all of them already "
                      "selected, for safe rest-pose Grab, Rotate and Scale")
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(items=[
        ('START', "Start", "Start a tongue editing session"),
        ('RESUME', "Resume", "Return to the active tongue editing session"),
        ('PIVOT', "Pivot to Base",
         "Put the rotate/scale pivot back on the root of the tongue, and "
         "select every tongue bone again"),
        ('FINISH', "Finish", "Save tongue placement and refresh RigLogic"),
        ('CANCEL', "Cancel", "Restore the tongue rest pose from before this session"),
    ], default='START')

    @classmethod
    def poll(cls, context):
        return _active_skeleton(context) is not None

    def execute(self, context):
        from .op_live import stop_running

        skel = _active_skeleton(context)
        if skel is None:
            self.report({'ERROR'}, "Build the rig first")
            return {'CANCELLED'}
        active_session = bool(skel.get(SESSION_ACTIVE_PROP))
        new_session = False

        if self.action == 'START':
            if active_session:
                self.action = 'RESUME'
            else:
                names = _tongue_bones(skel)
                if not names or names[0] != rest_tuning.TONGUE_BONES[0]:
                    self.report(
                        {'ERROR'},
                        "This skeleton has no tongue root "
                        f"('{rest_tuning.TONGUE_BONES[0]}') to tune")
                    return {'CANCELLED'}
                stop_running()
                _save_tongue_session(context, skel, names)
                # Bone ID properties written while this same armature is in
                # Edit Mode can be discarded when Blender syncs EditBones back
                # to Bones. Exit only after saving the user's bone selection.
                _mode_object(context)
                rest_tuning.snapshot_bones(skel, names)
                active_session = True
                new_session = True

        if self.action == 'RESUME' or (self.action == 'START' and active_session):
            if not active_session:
                self.report({'ERROR'}, "No tongue edit session to resume")
                return {'CANCELLED'}
            names = _session_tongue_bones(skel)
            if not _select_tongue_for_edit(context, skel):
                if new_session:
                    _mode_object(context)
                    rest_tuning.clear_snapshots(skel, names)
                    _restore_tongue_session(context, skel)
                self.report({'ERROR'}, "Could not enter tongue Edit Mode")
                return {'CANCELLED'}
            self.report({'INFO'},
                        f"{len(names)} purple tongue bones selected: "
                        "S scale, R rotate, G move")
            return {'FINISHED'}

        if self.action == 'PIVOT':
            if not active_session:
                self.report({'ERROR'}, "Start tongue editing first")
                return {'CANCELLED'}
            # Re-select as well as re-pivot. Whatever the artist clicked on the
            # way here, the tongue is moved as one piece, so this button is the
            # way back to that state - not a second thing to remember after it.
            if not _select_tongue_for_edit(context, skel):
                return {'CANCELLED'}
            self.report({'INFO'},
                        "Every tongue bone selected, pivot back on its base")
            return {'FINISHED'}

        if self.action == 'FINISH':
            if not active_session:
                self.report({'ERROR'}, "No tongue edit session to finish")
                return {'CANCELLED'}
            if not _tongue_structure_valid(skel):
                self.report(
                    {'ERROR'},
                    "Tongue structure changed; press Ctrl+Z until all bones "
                    "return, then Finish or Cancel",
                )
                return {'CANCELLED'}
            if not _tongue_lengths_valid(skel):
                self.report(
                    {'ERROR'},
                    "A tongue bone is collapsed; Undo/scale it up, or Cancel",
                )
                return {'CANCELLED'}
            count = 0
            error = None
            committed = False
            # Read before _restore_tongue_session, which deletes the session
            # properties this list lives in.
            names = _session_tongue_bones(skel)
            _mode_object(context)
            try:
                count = rest_tuning.commit_snapshots(skel, names)
                committed = True
                skel[rest_tuning.TONGUE_DONE_PROP] = True
                error = _refresh_live_rig(context, skel)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if not committed:
                    rest_tuning.clear_snapshots(skel, names)
                _restore_tongue_session(context, skel)
            if error:
                self.report({'WARNING'}, f"Tongue saved, rig refresh warning: {error}")
            else:
                self.report({'INFO'}, f"Tongue placement saved on {count} bones")
            return {'FINISHED'}

        if self.action == 'CANCEL':
            if not active_session:
                self.report({'ERROR'}, "No tongue edit session to cancel")
                return {'CANCELLED'}
            if not _tongue_structure_valid(skel):
                self.report(
                    {'ERROR'},
                    "Tongue structure changed; press Ctrl+Z until all bones "
                    "return, then Cancel",
                )
                return {'CANCELLED'}
            names = _session_tongue_bones(skel)
            if not _prepare_armature_edit(context, skel):
                self.report({'ERROR'}, "Could not restore tongue Edit Mode; try Cancel again")
                return {'CANCELLED'}
            error = None
            try:
                rest_tuning.restore_snapshots_edit_mode(skel, names)
                bpy.ops.object.mode_set(mode='OBJECT')
                rest_tuning.clear_snapshots(skel, names)
                error = _refresh_live_rig(context, skel)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                _mode_object(context)
                rest_tuning.clear_snapshots(skel, names)
                _restore_tongue_session(context, skel)
            if error:
                self.report({'WARNING'}, f"Tongue restored, rig refresh warning: {error}")
            else:
                self.report({'INFO'}, "Tongue edit cancelled; original placement restored")
            return {'FINISHED'}

        return {'CANCELLED'}


class MHFRT_OT_reset_rest_tuning(bpy.types.Operator):
    bl_idname = "mhfrt.reset_rest_tuning"
    bl_label = "Reset Placement"
    bl_description = "Remove this correction and restore its latest automatic cage fit"
    bl_options = {'REGISTER', 'UNDO'}

    feature: bpy.props.EnumProperty(items=[
        ('EYE_LEFT', "Left Eye", "Reset the character's left eye"),
        ('EYE_RIGHT', "Right Eye", "Reset the character's right eye"),
        ('TONGUE', "Tongue", "Reset all tongue bones"),
    ])

    @classmethod
    def poll(cls, context):
        skel = _active_skeleton(context)
        return bool(skel and not skel.get(SESSION_ACTIVE_PROP))

    def execute(self, context):
        skel = _active_skeleton(context)
        if self.feature == 'EYE_LEFT':
            names = rest_tuning.EYE_BONES['LEFT']
            flag = rest_tuning.EYE_LEFT_DONE_PROP
        elif self.feature == 'EYE_RIGHT':
            names = rest_tuning.EYE_BONES['RIGHT']
            flag = rest_tuning.EYE_RIGHT_DONE_PROP
        else:
            names = rest_tuning.TONGUE_BONES
            flag = rest_tuning.TONGUE_DONE_PROP

        state = _capture_context(context, skel)
        if not _prepare_armature_edit(context, skel):
            _restore_context(context, state, skel)
            self.report({'ERROR'}, "Could not enter armature Edit Mode")
            return {'CANCELLED'}
        error = None
        count = 0
        try:
            count = rest_tuning.reset_manual_tuning_edit_mode(skel, names)
            bpy.ops.object.mode_set(mode='OBJECT')
            _delete_prop(skel, flag)
            _delete_prop(skel, rest_tuning.TUNE_DONE_PROP)
            error = _refresh_live_rig(context, skel)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _restore_context(context, state, skel)
        if error:
            self.report({'WARNING'}, f"Placement reset, rig refresh warning: {error}")
        else:
            self.report({'INFO'}, f"Restored automatic placement on {count} bone(s)")
        return {'FINISHED'}


class MHFRT_OT_finish_rest_tuning(bpy.types.Operator):
    bl_idname = "mhfrt.finish_rest_tuning"
    bl_label = "Confirm Positions & Continue"
    bl_description = ("Accept the current automatic/manual positions and continue; "
                      "eye and tongue adjustments are optional")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        skel = _active_skeleton(context)
        return bool(skel and not skel.get(SESSION_ACTIVE_PROP))

    def execute(self, context):
        skel = _active_skeleton(context)
        _mode_object(context)
        skel[rest_tuning.TUNE_DONE_PROP] = True
        from ..ui import flow
        flow.advance(context, 'TUNE')
        self.report({'INFO'}, "Eye and tongue positions confirmed")
        return {'FINISHED'}


_classes = (
    MHFRT_OT_place_eye_center,
    MHFRT_OT_tune_tongue,
    MHFRT_OT_reset_rest_tuning,
    MHFRT_OT_finish_rest_tuning,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
