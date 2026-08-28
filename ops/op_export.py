"""Game-engine FBX export: one skeleton and baked facial correctives.

Blender drivers and the add-on's Python RigLogic runtime do not travel through
FBX. The portable result is the evaluated animation: bone transforms plus FBX
blend-shape DeformPercent curves. This module creates temporary visual bakes
for every character clip, writes the Unity or Unreal delivery layout, then
restores the artist's file exactly.

Nothing here assumes Rigify. Source clips are ordinary armature Actions/NLA
strips and control-board Actions discovered from their real data paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
import traceback
import uuid

import bpy
from bpy_extras import anim_utils
from bpy_extras.io_utils import ExportHelper

from ..core import board
from ..core import organization
from ..props import (RIG_ID_PROP, RIG_CAGE_PROP, RIG_TARGET_PROP,
                     RIG_GUI_COLL_PROP, RIG_MERGED_PROP,
                     RIG_DESTINATION_PROP)
from . import op_anim, op_rig


TEMP_ACTION_PROP = "mhfrt_unity_export_temp"
TEMP_TRACK_PREFIX = "__MHFRT_Export_"
_FACE_PREFIXES = ("FaceAnim_", "Face_", "Facial_")
_FACE_SUFFIXES = ("_Bake", "_Face", "_Facial")
_OBJECT_CHANNELS = {
    "location", "rotation_euler", "rotation_quaternion",
    "rotation_axis_angle", "scale",
}


@dataclass
class BodyClip:
    name: str
    action: bpy.types.Action
    slot: object
    frame_start: int
    frame_end: int
    track: object = None
    strip: object = None


@dataclass
class FaceClip:
    name: str
    # (control object, location array index) -> source FCurve
    curves: dict
    frame_start: float
    frame_end: float
    action: bpy.types.Action | None = None


@dataclass
class ExportClip:
    name: str
    body: BodyClip | None
    face: FaceClip | None
    frame_start: int
    frame_end: int


def _channelbags(action):
    if action is None:
        return
    seen = set()
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for bag in strip.channelbags:
                pointer = bag.as_pointer()
                if pointer not in seen:
                    seen.add(pointer)
                    yield bag


def _bag_fcurves(action, slot):
    if action is None or slot is None:
        return ()
    bag = op_anim._channelbag(action, slot)
    return tuple(bag.fcurves) if bag is not None else ()


def _path_resolves(owner, fcurve):
    path = fcurve.data_path
    if fcurve.array_index:
        path = f"{path}[{fcurve.array_index}]"
    try:
        owner.path_resolve(path)
        return True
    except (ValueError, TypeError):
        return False


def _action_slot_for(action, owner):
    """Find the layered Action slot whose channels really target ``owner``."""
    ad = owner.animation_data
    if ad is not None and ad.action == action and ad.action_slot is not None:
        return ad.action_slot
    for bag in _channelbags(action):
        curves = tuple(bag.fcurves)
        if curves and all(_path_resolves(owner, fc) for fc in curves):
            return bag.slot
    return None


def _body_action_paths(action, slot):
    return {fc.data_path for fc in _bag_fcurves(action, slot)}


def _meaningful_body_action(action, skel, slot=None):
    if action is None or action.get(TEMP_ACTION_PROP):
        return False
    if action == op_anim.face_bake_action(skel):
        return False
    slot = slot or _action_slot_for(action, skel)
    if slot is None:
        return False
    paths = _body_action_paths(action, slot)
    if any(path.startswith('pose.bones["') for path in paths):
        return True
    # Root-only actions and custom rig-property animation are valid only when
    # the slot itself belongs to this armature. This prevents an unrelated
    # object's generic location action from becoming a character clip.
    display = str(getattr(slot, "name_display", ""))
    if display == skel.name:
        return any(
            path in _OBJECT_CHANNELS
            or (path.startswith('["')
                and op_anim.BAKE_MARK_PROP not in path)
            for path in paths
        )
    return False


def _integer_range(frame_range):
    start, end = frame_range
    return int(math.floor(float(start))), int(math.ceil(float(end)))


def discover_body_clips(skel):
    """All armature NLA clips plus assigned/unassigned compatible Actions."""
    clips = []
    used_actions = set()
    ad = skel.animation_data

    if ad is not None:
        for track in ad.nla_tracks:
            for strip in track.strips:
                action = strip.action
                slot = getattr(strip, "action_slot", None)
                if not _meaningful_body_action(action, skel, slot):
                    continue
                start, end = _integer_range((strip.frame_start,
                                             strip.frame_end))
                clips.append(BodyClip(
                    strip.name or action.name,
                    action,
                    slot,
                    start,
                    max(start + 1, end),
                    track,
                    strip,
                ))
                used_actions.add(action.as_pointer())

        action = ad.action
        slot = ad.action_slot
        if (_meaningful_body_action(action, skel, slot)
                and action.as_pointer() not in used_actions):
            start, end = _integer_range(action.frame_range)
            clips.append(BodyClip(
                action.name, action, slot, start, max(start + 1, end)))
            used_actions.add(action.as_pointer())

    for action in sorted(bpy.data.actions, key=lambda item: item.name.casefold()):
        if action.as_pointer() in used_actions:
            continue
        slot = _action_slot_for(action, skel)
        if not _meaningful_body_action(action, skel, slot):
            continue
        start, end = _integer_range(action.frame_range)
        clips.append(BodyClip(
            action.name, action, slot, start, max(start + 1, end)))
        used_actions.add(action.as_pointer())
    return clips


def _control_maps(skel):
    """(controls, by object name, by template name, by bone name).

    A bone board's controls all share one armature and one action slot, so they
    are told apart by the fcurve data path instead - hence the bone-name map.
    """
    rid = str(skel.get(RIG_ID_PROP) or "")
    by_template = board.controls_for_rig(RIG_ID_PROP, rid, skel=skel)
    controls = list(by_template.values())
    by_name = {control.name: control for control in controls
               if not board.is_pose_bone(control)}
    by_bone = {control.name: control for control in controls
               if board.is_pose_bone(control)}
    return controls, by_name, by_template, by_bone


def _slot_owner_objects(controls):
    """The objects that can carry a board action, deduplicated."""
    owners = {}
    for control in controls:
        owner = board.owner_object(control)
        if owner is not None:
            owners[owner.name] = owner
    return owners


def _face_curves_for_action(action, controls, by_name, by_template, by_bone):
    """{(control, location index): fcurve} for the board curves in one action.

    A loose-object board is matched slot by slot (one slot per control object);
    a bone board shares a single armature slot, so its curves are matched by the
    bone named in each data path - but only inside a slot that belongs to this
    rig's board, so an unrelated armature with similar bone names is ignored.
    """
    curves = {}
    owner_names = set(_slot_owner_objects(controls))
    for bag in _channelbags(action):
        slot = bag.slot
        display = str(getattr(slot, "name_display", ""))
        obj = by_name.get(display) or by_template.get(display)
        if obj is None and by_name:
            # The assigned slot remains authoritative after an object rename,
            # even if the slot's display name was not edited.
            obj = next(
                (candidate for candidate in controls
                 if not board.is_pose_bone(candidate)
                 and candidate.animation_data is not None
                 and candidate.animation_data.action == action
                 and candidate.animation_data.action_slot == slot),
                None,
            )
        bones = by_bone if display in owner_names else {}
        if obj is None and not bones:
            continue
        for fc in bag.fcurves:
            control = board.curve_control(
                obj, fc.data_path, int(fc.array_index), bones)
            if control is not None:
                curves[(control, int(fc.array_index))] = fc
    return curves


def discover_face_clips(skel):
    """Control-board Actions, including a composite of hand-keyed controls."""
    controls, by_name, by_template, by_bone = _control_maps(skel)
    if not controls:
        return [], controls

    clips = []
    for action in bpy.data.actions:
        if action.get(TEMP_ACTION_PROP):
            continue
        curves = _face_curves_for_action(
            action, controls, by_name, by_template, by_bone)
        if not curves:
            continue
        starts = []
        ends = []
        for fc in curves.values():
            if not fc.keyframe_points:
                continue
            lo, hi = fc.range()
            starts.append(float(lo))
            ends.append(float(hi))
        if starts:
            clips.append(FaceClip(
                action.name, curves, min(starts), max(ends), action))

    # Hand-keying several board objects normally creates one Action per object.
    # Their simultaneously assigned actions are one visible clip, not a dozen
    # unrelated takes, so combine them into a current-board source. A bone board
    # has a single owner, so this composite pass simply finds one action.
    active_curves = {}
    active_actions = set()
    for owner in _slot_owner_objects(controls).values():
        ad = owner.animation_data
        action = ad.action if ad else None
        slot = ad.action_slot if ad else None
        if action is None or slot is None:
            continue
        obj = None if board.is_board_armature(owner) else owner
        for fc in _bag_fcurves(action, slot):
            control = board.curve_control(
                obj, fc.data_path, int(fc.array_index), by_bone)
            if control is not None:
                active_curves[(control, int(fc.array_index))] = fc
                active_actions.add(action.as_pointer())
    if len(active_actions) > 1 and active_curves:
        starts, ends = [], []
        for fc in active_curves.values():
            if fc.keyframe_points:
                lo, hi = fc.range()
                starts.append(float(lo))
                ends.append(float(hi))
        if starts:
            # The individual Actions are only implementation details of
            # Blender's one-Action-per-object animation model. Exporting both
            # those pieces and the combined visible board state would create
            # duplicate, partial engine clips.
            clips = [
                clip for clip in clips
                if clip.action is None
                or clip.action.as_pointer() not in active_actions
            ]
            clips.append(FaceClip(
                "Board_Current", active_curves, min(starts), max(ends)))

    return clips, controls


def _pair_key(name):
    value = str(name or "").split("|")[-1].strip()
    changed = True
    while changed:
        changed = False
        for prefix in _FACE_PREFIXES:
            if value.casefold().startswith(prefix.casefold()):
                value = value[len(prefix):]
                changed = True
        for suffix in _FACE_SUFFIXES:
            if value.casefold().endswith(suffix.casefold()):
                value = value[:-len(suffix)]
                changed = True
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _display_clip_name(name):
    value = str(name or "Animation").split("|")[-1].strip()
    for prefix in _FACE_PREFIXES:
        if value.casefold().startswith(prefix.casefold()):
            value = value[len(prefix):]
            break
    for suffix in _FACE_SUFFIXES:
        if value.casefold().endswith(suffix.casefold()):
            value = value[:-len(suffix)]
            break
    value = re.sub(r"[^A-Za-z0-9_. -]+", "_", value).strip(" ._")
    return value or "Animation"


def pair_export_clips(body_clips, face_clips):
    """Pair body and board clips by name; keep every unmatched clip."""
    unused_faces = set(range(len(face_clips)))
    pairs = []
    for body in body_clips:
        key = _pair_key(body.name)
        matches = [index for index in unused_faces
                   if _pair_key(face_clips[index].name) == key and key]
        face = None
        if len(matches) == 1:
            index = matches[0]
            face = face_clips[index]
            unused_faces.remove(index)
        pairs.append([body, face])

    unmatched_body = [item for item in pairs if item[1] is None]
    # One body + one face is unambiguous even when their artist-facing names
    # differ (the common "Action" + imported FaceAnim_take case).
    if len(unmatched_body) == 1 and len(unused_faces) == 1:
        index = next(iter(unused_faces))
        unmatched_body[0][1] = face_clips[index]
        unused_faces.remove(index)

    for index in sorted(unused_faces):
        pairs.append([None, face_clips[index]])

    out = []
    used_names = set()
    for body, face in pairs:
        base = _display_clip_name(body.name if body else face.name)
        name = base
        suffix = 2
        while name.casefold() in used_names:
            name = f"{base}_{suffix:02d}"
            suffix += 1
        used_names.add(name.casefold())

        if body is not None:
            start = body.frame_start
            body_count = body.frame_end - body.frame_start + 1
        else:
            start = 1
            body_count = 0
        face_count = 0
        if face is not None:
            face_count = int(math.ceil(face.frame_end - face.frame_start)) + 1
        count = max(2, body_count, face_count)
        out.append(ExportClip(name, body, face, start, start + count - 1))
    return out


def _object_reaches_armature(obj, skel):
    current = obj
    seen = set()
    while current is not None and current.as_pointer() not in seen:
        if current == skel:
            return True
        seen.add(current.as_pointer())
        current = current.parent
    return False


def body_rigs(context, skel):
    """Armatures of this character that are NOT the one being exported.

    Non-empty only while a body rig is still separate from the facial one.  A
    game engine wants a character on ONE skeleton, so this is the signal that
    the Merge step has not been done yet - not something to quietly export as
    a second hierarchy.
    """
    target = skel.get(RIG_TARGET_PROP)
    if not isinstance(target, bpy.types.Object):
        target = None
    scene_objects = set(context.scene.objects)
    return [armature
            for armature in organization.character_armatures(
                skel, target, scene=context.scene)[1:]
            if armature in scene_objects]


def export_objects(context, skel):
    """Final meshes driven by the character rig, excluding MHFRT helpers.

    "Driven by" now includes the body and clothing the character arrived with:
    once the rigs are merged they share `skel`, so the same modifier test finds
    them - but a mesh parented to a bone rather than skinned, which real rigs
    do for props, was being dropped, and so was anything the artist had bound
    before the face was ever transferred.  Both come from the character scan.
    """
    cage = skel.get(RIG_CAGE_PROP)
    panel_objects = set()
    gui_coll = skel.get(RIG_GUI_COLL_PROP)
    if isinstance(gui_coll, bpy.types.Collection):
        panel_objects.update(obj.as_pointer() for obj in gui_coll.all_objects)
    # Keep the fallback collection-role scan for appended/older files whose
    # armature has lost the direct GUI collection pointer.
    for coll in bpy.data.collections:
        if coll.get(organization.COLL_ROLE_PROP) == organization.ROLE_PANEL:
            panel_objects.update(
                obj.as_pointer() for obj in coll.all_objects)

    out = [skel]
    for obj in context.scene.objects:
        if obj.type != 'MESH' or obj == cage:
            continue
        if (obj.as_pointer() in panel_objects
                or board.tag(obj, board.TEMPLATE_PROP)):
            continue
        role = str(obj.get(organization.OBJECT_ROLE_PROP, ""))
        if role in {organization.ROLE_CAGE, organization.ROLE_PANEL}:
            continue
        armature_driven = any(
            mod.type == 'ARMATURE' and mod.object == skel
            for mod in obj.modifiers
        )
        if armature_driven or _object_reaches_armature(obj, skel):
            out.append(obj)

    target = skel.get(RIG_TARGET_PROP)
    if not isinstance(target, bpy.types.Object):
        target = None
    known = {obj.as_pointer() for obj in out}
    scene_objects = set(context.scene.objects)
    for obj in organization.character_meshes([skel], cage=cage, target=target,
                                             scene=context.scene):
        if obj.as_pointer() not in known and obj in scene_objects:
            known.add(obj.as_pointer())
            out.append(obj)
    return out


def _shape_coords(key):
    import numpy as np
    values = np.empty(len(key.data) * 3, dtype=np.float32)
    key.data.foreach_get("co", values)
    return values


def _authored_shape_targets(skel):
    target = skel.get(RIG_TARGET_PROP)
    objects = op_rig._shape_key_objects(skel, target)
    targets = op_rig._shape_key_targets(skel, target, objects)
    relative_cache = {}
    out = []
    for key, input_index in targets:
        shape_keys = key.id_data
        if not shape_keys.use_relative:
            continue
        relative = key.relative_key
        if relative is None or relative == key:
            continue
        pointer = relative.as_pointer()
        basis = relative_cache.get(pointer)
        if basis is None:
            basis = _shape_coords(relative)
            relative_cache[pointer] = basis
        if len(relative.data) != len(key.data):
            continue
        if not (basis == _shape_coords(key)).all():
            out.append((key, int(input_index)))
    return out


def _snapshot_animation(obj):
    ad = obj.animation_data
    return {
        "had": ad is not None,
        "action": ad.action if ad else None,
        "slot": ad.action_slot if ad else None,
        "tracks": [
            (track, bool(track.mute),
             [(strip, bool(strip.mute)) for strip in track.strips])
            for track in ad.nla_tracks
        ] if ad else [],
    }


def _restore_animation(obj, state):
    ad = obj.animation_data
    if state["had"]:
        if ad is None:
            obj.animation_data_create()
            ad = obj.animation_data
        ad.action = state["action"]
        if state["action"] is not None and state["slot"] is not None:
            ad.action_slot = state["slot"]
        for track, muted, strips in state["tracks"]:
            try:
                track.mute = muted
                for strip, strip_muted in strips:
                    strip.mute = strip_muted
            except ReferenceError:
                pass
    elif ad is not None and ad.action is None and not ad.nla_tracks:
        obj.animation_data_clear()


def _mute_nla(obj, source_track=None, source_strip=None):
    ad = obj.animation_data
    if ad is None:
        return
    for track in ad.nla_tracks:
        track.mute = track != source_track
        for strip in track.strips:
            strip.mute = strip != source_strip


def _reset_pose(skel):
    for pbone in skel.pose.bones:
        pbone.matrix_basis.identity()


def _configure_body_source(skel, clip, base_matrix_basis):
    if skel.animation_data is None:
        skel.animation_data_create()
    ad = skel.animation_data
    _reset_pose(skel)
    skel.matrix_basis = base_matrix_basis.copy()
    if clip.body is None:
        ad.action = None
        _mute_nla(skel)
        return
    body = clip.body
    if body.strip is not None:
        ad.action = None
        _mute_nla(skel, body.track, body.strip)
    else:
        _mute_nla(skel)
        ad.action = body.action
        if body.slot is not None:
            ad.action_slot = body.slot


def _aligned_face_action(clip, controls):
    if clip.face is None:
        return None, {}
    action = bpy.data.actions.new(f"__MHFRT_ExportFace_{clip.name}")
    action[TEMP_ACTION_PROP] = True
    source = clip.face
    # Slots are per OWNER object: a bone board puts every control in the one
    # armature slot, told apart by its pose-bone data path.
    slots = {}
    bags = {}
    count = clip.frame_end - clip.frame_start + 1
    for control in controls:
        source_curves = {
            axis: source.curves[(control, axis)]
            for axis in board.animated_axes(control)
            if (control, axis) in source.curves
        }
        owner = board.owner_object(control)
        if not source_curves or owner is None:
            continue
        slot = slots.get(owner)
        if slot is None:
            slot = action.slots.new('OBJECT', owner.name)
            slots[owner] = slot
            bags[owner] = op_anim._ensure_action_channelbag(
                anim_utils, action, slot)
            op_anim._add_guard_curve(
                bags[owner], clip.frame_start, clip.frame_end)
        bag = bags[owner]
        data_path = board.location_data_path(control)
        for axis, source_curve in source_curves.items():
            fc = bag.fcurves.new(data_path=data_path, index=axis)
            fc.keyframe_points.add(count)
            co = []
            for offset in range(count):
                source_frame = min(
                    source.frame_end, source.frame_start + offset)
                co.extend((float(clip.frame_start + offset),
                           float(source_curve.evaluate(source_frame))))
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [1] * count)
            fc.update()
    return action, slots


def _assign_face_action(controls, action, slots):
    for control in controls:
        board.rest_pose(control)
    for owner in _slot_owner_objects(controls).values():
        slot = slots.get(owner)
        if owner.animation_data is None and slot is not None:
            owner.animation_data_create()
        ad = owner.animation_data
        if ad is None:
            continue
        _mute_nla(owner)
        ad.action = action if slot is not None else None
        if slot is not None:
            ad.action_slot = slot


def _rig_cache(skel):
    op_rig.invalidate()
    op_rig._rescan()
    cache = op_rig._caches.get(str(skel.get(RIG_ID_PROP) or ""))
    if cache is None:
        raise RuntimeError("The live facial rig cache could not be built")
    return cache


def _sample_shapes(scene, cache, targets, frame_start, frame_end):
    samples = [[] for _target in targets]
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        op_rig._evaluate_guarded([cache])
        for index, (key, _input_index) in enumerate(targets):
            samples[index].append(float(key.value))
    return samples


def _bake_visual_action(context, skel, clip):
    for obj in list(context.selected_objects):
        obj.select_set(False)
    skel.hide_viewport = False
    skel.hide_select = False
    try:
        skel.hide_set(False)
    except RuntimeError:
        pass
    skel.select_set(True)
    context.view_layer.objects.active = skel
    if context.object.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    result = bpy.ops.nla.bake(
        frame_start=clip.frame_start,
        frame_end=clip.frame_end,
        step=1,
        only_selected=False,
        visual_keying=True,
        clear_constraints=False,
        clear_parents=False,
        use_current_action=False,
        clean_curves=False,
        bake_types={'POSE', 'OBJECT'},
        channel_types={'LOCATION', 'ROTATION', 'SCALE'},
    )
    if 'FINISHED' not in result:
        raise RuntimeError(f"Could not visually bake clip '{clip.name}'")
    action = skel.animation_data.action
    slot = skel.animation_data.action_slot
    if action is None or slot is None:
        raise RuntimeError(f"Blender created no bake for '{clip.name}'")
    action.name = f"MHFRT_Export_{clip.name}"
    action[TEMP_ACTION_PROP] = True
    return action, slot


def _add_shape_channels(action, slot, properties, samples, frame_start):
    bag = op_anim._channelbag(action, slot)
    if bag is None:
        raise RuntimeError(f"No animation channel bag on '{action.name}'")
    for prop, values in zip(properties, samples):
        path = f'["{prop}"]'
        fc = bag.fcurves.new(data_path=path, index=0)
        fc.keyframe_points.add(len(values))
        co = []
        for offset, value in enumerate(values):
            co.extend((float(frame_start + offset), float(value)))
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set(
            "interpolation", [1] * len(values))
        fc.update()


def _existing_driver_paths(shape_keys):
    ad = shape_keys.animation_data
    return {fc.data_path for fc in ad.drivers} if ad else set()


def _add_shape_drivers(skel, targets, properties):
    created = []
    try:
        for (key, _input_index), prop in zip(targets, properties):
            shape_keys = key.id_data
            path = key.path_from_id("value")
            if path in _existing_driver_paths(shape_keys):
                # A native user/legacy driver already owns the key and will be
                # evaluated from the visually baked bones during FBX sampling.
                continue
            fc = shape_keys.driver_add(path)
            driver = fc.driver
            driver.type = 'SCRIPTED'
            variable = driver.variables.new()
            variable.name = "weight"
            variable.type = 'SINGLE_PROP'
            variable.targets[0].id = skel
            variable.targets[0].data_path = f'["{prop}"]'
            driver.expression = "weight"
            created.append((shape_keys, path))
    except Exception:
        _remove_shape_drivers(created)
        raise
    return created


def _remove_shape_drivers(created):
    for shape_keys, path in created:
        try:
            shape_keys.driver_remove(path)
        except (ReferenceError, RuntimeError):
            pass


def _create_export_tracks(skel, baked):
    if skel.animation_data is None:
        skel.animation_data_create()
    ad = skel.animation_data
    ad.action = None
    for track in ad.nla_tracks:
        track.mute = True
    tracks = []
    for clip, action in baked:
        start, end = _integer_range(action.frame_range)
        track = ad.nla_tracks.new()
        track.name = f"{TEMP_TRACK_PREFIX}{clip.name}"
        strip = track.strips.new(clip.name, start, action)
        strip.name = clip.name
        strip.action_frame_start = float(start)
        strip.action_frame_end = float(end)
        strip.frame_start = float(start)
        strip.frame_end = float(end)
        strip.mute = False
        track.mute = False
        tracks.append(track)
    return tracks


def _prepare_export_selection(context, objects):
    state = {}
    for obj in objects:
        state[obj] = (
            bool(obj.hide_get()), bool(obj.hide_viewport),
            bool(obj.hide_select), bool(obj.select_get()),
        )
        obj.hide_viewport = False
        obj.hide_select = False
        try:
            obj.hide_set(False)
        except RuntimeError:
            pass
    # Hidden board meshes can remain selected without appearing in
    # context.selected_objects. Clear the complete view-layer selection so
    # Blender's use_selection FBX option cannot pick up a hidden helper.
    for obj in context.view_layer.objects:
        if obj.select_get():
            obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    return state


def _restore_object_state(state):
    for obj, (hidden, hide_viewport, hide_select, selected) in state.items():
        try:
            obj.hide_viewport = hide_viewport
            obj.hide_select = hide_select
            obj.select_set(selected)
            obj.hide_set(hidden)
        except (ReferenceError, RuntimeError):
            pass


def _export_fbx(filepath, animated=True, profile='UNITY'):
    is_unreal = profile == 'UNREAL'
    common = dict(
        filepath=filepath,
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        apply_unit_scale=True,
        use_space_transform=True,
        bake_space_transform=False,
        use_mesh_modifiers=True,
        # Epic's importer warns when no smoothing data is present. Unity does
        # not need FBX smoothing groups, so preserve the existing lean profile.
        mesh_smooth_type='FACE' if is_unreal else 'OFF',
        use_tspace=False,
        use_custom_props=False,
        add_leaf_bones=False,
        use_armature_deform_only=True,
        armature_nodetype='NULL',
        # Unity is Y-up; Unreal is Z-up with the conventional Blender FBX
        # conversion used for its X-forward coordinate system.
        axis_forward='-Y' if is_unreal else '-Z',
        axis_up='Z' if is_unreal else 'Y',
        path_mode='AUTO',
        embed_textures=False,
    )
    if not animated:
        return bpy.ops.export_scene.fbx(bake_anim=False, **common)
    return bpy.ops.export_scene.fbx(
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=True,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        **common,
    )


def _safe_file_token(value):
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "Character"))
    return token.strip("_") or "Character"


def _unity_file_token(value):
    """Unity-friendly CamelCase filename with no spaces or punctuation."""
    parts = re.findall(r"[A-Za-z0-9]+", str(value or "Character"))
    if not parts:
        return "Character"
    if len(parts) == 1:
        return parts[0]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _check_fbx_result(result, filepath):
    if 'FINISHED' not in result:
        raise RuntimeError("Blender's FBX exporter did not finish")
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        raise RuntimeError(
            f"The FBX exporter produced no file: {os.path.basename(filepath)}")


def _neutral_export_state(context, skel, controls, targets, properties,
                          base_matrix_basis):
    """Put temporary export data at the neutral pose between FBX writes."""
    _reset_pose(skel)
    skel.matrix_basis = base_matrix_basis.copy()
    if skel.animation_data is not None:
        skel.animation_data.action = None
    _mute_nla(skel)
    _assign_face_action(controls, None, {})
    for prop in properties:
        skel[prop] = 0.0
    for key, _input in targets:
        key.mute = False
        key.value = 0.0
    context.view_layer.update()


def _unreal_animation_paths(filepath, baked_actions):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    character = _safe_file_token(stem)
    if character.casefold().startswith("sk_"):
        character = character[3:] or "Character"
    folder = os.path.join(
        os.path.dirname(filepath), f"{_safe_file_token(stem)}_Animations")
    paths = []
    used = set()
    for clip, action in baked_actions:
        # Epic's official Animation Sequence prefix is AS_. The Skeletal Mesh
        # prefix is deliberately removed from the shared character name:
        # SK_Hero.fbx -> AS_Hero_Walk.fbx.
        base = f"AS_{character}_{_safe_file_token(clip.name)}"
        token = base
        suffix = 2
        while token.casefold() in used:
            token = f"{base}_{suffix:02d}"
            suffix += 1
        used.add(token.casefold())
        paths.append((clip, action, os.path.join(folder, f"{token}.fbx")))
    return folder, paths


def _export_profile_files(context, profile, filepath, skel, objects, controls,
                          targets, properties, base_matrix_basis,
                          baked_actions, export_tracks):
    """Write Unity's multi-take FBX or Unreal's mesh + per-clip FBXs."""
    produced = []
    _neutral_export_state(
        context, skel, controls, targets, properties, base_matrix_basis)

    if profile == 'UNITY':
        if baked_actions:
            export_tracks.extend(
                _create_export_tracks(skel, baked_actions))
        result = _export_fbx(
            filepath, animated=bool(baked_actions), profile=profile)
        _check_fbx_result(result, filepath)
        return [filepath]

    # Epic's skeletal-mesh pipeline accepts every morph target in the base
    # asset, but its animation workflow is one animation per FBX. Keep the
    # skeletal mesh clean and write each evaluated clip beside it.
    result = _export_fbx(filepath, animated=False, profile=profile)
    _check_fbx_result(result, filepath)
    produced.append(filepath)

    if not baked_actions:
        return produced
    folder, animation_paths = _unreal_animation_paths(
        filepath, baked_actions)
    os.makedirs(folder, exist_ok=True)
    for clip, action, animation_path in animation_paths:
        tracks = _create_export_tracks(skel, [(clip, action)])
        export_tracks.extend(tracks)
        context.scene.frame_start = clip.frame_start
        context.scene.frame_end = clip.frame_end
        context.scene.frame_set(clip.frame_start)
        result = _export_fbx(
            animation_path, animated=True, profile=profile)
        _check_fbx_result(result, animation_path)
        produced.append(animation_path)
        ad = skel.animation_data
        if ad is not None:
            ad.action = None
            for track in tracks:
                try:
                    ad.nla_tracks.remove(track)
                except (ReferenceError, RuntimeError):
                    pass
        _neutral_export_state(
            context, skel, controls, targets, properties, base_matrix_basis)
    return produced


def _poll_character_export(context):
    mh = getattr(context.scene, "mhfrt", None)
    skel = mh.skeleton if mh else None
    return bool(skel is not None and skel.type == 'ARMATURE'
                and skel.get(RIG_ID_PROP))


def _execute_character_export(operator, context, profile):
    mh = context.scene.mhfrt
    skel = mh.skeleton
    label = "Unreal" if profile == 'UNREAL' else "Unity"
    if skel is None or not skel.get(RIG_ID_PROP):
        operator.report({'ERROR'}, "Build the facial rig first")
        return {'CANCELLED'}
    destination = str(skel.get(RIG_DESTINATION_PROP, ""))
    if destination == 'EXISTING' and not skel.get(RIG_MERGED_PROP):
        operator.report(
            {'ERROR'},
            "Merge Into One Armature before exporting the character")
        return {'CANCELLED'}
    # A character that came in already rigged still has its body armature, and
    # nothing this exporter can do makes two skeletons into one game asset.
    # Silently writing the head alone is how an artist finds out at import time.
    others = body_rigs(context, skel)
    if others:
        counts = organization.character_meshes(
            others, cage=skel.get(RIG_CAGE_PROP),
            target=skel.get(RIG_TARGET_PROP), scene=context.scene)
        operator.report(
            {'ERROR'},
            f"'{others[0].name}' still drives {len(counts)} of this "
            f"character's meshes (body, clothing...). {label} needs ONE "
            "skeleton: go to Rig > Merge Into One Armature, then export")
        return {'CANCELLED'}

    filepath = bpy.path.ensure_ext(operator.filepath, ".fbx")
    directory = os.path.dirname(filepath)
    if directory and not os.path.isdir(directory):
        operator.report({'ERROR'}, f"Export folder does not exist: {directory}")
        return {'CANCELLED'}

    objects = export_objects(context, skel)
    meshes = [obj for obj in objects if obj.type == 'MESH']
    if not meshes:
        operator.report(
            {'ERROR'},
            "No final character meshes are driven by this armature")
        return {'CANCELLED'}

    body_clips = discover_body_clips(skel)
    face_clips, controls = discover_face_clips(skel)
    clips = pair_export_clips(body_clips, face_clips)
    targets = _authored_shape_targets(skel)

    scene = context.scene
    scene_state = (
        scene.frame_start, scene.frame_end, scene.frame_current)
    active = context.view_layer.objects.active
    active_mode = active.mode if active is not None else 'OBJECT'
    selection_state = {
        obj: bool(obj.select_get()) for obj in context.view_layer.objects
    }
    skel_anim_state = _snapshot_animation(skel)
    # Animation state is per OWNER object (one entry for a whole bone board);
    # the pose to put back afterwards is per control.
    control_states = {owner: _snapshot_animation(owner)
                      for owner in _slot_owner_objects(controls).values()}
    control_locations = {control: control.location.copy()
                         for control in controls}
    pose_state = {
        pbone.name: pbone.matrix_basis.copy()
        for pbone in skel.pose.bones
    }
    base_matrix_basis = skel.matrix_basis.copy()
    shape_state = {
        key.as_pointer(): (key, float(key.value), bool(key.mute))
        for key, _input in targets
    }
    object_state = {}
    baked_actions = []
    aligned_actions = []
    export_tracks = []
    produced_files = []
    initial_action_pointers = {
        action.as_pointer() for action in bpy.data.actions
    }
    initial_track_pointers = {
        track.as_pointer()
        for track in (skel.animation_data.nla_tracks
                      if skel.animation_data else ())
    }
    created_drivers = []
    property_backup = {}
    runtime_was_updating = op_rig._updating
    token = uuid.uuid4().hex[:5]
    properties = [f"_mhx_{token}_{index:04d}"
                  for index in range(len(targets))]

    try:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        cache = _rig_cache(skel) if controls else None

        for prop in properties:
            property_backup[prop] = (
                prop in skel, skel.get(prop) if prop in skel else None)
            skel[prop] = 0.0
        # A muted sculpt key still belongs in the engine export. Mute is an
        # artist viewport state and is restored exactly in finally.
        for key, _input in targets:
            key.mute = False

        for clip in clips:
            _configure_body_source(skel, clip, base_matrix_basis)
            aligned, slots = _aligned_face_action(clip, controls)
            if aligned is not None:
                aligned_actions.append(aligned)
            _assign_face_action(controls, aligned, slots)
            scene.frame_start = clip.frame_start
            scene.frame_end = clip.frame_end
            scene.frame_set(clip.frame_start)

            samples = (_sample_shapes(
                scene, cache, targets,
                clip.frame_start, clip.frame_end)
                if cache is not None else
                [[0.0] * (clip.frame_end - clip.frame_start + 1)
                 for _target in targets])

            action, slot = _bake_visual_action(context, skel, clip)
            baked_actions.append((clip, action))
            _add_shape_channels(
                action, slot, properties, samples, clip.frame_start)

            # Detach the bake before the next source is configured.
            skel.animation_data.action = None
            _assign_face_action(controls, None, {})

        if clips:
            created_drivers = _add_shape_drivers(
                skel, targets, properties)
        # Temporary baked Actions now own bones and correctives. Pause the
        # live runtime during both static and animated FBX writes.
        op_rig._updating = True

        object_state = _prepare_export_selection(context, objects)
        context.view_layer.objects.active = skel
        produced_files = _export_profile_files(
            context, profile, filepath, skel, objects, controls, targets,
            properties, base_matrix_basis, baked_actions, export_tracks)
    except Exception as exc:  # noqa: BLE001 - restore before reporting
        traceback.print_exc()
        operator.report({'ERROR'}, f"{label} export failed: {exc}")
        return {'CANCELLED'}
    finally:
        op_rig._updating = True
        _remove_shape_drivers(created_drivers)

        ad = skel.animation_data
        if ad is not None:
            ad.action = None
            for track in export_tracks:
                try:
                    ad.nla_tracks.remove(track)
                except (ReferenceError, RuntimeError):
                    pass
            # Also catch a partially created track if an exception interrupted
            # _create_export_tracks before it returned.
            for track in list(ad.nla_tracks):
                if (track.as_pointer() not in initial_track_pointers
                        and track.name.startswith(TEMP_TRACK_PREFIX)):
                    ad.nla_tracks.remove(track)

        for prop, (existed, value) in property_backup.items():
            if existed:
                skel[prop] = value
            elif prop in skel:
                del skel[prop]

        _restore_animation(skel, skel_anim_state)
        for owner, state in control_states.items():
            _restore_animation(owner, state)
        for control, location in control_locations.items():
            try:
                control.location = location
            except ReferenceError:
                pass

        for _clip, action in baked_actions:
            if action.name in bpy.data.actions:
                bpy.data.actions.remove(action)
        for action in aligned_actions:
            if action.name in bpy.data.actions:
                bpy.data.actions.remove(action)
        # Also catch a partially created bake/alignment Action that did not
        # make it into the tracking lists before an exception.
        for action in list(bpy.data.actions):
            if (action.as_pointer() not in initial_action_pointers
                    and action.get(TEMP_ACTION_PROP)):
                bpy.data.actions.remove(action)

        skel.matrix_basis = base_matrix_basis
        for name, matrix in pose_state.items():
            pbone = skel.pose.bones.get(name)
            if pbone is not None:
                pbone.matrix_basis = matrix
        for key, value, muted in shape_state.values():
            try:
                key.value = value
                key.mute = muted
            except ReferenceError:
                pass

        scene.frame_start, scene.frame_end, original_frame = scene_state
        try:
            scene.frame_set(original_frame)
        except (ReferenceError, RuntimeError):
            pass
        op_rig._updating = runtime_was_updating
        op_rig.invalidate()
        if not runtime_was_updating:
            try:
                op_rig._rescan()
                op_rig._evaluate_guarded(op_rig._caches.values())
            except (ReferenceError, RuntimeError):
                pass

        _restore_object_state(object_state)
        for obj in context.view_layer.objects:
            try:
                obj.select_set(selection_state.get(obj, False))
            except (ReferenceError, RuntimeError):
                pass
        if active is not None and context.view_layer.objects.get(active.name):
            context.view_layer.objects.active = active
            if active_mode != 'OBJECT':
                try:
                    bpy.ops.object.mode_set(mode=active_mode)
                except RuntimeError:
                    pass

    clip_text = (f"{len(clips)} clip(s)" if clips
                 else "static character")
    if profile == 'UNREAL':
        operator.report(
            {'INFO'},
            f"Unreal FBX: skeletal mesh + {len(clips)} animation file(s), "
            f"{len(targets)} corrective(s) -> "
            f"{os.path.basename(filepath)}")
    else:
        operator.report(
            {'INFO'},
            f"Unity FBX: {clip_text}, {len(meshes)} mesh(es), "
            f"{len(targets)} corrective(s) -> "
            f"{os.path.basename(filepath)}")
    return {'FINISHED'}


class MHFRT_OT_export_unity_character(bpy.types.Operator, ExportHelper):
    bl_idname = "mhfrt.export_unity_character"
    bl_label = "Export Character to Unity"
    bl_description = (
        "Export one Unity FBX. Every armature/board clip becomes a Unity "
        "clip with visually baked bones and matching corrective blend-shape "
        "animation")
    bl_options = {'REGISTER'}

    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(
        default="*.fbx", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_character_export(context)

    def invoke(self, context, event):
        skel = context.scene.mhfrt.skeleton
        if not self.filepath:
            self.filepath = f"{_unity_file_token(skel.name)}.fbx"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        return _execute_character_export(self, context, 'UNITY')


class MHFRT_OT_export_unreal_character(bpy.types.Operator, ExportHelper):
    bl_idname = "mhfrt.export_unreal_character"
    bl_label = "Export Character to Unreal Engine"
    bl_description = (
        "Export an Unreal skeletal-mesh FBX plus one FBX per animation clip, "
        "with baked bones and matching corrective morph-target curves")
    bl_options = {'REGISTER'}

    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(
        default="*.fbx", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_character_export(context)

    def invoke(self, context, event):
        skel = context.scene.mhfrt.skeleton
        if not self.filepath:
            self.filepath = f"SK_{_safe_file_token(skel.name)}.fbx"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        return _execute_character_export(self, context, 'UNREAL')


_classes = (
    MHFRT_OT_export_unity_character,
    MHFRT_OT_export_unreal_character,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
