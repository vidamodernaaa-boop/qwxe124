"""Always-on facial rig runtime - multi-character, append-proof.

Every fitted skeleton IS a live rig, like a real production rig:

* fitting a skeleton (op_skeleton) calls setup_rig(): the skeleton gets a
  unique rig id, its OWN control board (imported and tagged per character,
  placed beside that character's head), and ID-pointer properties linking it
  to its cage / target / board - so appending any piece of a character into
  another file pulls the whole working chain along,
* ONE persistent depsgraph handler watches every rig's controls and
  re-evaluates only the rig whose control moved - any number of characters
  stay live simultaneously, with no on/off switch,
* rigs are discovered by their tags, never by hardcoded names: after file
  load, undo, or append, the runtime lazily re-scans and keeps working.

Evaluation: GUI controls -> riglogic (numpy) -> joint deltas -> pose bones.
"""

import json
import math
import os
import time
import uuid
from contextlib import contextmanager

import numpy as np
import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector

from .. import paths
from ..props import (RIG_ID_PROP, RIG_CAGE_PROP, RIG_TARGET_PROP,
                     RIG_GUI_COLL_PROP, RIG_INTENSITY_PROP, RIG_MERGED_PROP,
                     RIG_MERGE_BONE_PROP,
                     RIG_SOURCE_SCALE_PROP, RIG_SOURCE_BASIS_PROP)
from ..core import riglogic
from ..core import dna_apply
from ..core import board
from ..core import organization
from ..core import render_state
from .op_weights import ARM_MOD_NAME
from .op_morphs import (MORPH_DONE_PROP, MORPH_EXTRA_PROP,
                        object_belongs_to_rig)

CENTER_EYE_CONTROL = "CTRL_C_eye"
CENTER_EYE_LINKS = {
    "tx": ("CTRL_L_eye.tx", "CTRL_R_eye.tx"),
    "ty": ("CTRL_L_eye.ty", "CTRL_R_eye.ty"),
}
# The bundled DNA maps all four side-eye GUI channels over [-1, 1].  Clamp
# additive master + individual offsets so RigLogic never falls outside its
# piecewise segments and snaps an eye back to neutral.
EYE_GUI_MIN = -1.0
EYE_GUI_MAX = 1.0

# ---- look-at eye aim -------------------------------------------------------
# CTRL_C_eyesAim is NOT a RigLogic control - the DNA has no channel for it. It
# is a target floating in front of the face, and the rig only responds to it
# because the runtime turns it back into ordinary eye channel values: for each
# eye, the direction from the eye joint to its aim handle is measured in the
# eye's own neutral frame and written into CTRL_L_eye / CTRL_R_eye.
#
# What the handle commands is its DISPLACEMENT from its own rest, not its raw
# position. An absolute solve - point the eye wherever the handle happens to be
# - is only quiet at rest if the handle's rest sits exactly on the eye's neutral
# gaze ray, and it does not: the chain's layout is authored for Ada and dropped
# onto the character rigidly, while the fit gives every character its own eye
# positions and its own eye-joint orientations. Raising the switch then
# commanded a gaze nobody asked for (5.5 degrees on a head whose sockets sat 7
# degrees off Ada's) because the rig was faithfully aiming at a handle that was
# never in front of the eye.
#
# So the rest direction is measured once per rig and the solve reports the
# rotation FROM it (see _eye_rest_direction). Untouched handle, zero rotation,
# no movement - on any character, with the board exactly as authored. The board
# was the wrong place to fix this: v3.12.0 moved each circle onto its eye's axis
# instead, which zeroed the solve and wrecked the layout - the circles left the
# plane of their frame and the frame left the eye line.
#
# CTRL_lookAtSwitch, a board handle whose Y travels 0..1, decides how much of
# this applies. The reference treats it as a switch and only acts at the very
# top, which makes the eyes jump the moment the handle arrives; here the handle
# is a BLEND, so the eyes ease onto the target across its travel and can be
# keyframed into and out of a look. At 0 the solve does not run at all and the
# eye controls behave exactly as they did before any of this existed.
LOOK_AT_SWITCH = "CTRL_lookAtSwitch"
# (aim handle, eye joint, eye control) per eye.
EYE_AIM_PAIRS = (
    ("CTRL_L_eyeAim", "FACIAL_L_Eye", "CTRL_L_eye"),
    ("CTRL_R_eyeAim", "FACIAL_R_Eye", "CTRL_R_eye"),
)
# The response to fall back on when the DNA cannot be probed: the reference
# implementation's flat 60 degrees of yaw and 30 of pitch, decoupled and
# symmetric.  Shaped like a measured one (see _eye_channel_columns) so the solve
# has a single path.  The real numbers are none of those things - on this DNA,
# 42.3 out against 37.6 in and 29.7 up against 39.9 down.
EYE_AIM_FALLBACK_COLUMNS = {
    (0, 1.0): np.array([math.radians(60.0), 0.0]),
    (0, -1.0): np.array([math.radians(60.0), 0.0]),
    (1, 1.0): np.array([0.0, math.radians(30.0)]),
    (1, -1.0): np.array([0.0, math.radians(30.0)]),
}
# A channel whose own axis moves less than this is a failed probe, not a stiff
# eye; fall back rather than invert it.
EYE_AIM_MIN_GAIN = math.radians(1.0)
EYE_AIM_EPS = 1e-6
# An eye joint looks down its own local +Z (verified on the DNA's own bones).
FORWARD = Vector((0.0, 0.0, 1.0))

# Runtime-only optimization marker stored on the owning Key datablock. The marker
# lets us restore the user's original mute state when the key becomes active,
# gains sculpted geometry, or the add-on unregisters.
AUTO_MUTE_PROP = "mhfrt_runtime_auto_muted"

_caches = {}          # rig_id -> cache dict
# Which rig a depsgraph update belongs to.  Keyed by the OBJECT POINTER, not
# the name: a name is something the artist can change at any moment, and when
# they renamed a control board every update from it stopped matching - the
# controls moved and the face did not follow, with nothing in the UI to say
# why.  Pointers survive renames.  A pointer freed and reused by an unrelated
# object can only cause one wasted no-op evaluation (the GUI read that follows
# compares against the last values and skips), never a wrong one.
_control_map = {}     # control object pointer -> rig_id
_scan_needed = True
# True when the pending scan must rebuild EVERY cache (undo, load, an operator
# that changed a rig's contents) rather than only picking up rigs it has never
# seen.  A full rebuild is ~85 ms per rig in the file.
_scan_full = True
_updating = False
# How long a deferred timer waits before asking again while a render job owns
# the scene. Everything these timers do - running operators, changing mode,
# moving the selection - belongs to the artist's session, not to a render.
_RENDER_RETRY = 0.5
_last_repaired = 0    # rigs re-stamped by the last duplicate-id resolve
# What the last automatic repair did, for the panel to show.  A repair that
# nobody can see is how the same bug gets reported twice as "it randomly
# worked this time", so every automatic split says so until the artist acts.
_repair_notes = []
_MAX_REPAIR_NOTES = 3


def note_repair(message):
    """Record something the add-on fixed on its own, for the panel."""
    text = str(message).strip()
    if text and text not in _repair_notes:
        _repair_notes.append(text)
        del _repair_notes[:-_MAX_REPAIR_NOTES]


def repair_notes():
    return tuple(_repair_notes)


def clear_repair_notes():
    _repair_notes.clear()


# ------------------------------------------------------------ cache build ---

def _gui_channel_source(control, channel):
    return board.channel_source(control, channel)


def _gui_channel_value(source, divisor, evaluated=None):
    return board.channel_value(source, divisor, evaluated)


def _is_rig_skeleton(obj):
    """A tagged armature that DEFORMS the character, not its control board.

    The bone board is an armature carrying the same rig id (that is how its
    controls are found), so every skeleton lookup has to exclude it - treating
    it as a skeleton would build a second, bone-less cache under the same id and
    silently replace the real one.
    """
    return board.is_rig_skeleton(obj, RIG_ID_PROP)


def _rig_skeletons():
    return [o for o in bpy.data.objects if _is_rig_skeleton(o)]


def _build_cache(skel):
    rid = skel.get(RIG_ID_PROP)
    m = riglogic.meta()
    controls = board.controls_for_rig(RIG_ID_PROP, rid, skel=skel)
    if not controls:
        return None
    sources = []
    for gui_name in m["gui_names"]:
        obj_name, _, channel = gui_name.rpartition(".")
        sources.append(_gui_channel_source(controls.get(obj_name), channel))
    gui_index = {name: index for index, name in enumerate(m["gui_names"])}
    center_eye_links = []
    for channel, target_names in CENTER_EYE_LINKS.items():
        indices = tuple(gui_index[name] for name in target_names
                        if name in gui_index)
        if indices:
            center_eye_links.append((channel, indices))
    divisor = board.gui_scale(skel)
    # joint-translation scale = unit conv * auto head-size ratio * armature
    # unit ratio (1.0 unless merged into an armature at another scale) *
    # manual mul.  Pose-bone locations live in the armature's own space.
    try:
        source_scale = max(1e-6, float(skel.get(RIG_SOURCE_SCALE_PROP, 1.0)))
    except (TypeError, ValueError):
        source_scale = 1.0
    base_scale = (float(skel.get("dna_scale", 0.01))
                  * float(skel.get("mhfrt_riglogic_scale", 1.0))
                  * source_scale)
    target = skel.get(RIG_TARGET_PROP)
    if not isinstance(target, bpy.types.Object):
        target = None
    targets = dna_apply.build_targets(
        skel, m["joint_names"],
        dna_apply.source_to_rig_basis(skel, RIG_SOURCE_BASIS_PROP))
    pose_constants = dna_apply.build_pose_constants(targets)
    # After the targets: the look-at inverts how far each eye channel really
    # turns its joint, and that is measured through the same pose constants.
    eye_aim = _build_eye_aim(skel, controls, gui_index, targets)
    shape_key_objects = _shape_key_objects(skel, target)
    shape_key_targets = _shape_key_targets(
        skel, target, objects=shape_key_objects)
    shape_key_state = _build_shape_key_state(
        shape_key_objects, shape_key_targets)
    # The objects that can carry board animation: one per loose-object control,
    # a single armature for a bone board (see core/board.owner_object).
    board_owners = {}
    for control in controls.values():
        owner = board.owner_object(control)
        if owner is not None:
            board_owners[owner.name] = owner
    # A board built before the locks existed gets protected here, so an existing
    # character does not need a rebuild to stop Alt+S blowing its panel up, and
    # a board built before the design was written down records it now - from
    # what it looks like today, which is the best answer available for a scene
    # that is already open. Both are one-shot and version-gated; the third,
    # wiring the follow-head switches, is skipped when the character has no head
    # bone to ride. All of it is skipped while the artist is actually placing:
    # re-locking under their cursor would cancel the drag.
    for owner in board_owners.values():
        if not board.is_board_armature(owner):
            continue
        if board.layout_unlocked(owner):
            continue
        board.stamp_design(owner)
        board.ensure_board_locked(owner)
        if not board.follow_head_installed(owner) and follow_head_bone(skel):
            # NOT done here. A cache is built from inside the depsgraph handler,
            # and adding a constraint plus a driver is a structural change to
            # the very graph that is mid-evaluation - the same reason character
            # switching is queued rather than run (see _apply_pending_switch).
            # Locks and ID properties are safe there; a new relationship is not.
            #
            # Gated on there BEING a head bone, or a skeleton that can never
            # have one would queue a timer on every full rescan for the rest of
            # the session - each one arriving only to find nothing to do.
            _queue_follow_head(skel)
    return {
        "skel": skel,
        "targets": targets,
        "eye_aim": eye_aim,
        "board_owners": tuple(board_owners.values()),
        "pose_constants": pose_constants,   # stacked per-bone RigLogic constants
        "shape_key_targets": shape_key_targets,
        "shape_key_objects": shape_key_objects,
        **shape_key_state,
        "sources": sources,
        "controls": controls,
        "base_scale": base_scale,
        "scale": base_scale * float(skel.get(RIG_INTENSITY_PROP, 1.0)),
        "divisor": divisor,
        "gui_count": m["gui_count"],
        "center_eye_links": tuple(
            (_gui_channel_source(controls.get(CENTER_EYE_CONTROL), channel),
             indices)
            for channel, indices in center_eye_links
        ),
    }


def _shape_key_objects(skel, target):
    rid = skel.get(RIG_ID_PROP) if skel is not None else None
    objects = []
    if target is not None and target.type == 'MESH':
        objects.append(target)
    if rid:
        for obj in bpy.data.objects:
            if obj in objects or obj.type != 'MESH':
                continue
            if (obj.get(MORPH_EXTRA_PROP)
                    and object_belongs_to_rig(obj, rid)):
                objects.append(obj)
    return objects


def _shape_key_targets(skel, target, objects=None):
    if objects is None:
        objects = _shape_key_objects(skel, target)

    input_by_name = riglogic.morph_input_by_key_name(head_only=True)
    out = []
    for obj in objects:
        shape_keys = obj.data.shape_keys
        if shape_keys is None:
            continue
        if not shape_keys.get(MORPH_DONE_PROP):
            continue
        # keys baked to native drivers have one owner already - skip them
        ad = shape_keys.animation_data
        driven_paths = ({fc.data_path for fc in ad.drivers}
                        if ad and ad.drivers else ())
        for key in shape_keys.key_blocks[1:]:
            input_index = input_by_name.get(key.name)
            if input_index is None:
                continue
            if driven_paths and \
                    f'key_blocks["{key.name}"].value' in driven_paths:
                continue
            out.append((key, int(input_index)))
    return out


def _active_shape_key_state(objects, target_by_pointer):
    """Return driven active keys plus a cheap picker/topology signature.

    Blender's Shape Key Move operator keeps the same active KeyBlock pointer
    while changing its collection index.  Remembering both lets the batched
    value writer rebuild its positional indices after an ordinary UI reorder.
    The name is included so renaming the selected key also reconciles our
    runtime mute ownership immediately.
    """
    active = set()
    state = []
    for obj in objects:
        key = obj.active_shape_key
        pointer = key.as_pointer() if key is not None else 0
        state.append((
            obj.as_pointer(), int(obj.active_shape_key_index), pointer,
            key.name if key is not None else "",
        ))
        if pointer in target_by_pointer:
            active.add(pointer)
    return active, tuple(state)


def _shape_key_coords(key):
    coords = np.empty(len(key.data) * 3, dtype=np.float32)
    key.data.foreach_get("co", coords)
    return coords


def _shape_key_has_delta(key, relative_cache):
    """Exact geometry test; even the smallest authored delta stays live."""
    shape_keys = key.id_data
    if not shape_keys.use_relative:
        return True
    relative = key.relative_key
    if relative is None or relative == key or len(relative.data) != len(key.data):
        return True
    pointer = relative.as_pointer()
    basis = relative_cache.get(pointer)
    if basis is None:
        basis = _shape_key_coords(relative)
        relative_cache[pointer] = basis
    return not np.array_equal(_shape_key_coords(key), basis)


def _auto_mute_records(targets):
    records = {}
    for key, _input_index in targets:
        shape_keys = key.id_data
        pointer = shape_keys.as_pointer()
        if pointer in records:
            continue
        names = {
            name for name in shape_keys.get(AUTO_MUTE_PROP, "").split("\n")
            if name
        }
        owned = {
            block.as_pointer(): block for block in shape_keys.key_blocks
            if block.name in names
        }
        records[pointer] = {
            "shape_keys": shape_keys,
            # Pointer ownership survives a UI rename.  Names are used only as
            # a persistent fallback across file load / add-on reload.
            "owned": owned,
        }
    return records


def _store_auto_mute_names(record):
    shape_keys = record["shape_keys"]
    owned = record.get("owned", {})
    current = {block.as_pointer(): block for block in shape_keys.key_blocks}
    owned = {pointer: current[pointer]
             for pointer in owned if pointer in current}
    record["owned"] = owned
    names = {block.name for block in owned.values()}
    if names:
        value = "\n".join(sorted(names))
        if shape_keys.get(AUTO_MUTE_PROP, "") != value:
            shape_keys[AUTO_MUTE_PROP] = value
    elif AUTO_MUTE_PROP in shape_keys:
        del shape_keys[AUTO_MUTE_PROP]


def _set_shape_key_auto_muted(key, should_mute, record):
    """Mute/unmute without disturbing a mute state the artist set themselves."""
    owned = record["owned"]
    pointer = key.as_pointer()
    auto_muted = pointer in owned
    if should_mute:
        if not key.mute:
            key.mute = True
            owned[pointer] = key
            return True
        return False
    if not auto_muted:
        return False
    if key.mute:
        key.mute = False
    owned.pop(pointer, None)
    return True


def _shape_key_write_groups(targets):
    """Group mapped keys by Key datablock for safe foreach value writes."""
    grouped = {}
    for key, input_index in targets:
        shape_keys = key.id_data
        pointer = shape_keys.as_pointer()
        group = grouped.get(pointer)
        if group is None:
            blocks = shape_keys.key_blocks
            group = {
                "shape_keys": shape_keys,
                "blocks": blocks,
                "block_indices": [],
                "input_indices": [],
                "values": np.empty(len(blocks), dtype=np.float64),
            }
            grouped[pointer] = group
        index = group["blocks"].find(key.name)
        if index < 0:
            continue
        group["block_indices"].append(index)
        group["input_indices"].append(input_index)
    out = []
    for group in grouped.values():
        group["block_indices"] = np.asarray(
            group["block_indices"], dtype=np.intp)
        group["input_indices"] = np.asarray(
            group["input_indices"], dtype=np.intp)
        out.append(group)
    return tuple(out)


def _build_shape_key_state(objects, targets):
    """Auto-mute exact-empty keys while retaining their live input values.

    Blender otherwise blends hundreds of non-zero but Basis-identical keys on
    every frame.  Keys stay present, their values keep tracking RigLogic (so
    downstream drivers retain the same signal), and the active picker key is
    always unmuted for Sculpt/Edit preview.
    """
    target_by_pointer = {
        key.as_pointer(): (key, input_index) for key, input_index in targets
    }
    active, active_state = _active_shape_key_state(
        objects, target_by_pointer)
    auto_records = _auto_mute_records(targets)
    relative_cache = {}
    for pointer, (key, _input_index) in target_by_pointer.items():
        # A key we auto-muted on an earlier cache build remained inaccessible
        # to normal editing while inactive, so its exact-empty classification
        # is still valid.  Active keys and user-unmuted keys are rechecked.
        record = auto_records[key.id_data.as_pointer()]
        known_empty = pointer in record["owned"] and key.mute \
            and pointer not in active
        empty = known_empty or not _shape_key_has_delta(key, relative_cache)
        _set_shape_key_auto_muted(
            key, should_mute=empty and pointer not in active, record=record)
    for record in auto_records.values():
        _store_auto_mute_names(record)
    return {
        "shape_key_target_by_pointer": target_by_pointer,
        "shape_key_active_pointers": active,
        "shape_key_active_state": active_state,
        "shape_key_auto_mute_records": auto_records,
        "shape_key_groups": _shape_key_write_groups(targets),
    }


def _sync_active_shape_key_mutes(cache):
    """Keep active picker keys previewable and classify the key just left."""
    target_by_pointer = cache.get("shape_key_target_by_pointer", {})
    current, current_state = _active_shape_key_state(
        cache.get("shape_key_objects", ()), target_by_pointer)
    previous = cache.get("shape_key_active_pointers", set())
    previous_state = cache.get("shape_key_active_state", ())
    topology_changed = current_state != previous_state
    if current == previous and not topology_changed:
        return False

    changed = False
    relative_cache = {}
    auto_records = cache.get("shape_key_auto_mute_records", {})
    dirty_records = set()
    # A key leaving the active slot may just have been sculpted.  Only mute it
    # when an exact comparison still proves it has no deformation.
    for pointer in previous - current:
        target = target_by_pointer.get(pointer)
        if target is None:
            continue
        key = target[0]
        empty = not _shape_key_has_delta(key, relative_cache)
        record = auto_records.get(key.id_data.as_pointer())
        if record is None:
            continue
        if _set_shape_key_auto_muted(key, empty, record):
            changed = True
            dirty_records.add(key.id_data.as_pointer())

    # Auto-muted keys become live previews as soon as artists select them.
    for pointer in current:
        target = target_by_pointer.get(pointer)
        if target is not None:
            key = target[0]
            record = auto_records.get(key.id_data.as_pointer())
            if record is not None and _set_shape_key_auto_muted(
                    key, False, record):
                changed = True
                dirty_records.add(key.id_data.as_pointer())

    cache["shape_key_active_pointers"] = current
    cache["shape_key_active_state"] = current_state
    for pointer in dirty_records:
        _store_auto_mute_names(auto_records[pointer])
    if topology_changed:
        # Collection foreach writes are positional.  Rebuild after picker
        # changes because the normal Blender reorder operator moves the active
        # key and therefore changes this state even when its pointer is stable.
        cache["shape_key_groups"] = _shape_key_write_groups(
            cache.get("shape_key_targets", ()))
        # Also persists a selected key's current name after a rename.
        for record in auto_records.values():
            _store_auto_mute_names(record)
    return changed or topology_changed


def prepare_shape_key_preview(obj):
    """Immediately unmute a picker-selected empty key before entering a mode."""
    if obj is None or obj.type != 'MESH' or obj.data.shape_keys is None:
        return
    key = obj.active_shape_key
    if key is None:
        return
    shape_keys = obj.data.shape_keys
    shape_keys_pointer = shape_keys.as_pointer()
    key_pointer = key.as_pointer()
    for cache in _caches.values():
        record = cache.get("shape_key_auto_mute_records", {}).get(
            shape_keys_pointer)
        if record is None or key_pointer not in record.get("owned", {}):
            continue
        if _set_shape_key_auto_muted(key, False, record):
            _store_auto_mute_names(record)
        return

    # Fallback for a marker restored from a saved file before caches exist.
    names = {
        name for name in shape_keys.get(AUTO_MUTE_PROP, "").split("\n")
        if name
    }
    if key.name not in names:
        return
    if key.mute:
        key.mute = False
    names.discard(key.name)
    if names:
        shape_keys[AUTO_MUTE_PROP] = "\n".join(sorted(names))
    elif AUTO_MUTE_PROP in shape_keys:
        del shape_keys[AUTO_MUTE_PROP]


# ------------------------------------------------------ duplicate rig ids ---
#
# A rig id is meant to name exactly one skeleton - everything else in the
# add-on is keyed by it.  Blender's Scene > New > **Full Copy** (and a
# whole-character Duplicate Objects) copies custom properties verbatim, so
# after one click two skeletons answer to the same id and the failure is total,
# not partial: `_caches` keeps whichever skeleton is scanned LAST, and
# `board.controls_for_rig` gathers controls from both boards and dedupes by
# template name, so every control resolves to whichever copy sorts last in
# `bpy.data.objects`.  The board the artist can see ends up driving the twin in
# the other scene, and moving a handle poses nothing at all.
#
# Ownership is recoverable because the copy does NOT share its ID pointers:
# Blender remaps `mhfrt_cage` / `mhfrt_target` / `mhfrt_gui_coll` (and the
# character's outliner root) to the copies as it duplicates, so each skeleton
# already names its own half of the file.  Everything below reads that existing
# evidence - no name heuristics, and no "the .001 one is the copy".

# Evidence strength.  The top rank is proof of a live relationship: only one
# armature can deform a mesh or be its ancestor.  A pointer is nearly as good
# but survives a lone Shift+D of the skeleton still naming the ORIGINAL's
# objects, so it must lose to a binding.  The collection is a last-resort hint
# that settles the objects nothing else names.
_CLAIM_BOUND = 3
_CLAIM_POINTER = 2
_CLAIM_COLLECTION = 1


def _claim_scores(skel):
    """``{object name: (score, object)}`` - what `skel` can prove is its own."""
    scores = {}

    def claim(obj, score):
        if not isinstance(obj, bpy.types.Object):
            return
        if scores.get(obj.name, (0, None))[0] < score:
            scores[obj.name] = (score, obj)

    claim(skel, _CLAIM_BOUND)
    for obj in bpy.data.objects:
        if obj == skel:
            continue
        if any(mod.type == 'ARMATURE' and mod.object == skel
               for mod in obj.modifiers):
            claim(obj, _CLAIM_BOUND)      # cage, head, attached parts, extras
            continue
        parent = obj.parent               # the board is parented to its rig
        while parent is not None:
            if parent == skel:
                claim(obj, _CLAIM_BOUND)
                break
            parent = parent.parent
    claim(skel.get(RIG_CAGE_PROP), _CLAIM_POINTER)
    claim(skel.get(RIG_TARGET_PROP), _CLAIM_POINTER)
    gui_coll = skel.get(RIG_GUI_COLL_PROP)
    if isinstance(gui_coll, bpy.types.Collection):
        for obj in gui_coll.all_objects:
            claim(obj, _CLAIM_POINTER)
    root = skel.get(organization.CHARACTER_COLL_PROP)
    if isinstance(root, bpy.types.Collection):
        for obj in root.all_objects:
            claim(obj, _CLAIM_COLLECTION)
    # The ~460 handle shapes are linked to no collection and no pointer names
    # them: the bones that draw them are the only way in (see
    # remove_board_widgets).  They are exactly as owned as their board.
    for score, obj in list(scores.values()):
        if board.is_board_armature(obj) and obj.pose is not None:
            for pose_bone in obj.pose.bones:
                claim(pose_bone.custom_shape, score)
    return scores


def _writable(obj):
    """False for a library-linked datablock - its properties are read-only."""
    return obj is not None and obj.library is None and obj.override_library is None


def _restamp(objects, old_id, new_id):
    """Move `old_id` to `new_id` on the tags these objects carry.

    Only a tag that really holds the old id is written: a user mesh that merely
    sits in the character's collection keeps its own data untouched.
    """
    for obj in objects:
        if not _writable(obj):
            continue
        if str(obj.get(RIG_ID_PROP) or "") == old_id:
            obj[RIG_ID_PROP] = new_id
        if str(obj.get(board.WIDGET_OWNER_PROP) or "") == old_id:
            obj[board.WIDGET_OWNER_PROP] = new_id


def _clear_display_name(skel):
    """Forget a copy's inherited list name so it is given a fresh unique one.

    The rig id is what the runtime keys on, but the artist reads the NAME: a
    Full Copy hands the copy the original's stored name too, so splitting the
    ids alone left the list showing 'MHFR, MHFR.001, MHFR.002, MHFR, MHFR.001,
    MHFR.002' with no way to tell the pairs apart.  Cleared here, and
    sync_rig_ui_state mints a unique one on the next draw.
    """
    from ..core import organization as org
    owners = [skel, skel.get(org.CHARACTER_COLL_PROP),
              org._root_from_objects(skel)]
    for owner in owners:
        if owner is None or not isinstance(
                owner, (bpy.types.Object, bpy.types.Collection)):
            continue
        if owner.library is not None:
            continue
        if RIG_NAME_PROP in owner:
            del owner[RIG_NAME_PROP]


def _drop_foreign_pointers(skel, owned_names):
    """Clear links this skeleton kept to objects that turned out to be another
    rig's - what a lone Shift+D of a skeleton leaves behind.

    Left in place they would list a second, boardless character pointing at
    someone else's cage, and Remove Rig deletes by pointer AND by id.
    """
    if not _writable(skel):
        return
    for prop in (RIG_CAGE_PROP, RIG_TARGET_PROP, RIG_GUI_COLL_PROP):
        value = skel.get(prop)
        if isinstance(value, bpy.types.Object):
            stale = value.name not in owned_names
        elif isinstance(value, bpy.types.Collection):
            stale = not any(obj.name in owned_names
                            for obj in value.all_objects)
        else:
            continue
        if stale:
            del skel[prop]


def _split_duplicate_skeletons():
    """Give every rig skeleton an id of its own again.  Returns the count."""
    groups = {}
    for skel in _rig_skeletons():          # already in bpy.data.objects order
        groups.setdefault(str(skel.get(RIG_ID_PROP) or ""), []).append(skel)
    repaired = 0
    for rid, skels in groups.items():
        if not rid or len(skels) < 2:
            continue
        scores = {skel.name: _claim_scores(skel) for skel in skels}
        # The id stays with the skeleton that has the most objects actually
        # bound to it; file order breaks the tie, and a copy is always created
        # after what it was copied from.  Which one wins only decides how many
        # tags get rewritten - the split itself is the same either way.  A
        # library-linked rig outranks both: its properties cannot be written,
        # so it is the only one that CAN keep the id.
        keeper = max(skels, key=lambda skel: (
            not _writable(skel),
            sum(1 for score, _ in scores[skel.name].values()
                if score >= _CLAIM_BOUND)))
        for skel in skels:
            if skel == keeper or not _writable(skel):
                continue
            mine = scores[skel.name]
            owned = {skel.name: skel}
            for name, (score, obj) in mine.items():
                rival = max(scores[other.name].get(name, (0, None))[0]
                            for other in skels if other != skel)
                if score > rival:
                    owned[name] = obj
            _restamp(owned.values(), rid, uuid.uuid4().hex[:12])
            _drop_foreign_pointers(skel, owned.keys())
            _clear_display_name(skel)
            repaired += 1
    return repaired


def _split_duplicate_boards():
    """One skeleton, two boards on its id: keep the one it points at.

    Duplicating just the control board leaves the runtime picking whichever
    armature sorts last.  A board the skeleton does not claim is re-stamped so
    it stops answering - it owns no skeleton, so it simply goes inert.
    """
    repaired = 0
    for skel in _rig_skeletons():
        rid = str(skel.get(RIG_ID_PROP) or "")
        if not rid:
            continue
        boards = [obj for obj in bpy.data.objects
                  if board.is_board_armature(obj)
                  and str(obj.get(RIG_ID_PROP) or "") == rid]
        if len(boards) < 2:
            continue
        # The GUI collection the skeleton points at is the tie-breaker: a
        # duplicated board keeps its parent, so "parented to this rig" no
        # longer tells the two apart.  With no pointer there is no evidence at
        # all, and guessing would be worse than leaving it: picking wrong
        # re-stamps the board the artist is actually posing and takes their
        # rig off its controls.  Leave it and let Refresh report it.
        keeper = board.own_board_armature(skel, RIG_ID_PROP)
        if keeper is None:
            continue
        for obj in boards:
            if obj == keeper:
                continue
            widgets = [obj]
            if obj.pose is not None:
                widgets += [pose_bone.custom_shape for pose_bone in obj.pose.bones
                            if pose_bone.custom_shape is not None]
            _restamp(widgets, rid, uuid.uuid4().hex[:12])
            repaired += 1
    return repaired


def _shared_board_warnings():
    """Warn when two rigs are left sharing ONE pre-3.0 loose-object board.

    The old board is hundreds of separate objects living in a GUI collection,
    and duplicating objects does not duplicate the collection they are in - so
    both skeletons still point at the same one and neither can prove which
    handles are its own. There is no evidence left to split on, and guessing
    would take a live character off its controls.

    The bone board has none of this: it is a single armature parented to its
    rig, which is exactly the evidence the split needs. So this names the one
    action that fixes it rather than pretending the copy works.
    """
    seen = {}
    for skel in _rig_skeletons():
        gui = skel.get(RIG_GUI_COLL_PROP)
        if isinstance(gui, bpy.types.Collection) and board_is_legacy(skel):
            seen.setdefault(gui.name, []).append(skel.name)
    return [
        f"{len(names)} rigs still share one old-style control board "
        f"({', '.join(sorted(names)[:3])}). Only one of them can be driven - "
        "select each and use Rig > Rebuild Board so each gets bones of its own"
        for names in seen.values() if len(names) > 1
    ]


def _shared_armature_warnings():
    """Warn about rigs that share one Armature datablock (Alt+D).

    A LINKED duplicate is not the same problem as a plain one and a fresh rig
    id cannot fix it: the copy shares the bones themselves, and the whole point
    of a transferred rig is that each character's bones are fitted to ITS head.
    Fitting one would move the other's face. It is also not safe to fix from
    here - un-sharing means creating a datablock, which this runs too deep in
    the depsgraph to do - so it is named and left to the artist.
    """
    by_data = {}
    for skel in _rig_skeletons():
        try:
            by_data.setdefault(skel.data.as_pointer(), []).append(skel)
        except (AttributeError, ReferenceError):
            continue
    out = []
    for skels in by_data.values():
        if len(skels) < 2:
            continue
        names = ", ".join(sorted(s.name for s in skels)[:3])
        out.append(
            f"{len(skels)} rigs share one armature ({names}) - a linked "
            "duplicate (Alt+D). Give each its own with Object > Relations > "
            "Make Single User > Object & Data, or they cannot be fitted apart")
    return out


def resolve_duplicate_rig_ids():
    """Repair every rig id that more than one rig answers to.

    Called from :func:`_rescan`, so it runs automatically after a file load,
    an undo, an append and on Refresh Rigs - the moments a duplicate can first
    appear.  Idempotent: a file with no duplicates writes nothing.
    """
    return _split_duplicate_skeletons() + _split_duplicate_boards()


def _rescan(full=True):
    """Bring the rig caches in line with the tags present in the file.

    ``full`` rebuilds every cache - the right thing after an undo, a load, or
    any operator that changed a rig's contents.  Otherwise only rigs with no
    cache yet are built and rigs that have gone away are dropped: building one
    cache is ~85 ms, so the old unconditional full rebuild made every rig in
    the file N times more expensive to create and turned "a character was
    appended" into a freeze proportional to the whole file.
    """
    global _scan_needed, _scan_full, _last_repaired
    # Reconcile current names before replacing live pointer-owned records.
    for cache in _caches.values():
        for record in cache.get("shape_key_auto_mute_records", {}).values():
            try:
                _store_auto_mute_names(record)
            except ReferenceError:
                pass
    if full:
        _caches.clear()
        _control_map.clear()
        # Splitting shared ids rewrites tags, so it belongs with a full pass.
        _last_repaired = resolve_duplicate_rig_ids()
        if _last_repaired:
            note_repair(f"{_last_repaired} duplicated rig(s) were given a rig "
                        "id of their own")
        for message in _shared_armature_warnings() + _shared_board_warnings():
            note_repair(message)
    if not riglogic.available():
        _caches.clear()
        _control_map.clear()
        _scan_needed = False
        _scan_full = False
        return
    skeletons = _rig_skeletons()
    live = set()
    for skel in skeletons:
        rid = str(skel.get(RIG_ID_PROP) or "")
        if not rid:
            continue
        live.add(rid)
        if rid in _caches:
            continue                    # already built and still valid
        cache = _build_cache(skel)
        if cache is None:
            continue
        _caches[rid] = cache
    # A rig whose skeleton is gone must not keep answering: its cache holds
    # PoseBone wrappers into data that may already be freed.
    for rid in [r for r in _caches if r not in live]:
        cache = _caches.pop(rid)
        cache.clear()
    _control_map.clear()
    for rid, cache in _caches.items():
        # Keyed by OBJECT name: a bone board reports its armature, never
        # the individual pose bones, so every control maps to one entry.
        for control in cache["controls"].values():
            owner = board.owner_object(control)
            if owner is not None:
                _control_map[owner.as_pointer()] = rid
    _scan_needed = False
    _scan_full = False


def invalidate():
    """Ask for a lazy FULL re-scan (cheap; the next depsgraph tick rebuilds).

    Used by every operator that changes what is inside a rig - morph objects,
    board rebuilds, merges, bakes - where an existing cache has gone stale.
    """
    global _scan_needed, _scan_full
    _scan_needed = True
    _scan_full = True


def invalidate_new_rigs():
    """Ask only for rigs that have no cache yet to be built."""
    global _scan_needed
    _scan_needed = True


def drop_cache(rid):
    """Forget one rig's cache BEFORE its objects are deleted.

    A bone board's controls are PoseBones, and a PoseBone is not an ID: when
    its armature object is removed, Blender invalidates the Object wrapper but
    the pose-bone wrappers held elsewhere keep pointing into freed memory, so
    the next garbage collection dereferences it and Blender goes down.  Every
    path that deletes a board or a skeleton calls this first, and the cache is
    emptied in place so the wrappers are released right here - while the data
    they point at is still alive - rather than whenever the collector runs.
    """
    global _scan_needed
    _scan_needed = True
    if not rid:
        return
    cache = _caches.pop(rid, None)
    for key, owner_rid in list(_control_map.items()):
        if owner_rid == rid:
            del _control_map[key]
    if cache is not None:
        cache.clear()


# -------------------------------------------------------------- evaluation ---

# Per-channel Bone Intensity (written by op_morphs): each row scales one DNA
# GUI channel by its factor before RigLogic runs, but only while that channel
# sits inside the row's value window - which is what keeps a left/right pair
# living on one channel independent.  A channel dialled to 0 moves no bones at
# all.  Files saved by the older binary isolate toggle carry only the index
# list: a missing factor means fully muted, a missing window means the whole
# channel.
CHANNEL_GUI_INDICES_PROP = "mhfrt_muted_gui_indices"
CHANNEL_GUI_FACTORS_PROP = "mhfrt_muted_gui_factors"
CHANNEL_GUI_LOWS_PROP = "mhfrt_channel_gui_lows"
CHANNEL_GUI_HIGHS_PROP = "mhfrt_channel_gui_highs"
CHANNEL_WINDOW_EPS = 1e-6


def _channel_factors(skel):
    """[(gui channel, lo, hi, factor)] rows below full bone strength."""
    if skel is None:
        return ()
    indices = skel.get(CHANNEL_GUI_INDICES_PROP)
    if not indices:
        return ()
    factors = skel.get(CHANNEL_GUI_FACTORS_PROP)
    lows = skel.get(CHANNEL_GUI_LOWS_PROP)
    highs = skel.get(CHANNEL_GUI_HIGHS_PROP)

    def at(values, k, fallback):
        if values is None or k >= len(values):
            return fallback
        return float(values[k])

    return tuple(
        (int(index), at(lows, k, -np.inf), at(highs, k, np.inf),
         at(factors, k, 0.0))
        for k, index in enumerate(indices)
    )


def _eye_channel_columns(targets, joint_name, index_x, index_y):
    """What one unit of each eye channel does to this eye's gaze.

    ``{(axis, direction): [yaw, pitch]}`` in radians per unit, axis 0 for tx and
    1 for ty, direction +1 and -1 - four columns, because the DNA's eye is not
    symmetric.

    Measured, not assumed.  The flat 60 degrees of yaw and 30 of pitch this used
    to hardcode are simply not what the DNA does: Ada's eye is anatomical, 42.3
    degrees outward against 37.6 inward and 29.7 up against 39.9 down.
    Inverting the wrong number does not break the rig, it just misses - ask for
    a 34 degree glance with the 60 degree assumption and the eye travels 24, so
    the gaze never lands on the target the artist positioned.

    Each half of each channel is exactly linear (checked against the bake at
    0.25 / 0.5 / 0.75 / 1.0: the radians-per-unit are constant to three
    decimals), so these columns are not a linearisation - inside a quadrant they
    ARE the map, and :func:`_eye_aim_channels` inverts them exactly.

    Measured per rig rather than per DNA: the gaze is read in the FITTED bone's
    own rest frame, through the same change of basis the pose path uses, so an
    eye the wrap left sitting at another angle reports its own numbers -
    including the small cross-axis terms a tilted socket picks up.
    """
    entry = next((t for t in targets if t["name"] == joint_name), None)
    if entry is None:
        return None
    try:
        constants = dna_apply.build_pose_constants([entry])
        gui_count = riglogic.meta()["gui_count"]
        columns = {}
        for axis, index in ((0, index_x), (1, index_y)):
            for direction in (1.0, -1.0):
                gui = np.zeros(gui_count)
                gui[index] = direction
                deltas = np.asarray(riglogic.evaluate_gui(gui), dtype=float)
                basis = dna_apply.pose_bases(
                    constants, deltas[constants["index"]])[0]
                # Where the eye's own forward axis ends up, read back as the
                # same yaw/pitch pair the solve measures a target with.
                gaze = basis @ np.array([0.0, 0.0, 1.0])
                angles = np.array([
                    math.atan2(gaze[0], gaze[2]),
                    math.atan2(gaze[1], math.hypot(gaze[0], gaze[2])),
                ])
                # Per UNIT of a value with this sign: at -1 the angles are the
                # negative of one unit's worth.
                columns[(axis, direction)] = angles / direction
    except (ValueError, IndexError, np.linalg.LinAlgError):
        return None
    # A channel that turns the eye by nothing is a failed probe, not a stiff
    # eye; one such column and the whole 2x2 stops being invertible.
    for (axis, direction), column in columns.items():
        if abs(column[axis]) < EYE_AIM_MIN_GAIN:
            return None
    return columns


def _eye_aim_quadrants(columns):
    """The inverse of the response map on each of the four sign quadrants.

    ``{(sign x, sign y): 2x2}``.  The map from channel values to gaze angles is
    linear on each quadrant and switches column at zero, so a quadrant is
    exactly where it can be inverted; a singular one is dropped rather than
    faked.
    """
    quadrants = {}
    for sign_x in (1.0, -1.0):
        for sign_y in (1.0, -1.0):
            matrix = np.column_stack((columns[(0, sign_x)],
                                      columns[(1, sign_y)]))
            try:
                quadrants[(sign_x, sign_y)] = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                continue
    return quadrants


def _handle_rest_head(pose_bone):
    """Where this handle sits with every control above it on its stored rest.

    The BOARD's rest, not the bone's: ``_import_gui`` records the authored rest
    of each control in ``board.REST_PROP``, and on this chain that is 0.17 mm of
    authoring dust on the two GRP_ bones.  Taking the bone rest alone would
    leave the dust looking like an artist's drag - 0.03 degrees of gaze, which
    is invisible but is exactly the kind of "nearly zero" this is meant to end.

    Every ancestor is at rest in the configuration being described, so each
    one's rest basis is its pose basis and the offsets simply accumulate.
    """
    head = pose_bone.bone.head_local.copy()
    node = pose_bone
    while node is not None:
        rest = board.tag(node, board.REST_PROP)
        if rest is not None:
            head = head + node.bone.matrix_local.to_3x3() @ Vector(rest[:3])
        node = node.parent
    return head


def _eye_rest_direction(skel, arm_obj, handle_name, joint_name):
    """Where this eye's handle sits at rest, as a unit vector in the eye's frame.

    The baseline the solve measures against.  Both ends are taken at REST - the
    eye joint's rest head and frame, the handle's rest head - so this is a
    property of the rig and not of the frame the artist is parked on.

    In the EYE's frame rather than the world, deliberately: that frame turns
    with the head, so the baseline turns with it too, and a world-fixed handle
    keeps reading as a rotation away from rest as the head turns.  That is what
    makes the eyes hold their target through a head turn instead of swinging
    along with it.
    """
    bone = skel.data.bones.get(joint_name)
    handle = arm_obj.pose.bones.get(handle_name)
    if bone is None or handle is None:
        return None
    reference = skel.matrix_world @ bone.matrix_local
    direction = ((arm_obj.matrix_world @ _handle_rest_head(handle))
                 - (skel.matrix_world @ bone.head_local))
    if direction.length < EYE_AIM_EPS:
        return None
    local = (reference.to_3x3().inverted_safe()
             @ direction.normalized()).normalized()
    return local if local.length > 0.5 else None


def _eye_response(aim, eye):
    """This eye's response map, probed on first use and kept for the cache.

    Deferred rather than built with the rest of the cache: the four probes are
    four full RigLogic evaluations each, about 23 ms for both eyes, and a rig
    whose look-at switch is never raised should not pay that on every rescan -
    and a rescan is an ordinary scene edit.  Nothing can go stale behind the
    memo, because the cache holding it is discarded whenever the rig changes.
    """
    response = eye.get("response")
    if response is None:
        columns = _eye_channel_columns(
            aim["targets"], eye["joint"], eye["x"], eye["y"])
        if columns is None:
            columns = EYE_AIM_FALLBACK_COLUMNS
        response = (columns, _eye_aim_quadrants(columns))
        eye["response"] = response
    return response


def _build_eye_aim(skel, controls, gui_index, targets):
    """What the look-at solve needs, resolved once per rig cache.

    None when this character has no aim chain - a pre-3.0 loose-object board,
    or a DNA whose eye channels are named differently.
    """
    switch = controls.get(LOOK_AT_SWITCH)
    if switch is None or not board.is_pose_bone(switch):
        return None
    armature = board.control_armature(switch)
    if armature is None:
        return None
    eyes = []
    for handle_name, joint_name, control_name in EYE_AIM_PAIRS:
        index_x = gui_index.get(f"{control_name}.tx")
        index_y = gui_index.get(f"{control_name}.ty")
        if (index_x is None or index_y is None
                or handle_name not in armature.pose.bones
                or joint_name not in skel.pose.bones):
            continue
        rest_direction = _eye_rest_direction(
            skel, armature, handle_name, joint_name)
        if rest_direction is None:
            continue
        eyes.append({
            "handle": handle_name,
            "joint": joint_name,
            "x": index_x,
            "y": index_y,
            "rest": rest_direction,
            "response": None,       # probed by _eye_response on first look-at
        })
    if not eyes:
        return None
    return {"switch": switch, "board": armature, "skel": skel,
            "targets": targets, "eyes": eyes}


def look_at_weight(cache, evaluated=None):
    """How far this character's CTRL_lookAtSwitch handle is up, 0..1.

    0 means the aim handles are ignored entirely; 1 means the eyes point
    exactly at them.  Everything between blends, so pushing the handle up eases
    the eyes onto the target instead of snapping them there.

    Divided by the board's unit, like every other channel on it
    (``board.channel_value``) - this switch is a slider whose travel is
    ``1.0 * unit``, and reading it raw made the blend depend on the size of the
    character's head. See :func:`board.switch_unit` for the measurements.
    """
    aim = cache.get("eye_aim") if cache else None
    if aim is None:
        return 0.0
    try:
        switch = aim["switch"]
        if evaluated is not None:       # rendering; see _evaluated_controls
            switch = evaluated.get(switch.name, switch)
        unit = board.switch_unit(aim["board"])
        return min(1.0, max(0.0, float(switch.location.y) / unit))
    except (ReferenceError, AttributeError, ZeroDivisionError):
        return 0.0


def look_at_enabled(cache):
    """True while the aim handles have any say at all."""
    return look_at_weight(cache) > 0.0


# True while one of our own depsgraph / frame-change handlers is on the stack.
# Read only by _safe_depsgraph; see the crash it exists to stop.
_in_handler = False


@contextmanager
def _handler_scope():
    global _in_handler
    was = _in_handler
    _in_handler = True
    try:
        yield
    finally:
        _in_handler = was


def _safe_depsgraph(depsgraph=None):
    """An evaluated depsgraph it is safe to read from HERE, or None.

    ``bpy.context.evaluated_depsgraph_get()`` is not a getter.  It runs
    ``CTX_data_ensure_evaluated_depsgraph`` -> ``scene_graph_update_tagged``,
    i.e. a full depsgraph evaluation.  Called from inside a
    ``depsgraph_update_post`` callback it therefore RE-ENTERS the evaluation
    that is still running, and Blender dies in
    ``BKE_object_eval_eval_base_flags`` reading a NULL view-layer base - the
    graph's base indices no longer match the view layer it is walking.

    Reported by Souhail (2026-08-07) after loading a .mhfrt and toggling the
    landmark overlay: appending a character rebuilds the view layer's base
    array, the handler fires mid-flush, and ``_eye_aim_gui`` asked the context
    for a depsgraph. Same family as
    [nested bpy.ops flushing the depsgraph], different door in.

    So: every handler passes the depsgraph it was HANDED - that one is already
    evaluated and costs nothing - and asking the context is allowed only when
    no handler of ours is on the stack.  Inside one, callers get None and fall
    back to un-evaluated data for a tick, which the next tick corrects.
    """
    if depsgraph is not None:
        return depsgraph
    if _in_handler:
        return None
    try:
        return bpy.context.evaluated_depsgraph_get()
    except (AttributeError, RuntimeError):
        return None


def _eye_aim_gui(aim, depsgraph=None):
    """{gui index: value} the aim handles ask of the eye channels.

    For each eye: measure the direction from the eye joint to its aim handle in
    the eye's own neutral frame, then invert the eye's measured response (see
    :func:`_eye_channel_columns`) to get the two channel values that put the
    gaze there - which is what makes the eye land ON the handle rather than
    somewhere short of it.  The neutral frame is built from the
    PARENT's current pose, not from rest, which is what lets the eyes hold a
    world-fixed target while the head turns.

    With the handles untouched this returns zero on any character, whatever the
    chain's authored layout: what is measured is the rotation away from the
    handle's own rest direction, and at rest that rotation is the identity.
    """
    values = {}
    depsgraph = _safe_depsgraph(depsgraph)
    if depsgraph is None:
        # Inside a handler with nothing handed down. The gaze holds last tick's
        # solve for one update rather than taking the whole session with it.
        return values
    try:
        skeleton = aim["skel"].evaluated_get(depsgraph)
        armature = aim["board"].evaluated_get(depsgraph)
        skeleton_matrix = skeleton.matrix_world
        board_matrix = armature.matrix_world
        for eye_aim in aim["eyes"]:
            handle = armature.pose.bones.get(eye_aim["handle"])
            eye = skeleton.pose.bones.get(eye_aim["joint"])
            if handle is None or eye is None:
                continue
            parent = eye.parent
            if parent is not None:
                rest_to_parent = (parent.bone.matrix_local.inverted_safe()
                                  @ eye.bone.matrix_local)
                reference = skeleton_matrix @ parent.matrix @ rest_to_parent
            else:
                reference = skeleton_matrix @ eye.bone.matrix_local

            direction = ((board_matrix @ handle.head)
                         - (skeleton_matrix @ eye.head))
            if direction.length < EYE_AIM_EPS:
                continue
            local = (reference.to_3x3().inverted_safe()
                     @ direction.normalized()).normalized()

            # The handle's rotation away from its rest, applied to the eye's
            # neutral gaze.  The shortest rotation carrying one unit vector to
            # another is the one an artist means by dragging a target: swing the
            # handle 20 degrees around the eye and the eye follows 20 degrees.
            # At rest the two vectors are equal, the rotation is the identity,
            # and the gaze comes back exactly forward.
            gaze = eye_aim["rest"].rotation_difference(local) @ FORWARD

            # atan2 rather than asin(x / horizontal): identical in front of the
            # face, and it keeps the sign of Z, so a target dragged BEHIND the
            # head reads as a full turn away instead of folding back onto the
            # mirrored forward angle and aiming the eye at nothing.
            yaw = math.atan2(gaze.x, gaze.z)
            pitch = math.atan2(gaze.y, math.hypot(gaze.x, gaze.z))

            value_x, value_y = _eye_aim_channels(
                yaw, pitch, _eye_response(aim, eye_aim))
            values[eye_aim["x"]] = value_x
            values[eye_aim["y"]] = value_y
    except (ReferenceError, AttributeError, RuntimeError):
        return {}
    return values


def _eye_aim_channels(yaw, pitch, response):
    """The two channel values that make this eye look exactly there.

    The channels are not a clean yaw/pitch pair: on an eye the wrap left tilted,
    a full tx also lifts the gaze by a degree or so.  Dividing each angle by its
    own gain therefore lands close but not on the target, and the error grows
    with the deflection - 1.2 degrees at a 38 degree glance, measured.  So the
    2x2 response is inverted instead of just its diagonal.

    The map is linear per sign quadrant, so the exact answer is one 2x2 product
    - the work is only in picking the quadrant.  A solution belongs to the
    quadrant whose signs it has; the search is over four candidates and the
    first consistent one is it.  Values outside the eye's reach stay consistent
    and simply clamp, which saturates the gaze at the limit the way dragging the
    handle past the end of its travel should.

    Falls back to the diagonal when no quadrant agrees - only reachable if the
    cross terms were to outweigh the direct ones, which would mean a channel
    that no longer does what its name says.
    """
    columns, quadrants = response
    angles = np.array([yaw, pitch])
    for signs, inverse in quadrants.items():
        solved = inverse @ angles
        # Sign-consistent, with zero belonging to either side.
        if (solved[0] * signs[0] >= -EYE_AIM_EPS
                and solved[1] * signs[1] >= -EYE_AIM_EPS):
            return (max(-1.0, min(1.0, float(solved[0]))),
                    max(-1.0, min(1.0, float(solved[1]))))
    gain_x = columns[(0, 1.0 if yaw >= 0.0 else -1.0)][0]
    gain_y = columns[(1, 1.0 if pitch >= 0.0 else -1.0)][1]
    return (max(-1.0, min(1.0, yaw / gain_x)),
            max(-1.0, min(1.0, pitch / gain_y)))


def _evaluated_controls(cache, depsgraph):
    """The board as the RENDER depsgraph sees it - {bone name: pose bone}.

    None for a viewport graph, where the originals are already up to date:
    Blender copies each evaluated pose back onto the original datablock so the
    UI can show it, and reading them costs nothing.

    A render graph copies nothing back.  During a render the original board
    bones still hold whatever the viewport last left there - measured, they sit
    on frame 1 for the whole job while the evaluated ones move - so a rig read
    from the originals renders the face frozen on the frame the artist happened
    to be on.  That went unnoticed for as long as the interface stayed unlocked
    during renders, because then the viewport kept evaluating alongside the
    render and refreshing them.  That is also exactly the two-threads-one-graph
    race that crashed Blender (see core.render_state), so the lock had to go on
    and this read had to stop depending on it.
    """
    if depsgraph is None or not _is_render_depsgraph(depsgraph):
        return None
    controls = {}
    for owner in cache.get("board_owners", ()):
        try:
            pose = owner.evaluated_get(depsgraph).pose
        except (ReferenceError, RuntimeError, AttributeError):
            continue
        if pose is None:
            continue
        for pose_bone in pose.bones:
            controls.setdefault(pose_bone.name, pose_bone)
    return controls or None


def _read_gui(cache, apply_weights=True, depsgraph=None):
    div = cache["divisor"]
    evaluated = _evaluated_controls(cache, depsgraph)
    gui = np.zeros(cache["gui_count"])
    for i, source in enumerate(cache["sources"]):
        gui[i] = _gui_channel_value(source, div, evaluated)
    for source, indices in cache.get("center_eye_links", ()):
        offset = _gui_channel_value(source, div, evaluated)
        if abs(offset) <= 1e-12:
            continue
        for index in indices:
            gui[index] = np.clip(
                gui[index] + offset, EYE_GUI_MIN, EYE_GUI_MAX)
    # The aim handles then pull the eye channels towards the target by however
    # far the look-at handle is up: at 1 the eyes point exactly at it, master
    # centre-eye offset and all, and anywhere below that they sit between the
    # posed value and the solved one.  Blending the VALUE rather than switching
    # between two behaviours is what makes the handle usable as an ease-in.
    weight = look_at_weight(cache, evaluated)
    if weight > 0.0:
        for index, value in _eye_aim_gui(cache["eye_aim"], depsgraph).items():
            gui[index] += (value - gui[index]) * weight
    # Bone Intensity: the artist dialled one morph channel's bone movement
    # down.  We scale the DNA GUI channels that feed it, so RigLogic evaluates
    # as if the board control had moved only that far - at 0 as if it sat at
    # rest - without moving the controllers on screen.  A row applies only
    # while the channel is inside its window, so muting mouth_left leaves the
    # mouth_right half of the same channel alone; where rows overlap the
    # smallest factor wins, matching how op_morphs merges them.  Read fresh
    # from the skeleton every tick so a slider drag is a light one-eval
    # refresh, not a full rescan.  ``apply_weights=False`` gives the clean,
    # full-strength GUI - used to drive the shape keys (a muted channel keeps
    # deforming its own shape) and to rank the Morph picker so the list holds
    # still.
    skel = cache.get("skel")
    if apply_weights and skel is not None:
        gui_count = cache["gui_count"]
        scales = {}
        for index, lo, hi, factor in _channel_factors(skel):
            if not 0 <= index < gui_count:
                continue
            value = gui[index]
            if lo - CHANNEL_WINDOW_EPS <= value <= hi + CHANNEL_WINDOW_EPS:
                scales[index] = min(scales.get(index, 1.0), factor)
        for index, factor in scales.items():
            gui[index] *= factor
    return gui


SK_EPS = 1e-6         # shape-key value change threshold
SHAPE_BATCH_MIN = 24   # keep tiny manual edits on the cheaper scalar path


def update_rig(cache, gui=None, depsgraph=None):
    if gui is None:
        gui = _read_gui(cache, depsgraph=depsgraph)
    inputs, deltas = riglogic.evaluate_gui_with_inputs(
        gui, cache.get("last_inputs"), cache.get("last_flat"))
    # Bone Intensity scales BONES only: the shape keys (and the Morph picker)
    # read a second, full-strength input vector, so a channel muted down to 0
    # keeps driving its own sculpted shape - the whole point of muting it.
    # Ranking the picker from the clean pose also keeps the list steady: from
    # the scaled result, muting a channel would drop it and every morph
    # sharing its board channels out of the list.  The extra inputs-only pass
    # runs only while some channel is actually dialled down.
    skel = cache.get("skel")
    if skel is not None and skel.get(CHANNEL_GUI_INDICES_PROP):
        clean = riglogic.raw_to_inputs(
            riglogic.gui_to_raw(_read_gui(cache, apply_weights=False,
                                          depsgraph=depsgraph)))
    else:
        clean = inputs
    _apply_shape_keys(cache, clean,
                      cache.get("last_inputs_unmuted", cache.get("last_inputs")),
                      depsgraph=depsgraph)
    _apply_bones(cache, deltas, depsgraph=depsgraph)
    cache["last_inputs"] = inputs
    cache["last_flat"] = deltas.reshape(-1)
    cache["last_gui"] = gui
    cache["last_inputs_unmuted"] = clean


def _apply_bones(cache, deltas, depsgraph=None):
    """Push the RigLogic joint deltas to the pose bones.

    The whole rig is solved in one vectorised pass (838 bones is a handful of
    small matrix products) and only the bones whose basis actually moved get an
    RNA write, so a single control drag still costs a few writes. Body channels
    on a merged rig are never touched: each bone is addressed individually.

    `depsgraph` is handed down so a render reads the bones this rig does NOT
    drive off the frame being rendered - on a merged rig those are the artist's
    body and head controls, and reading them off the original froze every one
    of them for the whole job (see core.dna_apply.evaluated_pose_bones).
    """
    dna_apply.apply_joint_outputs(
        cache["skel"], cache["targets"], cache["pose_constants"],
        deltas, cache["scale"], depsgraph=depsgraph)


def _apply_shape_keys_scalar(items, inputs):
    for key, input_index in items:
        if input_index >= len(inputs):
            continue
        value = float(inputs[input_index])
        if abs(float(key.value) - value) <= SK_EPS:
            continue
        key.value = value


def _shape_key_groups_valid(groups, selected_by_group):
    for group, selected in zip(groups, selected_by_group):
        if not len(selected):
            continue
        blocks = group["blocks"]
        if len(blocks) != len(group["values"]):
            return False
    return True


def _evaluated_key_blocks(group, depsgraph):
    """This Key datablock as the RENDER depsgraph has it, or None.

    Same asymmetry as the pose (core.dna_apply.evaluated_pose_bones), same
    read-modify-write to go with it: the value array is read whole and written
    whole, so a key the artist keyframed themselves - a corrective on the same
    mesh, say - would be read off the stale original and pushed back over the
    render's own copy of it.  None for a viewport graph, where the original is
    already the authority.
    """
    if depsgraph is None or not _is_render_depsgraph(depsgraph):
        return None
    try:
        blocks = group["shape_keys"].evaluated_get(depsgraph).key_blocks
    except (ReferenceError, RuntimeError, AttributeError):
        return None
    return blocks if len(blocks) == len(group["values"]) else None


def _apply_shape_keys_batched(cache, inputs, changed, depsgraph=None):
    groups = cache.get("shape_key_groups", ())
    selected_by_group = []
    for group in groups:
        input_indices = group["input_indices"]
        valid = input_indices < len(inputs)
        selected = np.flatnonzero(valid)
        if changed is not None and len(selected):
            selected = selected[changed[input_indices[selected]]]
        selected_by_group.append(selected)

    if not _shape_key_groups_valid(groups, selected_by_group):
        return False

    for group, selected in zip(groups, selected_by_group):
        if not len(selected):
            continue
        blocks = group["blocks"]
        values = group["values"]
        source = _evaluated_key_blocks(group, depsgraph) or blocks
        source.foreach_get("value", values)
        destinations = group["block_indices"][selected]
        sources = group["input_indices"][selected]
        new_values = inputs[sources]
        different = np.abs(values[destinations] - new_values) > SK_EPS
        if not different.any():
            continue
        values[destinations[different]] = new_values[different]
        blocks.foreach_set("value", values)
    return True


def _apply_shape_keys(cache, inputs, last=None, depsgraph=None):
    targets = cache.get("shape_key_targets", ())
    if not targets:
        return
    if last is None:
        last = cache.get("last_inputs")
    if last is not None and len(last) == len(inputs):
        changed = np.abs(inputs - last) > SK_EPS
        items = [t for t in targets if t[1] < len(changed) and changed[t[1]]]
    else:
        changed = None
        items = targets      # first pass after (re)scan: sync everything
    # The scalar path assigns each value through RNA, and every one of those
    # runs the property's update - which queues a window-manager notifier. On a
    # render thread that means appending to a list the main thread is walking,
    # so a render takes the batched route however few keys moved: foreach_set
    # writes them without a single update callback (see dna_apply and
    # core.render_state).
    if len(items) <= SHAPE_BATCH_MIN and not render_state.is_rendering():
        _apply_shape_keys_scalar(items, inputs)
    elif not _apply_shape_keys_batched(cache, inputs, changed, depsgraph):
        # External key deletion invalidated the collection buffer length.
        # KeyBlock references still preserve this frame through the scalar path;
        # the next handler tick rebuilds the batched topology.
        invalidate()            # the cached key topology no longer matches
        _apply_shape_keys_scalar(items, inputs)


def _evaluate_guarded(caches, gui_values=None, depsgraph=None):
    """Evaluate rigs without re-triggering ourselves via the depsgraph.

    Pass `depsgraph` whenever there is one to hand - a handler always has one.
    It is what keeps the eye-aim read off ``context.evaluated_depsgraph_get()``;
    see :func:`_safe_depsgraph`.
    """
    global _updating
    _updating = True
    try:
        if gui_values is None:
            for cache in caches:
                update_rig(cache, depsgraph=depsgraph)
        else:
            for cache, gui in zip(caches, gui_values):
                update_rig(cache, gui, depsgraph)
    except ReferenceError:
        invalidate()            # something we cached was deleted/undone
    finally:
        _updating = False


def apply_scale_mul(context):
    """Expression-intensity slider: write through to the ACTIVE character's
    skeleton and re-evaluate it live."""
    mh = getattr(context.scene, "mhfrt", None)
    skel = mh.skeleton if mh else None
    if skel is None or RIG_ID_PROP not in skel:
        return
    skel[RIG_INTENSITY_PROP] = float(mh.riglogic_scale_mul)
    cache = _caches.get(skel[RIG_ID_PROP])
    if cache is not None:
        cache["scale"] = cache["base_scale"] * skel[RIG_INTENSITY_PROP]
        _evaluate_guarded([cache])


# ----------------------------------------------------------------- handler ---

@persistent
def _on_invalidate(*_args):
    """Undo/redo (and file load) rebuild datablocks in memory: every cached
    reference - especially pose bones, which Python cannot even detect as
    freed - becomes a dangling pointer, and touching one hard-crashes
    Blender. Drop everything IMMEDIATELY (undo_pre, before any depsgraph
    event can fire) and re-scan lazily from fresh data."""
    global _scan_needed, _scan_full, _pick_primed
    _caches.clear()
    _control_map.clear()
    _scan_needed = True
    _scan_full = True
    bump_rig_topology()
    _last_pick_by_layer.clear()     # those pointers belong to freed data
    # A queued eye-target distance describes a character in the file that is
    # going away. Names survive a load, so leaving it would move the chain on
    # whatever same-named skeleton the NEW file happens to have; and after an
    # undo the value it was going to write is the one just undone.
    _pending_eye_aim.clear()
    # Re-prime rather than act on the selection the file (or the undo step)
    # arrives with: opening a file must not yank the panel off the character
    # its saved workflow bookmark is about to restore.
    _pick_primed = False
    # A file that has just arrived may hold rigs; check on the next beat, when
    # bpy.data is settled, in case nothing else makes the runtime look before
    # the artist hits Render (see ensure_render_safety).
    if not bpy.app.background \
            and not bpy.app.timers.is_registered(_apply_render_safety):
        bpy.app.timers.register(_apply_render_safety, first_interval=0.0)


def _apply_render_safety():
    """Timer body for the above - the scan itself, off the handler stack."""
    try:
        ensure_render_safety(deep=True)
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    return None


def _is_cached_owner(cache, idb):
    """Is `idb` the very skeleton this cache was built for?

    Two skeletons can carry one rig id - that is what a duplicate IS - and the
    cache is keyed by the id, so the id alone cannot tell them apart.  Only the
    object identity can, which is why this compares pointers and not tags.
    """
    skel = cache.get("skel") if cache else None
    try:
        return skel is not None and skel.as_pointer() == _orig_pointer(idb)
    except ReferenceError:
        return False


def _is_cached_control_owner(cache, idb):
    """Is `idb` one of the board owners this cache was built from?"""
    pointer = _orig_pointer(idb)
    for owner in (cache.get("board_owners", ()) if cache else ()):
        try:
            if owner.as_pointer() == pointer:
                return True
        except ReferenceError:
            continue
    return False


def _orig_pointer(idb):
    """The ORIGINAL datablock's address for an id from depsgraph.updates.

    Updates hand out the evaluated copy-on-write copy; `_control_map` is built
    from the originals, so the two only meet through `.original`.
    """
    try:
        return idb.original.as_pointer()
    except (AttributeError, ReferenceError):
        return idb.as_pointer()


def _is_render_depsgraph(depsgraph):
    return getattr(depsgraph, "mode", 'VIEWPORT') == 'RENDER'


def ensure_render_safety(scene=None, deep=False):
    """Make this scene safe to render a live rig from. See core.render_state.

    The rig writes bones and tags the armature from the render job's thread;
    Blender's interface lock is what stops its main thread from flushing the
    same depsgraph at the same time.  The flag is read when the job STARTS -
    measured: setting it from a ``render_init`` handler leaves
    ``wm.is_interface_locked`` false for the whole job - so it has to be on
    before the artist ever presses the button.  Hence: every depsgraph tick,
    every frame change, after a file load, and before every save.

    `deep` looks for rig tags in the file when no cache is built yet.  Opening
    a file and going straight to Render is a real path: the caches are built on
    the first depsgraph tick, and nothing guarantees one happens first.  It is
    off for the per-tick callers, where it would be a scan of every object in
    the file for nothing.

    `scene` is resolved AFTER the guards: asking ``bpy.context`` is only safe
    on the main thread, and a render thread must not get that far.
    """
    if render_state.is_rendering():
        return
    if not _caches and not (deep and _rig_skeletons()):
        return
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if render_state.ensure_lock(scene):
        print("[MHFRT] Lock Interface switched on for scene %r: a live rig is "
              "posed from Python, and rendering one without the lock crashes "
              "Blender. Preferences > Lock Interface While Rendering turns "
              "this off." % getattr(scene, "name", "?"))


def has_live_rigs():
    """True when at least one live rig is being driven in this file."""
    return bool(_caches)


@persistent
def _rig_handler(scene, depsgraph):
    if _updating:
        return
    if render_state.is_rendering() and not _is_render_depsgraph(depsgraph):
        # A render job is evaluating these rigs on its own thread. Driving them
        # from the viewport graph at the same time is the race in
        # core.render_state; the viewport catches up when the render ends.
        return
    with _handler_scope():
        _rig_handler_body(scene, depsgraph)


def _rig_handler_body(scene, depsgraph):
    try:
        ensure_render_safety(scene)
        if _scan_needed:
            was_full = _scan_full
            built = set(_caches)
            _rescan(full=was_full)
            # Again, because THIS is the tick a freshly opened file learns it
            # has rigs at all - and the artist can reach for F12 before the
            # next one comes.
            ensure_render_safety(scene)
            # Only rigs that were actually (re)built need evaluating.  Waking
            # every rig in the file because one character was appended is what
            # made adding the Nth rig cost N times as much as the first.
            fresh = [c for rid, c in _caches.items()
                     if was_full or rid not in built]
            if fresh:
                _evaluate_guarded(fresh, depsgraph=depsgraph)
            if was_full:
                return          # every rig was just evaluated from scratch
            # An incremental scan must NOT swallow this tick's updates: the
            # control the artist is dragging right now is in them, and
            # returning here left the board posed with the face unmoved until
            # something else happened to fire the handler again.
        shape_hit = {
            rid for rid, cache in _caches.items()
            if _sync_active_shape_key_mutes(cache)
        }
        hit = set(shape_hit)
        needs_full = False
        for u in depsgraph.updates:
            idb = u.id
            if not isinstance(idb, bpy.types.Object):
                continue
            # .original: depsgraph updates report the EVALUATED copy, whose
            # pointer is not the one the map was built from.  (This is why the
            # map used names before - and why renaming a board silently killed
            # its rig.)
            rid = _control_map.get(_orig_pointer(idb))
            if rid is not None:
                hit.add(rid)
            elif board.is_board_armature(idb) or idb.get(board.TEMPLATE_PROP):
                # A control the map does not know: a board that was appended,
                # rebuilt, or re-tagged since the last scan.  Drive its rig now
                # if we have it, and re-index either way - never sit there
                # doing nothing while the artist poses a live control.
                own = str(idb.get(RIG_ID_PROP) or "")
                if own and own in _caches:
                    hit.add(own)
                    if not _is_cached_control_owner(_caches[own], idb):
                        # A second board on a live id: the artist duplicated
                        # the board (or the whole character). Same reasoning as
                        # the skeleton case - only a full pass repairs it.
                        needs_full = True
                invalidate_new_rigs()
            elif _is_rig_skeleton(idb):
                own = str(idb.get(RIG_ID_PROP) or "")
                if own not in _caches:
                    # a character was appended/linked in: build ITS cache, and
                    # leave every other rig's alone
                    invalidate_new_rigs()
                elif not _is_cached_owner(_caches[own], idb):
                    # A SECOND skeleton answering to a LIVE id - the artist
                    # just duplicated a character. Only a FULL pass runs the
                    # id repair, and an incremental one cannot: the id is
                    # already known, which is exactly why this used to sit
                    # there doing nothing and hand back a dead copy.
                    needs_full = True
                elif look_at_enabled(_caches[own]):
                    # The aim is solved against the eye joints' CURRENT pose, so
                    # turning the head has to re-solve it or the eyes would
                    # swing with the head instead of holding their target.
                    # Only while the switch is up: otherwise this would re-read
                    # every rig every time its own bones were written.
                    hit.add(own)
        if needs_full:
            # Acted on AFTER the loop, never inside it: the repair rewrites
            # tags and clears the caches this loop is still reading.
            invalidate()
            _rescan(full=True)
            _evaluate_guarded(_caches.values(), depsgraph=depsgraph)
            return
        if hit:
            changed = []
            gui_values = []
            for rid in hit:
                cache = _caches.get(rid)
                if cache is None:
                    continue
                _clamp_board_to_frames(cache)
                gui = _read_gui(cache, depsgraph=depsgraph)
                last = cache.get("last_gui")
                if (rid in shape_hit or last is None
                        or not np.array_equal(gui, last)):
                    changed.append(cache)
                    gui_values.append(gui)
            if changed:
                _evaluate_guarded(changed, gui_values, depsgraph=depsgraph)
    except ReferenceError:
        invalidate()


def _clamp_board_to_frames(cache):
    """Snap any handle the artist dragged past its frame back onto the edge.

    Skipped while an action drives the board: the fcurve owns the value then,
    and clamping would just be overwritten on the next frame.

    Skipped during a render as well.  Nothing is being dragged then, and the
    write would go onto the ORIGINAL board - which re-syncs the render's copy
    from it and drops the animation off every channel that copy carries, the
    same way it froze the artist's bones (core.dna_apply.evaluated_pose_bones).
    """
    if render_state.is_rendering():
        return
    for owner in cache.get("board_owners", ()):
        try:
            animation = owner.animation_data
        except ReferenceError:
            return
        if animation is not None and animation.action is not None:
            return
    for control in cache["controls"].values():
        board.clamp_into_frame(control)


@persistent
def _frame_change_handler(scene, depsgraph=None):
    """Evaluate animated control locations after Blender changes frame.

    Manual transforms produce Object depsgraph updates, but keyed interpolation
    does not reliably produce one for these board objects.  Frame evaluation is
    therefore explicit so scrubbing, playback, NLA and rendering all drive the
    facial bones and morph values.
    """
    if _updating:
        return
    with _handler_scope():
        _frame_change_body(depsgraph)


def _frame_change_body(depsgraph):
    try:
        ensure_render_safety()
        if _scan_needed:
            _rescan(full=_scan_full)
            _evaluate_guarded(_caches.values(), depsgraph=depsgraph)
            return
        changed = []
        gui_values = []
        for cache in _caches.values():
            shape_state_changed = _sync_active_shape_key_mutes(cache)
            gui = _read_gui(cache, depsgraph=depsgraph)
            last = cache.get("last_gui")
            if (shape_state_changed or last is None
                    or not np.array_equal(gui, last)):
                changed.append(cache)
                gui_values.append(gui)
        if changed:
            _evaluate_guarded(changed, gui_values, depsgraph=depsgraph)
    except ReferenceError:
        invalidate()


# ---------------------------------------------------------------- board ---

def _rig_gui_collection(skel):
    coll = skel.get(RIG_GUI_COLL_PROP)
    return coll if isinstance(coll, bpy.types.Collection) else None


def _has_board(skel):
    """True when this rig already owns controls, whatever layout they use.

    The GUI collection pointer alone is not enough: an artist may reorganise
    the board into their own collections, and a rig whose pointer was lost
    must not silently gain a second board.
    """
    return bool(board.controls_for_rig(RIG_ID_PROP, skel.get(RIG_ID_PROP),
                                       skel=skel))


def _world_bounds(obj):
    """(min, max) world-space bounds of what is actually on screen.

    A mesh is measured from its EVALUATED geometry rather than ``bound_box``.
    The cached box follows the shape keys but not the modifier stack, and the
    head target here is bound to the facial rig and may carry the artist's own
    modifiers - measuring the box would put the board beside a head of the
    wrong size.  The two agree whenever no modifier deforms the mesh, so this
    is the same measurement the reference importer makes, just taken after the
    stack instead of before it.
    """
    matrix = obj.matrix_world
    depsgraph = _safe_depsgraph()
    if obj.type == 'MESH' and depsgraph is not None:
        try:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            count = len(mesh.vertices)
            if count:
                coords = np.empty(count * 3)
                mesh.vertices.foreach_get("co", coords)
                evaluated.to_mesh_clear()
                points = coords.reshape(-1, 3)
                low = Vector(points.min(axis=0).tolist())
                high = Vector(points.max(axis=0).tolist())
                corners = [matrix @ Vector((x, y, z))
                           for x in (low.x, high.x)
                           for y in (low.y, high.y)
                           for z in (low.z, high.z)]
                return (Vector(min(c[i] for c in corners) for i in range(3)),
                        Vector(max(c[i] for c in corners) for i in range(3)))
            evaluated.to_mesh_clear()
        except (RuntimeError, AttributeError, ReferenceError):
            pass
    corners = [matrix @ Vector(corner) for corner in obj.bound_box]
    return (Vector(min(c[i] for c in corners) for i in range(3)),
            Vector(max(c[i] for c in corners) for i in range(3)))


# The eye-aim chain is a second root on the board, floating in front of the
# character's eyes rather than sitting on the panel; it has to be moved on its
# own. Names are the reference importer's EYE_AIM_BONES.
_EYE_AIM_BONES = (
    "LOC_R_eyeUIDriver", "LOC_L_eyeUIDriver", "LOC_C_eyeUIDriver",
    "LOC_R_eyeDriver", "LOC_L_eyeDriver", "LOC_C_eyeDriver",
    "LOC_R_eyeAimDriver", "LOC_L_eyeAimDriver",
    "LOC_R_eyeAimUp", "LOC_L_eyeAimUp",
    "GRP_convergenceGUI", "GRP_L_eyeAim", "GRP_R_eyeAim",
    "FRM_convergenceGUI", "FRM_convergenceSwitch", "TEXT_convergence",
    "CTRL_C_eyesAim", "CTRL_L_eyeAim", "CTRL_R_eyeAim",
    "CTRL_convergenceSwitch",
)
# How far in front of the eyes the aim control sits, in metres on a human-sized
# head (the reference importer's constant), scaled with the character.  The
# default only: `eye_aim_distance` returns whatever this character was set to.
EYE_AIM_DISTANCE = 0.3
# Below this the aim chain has collapsed to a point and its circle separation
# can no longer be measured, so a fit is refused rather than dividing by it.
EYE_AIM_MIN_SPAN = 1.0e-6


@contextmanager
def _panel_only(arm_obj):
    """Measure the PANEL, with the eye-aim chain out of the way.

    An armature's ``bound_box`` spans its VISIBLE bones only, and the aim chain
    floats 0.3 m in FRONT of the face rather than sitting on the panel - so
    whether it is drawn moves the board's measured extent by ~22 cm in X and
    ~19 cm in Y, which is enough to shove the panel right off the head.

    It used to be excluded by accident: the bundled asset ships the chain hidden
    (Character DNA hides it below CTRL_lookAtSwitch 0.99) and nothing here ever
    revealed it.  Now that ``board.expose_eye_aim`` does, the exclusion has to be
    deliberate, or the placement silently depends on a visibility flag.  The
    chain gets its own position from ``_place_eye_aim`` immediately afterwards.
    """
    collections = getattr(arm_obj.data, "collections_all",
                          getattr(arm_obj.data, "collections", ()))
    coll = next((c for c in collections
                 if c.name == board.EYE_AIM_COLLECTION), None)
    was_visible = coll.is_visible if coll is not None else None
    hidden = [pb for pb in arm_obj.pose.bones
              if pb.name in board.EYE_AIM_BONES and not board.bone_hidden(pb)]
    if coll is not None:
        coll.is_visible = False
    for pose_bone in hidden:
        board.set_bone_hidden(pose_bone, True)
    bpy.context.view_layer.update()      # bound_box is evaluated, not cached
    try:
        yield
    finally:
        for pose_bone in hidden:
            board.set_bone_hidden(pose_bone, False)
        if coll is not None:
            coll.is_visible = was_visible
        bpy.context.view_layer.update()


def _eye_pair_world(skel):
    """This character's two eye joints AT REST, in world space, as (left, right).

    The one landmark on the rig that means "the face is HERE" whatever the
    anchor mesh happens to be - and, taken as a pair rather than a midpoint,
    also how wide the face's eyes are, which is what the eye-aim circles are
    sized against.

    At REST, unlike ``_eyes_world`` below, because what it measures is written
    into the aim chain's own REST: read the posed heads instead and pressing
    Fit with the head turned would bake that turn into the rig, leaving the
    target off to one side the moment the head came back.  It is also the
    frame ``_eye_rest_direction`` captures the solve's baseline in, so the two
    agree by construction.  The upshot is that fitting is safe from any pose.
    """
    if skel is None or getattr(skel, "data", None) is None:
        return None
    left = skel.data.bones.get("FACIAL_L_Eye")
    right = skel.data.bones.get("FACIAL_R_Eye")
    if left is None or right is None:
        return None
    matrix = skel.matrix_world
    return matrix @ left.head_local, matrix @ right.head_local


def _eyes_world(skel):
    """The midpoint between this character's eye joints, POSED, in world space.

    The board is placed against the anchor mesh's bounding box, which is the
    evaluated, posed mesh, so the height that lines up with it has to be posed
    too.  The eye-aim chain has the opposite requirement and uses
    ``_eye_pair_world``.
    """
    if skel is None or getattr(skel, "pose", None) is None:
        return None
    left = skel.pose.bones.get("FACIAL_L_Eye")
    right = skel.pose.bones.get("FACIAL_R_Eye")
    if left is None or right is None:
        return None
    return skel.matrix_world @ ((left.head + right.head) * 0.5)


def eye_span(skel):
    """How far apart this character's eye joints are, in metres. 0 if unknown."""
    pair = _eye_pair_world(skel)
    return 0.0 if pair is None else (pair[0] - pair[1]).length


def board_scale_for(skel, arm_obj):
    """How much bigger this character's panel should be than the authored one.

    The EYES decide, exactly as they decide the look-at target's size - the
    panel and the aim frame were authored as one design and stay in it, so
    whatever the aim chain is scaled to, the panel is scaled to as well.

    Measured, not remembered: the authored separation of the two aim circles is
    recovered by dividing the board's current one by the unit already baked
    into it, so this returns the same number whether the board has just been
    appended or has been placed onto three characters already.

    Falls back to the wrap's head-size ratio - the pre-4.17.0 rule - for a rig
    with no eye joints to ask or a board with no aim chain to measure.  That
    ratio is a whole-head average, which is why it is second choice: a head
    that is wide but short scales the panel by neither of its own dimensions,
    where the eye spacing is the same thing the face's own controls are
    proportioned to.
    """
    ratio = float(skel.get("mhfrt_riglogic_scale", 1.0)) or 1.0
    span = eye_aim_span(skel, arm_obj)
    eyes = eye_span(skel)
    unit = board.placement_scale(arm_obj)
    if span < EYE_AIM_MIN_SPAN or eyes < EYE_AIM_MIN_SPAN or unit <= 0.0:
        return ratio
    authored = span / unit
    if authored < EYE_AIM_MIN_SPAN:
        return ratio
    wanted = eyes / authored
    return wanted if wanted > EYE_AIM_MIN_SPAN else ratio


def _place_bone_board(skel, arm_obj, anchor, place_eye_aim=True):
    """Sit the board beside this character's head, the way the DNA rig does.

    The reference importer's rule, from its ``position_face_board``: put the
    board's LEFT edge exactly on the head's RIGHT edge so the panel sits next to
    the face without touching it, and line the two bounding-box centres up
    vertically.  Reproduced here so a transferred character's panel lands where
    an artist expects it.

    Three deliberate differences, all of which reduce to the reference
    behaviour on a normally-sized character:

    * the board is scaled to the character.  The reference rig is always
      MetaHuman-sized; this add-on transfers onto heads of any size, and an
      unscaled panel beside a giant head is unusable.  The factor is the EYE
      SPACING one (``board_scale_for``), the same one the look-at target is
      fitted with, so the panel and the aim frame stay in the proportion they
      were authored in.  It is 1.0 for a MetaHuman-sized face, so nothing moves
      in the ordinary case.
    * depth (Y) is lined up with the head as well.  The reference leaves it at
      the authored value because its character is always at the origin - on the
      bundled asset that value already sits within 9 mm of the head centre - but
      a transferred character can be anywhere in the scene, and a board left at
      Ada's depth would float away from it.
    * height (Z) comes from the EYE JOINTS, not from the anchor's bounding box,
      and so - through ``_place_eye_aim`` - does the whole size and position of
      the look-at target.
      The reference is always handed a head, where the two agree.  This add-on
      is routinely handed a whole body - v3.18.0 made that a supported target -
      and the centre of a body is somewhere around the waist, which is where
      the panel went: correctly placed to the character's right, and level with
      their hips.  The eyes are the one landmark that means "the face is here"
      whatever mesh the anchor turns out to be, and it puts the panel at the
      same height as the eye-aim control, which is placed from the same point.

    Only the board OBJECT is transformed. Pose-bone locations are bone-local, so
    moving and scaling the object never changes a single GUI value. Any existing
    parenting is preserved - a merged rig hangs the board off the head bone.
    Re-running is idempotent: the placement is recomputed from scratch.

    ``place_eye_aim=False`` leaves the aim chain where it is.  It is placed once
    at build time and is right; re-running it costs an Edit Mode round trip on
    435 bones for no gain, so Panel Layout > Reset Placement skips it.
    """
    if anchor is None:
        return
    ratio = board_scale_for(skel, arm_obj)
    previous = board.placement_matrix(arm_obj)
    undo = previous.inverted_safe()

    # Measure the board as if it were AUTHORED, at the wanted scale: the rest
    # already carries the last placement, so undoing it here is what keeps
    # calling this twice from stacking one placement on top of another.
    arm_obj.matrix_world = Matrix.Scale(ratio, 4) @ undo
    head_low, head_high = _world_bounds(anchor)
    with _panel_only(arm_obj):
        board_low, board_high = _world_bounds(arm_obj)
    head_centre = (head_low + head_high) * 0.5
    board_centre = (board_low + board_high) * 0.5
    # Eye height, or the anchor's centre on a rig with no eye joints to ask.
    eyes = _eyes_world(skel)
    eyes_z = eyes.z if eyes is not None else head_centre.z

    offset = Vector((
        head_high.x - board_low.x,
        head_centre.y - board_centre.y,
        eyes_z - board_centre.z,
    ))
    wanted = Matrix.Translation(offset) @ Matrix.Scale(ratio, 4)
    _flatten_board(skel, arm_obj, wanted, previous)
    if place_eye_aim:
        _place_eye_aim(skel, arm_obj, ratio)


def place_eye_aim(skel, arm_obj=None, distance=None, fit=True):
    """Public entry: re-fit and re-place this character's look-at target.

    Returns the distance it ended up at, or None when there is nothing to
    place.  ``fit=False`` moves it without resizing - what the distance
    control does, since changing how far out the target sits is not a reason
    to change how big it is.
    """
    if arm_obj is None:
        arm_obj = board_armature(skel)
    if skel is None or arm_obj is None or _eye_pair_world(skel) is None:
        return None
    ratio = board.placement_scale(arm_obj)
    if distance is None:
        distance = eye_aim_distance(skel, ratio)
    _place_eye_aim(skel, arm_obj, ratio, distance=distance, fit=fit)
    return distance


# Distances set by the slider, waiting for a moment when moving the chain is
# safe. {skeleton name: metres}, plus when the last one was set.
_pending_eye_aim = {}
_eye_aim_touched = 0.0
# How long the value has to hold still before it is applied, and how long the
# apply may wait for Object Mode before giving up on being polite.
_EYE_AIM_SETTLE = 0.25
_EYE_AIM_PATIENCE = 30.0


def queue_eye_aim_distance(skel, distance):
    """Set the look-at target's distance, and move it once the artist stops.

    Two reasons this is not done in the setter.  A slider fires its setter on
    every mouse move, and moving the chain is an Edit Mode round trip on the
    board - a mode switch, dozens of times a second, underneath the live modal
    operator that is drawing the slider.  And a mode switch would throw an
    artist posing that same board out of Pose Mode, so it waits for Object
    Mode rather than taking it.

    The number is stored immediately, so the slider always reads back what was
    typed; only the geometry lags, and only until the drag settles.
    """
    global _eye_aim_touched
    if skel is None:
        return
    skel[board.EYE_AIM_DISTANCE_PROP] = float(distance)
    _pending_eye_aim[skel.name] = (float(distance), time.monotonic())
    _eye_aim_touched = time.monotonic()
    if not bpy.app.timers.is_registered(_apply_pending_eye_aim):
        bpy.app.timers.register(_apply_pending_eye_aim,
                                first_interval=_EYE_AIM_SETTLE)


def _apply_pending_eye_aim():
    """Move every queued look-at target, once it is safe to.

    Returns a delay to be asked again, or None when there is nothing left.
    """
    if render_state.is_rendering():
        return _RENDER_RETRY
    now = time.monotonic()
    if now - _eye_aim_touched < _EYE_AIM_SETTLE:
        return _EYE_AIM_SETTLE          # still dragging
    mode = getattr(bpy.context, "mode", 'OBJECT')
    waiting = {name: (distance, queued)
               for name, (distance, queued) in _pending_eye_aim.items()
               if now - queued < _EYE_AIM_PATIENCE}
    if mode != 'OBJECT' and waiting:
        # Someone is in Pose or Edit Mode. Come back rather than drag them out
        # of it - unless they have been there so long that the slider and the
        # rig have been out of step for half a minute, which is worse.
        _pending_eye_aim.clear()
        _pending_eye_aim.update(waiting)
        return _EYE_AIM_SETTLE
    pending = dict(_pending_eye_aim)
    _pending_eye_aim.clear()
    for name, (distance, _queued) in pending.items():
        skel = bpy.data.objects.get(name)
        if skel is None or skel.type != 'ARMATURE':
            continue
        # Distance only: how far out the target sits is not a reason to
        # change how big it is.
        place_eye_aim(skel, distance=distance, fit=False)
    return None


def bake_panel_bars(arm_obj):
    """Make the three panel bars read 0/0/0, identity, 1/1/1 at rest.

    The maths, and why only these three bones can have it, is in
    ``board.plan_bar_bake``.  Here is the mechanism, for one bar B whose
    authored basis is ``M = T(l) . R . S(s)`` with s uniform:

    * B's own rest absorbs the translation and the rotation
      (``rest_B' = rest_B . T(l) . R``).  Its pose matrix then loses exactly
      ``S(s)``, and the widget is drawn at
      ``bone length x custom_shape_scale x pose_scale`` (every bone on this
      board has ``use_custom_shape_bone_size`` on), so the same factor goes onto
      ``custom_shape_scale_xyz`` and the identical handle is drawn.  Onto the
      widget scale rather than the bone LENGTH, which would work equally well on
      paper: ``CTRL_faceGUI`` would become a 61-micron bone, and a bone that
      small is awkward to see, awkward to select, and a poor thing to leave in
      an artist's armature.
    * a rest cannot hold a scale - Blender's bone offset is a rotation and a
      translation, nothing else - so ``S(s)`` moves down into B's single child
      frame: its rest offset from B grows by s, and its own pose location and
      scale grow by s.  That frame's pose matrix is then unchanged, which is
      what leaves its whole subtree, and every GUI channel in it, alone.

    Returns the number of bars folded.  Idempotent: a bar already at identity
    has nothing to fold and is skipped.
    """
    plan = board.plan_bar_bake(arm_obj)
    if not plan:
        return 0
    context = bpy.context
    live = {}
    for template, _basis in plan.items():
        pose_bone = arm_obj.pose.bones.get(template)
        if pose_bone is not None:
            live[template] = pose_bone.matrix_basis.copy()

    with organization.shown_for_edit(arm_obj, view_layer=context.view_layer):
        was_active = context.view_layer.objects.active
        context.view_layer.objects.active = arm_obj
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except RuntimeError:
            context.view_layer.objects.active = was_active
            return 0
        edit_bones = arm_obj.data.edit_bones
        # A connected bone's head is pinned to its parent's tail, so writing one
        # would silently drag the other. None of the board's bones is connected;
        # this makes sure of it rather than assuming it.
        connected = [bone for bone in edit_bones if bone.use_connect]
        for bone in connected:
            bone.use_connect = False
        rest = {bone.name: bone.matrix.copy() for bone in edit_bones}
        length = {bone.name: bone.length for bone in edit_bones}
        children = {}
        for bone in edit_bones:
            if bone.parent is not None:
                children.setdefault(bone.parent.name, []).append(bone.name)

        def subtree(name):
            out, stack = [], [name]
            while stack:
                current = stack.pop()
                out.append(current)
                stack.extend(children.get(current, ()))
            return out

        def write(name, matrix, bone_length):
            bone = edit_bones[name]
            bone.matrix = matrix
            bone.length = max(bone_length, 1.0e-6)
            rest[name] = matrix

        folded = 0
        for template, basis in plan.items():
            bar = edit_bones.get(template)
            if bar is None:
                continue
            scale = basis.to_scale().x
            if abs(scale) < 1.0e-9:
                continue
            # the basis without its scale: translation and rotation only, which
            # is all a bone rest can hold
            rigid = (Matrix.Translation(basis.translation)
                     @ basis.to_quaternion().to_matrix().to_4x4())
            bar_rest = rest[template]
            bar_rest_new = bar_rest @ rigid
            for name in children.get(template, ()):
                offset = bar_rest.inverted_safe() @ rest[name]
                offset.translation = offset.translation * scale
                child_new = bar_rest_new @ offset
                delta = child_new @ rest[name].inverted_safe()
                for descendant in subtree(name):
                    write(descendant, delta @ rest[descendant],
                          length[descendant])
            write(template, bar_rest_new, length[template])
            folded += 1

        for bone in connected:
            bone.use_connect = True
        bpy.ops.object.mode_set(mode='OBJECT')
        if was_active is not None \
                and context.view_layer.objects.get(was_active.name):
            context.view_layer.objects.active = was_active

    for template, basis in plan.items():
        scale = basis.to_scale().x
        for name in [b.name for b in arm_obj.pose.bones
                     if b.parent is not None and b.parent.name == template]:
            child = arm_obj.pose.bones[name]
            child.location = child.location * scale
            child.scale = [v * scale for v in child.scale]
            # The stamps are moved ARITHMETICALLY, never re-read off the live
            # pose: a frame the artist has redesigned would otherwise have that
            # redesign enshrined as the authored design and Repair would stop
            # being able to undo it.
            child_rest = board.tag(child, board.REST_PROP)
            if child_rest is not None:
                try:
                    child[board.REST_PROP] = tuple(
                        float(v) * scale for v in child_rest[:3])
                except (TypeError, ValueError):
                    pass
            authored = board.design_rest(child)
            if authored is not None:
                quat, authored_scale = authored
                child[board.SHAPE_PROP] = (*quat,
                                           *(v * scale for v in authored_scale))
        bar = arm_obj.pose.bones.get(template)
        if bar is None:
            continue
        # The scale the pose gave up goes onto the widget, so the handle is
        # drawn at exactly the size it always was.
        current = getattr(bar, "custom_shape_scale_xyz", None)
        if current is not None:
            bar.custom_shape_scale_xyz = [v * scale for v in current]
        board.store_widget(bar)
        # Identity IS the bar's authored state now - that is the whole point, so
        # it is written as such rather than read back off whatever pose the bar
        # happens to be in.
        bar[board.REST_PROP] = (0.0, 0.0, 0.0)
        bar[board.SHAPE_PROP] = (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        # And the bar's LIVE pose restated in the new frame, so a board whose
        # bars the artist already arranged keeps that arrangement while a fresh
        # one lands exactly on the identity.
        bar.matrix_basis = basis.inverted_safe() @ live.get(template,
                                                            basis.copy())
    board.note_bar_bake(arm_obj, plan)
    return folded


def reflatten_board(skel, arm_obj):
    """Return a board to identity after its PARENT changed, without moving it.

    The merge re-parents the board from the facial skeleton to the survivor of
    the join, keeping its world transform - which necessarily leaves a
    non-identity ``matrix_basis`` behind.  This pushes that back into the rest
    so the object is clean again, and it does not re-place anything: the board
    is already where it belongs, and the merge is not the moment to move it.

    Restating the units is part of it and has to be: if the new parent is at a
    different scale, one armature unit is now worth something else, and
    ``dna_gui_scale`` moves with it.
    """
    if arm_obj is None or skel is None or not board.is_flattened(arm_obj):
        return False
    if arm_obj.matrix_basis == Matrix.Identity(4) \
            and arm_obj.matrix_parent_inverse == Matrix.Identity(4):
        return False
    # ``matrix_world`` is EVALUATED, not computed on read: after a re-parent it
    # still reports where the board used to hang until the depsgraph catches up,
    # and flattening a stale matrix moves the panel by the difference.
    bpy.context.view_layer.update()
    placement = board.placement_matrix(arm_obj)
    # ``wanted`` is where the AUTHORED board should end up, so it is the object's
    # current transform WITH the placement already in the rest - not the object
    # transform alone, which would throw the existing placement away and drop
    # the panel back on the character's origin.
    _flatten_board(skel, arm_obj, arm_obj.matrix_world @ placement, placement)
    return True


def _flatten_board(skel, arm_obj, wanted, previous):
    """Put the placement in the REST and leave the object at 0/0/0, 1/1/1.

    ``wanted`` is where the board should end up in WORLD space; ``previous`` is
    what is already baked into its rest.  Only the difference is applied, so
    re-placing is exact rather than cumulative.

    Expressed in the PARENT's space, because "the object is at identity" means
    ``matrix_basis`` is the identity, and for a board parented to the skeleton
    that puts its world matrix equal to the skeleton's.  The rest therefore has
    to carry the placement relative to the skeleton, which is also what makes
    the panel travel with the character exactly as it did before.

    Scaling the rest changes what one armature unit is worth, so every
    bone-local length moves with it and ``dna_gui_scale`` - the divisor
    ``channel_value`` reads through - is set to match.  That pairing is what
    makes this invisible to the rig: the GUI vector is identical afterwards.
    """
    parent = arm_obj.parent
    if parent is not None and arm_obj.parent_type == 'OBJECT':
        to_parent = parent.matrix_world.inverted_safe()
    elif parent is not None:
        # Bone- or vertex-parented boards keep the old behaviour: their parent
        # space is not a plain object matrix, and guessing at it would put the
        # panel somewhere nobody asked for.
        arm_obj.matrix_world = wanted
        return
    else:
        to_parent = Matrix.Identity(4)
    local = to_parent @ wanted
    delta = local @ previous.inverted_safe()

    context = bpy.context
    with organization.shown_for_edit(arm_obj, view_layer=context.view_layer):
        was_active = context.view_layer.objects.active
        context.view_layer.objects.active = arm_obj
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except RuntimeError:
            context.view_layer.objects.active = was_active
            arm_obj.matrix_world = wanted       # the old behaviour, unharmed
            return
        # Read every bone first: moving a head can drag a connected child's
        # tail, so writing while still reading would transform some bones twice.
        edit_bones = arm_obj.data.edit_bones
        moved = [(bone, delta @ bone.head, delta @ bone.tail, bone.roll)
                 for bone in edit_bones]
        for bone, head, tail, roll in moved:
            bone.head = head
            bone.tail = tail
            bone.roll = roll        # a translate + uniform scale cannot change
        bpy.ops.object.mode_set(mode='OBJECT')
        if was_active is not None \
                and context.view_layer.objects.get(was_active.name):
            context.view_layer.objects.active = was_active

    factor = local.to_scale().x / (previous.to_scale().x or 1.0)
    board.rescale_channels(arm_obj, factor)
    board.store_placement(arm_obj, local)
    skel["dna_gui_scale"] = board.placement_scale(arm_obj)
    arm_obj.matrix_parent_inverse = Matrix.Identity(4)
    arm_obj.matrix_basis = Matrix.Identity(4)
    # ``matrix_world`` is evaluated, not computed on read, so until the graph
    # catches up it still reports the temporary matrix the measurement above
    # used.  _place_eye_aim converts world coordinates through it to find where
    # to put the aim chain, and reading the stale one put the eye target
    # somewhere near the panel instead of in front of the character's eyes.
    context.view_layer.update()


def eye_aim_distance(skel, ratio=None):
    """How far in front of the eyes this character's aim target floats, in m.

    The artist's own value once they have set one, and until then the
    reference importer's 0.3 m scaled by the head-size ratio - which is what
    every character built before the control existed already has.
    """
    if skel is None:
        return EYE_AIM_DISTANCE
    stored = skel.get(board.EYE_AIM_DISTANCE_PROP)
    if stored is not None:
        try:
            value = float(stored)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    if ratio is None:
        ratio = float(skel.get("mhfrt_riglogic_scale", 1.0)) or 1.0
    return EYE_AIM_DISTANCE * ratio


def eye_aim_span(skel, arm_obj=None):
    """How far apart the two aim circles are right now, in metres.

    Measured from the REST heads, through the board's object matrix, because
    that is the geometry the fit sizes - where an artist has dragged the
    handles to is a pose, and a pose is not what is being measured here.
    """
    if arm_obj is None:
        arm_obj = board_armature(skel)
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return 0.0
    heads = []
    for name in board.EYE_AIM_HANDLES:
        pose_bone = arm_obj.pose.bones.get(name)
        if pose_bone is None:
            return 0.0
        heads.append(arm_obj.matrix_world @ _handle_rest_head(pose_bone))
    return (heads[0] - heads[1]).length


def _place_eye_aim(skel, arm_obj, ratio, distance=None, fit=True):
    """Size the eye-aim control to this character's eyes and float it in front.

    Two measurements of the character, and nothing an artist has to judge by
    eye - the same bargain the board's own placement already makes, where the
    panel's HEIGHT comes from the eye joints rather than from a bounding box:

    * SCALE.  The chain is scaled uniformly until the two circles,
      ``CTRL_L_eyeAim`` and ``CTRL_R_eyeAim``, are exactly as far apart as the
      character's own eye joints, so each circle sits on its own eye's line
      instead of near it.  Before this the chain carried Ada's spacing times
      the head-size ratio, which is a whole-head average and not the eye
      spacing: on the bundled character itself the circles are 60.8 mm apart
      against 63.8 mm of eye, so even the reference is a couple of millimetres
      out, and a head whose eyes are not in Ada's proportion is out by however
      much that differs.
    * POSITION.  The midpoint of the two circles goes to the midpoint of the
      two eyes, pushed ``distance`` metres forward.  Taking it from the circles
      rather than from the frame is what keeps them ON the eyes: the frame is
      not centred between them (it is 0.6 mm off on the bundled asset), and
      centring the frame instead leaves that error on the circles.

    That leaves ``distance`` as the one thing to set by hand, which is the
    point: it is a matter of taste - how far out the target should sit - and
    nothing in the character can answer it.

    The scale is recomputed from the geometry as it stands, never accumulated,
    so calling this twice is calling it once; and the rigid part stays rigid.
    v3.12.0 gave each eye's branch a second shift onto that eye's own gaze ray,
    to make the look-at solve read zero at rest.  It did read zero - and it
    wrecked the board: the two circles left the plane of the frame they sit in
    and stopped being symmetric about it, and the frame itself dropped off the
    eye line, because its new position came from the average of two tilted axes
    rather than from the eyes themselves.  The layout INSIDE this chain is
    authored, and it is not the solve's to spend.  The rest offset is corrected
    in ``_eye_aim_gui``, where it costs no geometry.

    ``fit=False`` moves the chain without resizing it - for the distance
    control, which has no business changing a size the artist can see.
    """
    eyes = _eye_pair_world(skel)
    if eyes is None or arm_obj.data is None:
        return
    eyes_mid = (eyes[0] + eyes[1]) * 0.5
    if distance is None:
        distance = eye_aim_distance(skel, ratio)

    # Uniform, and taken from the plain distance between the two joints: it is
    # the one number that means "this face's eyes are this far apart" whichever
    # way the head is turned. A rotation to match the eye axis as well would be
    # a different feature, and one that would tilt the frame and its label.
    scale = 1.0
    if fit:
        span = eye_aim_span(skel, arm_obj)
        wanted = (eyes[0] - eyes[1]).length
        if span < EYE_AIM_MIN_SPAN or wanted < EYE_AIM_MIN_SPAN:
            fit = False                 # nothing measurable; place only
        else:
            scale = wanted / span

    # Everything in the ARMATURE's own space, which is what an edit bone reads
    # and write. The pivot is the circles' midpoint including the fraction of a
    # millimetre of authoring dust their GRP_ parents carry as a stored rest
    # (see _handle_rest_head) - that dust is part of where the circle is drawn,
    # and rescale_channels below scales it by the same factor, so pivoting on
    # it is what lands the drawn midpoint exactly on the target.
    pivot = Vector((0.0, 0.0, 0.0))
    for name in board.EYE_AIM_HANDLES:
        pose_bone = arm_obj.pose.bones.get(name)
        if pose_bone is None:
            return
        pivot += _handle_rest_head(pose_bone) * 0.5
    to_board = arm_obj.matrix_world.inverted_safe()
    shift = to_board @ (eyes_mid + Vector((0.0, -distance, 0.0))) - pivot

    context = bpy.context
    with organization.shown_for_edit(arm_obj, view_layer=context.view_layer):
        previous = context.view_layer.objects.active
        context.view_layer.objects.active = arm_obj
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except RuntimeError:
            context.view_layer.objects.active = previous
            return
        edit_bones = arm_obj.data.edit_bones
        for name in _EYE_AIM_BONES:
            bone = edit_bones.get(name)
            if bone is None:
                continue
            bone.head = pivot + (bone.head - pivot) * scale + shift
            bone.tail = pivot + (bone.tail - pivot) * scale + shift
            # roll is invariant under a translation and a uniform scale
        bpy.ops.object.mode_set(mode='OBJECT')
        if previous is not None and                 context.view_layer.objects.get(previous.name):
            context.view_layer.objects.active = previous

    if fit and abs(scale - 1.0) > 1.0e-9:
        # A bone rest carries no scale, so growing the chain's geometry has to
        # be matched on everything else measured in those units: the handles'
        # pose locations (where the artist has dragged the target to) and the
        # convergence switch's Limit Location frame. None of these twenty bones
        # owns a DNA channel - checked against the DNA's gui_names - so unlike
        # the board-wide rescale there is no dna_gui_scale to move with it.
        board.rescale_channels(arm_obj, scale, names=_EYE_AIM_BONES)
    # The solve measures each handle against the rest direction captured when
    # the cache was built, and the chain has just moved: keep the old baseline
    # and the eyes would read a rotation away from a place the handle no longer
    # occupies, i.e. gaze off-target the moment look-at is switched on.
    invalidate()
    # Record where it was put, so the panel's slider reads the truth and a
    # later re-place (Reset Placement, Rebuild Control Board) reproduces it
    # instead of quietly reverting to the automatic default.
    skel[board.EYE_AIM_DISTANCE_PROP] = float(distance)


_GUI_RANGES = None


def _gui_channel_ranges():
    """``{control name: {channel: (lo, hi)}}`` - the span the DNA defines.

    Used to check each drawn frame against the range it is supposed to draw
    (see ``board.align_limits_to_dna``).  Constant per DNA, so built once.
    """
    global _GUI_RANGES
    if _GUI_RANGES is None:
        ranges = {}
        for index, gui_name in enumerate(riglogic.meta()["gui_names"]):
            template, _, channel = gui_name.rpartition(".")
            if template and channel:
                ranges.setdefault(template, {})[channel] = \
                    riglogic.gui_channel_range(index)
        _GUI_RANGES = ranges
    return _GUI_RANGES


# ------------------------------------------------------------ panel layout ---

def board_layout(skel):
    """The panel layout saved on this character, or None.

    Kept as JSON on the SKELETON (``board.LAYOUT_PROP``) rather than on the
    board, so it outlives the board: Rebuild Control Board deletes the whole
    armature, and a character restored from a .mhfrt without its payload has no
    board at all until one is imported.
    """
    raw = skel.get(board.LAYOUT_PROP) if skel is not None else None
    if not raw:
        return None
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("handles"):
        return None
    return data


def store_board_layout(skel, data):
    """Save (or, with a falsy `data`, forget) this character's panel layout."""
    if skel is None:
        return
    if data:
        skel[board.LAYOUT_PROP] = json.dumps(data, separators=(",", ":"))
    elif board.LAYOUT_PROP in skel:
        del skel[board.LAYOUT_PROP]


def board_armature(skel):
    """This character's control board armature, or None."""
    if skel is None:
        return None
    return board.board_armature_for_rig(RIG_ID_PROP, skel.get(RIG_ID_PROP),
                                        skel=skel)


# The facial hierarchy's own root.  In the source character its PARENT is the
# body's ``head`` bone; a standalone facial skeleton keeps only FACIAL_* bones
# (op_skeleton._strip_body_bones removes the rest), so this is what "the head"
# means on a rig that has no head bone left.
FACIAL_ROOT_BONE = "FACIAL_C_FacialRoot"
_HEAD_BONE_NAMES = ("head", "Head")


def follow_head_bone(skel):
    """The bone the control panel rides when Follow Head is switched on.

    Three answers in preference order, all of them "the thing that carries the
    face around":

    1. the bone the facial rig was MERGED under - on a merged character that is
       the body's own head, and it is the only one that moves when an animator
       turns the head in a body clip;
    2. a bone plainly called ``head``, for a skeleton that kept the DNA's body
       chain;
    3. ``FACIAL_C_FacialRoot`` - a standalone facial skeleton has nothing else,
       and it is exactly where the source rig's head bone used to be.

    Returns "" when none of them is there, which is a real answer: the switches
    then stay inert and the panel says why, instead of a constraint being wired
    to a bone that is not the head.
    """
    if skel is None or getattr(skel, "pose", None) is None:
        return ""
    bones = skel.pose.bones
    merged = str(skel.get(RIG_MERGE_BONE_PROP, "") or "")
    if merged and merged in bones:
        return merged
    for name in _HEAD_BONE_NAMES:
        if name in bones:
            return name
    return FACIAL_ROOT_BONE if FACIAL_ROOT_BONE in bones else ""


def install_follow_head(skel, arm_obj=None):
    """Wire this character's follow-head switches. Returns how many are live.

    There is nothing captured, so this is safe from any pose at any time - see
    the long note in ``core.board``. Every path that rebuilds, re-places or
    re-loads a board just calls it.
    """
    if arm_obj is None:
        arm_obj = board_armature(skel)
    bone_name = follow_head_bone(skel)
    if arm_obj is None or not bone_name:
        return 0
    return board.install_follow_head(arm_obj, skel, bone_name)


# Skeletons whose board still needs the follow-head rig, noticed inside the
# depsgraph handler and wired up from a timer once it has finished.
_pending_follow_head = set()


def _queue_follow_head(skel):
    if skel is None:
        return
    _pending_follow_head.add(skel.name)
    if not bpy.app.timers.is_registered(_apply_pending_follow_head):
        bpy.app.timers.register(_apply_pending_follow_head, first_interval=0.0)


def _apply_pending_follow_head():
    """Give existing characters the follow-head rig, outside the handler.

    A board built before v4.12.0 has the two switches and nothing behind them.
    Wiring one costs a constraint and a driver, so it happens here rather than
    in the middle of a depsgraph evaluation, and only once per board - the
    caller checks ``board.follow_head_installed`` first.
    """
    if render_state.is_rendering():
        return _RENDER_RETRY            # constraints and drivers can wait
    pending = sorted(_pending_follow_head)
    _pending_follow_head.clear()
    for name in pending:
        skel = bpy.data.objects.get(name)
        if skel is None or skel.type != 'ARMATURE':
            continue
        arm_obj = board_armature(skel)
        if arm_obj is None or board.follow_head_installed(arm_obj):
            continue
        if board.layout_unlocked(arm_obj):
            continue                    # mid-redesign; try again next rescan
        install_follow_head(skel, arm_obj)
    return None


def _import_gui(context, skel, anchor, layout=None):
    """Append this character's control board: the ``face_gui`` bone board.

    Appended as authored rather than rebuilt - the per-frame pose scales that
    size every handle (CTRL_faceGUI at 0.005, the FRM_* frames at up to 4) are
    not something a bone rest matrix can carry, so a regenerated board would
    move its handles the wrong distance inside their frames.

    It comes out of the source character, which already carries the board its
    own import built.  Only the armature is asked for; Blender brings the ~460
    handle shapes along because the pose bones reference them.

    ``layout`` is a panel layout to reproduce on the new board instead of
    leaving it at the authored positions - the one the old board was carrying
    when Rebuild Control Board replaced it.  Left out, the layout saved on the
    skeleton is used, so re-importing a board never costs the artist the panel
    arrangement they set up (see :func:`board_layout`).
    """
    path = paths.BUNDLED_BLEND
    if not os.path.isfile(path):
        return None
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(path, link=False) as (df, dt):
        if board.FACE_BOARD_ARMATURE not in df.objects:
            return None
        dt.objects = [board.FACE_BOARD_ARMATURE]
    arm_obj = next((o for o in dt.objects if o is not None), None)
    # The handle shapes arrive as dependencies rather than by name.
    appended = [obj for obj in bpy.data.objects if obj not in before]
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        for obj in appended:
            bpy.data.objects.remove(obj, do_unlink=True)
        return None
    # The source character's own importer anchored its board to ITS head with
    # CHILD_OF constraints. This copy belongs to this rig and is placed and
    # parented by _place_bone_board, so those have to go - left in place they
    # would drag the panel onto a different character's head.
    for pose_bone in arm_obj.pose.bones:
        for constraint in list(pose_bone.constraints):
            if constraint.type == 'CHILD_OF':
                pose_bone.constraints.remove(constraint)
    # The 29 bones drawn by an EMPTY are grouping nodes and eye-aim locators,
    # not handles; left out they pepper the forehead and eyes with black
    # crosses that cannot even be clicked.
    board.stow_helper_bones(arm_obj)
    board.expose_eye_aim(arm_obj)

    coll = bpy.data.collections.new(f"{skel.name}_GUI")
    context.scene.collection.children.link(coll)
    coll.objects.link(arm_obj)

    rid = skel[RIG_ID_PROP]
    arm_obj.name = f"{skel.name}_FaceBoard"
    arm_obj.data.name = arm_obj.name
    arm_obj[RIG_ID_PROP] = rid
    arm_obj[board.BOARD_ARMATURE_PROP] = True
    arm_obj[board.SOURCE_PROP] = board.BONES_SOURCE
    arm_obj.show_in_front = True
    arm_obj.hide_render = True

    # Widgets stay unlinked: they are referenced as custom bone shapes only.
    # Tagged with the rig id so removing or rebuilding the board takes them
    # with it - nothing else in the file can reach an unlinked object.
    for widget in appended:
        if widget is arm_obj:
            continue
        widget.use_fake_user = True
        widget.name = f"{skel.name}_{widget.name}"
        widget[board.WIDGET_OWNER_PROP] = rid

    dna_ranges = _gui_channel_ranges()
    for pose_bone in arm_obj.pose.bones:
        pose_bone[board.TEMPLATE_PROP] = pose_bone.name
        pose_bone[board.SOURCE_PROP] = board.BONES_SOURCE
        # Order matters: snap the authored float dust, correct the drawn frame
        # against the DNA from that clean rest, then clamp into the corrected
        # frame. Doing the frame first would measure it from a dusty rest.
        pose_bone.location = board.snap_dust(pose_bone)
        ranges = dna_ranges.get(pose_bone.name)
        if ranges:
            board.align_limits_to_dna(pose_bone, ranges)
        # Deliberate non-zero rests, such as the rigLogic/lookAt switches at
        # 1.0, are far above the dust neutral_location cleans up.
        location = board.neutral_location(pose_bone)
        pose_bone.location = location
        pose_bone[board.REST_PROP] = location

    # How this board is DRAWN - every bone's authored rotation, scale and widget
    # size - recorded off the asset while it is still pristine.  192 of these
    # bones carry a pose scale and 125 a pose rotation, and until v4.12.0
    # nothing in the file knew that, so one Alt+S in Pose Mode flattened the
    # panel with no way back.  Stamped here, before anything can touch it.
    board.stamp_design(arm_obj, force=True)
    # Then the three bars the artist actually grabs are folded into the rest, so
    # selecting one reads 0/0/0 and 1/1/1 instead of a scale of 0.005, and
    # re-stamped: identity is their authored state from here on.
    bake_panel_bars(arm_obj)

    # The board is authored 1:1 with the DNA's +/-1 control range. The
    # placement below restates that in the units it bakes into the rest.
    skel["dna_gui_scale"] = 1.0
    # Parent FIRST, then place. The placement is baked into the rest and the
    # object left at identity (_flatten_board), so it has to be computed in the
    # space the board will finally live in - which is the skeleton's.
    arm_obj.parent = skel
    arm_obj.matrix_parent_inverse = Matrix.Identity(4)
    arm_obj.matrix_basis = Matrix.Identity(4)
    _place_bone_board(skel, arm_obj, anchor)
    # AFTER the parenting, never before: a saved layout stores the board's
    # ``matrix_basis``, which is its transform RELATIVE to the skeleton.
    # Writing it while the board is still unparented would treat those numbers
    # as world coordinates and drop the panel wherever the character is not.
    if layout is None:
        layout = board_layout(skel)
    if layout:
        board.apply_layout(arm_obj, layout)
    # The two follow-head switches, wired to real Child Of constraints on the
    # panel and eye-aim roots.  AFTER the layout, so the bind pose the inverse
    # is computed from is where the panel actually ends up.
    install_follow_head(skel, arm_obj)
    # A fresh board arrives locked: everything on it except an animation
    # handle's location is placement, not animation, so nothing should move it
    # until the artist asks for Grab Panel Handles.
    board.set_board_locked(arm_obj, True)
    skel[RIG_GUI_COLL_PROP] = coll
    return coll


def _ensure_armature_mod(obj, skel):
    """Bind `obj` to the skeleton with an Armature modifier (deforms via its
    bone-named vertex groups). Idempotent.

    On a merged rig the character's own Armature modifier already points at
    this armature; adding a second one would deform the mesh twice, so an
    existing modifier for the same armature is reused whatever its name."""
    mod = obj.modifiers.get(ARM_MOD_NAME)
    if mod is not None and mod.type != 'ARMATURE':
        mod = None
    if mod is None:
        mod = next((m for m in obj.modifiers
                    if m.type == 'ARMATURE' and m.object == skel), None)
    if mod is None:
        mod = obj.modifiers.new(ARM_MOD_NAME, 'ARMATURE')
    mod.object = skel
    mod.use_vertex_groups = True
    mod.show_in_editmode = True     # posed result visible while editing
    mod.show_on_cage = True         # and editable directly on the deformed mesh
    return mod


# ------------------------------------------------------------- public API ---

def setup_rig(context, skel, cage=None, target=None):
    """Turn `skel` into (or refresh) a live character rig. Called by the
    skeleton fit/update; idempotent. Returns an error string or None."""
    if not riglogic.available():
        return "Rig logic data missing from the add-on"
    # Who this is being built for, resolved BEFORE anything is created: the
    # control board, its 460 handle shapes and the GUI collection are all
    # stamped as this character's by the tracker below, and that is what
    # authorises removal to delete them later.
    from ..core import registry as _registry
    from ..core import provenance as _provenance
    _record = (_registry.for_object(context.scene, skel)
               or _registry.for_object(context.scene, cage)
               or _registry.for_object(context.scene, target)
               or _registry.active(context.scene))
    with _provenance.track(_record):
        return _setup_rig(context, skel, cage, target, _record)


def _setup_rig(context, skel, cage, target, tracked_record):
    if RIG_ID_PROP not in skel:
        skel[RIG_ID_PROP] = uuid.uuid4().hex[:12]
    if RIG_INTENSITY_PROP not in skel:
        skel[RIG_INTENSITY_PROP] = 1.0
    rid = skel[RIG_ID_PROP]
    if cage is not None:
        cage[RIG_ID_PROP] = rid
        skel[RIG_CAGE_PROP] = cage
        _ensure_armature_mod(cage, skel)   # cage previews the rig too
    if target is not None:
        target[RIG_ID_PROP] = rid
        skel[RIG_TARGET_PROP] = target
    if cage is not None and target is not None:
        cage[RIG_TARGET_PROP] = target
        target[RIG_CAGE_PROP] = cage
        # The cage stands in for the head, so it hangs off whatever the head
        # hangs off - a body rig, a bone, an empty the character is moved with.
        organization.match_cage_parent(cage, target)

    if _rig_gui_collection(skel) is None and not _has_board(skel):
        if _import_gui(context, skel, target or cage) is None:
            return ("The control board is missing from the bundled character "
                    "(data/AdaHeadsv3.blend)")

    # A board built before the 'helpers' collection hid its shapeless bones one
    # by one, and a single Alt+H in Pose Mode brings all 29 crosses back with no
    # obvious way to re-hide them.  Refreshing moves them into the collection;
    # idempotent, so a board that already has one is untouched.
    existing_board = board.board_armature_for_rig(RIG_ID_PROP, rid, skel=skel)
    board.stow_helper_bones(existing_board)
    # The asset was saved with CTRL_lookAtSwitch at 0, so its whole eye-aim
    # chain arrives hidden and the look-at control looks lost.
    board.expose_eye_aim(existing_board)
    # How the board is drawn - recorded on a board that predates the stamps, so
    # a later Clear Scale is recoverable without a rebuild.
    board.stamp_design(existing_board)
    # And the panel bars folded to identity on a board built before that was
    # done, so selecting one reads 1/1/1 rather than 0.005. A no-op once they
    # are: plan_bar_bake then finds nothing left to fold.
    bake_panel_bars(existing_board)
    # The two follow-head switches.  Re-run unconditionally, which is exactly
    # what an Update Rig needs: the fit has just moved every bone's REST, and
    # the constraint is measured against that rest, so it is re-pointed at the
    # skeleton as it is now and there is nothing stale left behind.
    install_follow_head(skel, existing_board)

    # Tell the character record it now HAS a rig.  Without this the record
    # still reads as "setting up" forever and nothing that asks the registry
    # for this character's skeleton - the list, the wiring audit, removal -
    # can find it.
    from ..core import registry
    scene = context.scene
    record = (tracked_record
              or registry.for_object(scene, cage)
              or registry.for_object(scene, target)
              or registry.for_object(scene, skel)
              or registry.active(scene))
    if record is None:
        record = registry.new(scene)
        registry.set_active(scene, record.uid)
    record.skeleton = skel
    if cage is not None:
        record.cage = cage
    if target is not None:
        record.target = target
    gui = _rig_gui_collection(skel)
    if gui is not None:
        record.gui_coll = gui
    for datablock in (skel, cage, target, gui):
        registry.claim(record, datablock)

    organization.ensure_character_collections(
        context,
        cage=cage,
        target=target,
        skel=skel,
        gui_coll=gui,
        root=record.root,
        record=record,
    )
    if record.root is None:
        root = organization._root_from_objects(skel, cage, target,
                                               scene=scene)
        if root is not None:
            record.root = root
            registry.claim(record, root)

    # Only THIS rig is rebuilt.  A full rescan here is what made building the
    # 4th character take three times as long as the 1st.
    bump_rig_topology()
    drop_cache(skel[RIG_ID_PROP])
    _rescan(full=False)
    cache = _caches.get(skel[RIG_ID_PROP])
    if cache is None:
        return "Rig cache could not be built (control board missing?)"
    _evaluate_guarded([cache])
    # The character has a rig now: write that (and the ledger of everything the
    # build just created) into its file.
    from ..core import sidecar
    sidecar.touch(scene, record)
    return None


def board_is_legacy(skel):
    """True when this rig still drives the pre-3.0 loose-object board."""
    if skel is None:
        return False
    controls = board.controls_for_rig(RIG_ID_PROP, skel.get(RIG_ID_PROP),
                                      skel=skel)
    return bool(controls) and not any(
        board.is_pose_bone(control) for control in controls.values())


def _widgets_in_use_elsewhere(rid):
    """Handle shapes tagged to `rid` that ANOTHER rig's board also draws with.

    Scene > New > Full Copy duplicates the board armature but not the widgets:
    they are linked to no collection, so the copy keeps drawing its handles
    with the original's shape objects.  Deleting this rig would then leave the
    other character's board as 435 invisible bones.
    """
    keep = set()
    for arm_obj in bpy.data.objects:
        if not board.is_board_armature(arm_obj) or arm_obj.pose is None:
            continue
        if str(arm_obj.get(RIG_ID_PROP) or "") == str(rid):
            continue
        for pose_bone in arm_obj.pose.bones:
            if pose_bone.custom_shape is not None:
                keep.add(pose_bone.custom_shape.name)
    return keep


def remove_board_widgets(rid):
    """Delete the handle shapes of this rig's board, and their meshes.

    They are linked to no collection - only the bones reference them - so they
    survive every scene-level cleanup unless removed by name here.
    """
    removed = 0
    shared = _widgets_in_use_elsewhere(rid)
    for widget in board.widgets_for_rig(rid):
        if widget.name in shared:
            continue            # another character's board still draws with it
        mesh = widget.data if widget.type == 'MESH' else None
        try:
            bpy.data.objects.remove(widget, do_unlink=True)
        except (ReferenceError, RuntimeError):
            continue
        removed += 1
        if mesh is not None and mesh.users == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except (ReferenceError, RuntimeError):
                pass
    return removed


def _remove_board(skel):
    """Delete this rig's current control board. Returns how many objects went."""
    rid = skel.get(RIG_ID_PROP)
    drop_cache(rid)         # the cache points into the board (see drop_cache)
    # Resolve every owner while the board is still whole. A bone board's 435
    # controls all belong to ONE armature, so deleting the owner of the first
    # control leaves the other 434 pose bones dangling - and asking a dangling
    # pose bone which armature it belongs to reads freed memory.
    owners = {}
    for control in board.controls_for_rig(RIG_ID_PROP, rid,
                                          skel=skel).values():
        owner = board.owner_object(control)
        if owner is not None:
            owners[owner.name] = owner
    removed = 0
    for name, owner in owners.items():
        if name not in bpy.data.objects:
            continue
        data = owner.data if owner.type == 'ARMATURE' else None
        bpy.data.objects.remove(owner, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.armatures.remove(data)
        removed += 1
    remove_board_widgets(rid)
    coll = _rig_gui_collection(skel)
    if coll is not None and not coll.all_objects and not coll.children:
        bpy.data.collections.remove(coll)
    if RIG_GUI_COLL_PROP in skel:
        del skel[RIG_GUI_COLL_PROP]
    return removed


class MHFRT_OT_rebuild_board(bpy.types.Operator):
    bl_idname = "mhfrt.rebuild_board"
    bl_label = "Rebuild Control Board"
    bl_description = ("Replace this character's control board with the current "
                      "bone board - one armature of posable handles instead of "
                      "loose objects. The pose is carried over; any keyframes "
                      "on the old board are not")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        return bool(mh and mh.skeleton and mh.skeleton.get(RIG_ID_PROP))

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        # Read the pose off the old board first so the new one can reproduce it.
        _rescan(full=False)
        cache = _caches.get(skel.get(RIG_ID_PROP))
        gui = _read_gui(cache, apply_weights=False) if cache else None
        # Where the panels were arranged, too - the pose above is expression
        # channels only, and the panel bars carry none.  Taken from the LIVE
        # board rather than from a saved layout so an artist who never pressed
        # Save Layout still gets their arrangement back.
        layout = board.capture_layout(board_armature(skel))

        _remove_board(skel)
        cage = skel.get(RIG_CAGE_PROP)
        target = skel.get(RIG_TARGET_PROP)
        anchor = target if isinstance(target, bpy.types.Object) else cage
        if _import_gui(context, skel,
                       anchor if isinstance(anchor, bpy.types.Object)
                       else None, layout=layout) is None:
            self.report({'ERROR'},
                        "The control board is missing from the bundled "
                        "character (data/AdaHeadsv3.blend)")
            return {'CANCELLED'}

        organization.ensure_character_collections(
            context,
            cage=cage if isinstance(cage, bpy.types.Object) else None,
            target=target if isinstance(target, bpy.types.Object) else None,
            skel=skel,
            gui_coll=_rig_gui_collection(skel),
        )
        drop_cache(skel.get(RIG_ID_PROP))
        _rescan(full=False)
        cache = _caches.get(skel.get(RIG_ID_PROP))
        if cache is None:
            self.report({'ERROR'}, "The new board could not be read back")
            return {'CANCELLED'}
        restored = 0
        if gui is not None and len(gui) == cache["gui_count"]:
            divisor = cache["divisor"]
            for name, value in zip(riglogic.meta()["gui_names"], gui):
                if abs(float(value)) <= 1e-9:
                    continue
                template, _, channel = name.rpartition(".")
                if board.set_channel_value(
                        cache["controls"].get(template), channel,
                        float(value), divisor):
                    restored += 1
        _evaluate_guarded([cache])
        self.report({'INFO'},
                    f"Control board rebuilt as bones; {restored} channel(s) "
                    "carried over")
        return {'FINISHED'}


class MHFRT_OT_edit_board(bpy.types.Operator):
    bl_idname = "mhfrt.edit_board"
    bl_label = "Pose Control Board"
    bl_description = ("Select the control board of the rig chosen in the Rig "
                      "list and enter Pose Mode on it. Unhides the board and "
                      "the collections above it, which Pose Mode requires")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # A pre-3.0 loose-object board has no armature to pose, so this greys
        # out for those rather than failing when pressed.
        mh = getattr(context.scene, "mhfrt", None)
        skel = getattr(mh, "skeleton", None) if mh else None
        if skel is None:
            return False
        return board.board_armature_for_rig(
            RIG_ID_PROP, skel.get(RIG_ID_PROP), skel=skel) is not None

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        arm_obj = board.board_armature_for_rig(RIG_ID_PROP,
                                               skel.get(RIG_ID_PROP),
                                               skel=skel)
        if arm_obj is None:
            self.report({'ERROR'}, "This rig has no bone control board")
            return {'CANCELLED'}

        # Leave the current mode FIRST. mode_set acts on the active object, so
        # reassigning that while still in Pose or Edit mode on another object is
        # what produces "context is incorrect".
        active = context.view_layer.objects.active
        if active is not None and active.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass

        organization.reveal(arm_obj, view_layer=context.view_layer)
        if context.view_layer.objects.get(arm_obj.name) is not arm_obj:
            self.report({'ERROR'},
                        f"'{arm_obj.name}' is not in the current view layer - "
                        "link its collection into this scene first")
            return {'CANCELLED'}

        try:
            bpy.ops.object.select_all(action='DESELECT')
        except RuntimeError:
            pass
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

        try:
            bpy.ops.object.mode_set(mode='POSE')
        except RuntimeError as error:
            self.report({'ERROR'}, f"Could not enter Pose Mode: {error}")
            return {'CANCELLED'}

        # Arrive on a clean selection instead of 435 lit-up bones.
        try:
            bpy.ops.pose.select_all(action='DESELECT')
        except RuntimeError:
            pass
        self.report({'INFO'}, f"Posing {arm_obj.name}")
        return {'FINISHED'}


def _reveal_bone(arm_obj, pose_bone):
    """Make one bone selectable in the viewport, without rearranging the rig.

    Two things can hide it and both have to give: the per-bone flag, and the
    bone collections.  Visibility ORs across collections - a bone is drawn
    while ANY collection holding it is visible - so nothing needs turning on
    when even one of its collections already is, and when none is, the first
    one is enough.  Turning them all on would un-hide whatever else lives
    there, and the asset files these handles beside 197 other controls.
    """
    board.set_bone_hidden(pose_bone, False)
    board.set_bone_flag(pose_bone, "hide_select", False)
    collections = list(pose_bone.bone.collections)
    if collections and not any(coll.is_visible for coll in collections):
        collections[0].is_visible = True


class MHFRT_OT_board_layout(bpy.types.Operator):
    """Where this character's control panels sit - save it, put it back, or
    start again from the automatic placement beside the head."""

    bl_idname = "mhfrt.board_layout"
    bl_label = "Panel Layout"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
        items=[
            ('SELECT', "Grab Panel Handles", ""),
            ('SAVE', "Save Layout", ""),
            ('RESTORE', "Restore Layout", ""),
            ('RESET', "Reset Placement", ""),
            ('FORGET', "Forget Saved Layout", ""),
            ('REPAIR', "Repair Board", ""),
            ('REVEAL', "Show Controllers", ""),
        ],
        default='SAVE',
        options={'HIDDEN'},
    )

    _DESCRIPTIONS = {
        'SELECT': ("Enter Pose Mode on the control board with the three panel "
                   "bars selected and nothing else, ready to drag with G. The "
                   "expression handles step out of the way so only the frames "
                   "and the bars are visible. Each bar moves its own panel: "
                   "the main expressions board, the TWEAKERS board, and the "
                   "follow-head switches"),
        'SAVE': ("Remember the whole panel arrangement - the bars, and any "
                 "frame or label you moved, rotated or resized. It comes back "
                 "with Restore, survives Rebuild Control Board, and is written "
                 "into the .mhfrt character file"),
        'RESTORE': "Put the panel back exactly as it was when you saved it",
        'RESET': ("Send the whole panel back to its authored design and re-run "
                  "the automatic placement beside this character's head. A "
                  "saved layout is kept - Restore still works"),
        'FORGET': ("Discard the saved arrangement. The panels stay exactly "
                   "where they are"),
        'REPAIR': ("Put back the size and angle of every handle and frame on "
                   "the board, as authored. Use it after a Clear Scale or "
                   "Clear Rotation flattened the panel. Your expression pose "
                   "and your panel positions are left alone"),
        'REVEAL': ("Bring back every expression handle the redesign mode put "
                   "away. Use it if the panel came back from Save Layout with "
                   "its controllers still missing. Nothing else is touched"),
    }

    @classmethod
    def description(cls, context, properties):
        return cls._DESCRIPTIONS.get(properties.action, cls.__doc__)

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        skel = getattr(mh, "skeleton", None) if mh else None
        return skel is not None and board_armature(skel) is not None

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        arm_obj = board_armature(skel)
        if arm_obj is None:
            self.report({'ERROR'}, "This character has no bone control board")
            return {'CANCELLED'}
        return getattr(self, f"_{self.action.lower()}")(context, skel, arm_obj)

    # -- the five actions ---------------------------------------------------

    def _save(self, _context, skel, arm_obj):
        layout = board.capture_layout(arm_obj)
        if not layout:
            self.report({'ERROR'},
                        "This board has no panel bars to remember")
            return {'CANCELLED'}
        store_board_layout(skel, layout)
        # Saving ENDS the placing mode: the handles come back and locking is
        # what makes the layout permanent - from here Alt+G (and Alt+R, Alt+S,
        # Clear Transform) are refused, so the panel cannot be knocked back to
        # where it used to be by a shortcut aimed at something else.
        board.set_layout_solo(arm_obj, False)
        board.set_board_locked(arm_obj, True)
        # A board that predates the switches - or carries the old Child Of -
        # gets them here too, at no cost to one that is already wired.  Where
        # the panel has been dragged to is irrelevant: the constraint applies
        # the head's motion ON TOP of wherever the panel is.
        install_follow_head(skel, arm_obj)
        redesigned = len(layout.get("bones") or ()) + len(layout.get("shape") or ())
        detail = f" and {redesigned} redesigned bone(s)" if redesigned else ""
        self.report({'INFO'},
                    f"Saved and locked {len(layout['handles'])} panel(s)"
                    f"{detail} - this is their rest position now")
        return {'FINISHED'}

    def _restore(self, _context, skel, arm_obj):
        layout = board_layout(skel)
        if layout is None:
            self.report({'ERROR'}, "No panel layout has been saved yet")
            return {'CANCELLED'}
        board.set_layout_solo(arm_obj, False)
        board.set_board_locked(arm_obj, False)
        moved = board.apply_layout(arm_obj, layout)
        board.set_board_locked(arm_obj, True)
        install_follow_head(skel, arm_obj)
        self.report({'INFO'}, f"Restored {moved} panel(s)")
        return {'FINISHED'}

    def _repair(self, _context, _skel, arm_obj):
        # Python assignment ignores the locks, so this is only about leaving the
        # board in the state it was found in: a repair pressed mid-redesign must
        # not silently end the redesign.
        placing = board.layout_unlocked(arm_obj)
        fixed = board.restore_board_design(arm_obj)
        if not placing:
            board.ensure_board_locked(arm_obj)
        if not fixed:
            self.report({'INFO'},
                        "Every handle is already the size and angle it was "
                        "authored at")
            return {'CANCELLED'}
        self.report({'INFO'},
                    f"Put back the authored size and angle of {fixed} "
                    "handle(s)")
        return {'FINISHED'}

    def _reset(self, _context, skel, arm_obj):
        board.set_layout_solo(arm_obj, False)
        board.set_board_locked(arm_obj, False)
        n = board.reset_layout(arm_obj)
        board.set_board_locked(arm_obj, True)
        # The bars are bone-local; where the board itself sits beside the head
        # is a measurement of THIS character, so it is recomputed rather than
        # remembered - the same call Build Rig makes.  It writes matrix_world,
        # which is right for a board already parented to the skeleton (the
        # import path is the one that has to parent afterwards).
        cage = skel.get(RIG_CAGE_PROP)
        target = skel.get(RIG_TARGET_PROP)
        anchor = target if isinstance(target, bpy.types.Object) else cage
        placed = isinstance(anchor, bpy.types.Object)
        if placed:
            # The aim chain is re-fitted too. It is measured from the eye
            # joints, and "start again from the automatic placement" has to
            # mean the look-at target as much as the panel - the artist's own
            # distance is kept, being the one part of it that is a choice.
            _place_bone_board(skel, arm_obj, anchor)
        install_follow_head(skel, arm_obj)
        self.report({'INFO'},
                    f"Reset {n} panel bone(s)"
                    + (" and re-placed the board" if placed else ""))
        return {'FINISHED'}

    def _reveal(self, _context, _skel, arm_obj):
        # Deliberately does NOT end the redesign mode: an artist who pressed
        # this mid-placement wants to see their handles, not to have the panel
        # locked under them. The solo record is cleared all the same, so
        # leaving the mode later cannot put them away again.
        shown = board.reveal_controls(arm_obj)
        if arm_obj.library is None \
                and board.LAYOUT_SOLO_PROP in arm_obj.keys():
            del arm_obj[board.LAYOUT_SOLO_PROP]
        if not shown:
            self.report({'INFO'},
                        "Every controller on this board is already visible")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Brought back {shown} controller(s)")
        return {'FINISHED'}

    def _forget(self, _context, skel, _arm_obj):
        if board_layout(skel) is None:
            self.report({'INFO'}, "There was no saved layout")
            return {'CANCELLED'}
        store_board_layout(skel, None)
        self.report({'INFO'}, "Forgot the saved panel layout")
        return {'FINISHED'}

    def _select(self, context, _skel, arm_obj):
        handles = board.layout_handle_bones(arm_obj)
        if not handles:
            self.report({'ERROR'},
                        "This board has no panel bars - it may predate them, "
                        "or a Purge blanked its handle shapes. Rebuild Control "
                        "Board restores them")
            return {'CANCELLED'}

        # Leave the current mode FIRST: mode_set acts on the active object, so
        # reassigning that from inside Pose or Edit mode is what produces
        # "context is incorrect" (same order as Pose Control Board).
        active = context.view_layer.objects.active
        if active is not None and active.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass

        organization.reveal(arm_obj, view_layer=context.view_layer)
        if context.view_layer.objects.get(arm_obj.name) is not arm_obj:
            self.report({'ERROR'},
                        f"'{arm_obj.name}' is not in the current view layer - "
                        "link its collection into this scene first")
            return {'CANCELLED'}

        try:
            bpy.ops.object.select_all(action='DESELECT')
        except RuntimeError:
            pass
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj
        try:
            bpy.ops.object.mode_set(mode='POSE')
        except RuntimeError as error:
            self.report({'ERROR'}, f"Could not enter Pose Mode: {error}")
            return {'CANCELLED'}
        try:
            bpy.ops.pose.select_all(action='DESELECT')
        except RuntimeError:
            pass

        # Entering the mode is what FREES the board. Outside it the bars, and
        # every handle's rotation and scale, are locked, so the panel cannot be
        # dragged, cleared or resized by accident while the artist is animating
        # the face.
        board.set_board_locked(arm_obj, False)
        # And the 178 expression handles step aside, leaving the frames, the
        # labels and the bars - the panel's own structure, which is what is
        # actually being arranged.
        hidden = board.set_layout_solo(arm_obj, True)
        for _template, pose_bone in handles:
            _reveal_bone(arm_obj, pose_bone)
            board.set_bone_flag(pose_bone, "select", True)
        arm_obj.data.bones.active = handles[0][1].bone
        solo = f", {hidden} handle(s) hidden" if hidden else ""
        self.report({'INFO'},
                    f"{len(handles)} panel bar(s) unlocked{solo} - G moves, R "
                    "rotates, S scales. Save Layout when done")
        return {'FINISHED'}


class MHFRT_OT_eye_target(bpy.types.Operator):
    """Put the look-at target back on this character's eyes.

    The size is a measurement, not a taste: the two circles belong exactly as
    far apart as the eye joints are, and centred on them.  Only how far in
    front of the face the target floats is left to the artist.
    """

    bl_idname = "mhfrt.eye_target"
    bl_label = "Eye Target"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
        items=[
            ('FIT', "Fit to Eyes", ""),
            ('DEFAULT', "Reset Distance", ""),
        ],
        default='FIT',
        options={'HIDDEN'},
    )

    _DESCRIPTIONS = {
        'FIT': ("Scale the eye target until its two circles are exactly as "
                "far apart as this character's eyes, and centre them on the "
                "eye line at the distance below. Nothing to position by hand"),
        'DEFAULT': ("Put the distance back to the automatic one for a head "
                    "this size, and re-fit"),
    }

    @classmethod
    def description(cls, context, properties):
        return cls._DESCRIPTIONS.get(properties.action, cls.__doc__)

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        skel = getattr(mh, "skeleton", None) if mh else None
        return skel is not None and board_armature(skel) is not None

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        arm_obj = board_armature(skel)
        if arm_obj is None:
            self.report({'ERROR'}, "This character has no bone control board")
            return {'CANCELLED'}
        if _eye_pair_world(skel) is None:
            self.report({'ERROR'},
                        "This skeleton has no FACIAL_L_Eye / FACIAL_R_Eye to "
                        "measure - there is nothing to fit the target to")
            return {'CANCELLED'}
        if self.action == 'DEFAULT' and board.EYE_AIM_DISTANCE_PROP in skel:
            del skel[board.EYE_AIM_DISTANCE_PROP]
        _pending_eye_aim.pop(skel.name, None)
        distance = place_eye_aim(skel, arm_obj)
        if distance is None:
            self.report({'ERROR'}, "Could not place the eye target")
            return {'CANCELLED'}
        span = eye_aim_span(skel, arm_obj)
        self.report({'INFO'},
                    f"Eye target fitted: circles {span * 1000.0:.1f} mm apart, "
                    f"{distance * 100.0:.1f} cm in front of the eyes")
        return {'FINISHED'}


class MHFRT_OT_follow_head(bpy.types.Operator):
    """Make the control panel - or the eye target - ride this character's head.

    The board ships with two switches for exactly this and nothing was reading
    them.  Both are real Armature constraints now, driven by the switch handle
    itself, so the behaviour is the rig's and not the add-on's: it works with
    the add-on disabled, in an appended file, and in the standalone .blend the
    driver bake writes out.
    """

    bl_idname = "mhfrt.follow_head"
    bl_label = "Follow Head"
    bl_options = {'REGISTER', 'UNDO'}

    switch: bpy.props.StringProperty(
        name="Switch", default="CTRL_faceGUIfollowHead", options={'HIDDEN'})
    enable: bpy.props.BoolProperty(name="On", default=True, options={'HIDDEN'})

    @classmethod
    def description(cls, _context, properties):
        what = board.FOLLOW_HEAD_LABELS.get(properties.switch, "This control")
        return (f"{what}: turn it on and the panel stays with the head when "
                "the character looks around; turn it off and it holds still "
                "in front of them. Stored on the board's own switch handle, so "
                "it exports and survives the add-on being uninstalled")

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        skel = getattr(mh, "skeleton", None) if mh else None
        return skel is not None and board_armature(skel) is not None

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        arm_obj = board_armature(skel)
        if arm_obj is None:
            self.report({'ERROR'}, "This character has no bone control board")
            return {'CANCELLED'}
        bone_name = follow_head_bone(skel)
        if not bone_name:
            self.report({'ERROR'},
                        "This character's skeleton has no head bone to follow "
                        "- merge the facial rig into the body first")
            return {'CANCELLED'}
        if not board.follow_head_installed(arm_obj):
            install_follow_head(skel, arm_obj)
        if not board.set_follow_head_value(arm_obj, self.switch,
                                           1.0 if self.enable else 0.0):
            self.report({'ERROR'},
                        f"This board has no '{self.switch}' handle")
            return {'CANCELLED'}
        label = board.FOLLOW_HEAD_LABELS.get(self.switch, self.switch)
        self.report({'INFO'},
                    f"{label}: {'on' if self.enable else 'off'} "
                    f"(following '{bone_name}')")
        return {'FINISHED'}


class MHFRT_OT_save_character(bpy.types.Operator, ExportHelper):
    bl_idname = "mhfrt.save_character"
    bl_label = "Save Character (.mhfrt)"
    bl_description = (
        "Write the selected rig to a .mhfrt character file: its landmark "
        "curves, settings, and the record of your original rig and weights. "
        "The file lives outside the .blend, so the work survives a lost or "
        "corrupted scene and can be carried into another file")
    bl_options = {'REGISTER'}

    # ExportHelper supplies `filepath` AND `check_existing` - the latter is
    # what makes the file browser ask before writing over a character that is
    # already there.  Losing a saved rig to a silent overwrite would undo the
    # entire point of the format.
    filename_ext = ".mhfrt"
    filter_glob: bpy.props.StringProperty(default="*.mhfrt",
                                          options={'HIDDEN'})
    include_blend: bpy.props.BoolProperty(
        name="Include the character",
        description="Write the character itself into the file - head, cage, "
                    "fitted skeleton, control board, weights and morphs - so "
                    "it can be restored complete into any .blend. Turn off to "
                    "save only the landmark curves and settings (a few KB), "
                    "for moving setup between files that already have the "
                    "meshes",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        return bool(mh and len(mh.rig_ui_items)
                    and 0 <= mh.rig_ui_active_index < len(mh.rig_ui_items))

    def invoke(self, context, event):
        from ..core import registry
        mh = context.scene.mhfrt
        item = mh.rig_ui_items[mh.rig_ui_active_index]
        rec = registry.find(context.scene, item.rig_key)
        base = (rec.name if rec is not None else "character")
        folder = os.path.dirname(bpy.data.filepath) or os.path.expanduser("~")
        self.filepath = os.path.join(folder, f"{base}.mhfrt")
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        from ..core import registry, sidecar
        mh = context.scene.mhfrt
        idx = mh.rig_ui_active_index
        if not (0 <= idx < len(mh.rig_ui_items)):
            self.report({'ERROR'}, "Select a rig in the list first")
            return {'CANCELLED'}
        rec = registry.find(context.scene, mh.rig_ui_items[idx].rig_key)
        if rec is None:
            self.report({'ERROR'}, "That rig no longer exists")
            return {'CANCELLED'}
        # Only merge into a file that is already THIS character's: carrying
        # another character's original-state snapshots into it would be worse
        # than useless, it would be wrong.
        same = (os.path.normcase(os.path.abspath(self.filepath))
                == os.path.normcase(os.path.abspath(rec.file_path))
                if rec.file_path else False)
        try:
            written = sidecar.write(context.scene, rec, self.filepath,
                                    include_blend=self.include_blend,
                                    keep_existing=same)
        except OSError as error:
            self.report({'ERROR'}, f"Could not write the file: {error}")
            return {'CANCELLED'}
        # This is the character's file from now on: the automatic updates that
        # run as they work follow it here instead of leaving a second, staler
        # copy beside the .blend.
        rec.file_path = written
        rec.file_error = ""
        self.report({'INFO'}, f"Saved '{rec.name}' to "
                              f"{os.path.basename(written)}")
        return {'FINISHED'}


def select_loaded_character(context, record, arrived=()):
    """Leave the artist with exactly the character they just loaded selected.

    A .mhfrt can carry a whole character - head, cage, fitted rig, control
    board, body and clothing - and dropping all of that into a file without
    touching the selection leaves them hunting through the outliner for what
    just arrived.  So: everything else is deselected, everything that came in is
    selected, and the rig (failing that the head, failing that the cage) is made
    active, because that is what every operator afterwards reads.

    Objects that are not in the view layer cannot be selected at all - the
    board's ~460 handle shapes are in no collection by design, and a collection
    the artist has excluded is not there either - so they are skipped rather
    than raising.  Visibility is left alone: what is hidden stays hidden (the
    display bar owns that), it is simply selected as well.

    Returns how many objects ended up selected.
    """
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return 0
    # A collection linked a moment ago is not in the view layer until it is
    # rebuilt, and its objects cannot be selected before that.
    view_layer.update()

    for obj in list(getattr(context, "selected_objects", ()) or ()):
        try:
            obj.select_set(False)
        except (RuntimeError, ReferenceError):
            continue

    wanted, seen = [], set()

    def want(obj):
        if obj is None:
            return
        try:
            if obj.name in seen:
                return
            seen.add(obj.name)
        except ReferenceError:
            return
        wanted.append(obj)

    for obj in arrived or ():
        want(obj)
    # Nothing arrived (a light .mhfrt re-attaching to meshes already here):
    # select the character it just restored, which is the same intent.
    if not wanted and record is not None:
        if record.root is not None:
            for obj in record.root.all_objects:
                want(obj)
        for obj in (record.cage, record.target, record.skeleton):
            want(obj)

    selected = 0
    for obj in wanted:
        try:
            if view_layer.objects.get(obj.name) != obj:
                continue                    # not in this view layer
            obj.select_set(True)
            selected += 1
        except (RuntimeError, ReferenceError):
            continue

    if record is not None:
        for candidate in (record.skeleton, record.target, record.cage):
            try:
                if candidate is not None and \
                        view_layer.objects.get(candidate.name) == candidate:
                    view_layer.objects.active = candidate
                    break
            except (RuntimeError, ReferenceError):
                continue
        else:
            for obj in wanted:
                try:
                    if view_layer.objects.get(obj.name) == obj:
                        view_layer.objects.active = obj
                        break
                except (RuntimeError, ReferenceError):
                    continue
    return selected


class MHFRT_OT_load_character(bpy.types.Operator, ImportHelper):
    bl_idname = "mhfrt.load_character"
    bl_label = "Load Character (.mhfrt)"
    bl_description = (
        "Restore a character from a .mhfrt file: its landmark curves, "
        "settings and original-rig record come back and it appears in the "
        "list. Meshes are matched by name; anything missing is reported so "
        "you can point the slots at the right objects")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".mhfrt"
    filter_glob: bpy.props.StringProperty(default="*.mhfrt",
                                          options={'HIDDEN'})

    def execute(self, context):
        from ..core import sidecar, registry
        try:
            data, payload = sidecar.read(self.filepath)
        except ValueError as error:
            self.report({'ERROR'},
                        f"'{os.path.basename(self.filepath)}' {error}")
            return {'CANCELLED'}
        arrived = []
        with sidecar.suspend():
            record, missing, rebuilt_cage = sidecar.apply(
                context.scene, data, payload=payload, arrived=arrived)
        # This file IS the character's file now - the one its original-state
        # snapshots live in, so a character restored here can still be removed
        # cleanly afterwards.
        record.file_path = self.filepath
        registry.set_active(context.scene, record.uid)
        mh = context.scene.mhfrt
        if record.cage is not None or record.target is not None:
            mh.switching_character = True
            try:
                mh.cage = record.cage
                mh.target = record.target
                mh.skeleton = record.skeleton
            finally:
                mh.switching_character = False
            from ..core import landmarks as lmdata
            lmdata.load_active(mh)
        bump_rig_topology()
        sync_rig_ui_state(mh)
        # Everything that just arrived, selected and nothing else - otherwise a
        # whole character lands in the file and the artist has to go and find
        # it.
        selected = select_loaded_character(context, record, arrived)
        note = ""
        if rebuilt_cage:
            note = " - a fresh head cage was built for it"
        picked = f"; {selected} object(s) selected" if selected else ""
        if record.pending_landmarks:
            wanted = record.pending_target_name or "its head"
            self.report({'WARNING'},
                        f"Loaded '{record.name}'{note}. Its head mesh "
                        f"('{wanted}') is not in this file - pick your head in "
                        "Setup and its landmark curves come back with it")
        elif missing:
            self.report({'WARNING'},
                        f"Loaded '{record.name}'{note}, but these objects are "
                        f"not in this file: {', '.join(sorted(set(missing)))}")
        else:
            self.report({'INFO'},
                        f"Loaded character '{record.name}'{note}{picked}")
        return {'FINISHED'}



def character_label(skel):
    """Display name of one character: its head mesh, falling back to the
    skeleton (discovery stays tag-based - names are labels only)."""
    target = skel.get(RIG_TARGET_PROP)
    if isinstance(target, bpy.types.Object):
        return target.name
    return skel.name


_ROOT_SUFFIX_RE = None


def _root_label(root):
    """'Ada_MHFRT_Character.001' -> 'Ada' (labels only, never used to match)."""
    global _ROOT_SUFFIX_RE
    if _ROOT_SUFFIX_RE is None:
        import re
        _ROOT_SUFFIX_RE = re.compile(r"_MHFRT_Character(\.\d+)?$")
    return _ROOT_SUFFIX_RE.sub("", root.name) or root.name


ROOT_UID_PROP = "mhfrt_root_uid"


def _root_uid(root):
    """A character root's stable identity, minted on first sight.

    Survives renaming the collection, and a duplicated collection gets its own
    because the copy is re-stamped by :func:`_dedupe_roots`."""
    if root is None:
        return ""
    uid = root.get(ROOT_UID_PROP)
    if not uid and root.library is None:
        uid = uuid.uuid4().hex[:12]
        root[ROOT_UID_PROP] = uid
    return str(uid or "")


class _Character:
    """One row of the Rig list - a live view onto a registry record.

    The attribute names are unchanged so every existing caller keeps working;
    what changed is where the answers come from.
    """

    __slots__ = ("rec",)

    def __init__(self, rec):
        self.rec = rec

    @property
    def root(self):
        return self.rec.root

    @property
    def skel(self):
        return self.rec.skeleton

    @property
    def cage(self):
        return self.rec.cage

    @property
    def target(self):
        return self.rec.target

    @property
    def name(self):
        return self.rec.name

    @property
    def order(self):
        return self.rec.order

    @property
    def key(self):
        return self.rec.uid

    @property
    def label(self):
        from ..core import registry
        return registry.label(self.rec)


def list_characters(scene=None):
    """Every character in the file, straight out of the registry.

    No scanning, no inference, no ordering that can change under the artist.
    A character is here because a record says so, and it stays here until the
    record is removed - including a character that is still being set up and
    has no rig, no head, or nothing at all yet.
    """
    from ..core import registry
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is None or getattr(scene, "mhfrt", None) is None:
        return []
    registry.migrate(scene)
    adopt_duplicated_characters(scene)
    return [_Character(rec) for rec in registry.sorted_records(scene)]


def _duplicate_uid_groups(scene):
    """``{record: [skeletons]}`` for every record more than one rig claims.

    A duplicate copies the character uid verbatim, so the copy's objects claim
    the ORIGINAL's record. That is the half of the damage the rig-id repair
    does not touch, and it is the half the LIST is made of: left alone the copy
    never appears (its uid is already known, so the card rebuild skips it)
    while one record silently owns both sets of objects - and Remove Character
    on either would take the other's meshes with it.
    """
    from ..core import registry
    groups = {}
    for skel in _rig_skeletons():          # already in bpy.data.objects order
        if not _writable(skel):
            continue
        uid = str(skel.get(registry.UID_PROP) or "")
        if registry.find(scene, uid) is None:
            continue                # no record yet: migrate/cards own that
        groups.setdefault(uid, []).append(skel)
    return {uid: skels for uid, skels in groups.items() if len(skels) > 1}


def adopt_duplicated_characters(scene):
    """Give a duplicated character a record - and objects - of its own.

    Runs where the list is built rather than in the depsgraph handler, because
    this ADDS a record and re-stamps datablocks, and the handler is safe for
    neither.

    Ownership is settled by the same rival comparison the rig-id split uses,
    and for the same reason: a plain Shift+D leaves both copies sitting in ONE
    collection, so "is in this character's collection" is true of every object
    for BOTH skeletons. Claiming everything a skeleton can see would hand one
    copy all 469 objects and leave the original owning nothing.
    """
    from ..core import registry
    groups = _duplicate_uid_groups(scene)
    if not groups:
        return 0
    made = 0
    for uid, skels in groups.items():
        rec = registry.find(scene, uid)
        scores = {skel.name: _claim_scores(skel) for skel in skels}
        # The record stays with the skeleton it already points at; failing
        # that, with the one holding the most bound objects (file order breaks
        # the tie, and a copy is always created after its source).
        keeper = next((s for s in skels if _same_object(rec.skeleton, s)), None)
        if keeper is None:
            keeper = max(skels, key=lambda skel: sum(
                1 for score, _ in scores[skel.name].values()
                if score >= _CLAIM_BOUND))
        for skel in skels:
            if skel == keeper:
                continue
            copy = registry.new(scene, registry.unique_name(scene, rec.name))
            for name, (score, obj) in scores[skel.name].items():
                rival = max(scores[other.name].get(name, (0, None))[0]
                            for other in skels if other != skel)
                if score > rival:
                    registry.claim(copy, obj)
            copy.skeleton = skel
            registry.claim(copy, skel)
            for prop, field in ((RIG_CAGE_PROP, "cage"),
                                (RIG_TARGET_PROP, "target")):
                value = skel.get(prop)
                if isinstance(value, bpy.types.Object):
                    setattr(copy, field, value)
                    registry.claim(copy, value)
            root = skel.get(organization.CHARACTER_COLL_PROP)
            if isinstance(root, bpy.types.Collection) and root != rec.root:
                copy.root = root
                registry.claim(copy, root)
            registry.write_card(copy)
            note_repair(f"'{registry.label(copy)}' was duplicated and is now "
                        "its own character in the list")
            made += 1
    return made


def _same_object(a, b):
    try:
        return a is not None and b is not None and a == b
    except ReferenceError:
        return False


def _dedupe_roots(entries):
    """Kept for the older call sites; the registry makes duplicates impossible."""
    return entries


# --------------------------------------------------------- rig naming & list ---

RIG_NAME_PROP = "mhfrt_rig_name"
DEFAULT_RIG_NAME = "MHFR"


def stored_rig_name(root, skel):
    """Legacy shim: the name lives on the record now."""
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return ""
    from ..core import registry
    rec = (registry.for_object(scene, skel)
           or registry.for_collection(scene, root))
    return rec.name if rec is not None else ""


def set_rig_name_for_item(item, value):
    """Rename the record this row points at."""
    from ..core import registry
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    rec = registry.find(scene, item.rig_key)
    if rec is None:
        return
    value = str(value).strip() or registry.DEFAULT_NAME
    if any(o.name == value for o in registry.records(scene) if o is not rec):
        value = registry.unique_name(scene, value, skip=rec)
    rec.name = value
    item.rig_name = value


def _has_face_anim(skel):
    try:
        from .op_anim import face_animation_action
        return face_animation_action(skel) is not None
    except Exception:
        return False


def character_for_item(item):
    """The character a Rig-list row points at, or None."""
    from ..core import registry
    if item is None:
        return None
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return None
    rec = registry.find(scene, item.rig_key)
    if rec is None and item.skel_name:
        rec = registry.for_object(scene, bpy.data.objects.get(item.skel_name))
    if rec is None and item.root_name:
        rec = registry.for_collection(
            scene, bpy.data.collections.get(item.root_name))
    return _Character(rec) if rec is not None else None


def stored_rig_name(root, skel):
    """Legacy shim: the name lives on the record now."""
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return ""
    from ..core import registry
    rec = (registry.for_object(scene, skel)
           or registry.for_collection(scene, root))
    return rec.name if rec is not None else ""


def set_rig_name_for_item(item, value):
    """Rename the record this row points at."""
    from ..core import registry
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    rec = registry.find(scene, item.rig_key)
    if rec is None:
        return
    value = str(value).strip() or registry.DEFAULT_NAME
    if any(o.name == value for o in registry.records(scene) if o is not rec):
        value = registry.unique_name(scene, value, skip=rec)
    rec.name = value
    item.rig_name = value


def _has_face_anim(skel):
    try:
        from .op_anim import face_animation_action
        return face_animation_action(skel) is not None
    except Exception:
        return False


def character_for_item(item):
    """The character a Rig-list row points at, or None.

    Resolved by STABLE KEY first (the rig id, or the root collection's uid) and
    only then by the names cached in the row.  Names are a display convenience:
    a renamed rig used to make its own row unusable, and - far worse - a name
    that had since been taken over by a different datablock resolved to THAT
    one, so Remove Rig could delete the wrong character.
    """
    if item is None:
        return None
    entries = list_characters()
    key = str(item.rig_key or "")
    if key:
        for entry in entries:
            if entry.key == key:
                return entry
    skel = bpy.data.objects.get(item.skel_name) if item.skel_name else None
    root = (bpy.data.collections.get(item.root_name)
            if item.root_name else None)
    if skel is None and root is None:
        return None
    for entry in entries:
        if ((skel is not None and entry.skel == skel)
                or (root is not None and entry.root == root)):
            return entry
    return None


def _has_face_anim(skel):
    try:
        from .op_anim import face_animation_action
        return face_animation_action(skel) is not None
    except Exception:
        return False


def sync_rig_ui_state(mh):
    """Rebuild the Rig list rows from the registry.

    One row per record, in creation order, always.  Nothing here can add a
    character, drop one, or move one - it is a straight readout.
    """
    from ..core import registry
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    registry.migrate(scene)
    # A duplicated character has to become a record BEFORE the rows are built,
    # or the copy is simply missing from the list.  This runs from the sync
    # timer, which is the one place allowed to add records; the signature
    # includes the object count, so a duplicate always gets here.
    adopt_duplicated_characters(scene)
    items = mh.rig_ui_items
    items.clear()

    active_uid = mh.active_character_uid
    active_idx = 0
    for i, rec in enumerate(registry.sorted_records(scene)):
        item = items.add()
        item.rig_key = rec.uid
        item.rig_name = rec.name
        item.skel_name = rec.skeleton.name if rec.skeleton else ""
        item.root_name = rec.root.name if rec.root else ""
        item.is_setup = rec.skeleton is None
        item.is_new = rec.cage is None and rec.target is None
        item.has_anim = bool(rec.skeleton is not None
                             and _has_face_anim(rec.skeleton))
        if rec.uid == active_uid:
            active_idx = i
    if len(items):
        mh.rig_ui_active_index = min(active_idx, len(items) - 1)


_rig_ui_sig_cache = {}
_rig_ui_sync_pending = False

# Bumped whenever something that could change the SET of rigs happens: a rig
# built or removed, a file loaded, an undo, a refresh, an object leaving the
# scene.  The panel's signature reads this counter instead of enumerating every
# character in the file, which it used to do on every single redraw
# (~0.6 ms with four rigs, and it grows with the object count).
_rig_topology_serial = 0


def bump_rig_topology():
    """Mark the rig set as possibly changed; the panel re-syncs on next draw."""
    global _rig_topology_serial
    _rig_topology_serial += 1
    _rig_ui_sig_cache.clear()


def _rig_ui_signature(mh):
    """O(1) - never enumerates characters.

    Adding or deleting anything changes a datablock count, and everything the
    add-on itself does to the rig set bumps the serial, so a stale list is only
    possible after a rename made by hand outside the add-on - which is what
    the Refresh button is for.
    """
    return (
        _rig_topology_serial,
        len(bpy.data.objects),
        len(bpy.data.collections),
        len(mh.characters),
        mh.active_character_uid,
        tuple(r.name for r in mh.characters),
        mh.cage.name if mh.cage else "",
        mh.target.name if mh.target else "",
        mh.skeleton.name if mh.skeleton else "",
        bool(mh.starting_new_character),
    )


def request_rig_ui_sync(mh, context):
    """Panel-draw safe refresh: schedule a sync only when the rig set changed.
    The mutation runs in a 0-delay timer, never during draw."""
    global _rig_ui_sync_pending
    try:
        sig = _rig_ui_signature(mh)
    except (AttributeError, ReferenceError):
        return
    key = context.scene.as_pointer()
    if _rig_ui_sig_cache.get(key) == sig or _rig_ui_sync_pending:
        return
    _rig_ui_sync_pending = True

    def _do():
        global _rig_ui_sync_pending
        _rig_ui_sync_pending = False
        scene = getattr(bpy.context, "scene", None)
        mh2 = getattr(scene, "mhfrt", None) if scene else None
        if mh2 is None:
            return None
        try:
            sync_rig_ui_state(mh2)
            _rig_ui_sig_cache[scene.as_pointer()] = _rig_ui_signature(mh2)
        except (AttributeError, ReferenceError, RuntimeError):
            return None
        try:
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except (AttributeError, RuntimeError):
            pass
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


class MHFRT_OT_dismiss_repair_notes(bpy.types.Operator):
    bl_idname = "mhfrt.dismiss_repair_notes"
    bl_label = "Dismiss"
    bl_description = "Hide what the add-on repaired on its own"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, _context):
        clear_repair_notes()
        return {'FINISHED'}


class MHFRT_OT_activate_character(bpy.types.Operator):
    bl_idname = "mhfrt.activate_character"
    bl_label = "Switch Character"
    # Deliberately NOT an UNDO operator.  Which rig the panel is pointed at is
    # view state, like the active object - and this now runs from every
    # selection change, so pushing an undo step here meant Blender took a full
    # memfile snapshot of a scene holding every character each time the artist
    # clicked another rig's mesh.  That was the single biggest cost in "select
    # something belonging to another rig and the UI hangs".
    bl_options = {'REGISTER', 'INTERNAL'}

    skeleton_name: bpy.props.StringProperty(options={'HIDDEN'})
    root_name: bpy.props.StringProperty(options={'HIDDEN'})
    rig_key: bpy.props.StringProperty(options={'HIDDEN'})
    list_index: bpy.props.IntProperty(default=-1, options={'HIDDEN'})

    @classmethod
    def description(cls, context, properties):
        skel = bpy.data.objects.get(properties.skeleton_name)
        if skel is not None:
            name = character_label(skel)
        else:
            root = bpy.data.collections.get(properties.root_name)
            name = root.name if root is not None else "this character"
        return (f"Work on '{name}'. Every step - landmarks, wrap, rig, "
                "morphs, animation import - applies only to the character "
                "selected here")

    def _resolve(self):
        """The record this row points at.  A uid, or nothing."""
        from ..core import registry
        scene = bpy.context.scene
        rec = registry.find(scene, self.rig_key)
        if rec is None and self.skeleton_name:
            rec = registry.for_object(
                scene, bpy.data.objects.get(self.skeleton_name))
        if rec is None and self.root_name:
            rec = registry.for_collection(
                scene, bpy.data.collections.get(self.root_name))
        if rec is None:
            return None, None
        return rec.skeleton, _Character(rec)

    def _reset_row(self, mh):
        """The row is stale - put the highlight back on the live rig."""
        try:
            sync_rig_ui_state(mh)
        except (AttributeError, ReferenceError, RuntimeError):
            pass

    def execute(self, context):
        from .op_live import stop_running
        from .op_pairs import is_running as landmarks_running
        from ..props import activate_rig, activate_pair
        if landmarks_running():
            self.report({'ERROR'},
                        "Finish landmark mode first (Esc in the split view)")
            return {'CANCELLED'}
        mh = context.scene.mhfrt
        skel, entry = self._resolve()
        if skel is None and entry is None:
            # Deleted or renamed outside the add-on: re-sync rather than leave
            # the list highlighting a row that points at nothing.
            self._reset_row(mh)
            self.report({'ERROR'}, "That character no longer exists - "
                                   "the list has been refreshed")
            return {'CANCELLED'}
        # Highlight by IDENTITY, not by the row index baked in at draw time.
        # The list can be rebuilt between the draw and the click, and trusting
        # a stale index put the highlight on somebody else's row.
        key = str(self.rig_key or "")
        moved = False
        if key:
            for i, row in enumerate(mh.rig_ui_items):
                if row.rig_key == key:
                    mh.rig_ui_active_index = i
                    moved = True
                    break
        if not moved and 0 <= self.list_index < len(mh.rig_ui_items):
            mh.rig_ui_active_index = self.list_index
        stop_running()          # the live session's meshes are about to change
        from ..core import registry
        registry.set_active(context.scene, entry.key)
        if skel is not None:
            result = activate_rig(context, skel)
        else:
            result = activate_pair(context, entry.cage, entry.target)
        name = entry.label or entry.name
        if 'FINISHED' not in result:
            self.report({'ERROR'},
                        "Finish or Cancel Tongue Edit before switching")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Active character: {name}")
        return {'FINISHED'}


class MHFRT_OT_new_character(bpy.types.Operator):
    bl_idname = "mhfrt.new_character"
    bl_label = "New Character"
    bl_description = ("Start rigging another character in this file: the panel "
                      "returns to Setup with empty slots. Characters already "
                      "rigged stay live and untouched - switch back to them "
                      "any time from this list")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .op_live import stop_running
        from .op_pairs import is_running as landmarks_running
        from ..core import landmarks as lmdata
        from ..core import rest_tuning
        if landmarks_running():
            self.report({'ERROR'},
                        "Finish landmark mode first (Esc in the split view)")
            return {'CANCELLED'}
        mh = context.scene.mhfrt
        session = next(
            (obj for obj in bpy.data.objects
             if obj.type == 'ARMATURE'
             and obj.get(rest_tuning.TONGUE_SESSION_PROP)),
            None,
        )
        if session is not None:
            self.report({'ERROR'},
                        "Finish or Cancel Tongue Edit before starting "
                        "a new character")
            return {'CANCELLED'}
        stop_running()
        lmdata.save_active(mh)              # keep the current pair's landmarks
        # Hand the outgoing rig its display state back before letting go of it,
        # or switching to it later would restore whatever the NEXT rig ended up
        # with.
        from ..props import _save_display_toggles, reset_display_toggles
        _save_display_toggles(mh)
        mh.switching_character = True
        try:
            mh.skeleton = None
            mh.cage = None
            mh.target = None
            mh.merge_body_armature = None
            mh.merge_head_bone = ""
            mh.merge_deform_bone = ""
            mh.rig_destination = 'UNDECIDED'
        finally:
            mh.switching_character = False
        lmdata.clear(mh, save=False)
        # forget the previous pair so its saved landmarks can never be
        # overwritten by this fresh empty set
        mh.active_cage = None
        mh.active_target = None
        # Clean slate: bones shown, landmark overlay on, nothing X-rayed,
        # both meshes visible - never the previous rig's leftovers.
        reset_display_toggles(mh, context)
        mh.ui_tab = 'SETUP'
        # Give the list a row of its own to sit on until this character has a
        # cage; otherwise it keeps the PREVIOUS character highlighted and the
        # new one looks like it was never created.
        mh.starting_new_character = True
        # A character exists from HERE.  Not when a cage arrives, not when a
        # rig is built - the row is real and it cannot vanish on its own.
        from ..core import registry
        record = registry.active(context.scene)
        blank = (record is not None and record.cage is None
                 and record.target is None and record.skeleton is None)
        if not blank:
            record = registry.new(context.scene)
        registry.set_active(context.scene, record.uid)
        bump_rig_topology()
        sync_rig_ui_state(mh)
        from ..ui import flow
        flow._redraw(context)
        self.report({'INFO'}, "New character - load a cage and pick its head")
        return {'FINISHED'}


def _character_for_object(obj):
    """(skeleton_name, record_uid) for whatever the artist just clicked."""
    from ..core import registry
    if obj is None:
        return "", ""
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return "", ""
    rec = registry.for_object(scene, obj)
    if rec is None:
        # anything filed inside a character's collection counts too
        for coll in obj.users_collection:
            rec = registry.for_collection(scene, coll)
            if rec is not None:
                break
    if rec is None:
        return "", ""
    return (rec.skeleton.name if rec.skeleton else ""), rec.uid


def _character_for_collection(coll):
    from ..core import registry
    scene = getattr(bpy.context, "scene", None)
    if scene is None or coll is None:
        return "", ""
    rec = registry.for_collection(scene, coll)
    if rec is None:
        from ..core import organization as org
        root = org._ancestor_root(coll)
        rec = registry.for_collection(scene, root) if root else None
    return ("", rec.uid) if rec is not None else ("", "")


def _active_character_root(mh):
    """The character collection the panel is already working on."""
    from ..core import organization as org
    return org._root_from_objects(mh.skeleton, mh.cage, mh.target)


_pending_switch = {"skel": "", "root": ""}
# Last thing the artist picked, PER VIEW LAYER.  A scene or a view layer has
# its own active object, its own active collection and its own selection, so a
# single global record meant stepping into a second scene (or a second window)
# looked like a brand-new pick of whatever happened to be selected there - and
# switching back re-fired the pick from the other scene.  Primed (rather than
# acted on) after a load or an undo, so re-opening a file never yanks the panel
# off the character its saved workflow bookmark restores.
_EMPTY_PICK = {"active": 0, "coll": 0, "selection": ()}
_last_pick_by_layer = {}
_pick_primed = False


def _pick_state(view_layer):
    key = view_layer.as_pointer()
    state = _last_pick_by_layer.get(key)
    if state is None:
        state = dict(_EMPTY_PICK)
        # Unbounded growth is not a concern (a file has a handful of view
        # layers) but a deleted one should not linger forever either.
        if len(_last_pick_by_layer) > 64:
            _last_pick_by_layer.clear()
        _last_pick_by_layer[key] = state
    return state


def _selected_objects(context, view_layer):
    """The selection, from whichever of the two C-side lists is available.

    ``context.selected_objects`` is missing in a restricted handler context;
    ``ViewLayerObjects.selected`` is not, and both are built in C - a full
    selection read costs microseconds, so this is safe on every redraw.
    """
    try:
        return context.selected_objects
    except AttributeError:
        pass
    try:
        return view_layer.objects.selected
    except AttributeError:
        return ()


def _selection_pick(context):
    """Which character did the artist just pick?  ("", "") when nothing did.

    Three signals are read, and only the ones that actually CHANGED are
    consulted - a click must not be overruled by an active object that has been
    sitting there all along:

    * the active object - a plain viewport or Outliner click on an object,
    * the active collection - an Outliner click on a collection row,
    * the selection - box-select and Shift+click extend it without ever
      touching the active object, which is why watching the active object
      alone missed them.
    """
    global _pick_primed
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return "", ""
    last = _pick_state(view_layer)
    active = view_layer.objects.active
    layer_coll = getattr(view_layer, "active_layer_collection", None)
    coll = getattr(layer_coll, "collection", None)
    selected = _selected_objects(context, view_layer)
    was_selected = set(last["selection"])
    key = {
        "active": active.as_pointer() if active is not None else 0,
        "coll": coll.as_pointer() if coll is not None else 0,
        "selection": tuple(obj.as_pointer() for obj in selected),
    }
    changed = {name for name, value in key.items() if last[name] != value}
    last.update(key)
    if not _pick_primed:
        _pick_primed = True
        return "", ""
    if not changed:
        return "", ""
    if "active" in changed and active is not None:
        found = _character_for_object(active)
        if any(found):
            return found
    if "coll" in changed:
        found = _character_for_collection(coll)
        if any(found):
            return found
    if "selection" in changed:
        # ONLY what was just added counts.  Shift+click and box-select extend a
        # selection, so the objects that were already in it are the character
        # the artist is LEAVING - answering with one of those is how pressing
        # New Character and then clicking the next character's mesh used to
        # snap the panel straight back to the old rig.
        for obj in selected:
            if obj.as_pointer() in was_selected:
                continue
            found = _character_for_object(obj)
            if any(found):
                return found
    # Deliberately no fallback to the rest of the selection: a pick that lands
    # on nothing belonging to a character means "no pick", never "some other
    # thing that happens to still be selected".
    return "", ""


def queue_character_pick(context, mh):
    """Make whatever the artist just picked the active character.

    Draw-safe and handler-safe: it only records the request and registers the
    0-delay timer that runs the operator (see _apply_pending_switch).
    """
    global _pending_switch
    if mh is None or mh.switching_character:
        return
    skel_name, root_name = _selection_pick(context)
    if not skel_name and not root_name:
        return
    # Already there?  The switch operator is idempotent but it is an UNDO
    # operator, and this now fires on every selection change.
    active = mh.skeleton
    if skel_name:
        if active is not None and active.name == skel_name:
            return
    elif root_name:
        if mh.active_character_uid == root_name:
            return
    _pending_switch = {"skel": skel_name, "root": root_name}
    if not bpy.app.timers.is_registered(_apply_pending_switch):
        bpy.app.timers.register(_apply_pending_switch, first_interval=0.0)


def _apply_pending_switch():
    """Trigger the queued character switch outside the depsgraph handler.

    Running bpy.ops in a depsgraph_update_post handler is unsafe; a timer
    invocation runs after the event finishes so switching is stable and
    UNDO records exactly once."""
    global _pending_switch
    from .op_pairs import is_running as landmarks_running
    from ..core import rest_tuning
    if render_state.is_rendering():
        # An operator that relinks collections and pushes undo, run while a
        # render job holds the scene, is the worst version of the race in
        # core.render_state. The request keeps until the render is done.
        return _RENDER_RETRY
    request = _pending_switch
    _pending_switch = {"skel": "", "root": ""}
    if landmarks_running():
        return None
    skel_name = str(request.get("skel", ""))
    root_name = str(request.get("root", ""))
    if not skel_name and not root_name:
        return None
    scene = bpy.context.scene
    mh = getattr(scene, "mhfrt", None) if scene else None
    if mh is None or mh.switching_character:
        return None
    active_skel = mh.skeleton
    if active_skel is not None \
            and active_skel.get(rest_tuning.TONGUE_SESSION_PROP):
        # never yank the artist out of the tongue edit
        return None
    if skel_name:
        target = bpy.data.objects.get(skel_name)
        if target is not None and target == active_skel:
            return None
    try:
        bpy.ops.mhfrt.activate_character(
            'INVOKE_DEFAULT',
            skeleton_name=skel_name, rig_key=root_name)
    except RuntimeError:
        pass
    return None


@persistent
def _viewport_click_sync_handler(scene, depsgraph=None):
    """Picking anything that belongs to a character activates that character.

    A cage, head mesh, skeleton, board control, part mesh or bone identifies
    its character by tag; anything else the artist keeps inside the character's
    collection - body, clothing, hair, props, tagged or not - identifies it by
    collection.  Whichever they pick becomes the Characters-list target for the
    whole panel.

    The panel's own list draw calls the same check (see
    ui/panel._draw_characters), because a pure selection change produces no
    depsgraph update at all - but it always produces a redraw.
    """
    # Purely about what the artist has selected, so it has nothing to do during
    # a render - and it reads ``bpy.context``, which a render thread must not.
    if render_state.is_rendering() or _is_render_depsgraph(depsgraph):
        return
    queue_character_pick(bpy.context, getattr(scene, "mhfrt", None))


# ----------------------------------------------------------- remove a rig ---

def _strip_addon_props(datablock):
    """Delete every MHFRT / DNA custom property from an ID datablock."""
    if datablock is None:
        return
    for key in list(datablock.keys()):
        if key.startswith("mhfrt_") or key.startswith("dna_"):
            try:
                del datablock[key]
            except (KeyError, TypeError):
                pass


def _clear_driven_morphs(obj):
    """Cut the morph drivers and return the keys to neutral.

    The sculpted shapes themselves stay: they are the artist's modelling work,
    not add-on machinery, and a removal that deleted a day of corrective
    sculpting would be worse than the one it replaced.
    """
    keys = obj.data.shape_keys if obj.data else None
    if keys is None:
        return
    if keys.animation_data is not None:
        keys.animation_data_clear()
    for kb in keys.key_blocks[1:]:
        kb.value = 0.0
    _strip_addon_props(keys)


def _clean_kept_mesh(obj, deleted_skels, kept_skel=None):
    """Turn a rig-bound mesh back into a plain object: un-parent it from a
    deleted skeleton (keeping its world place), drop the add-on Armature
    modifier, clear the driven morph shape-key drivers (keep the sculpted
    shapes, return them to neutral) and strip every add-on property.

    This is the fallback for a mesh with no ledger entry - a character built by
    a version of this add-on that did not yet write down what it changed.  A
    ledgered mesh goes through provenance.restore_object instead, which knows
    which collections it came from and what its weights were.

    `kept_skel` is a merged armature that survives (it is the character's own
    body rig): its modifier is the mesh's real skinning and must stay - only
    the add-on's name on it goes."""
    if obj.parent is not None and obj.parent in deleted_skels:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw
    for mod in list(obj.modifiers):
        if mod.type != 'ARMATURE':
            continue
        if kept_skel is not None and mod.object == kept_skel:
            if mod.name == ARM_MOD_NAME:
                mod.name = "Armature"
            continue
        if (mod.name == ARM_MOD_NAME or mod.object is None
                or mod.object in deleted_skels):
            obj.modifiers.remove(mod)
    _clear_driven_morphs(obj)
    _strip_addon_props(obj)
    _strip_addon_props(obj.data)


def _iter_child_collections(coll):
    out = []
    for child in coll.children:
        out.append(child)
        out.extend(_iter_child_collections(child))
    return out


def _character_parent_collection(root, context):
    if root is not None:
        parents = organization._collection_parents(root)
        if parents:
            return parents[0]
    return context.scene.collection


def _reset_active_pair(context, mh):
    """Return the panel to a clean Setup after the active rig was removed."""
    from ..core import landmarks as lmdata
    mh.switching_character = True
    try:
        mh.skeleton = None
        mh.cage = None
        mh.target = None
        mh.merge_body_armature = None
        mh.merge_head_bone = ""
        mh.merge_deform_bone = ""
        mh.rig_destination = 'UNDECIDED'
    finally:
        mh.switching_character = False
    lmdata.clear(mh, save=False)
    mh.active_cage = None
    mh.active_target = None
    # The removed rig's display state must not become the next one's.
    from ..props import reset_display_toggles
    reset_display_toggles(mh, context)
    mh.ui_tab = 'SETUP'


# op_attach's marker for a mesh the artist attached to the rig (their eyes,
# teeth, tongue).  Spelled out rather than imported: op_attach imports this
# module, and it is the same trick core/organization.py already uses.
ATTACH_PART_PROP = "mhfrt_attached_part"


def _legacy_addon_objects(rid, cage, gui_coll, kept_skel, keep_set):
    """The add-on's own objects in a file made before creation was stamped.

    Everything built by this version of the add-on carries a made-by stamp and
    needs none of this.  A character built by an earlier one carries no stamps
    at all, and refusing to remove it would be its own bug - so the old rules
    stay, as a supplement to the stamps and never as a replacement for them.
    """
    out = set()
    if cage is not None and cage not in keep_set:
        out.add(cage)
    if rid:
        for obj in bpy.data.objects:
            if obj in keep_set or obj == kept_skel:
                continue
            # never delete a user mesh, even if it is rig-tagged
            if obj.type == 'MESH' and (obj.get(MORPH_EXTRA_PROP)
                                       or obj.get(ATTACH_PART_PROP)):
                continue
            if str(obj.get(RIG_ID_PROP) or "") == str(rid):
                out.add(obj)
    if gui_coll is not None:
        for obj in list(gui_coll.objects):
            if obj not in keep_set and obj != kept_skel:
                out.add(obj)
    return out


def _legacy_addon_collections(root, gui_coll):
    """Our collections in a file that predates the made-by stamp.

    Recognised by the properties only this add-on writes: the character root
    and the numbered role collections under it.  A collection the ARTIST made
    and dropped inside the root - a Clothes folder, a Hair folder - carries
    neither, and is left exactly where it is.
    """
    from ..core import provenance
    out = set()
    if root is not None and root.get(organization.ROOT_COLL_PROP):
        out.add(root)
        for child in _iter_child_collections(root):
            if child.get(organization.COLL_ROLE_PROP) \
                    or child.get(organization.ROOT_COLL_PROP) \
                    or provenance.made_by(child):
                out.add(child)
    if gui_coll is not None:
        out.add(gui_coll)
        out.update(_iter_child_collections(gui_coll))
    return out


def _custom_shape_names(armatures):
    """The handle-shape objects these armatures draw their bones with.

    A custom shape belongs to no collection - a pose bone referencing it is its
    only user - so nothing that scans collections or rig ids will ever find it.
    :func:`remove_board_widgets` catches the ones the board tagged; this asks
    the armatures themselves, which needs no tag and therefore also works on a
    file built before the tags existed.
    """
    names = set()
    for arm in armatures:
        try:
            if arm is None or arm.type != 'ARMATURE' or arm.pose is None:
                continue
            for pose_bone in arm.pose.bones:
                shape = pose_bone.custom_shape
                if shape is not None:
                    names.add(shape.name)
        except (ReferenceError, AttributeError):
            continue
    return names


def _sweep_orphan_shapes(names):
    """Delete our handle shapes once nothing draws with them any more.

    Run at the END of a removal, never before it: on a merged rig the facial
    bones live inside the artist's own armature, so until they are stripped out
    that armature still counts as a user of every shape they reference.  Asking
    the question too early left one loose object per removal, in no collection,
    invisible in the outliner's scene view and impossible to select.
    """
    if not names:
        return 0
    in_use = set()
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE' or obj.pose is None:
            continue
        for pose_bone in obj.pose.bones:
            shape = pose_bone.custom_shape
            if shape is not None:
                in_use.add(shape.name)
    removed = 0
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None or name in in_use:
            continue
        if obj.users_collection:
            continue            # it is in the scene: not a loose handle shape
        mesh = obj.data if obj.type == 'MESH' else None
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except (ReferenceError, RuntimeError):
            continue
        removed += 1
        if mesh is not None and mesh.users == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except (ReferenceError, RuntimeError):
                pass
    return removed


def _rehome(obj, fallback):
    """Never leave an object we are keeping in no collection at all."""
    try:
        if obj.users_collection:
            return False
        home = fallback if fallback is not None else \
            bpy.context.scene.collection
        home.objects.link(obj)
        return True
    except (RuntimeError, ReferenceError, AttributeError):
        return False


def _purge_rig(context, skel, root, cage, target, rig_name, record=None):
    """Take this add-on back out of the artist's scene, and nothing else.

    Removal used to be a demolition: delete the character's root collection and
    every collection under it, and rescue whichever meshes a handful of rules
    named.  But the organizer MOVES the artist's body, clothing and hair into
    the character root and unlinks them from the collections they came from, so
    everything the rules did not name went with the root.  Removing the rig
    removed the character.

    It is now bookkeeping, not rules.  Two facts were written down as they
    happened (see core/provenance.py):

    * everything the add-on CREATED carries this character's made-by stamp, and
      the delete set is exactly that stamped set - an object without the stamp
      is the artist's and is never deleted, whatever collection it sits in;
    * everything the add-on CHANGED has a ledger entry saying what it was -
      which collections held it, its parent, its modifiers, its vertex weights
      before the merge rebalanced them - and every one of those is replayed.

    A file built by an older version has neither, so the old rules still run
    underneath as a fallback, with the one difference that matters: a
    collection we did not create is never deleted, and an object we are keeping
    is never left without one.

    Returns (kept mesh count, notes about what was restored)."""
    org = organization
    from ..core import registry, provenance

    ledger = provenance.read(record)
    entries = ledger.get("objects", {})
    restored = []

    # The bones we created - the only vertex groups removal may ever delete.
    our_bones = set()
    if skel is not None and skel.type == 'ARMATURE' and skel.data is not None:
        try:
            our_bones = {b.name for b in skel.data.bones}
        except (AttributeError, ReferenceError):
            our_bones = set()

    rid = skel.get(RIG_ID_PROP) if skel is not None else None
    gui_coll = None
    if skel is not None:
        raw = skel.get(RIG_GUI_COLL_PROP)
        gui_coll = raw if isinstance(raw, bpy.types.Collection) else None

    # ---- what the artist owns: never deleted, always put back ------------
    keep, seen = [], set()

    def _add_keep(obj):
        if obj is None or obj.name in seen:
            return
        # The appended cage is add-on data, never a kept user mesh - even when
        # the artist registered it as a morph object (MORPH_EXTRA_PROP) so they
        # could sculpt on it.  Without this guard the MORPH_EXTRA_PROP keep-loop
        # below (and an Extras-collection listing) would preserve it.
        if cage is not None and obj == cage:
            return
        # A made-by stamp says we created it - unless the ledger ALSO has a
        # "before" entry for it, which only exists for something that was here
        # first.  When the two disagree the ledger wins: it is the record of
        # the artist handing us their object, and no stamp outranks that.
        if provenance.is_ours(record, obj) \
                and provenance.entry_for(record, obj, ledger) is None:
            return
        seen.add(obj.name)
        keep.append(obj)

    _add_keep(target)          # the head is always the user's, tagged or not
    # Anything we wrote a "before" entry for was here before we were, by
    # definition.  This is what brings the body, the clothing and the hair
    # back - the organizer filed them under the character and they were never
    # on any keep list.
    for key, entry in entries.items():
        _add_keep(provenance.resolve(record, key, entry))
    if root is not None:
        for child in root.children:
            role = child.get(org.COLL_ROLE_PROP)
            if role in {org.ROLE_HEAD, org.ROLE_EXTRAS, org.ROLE_PARTS,
                        org.ROLE_BODY}:
                for obj in list(child.objects):
                    _add_keep(obj)
            elif role is None:
                # A folder the artist made and dropped inside the character.
                for obj in list(child.all_objects):
                    _add_keep(obj)
    if rid:
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            if obj.get(MORPH_EXTRA_PROP) or obj.get(ATTACH_PART_PROP):
                if str(obj.get(RIG_ID_PROP) or "") == str(rid):
                    _add_keep(obj)
    keep_set = set(keep)

    # ---- what we created: the delete set ---------------------------------
    # Before anything is removed: the cache points into the board's pose bones
    # and the skeleton's, both of which are about to go (see drop_cache).
    drop_cache(rid)
    created_objs, created_colls, created_data = \
        provenance.created_datablocks(record)

    kept_skel = None
    if skel is not None:
        if skel.get(RIG_MERGED_PROP) or provenance.entry_for(record, skel):
            # One-armature rig: the skeleton IS the artist's body armature.
            # Peel the facial bones back out instead of deleting their rig.
            # Kept unconditionally - not through _add_keep, whose stamp check
            # must never get a say over the artist's own body rig.
            kept_skel = skel
            if skel.name not in seen:
                seen.add(skel.name)
                keep.append(skel)
            keep_set.add(skel)

    delete_objs = {obj for obj in created_objs
                   if obj not in keep_set and obj is not kept_skel}
    delete_objs |= _legacy_addon_objects(rid, cage, gui_coll, kept_skel,
                                         keep_set)
    # The board's handles again, from the other end: whatever our armatures
    # draw their bones with.  Only NAMES here - whether each one is still in
    # use cannot be answered until the facial bones are out of the merged
    # armature, so the sweep itself runs at the very end.
    shape_names = _custom_shape_names(
        [skel] + [obj for obj in delete_objs if obj.type == 'ARMATURE'])
    # The last word, whatever any rule above decided: an object the ledger
    # describes existed before this add-on did, and cannot be deleted by it.
    delete_objs = {obj for obj in delete_objs
                   if provenance.entry_for(record, obj, ledger) is None}

    # Everything below this line crosses a bpy.ops call - Edit Mode on the
    # merged armature - and a bpy.ops call from Python pushes an undo step.
    # A global undo step reloads bpy.data, and every Python reference held
    # across it goes stale ("StructRNA of type Object has been removed") even
    # though the object is perfectly alive.  So the working sets are carried
    # over that line as NAMES and resolved back afterwards.  Getting this wrong
    # meant the shirt was found, then failed to be restored, and kept our
    # properties forever.
    keep_names = [obj.name for obj in keep]
    delete_names = {obj.name for obj in delete_objs}
    created_coll_names = [coll.name for coll in created_colls]
    legacy_coll_names = {coll.name for coll in
                         _legacy_addon_collections(root, gui_coll)}
    created_data_names = _datablock_names(created_data)
    kept_skel_name = kept_skel.name if kept_skel is not None else ""
    root_name = root.name if root is not None else ""

    if skel is not None:
        try:
            from .op_anim import remove_face_animation
            remove_face_animation(skel)     # deletes the imported action too
        except Exception:
            pass

    # The board's handle shapes are in no collection and carry no rig id, so
    # neither scan above reaches them (see remove_board_widgets).
    remove_board_widgets(rid)

    # ---- take the facial bones back out of the artist's own armature -----
    if kept_skel is not None:
        from . import op_merge
        entry = provenance.entry_for(record, kept_skel, ledger)
        op_merge.strip_facial_from_armature(context, kept_skel)
        kept_skel = bpy.data.objects.get(kept_skel_name)
        if kept_skel is not None and entry is not None:
            # The join re-derives bone rolls: put the artist's own bones back
            # to the rest they had before the merge, to the micron.
            moved = provenance.restore_armature_rest(context, kept_skel, entry)
            kept_skel = bpy.data.objects.get(kept_skel_name)
            if moved:
                restored.append(f"put {moved} bone(s) of "
                                f"'{kept_skel_name}' back to their original "
                                "rest position")
            if kept_skel is not None:
                provenance.restore_props(kept_skel, entry)
        elif kept_skel is not None:
            _strip_addon_props(kept_skel)
            _strip_addon_props(kept_skel.data)

    # ---- back to live datablocks -----------------------------------------
    keep = [bpy.data.objects[name] for name in keep_names
            if name in bpy.data.objects]
    delete_objs = {bpy.data.objects[name] for name in delete_names
                   if name in bpy.data.objects}
    deleted_skels = {obj for obj in delete_objs if obj.type == 'ARMATURE'}
    created_colls = [bpy.data.collections[name] for name in created_coll_names
                     if name in bpy.data.collections]
    doomed_colls = set(created_colls)
    doomed_colls |= {bpy.data.collections[name] for name in legacy_coll_names
                     if name in bpy.data.collections}
    kept_skel = bpy.data.objects.get(kept_skel_name) if kept_skel_name else None
    root = bpy.data.collections.get(root_name) if root_name else None

    # ---- hand every kept object back the way it was ----------------------
    fallback = _character_parent_collection(root, context)
    legacy_keeps = []
    for obj in keep:
        entry = provenance.entry_for(record, obj, ledger)
        if entry is None:
            legacy_keeps.append(obj)
            continue
        if obj.type == 'MESH':
            _clear_driven_morphs(obj)
        if obj is not kept_skel:
            restored.extend(provenance.restore_object(
                context, record, obj, entry, our_names=our_bones,
                doomed=delete_objs, fallback=fallback, data=ledger))
        else:
            provenance.restore_collections(record, obj, entry,
                                           fallback=fallback)

    # ---- meshes with no ledger entry: the pre-ledger fallback -------------
    # Their original collections were never written down, so the best that can
    # be done is what removal always did: clean them and put them somewhere
    # visible.  Only ever objects that would otherwise be left homeless.
    new_coll = None
    for obj in legacy_keeps:
        if obj.type == 'MESH':
            _clean_kept_mesh(obj, deleted_skels, kept_skel)
        else:
            _strip_addon_props(obj)
            if obj.data is not None:
                _strip_addon_props(obj.data)
    homeless = [obj for obj in legacy_keeps
                if all(coll in created_colls or coll == root
                       or coll.get(org.COLL_ROLE_PROP)
                       for coll in obj.users_collection)]
    if homeless:
        parent = fallback
        new_coll = bpy.data.collections.new(
            (rig_name or DEFAULT_RIG_NAME).strip() or DEFAULT_RIG_NAME)
        parent.children.link(new_coll)
        for obj in homeless:
            for coll in list(obj.users_collection):
                coll.objects.unlink(obj)
            new_coll.objects.link(obj)

    # ---- delete the add-on objects ---------------------------------------
    for obj in delete_objs:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except (ReferenceError, RuntimeError):
            pass

    # ---- delete the add-on collections -----------------------------------
    for coll in doomed_colls:
        if coll is new_coll:
            continue
        # An artist's collection is never deleted, and neither is one still
        # holding something of theirs.
        try:
            survivors = [obj for obj in coll.objects if obj not in delete_objs]
        except ReferenceError:
            continue
        for obj in survivors:
            _rehome(obj, fallback)
        try:
            bpy.data.collections.remove(coll)
        except (ReferenceError, RuntimeError):
            pass

    # ---- nothing of ours left orphaned in the file -----------------------
    # The handle shapes last of all: the facial bones are out of the merged
    # armature by now, so "does anything still draw with this?" finally has the
    # right answer.
    _sweep_orphan_shapes(shape_names)
    _purge_created_data(created_data_names)

    # A kept object can still be homeless if its collections were all ours and
    # it had no ledger entry to send it home.  Nothing leaves this function in
    # that state.
    for obj in keep:
        try:
            _rehome(obj, fallback)
        except ReferenceError:
            continue

    drop_cache(rid)

    # ---- give the artist their own rig and weights back -------------------
    if target is not None and provenance.entry_for(record, target,
                                                   ledger) is None:
        try:
            restored = restored + registry.restore_baseline(record, target,
                                                            our_bones)
        except (ReferenceError, RuntimeError, AttributeError):
            pass
    return len(keep), restored


_DATA_KINDS = ("meshes", "armatures", "actions")


def _datablock_names(created_data):
    """[(bpy.data collection name, datablock name)] - safe across an undo push."""
    out = []
    for datablock in created_data:
        for kind in _DATA_KINDS:
            source = getattr(bpy.data, kind)
            try:
                if source.get(datablock.name) == datablock:
                    out.append((kind, datablock.name))
                    break
            except (ReferenceError, AttributeError, TypeError):
                continue
    return out


def _purge_created_data(created_data_names):
    """Delete the meshes/armatures/actions we made, once nothing uses them.

    Their objects are gone by now; without this the datablocks ride along in
    the file as zero-user orphans until the next save decides to drop them.
    """
    for kind, name in created_data_names:
        source = getattr(bpy.data, kind, None)
        datablock = source.get(name) if source is not None else None
        if datablock is None:
            continue
        try:
            if datablock.users:
                continue
            source.remove(datablock)
        except (ReferenceError, RuntimeError):
            pass


class MHFRT_OT_remove_rig(bpy.types.Operator):
    bl_idname = "mhfrt.remove_rig"
    bl_label = "Remove Rig"
    bl_description = (
        "Take this add-on back out of your scene. Everything it created is "
        "deleted - the head cage, the fitted skeleton, the control board, the "
        "driven morph drivers, the imported animation, the landmarks and the "
        "MHFRT collections - and everything it CHANGED is put back the way you "
        "had it: your body, clothing and hair return to the collections they "
        "came from, your head gets its own armature and weights back, and a "
        "merged armature has the facial bones peeled out and its own bones "
        "restored. Your character is not deleted. Its .mhfrt file stays on "
        "disk, so the rig can be loaded back")
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        mh = context.scene.mhfrt
        idx = mh.rig_ui_active_index
        if not (0 <= idx < len(mh.rig_ui_items)):
            self.report({'ERROR'}, "Select a rig in the list first")
            return {'CANCELLED'}
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from .op_live import stop_running
        from .op_pairs import is_running as landmarks_running
        from ..core import rest_tuning
        mh = context.scene.mhfrt
        idx = mh.rig_ui_active_index
        if not (0 <= idx < len(mh.rig_ui_items)):
            self.report({'ERROR'}, "Select a rig in the list first")
            return {'CANCELLED'}
        item = mh.rig_ui_items[idx]
        rig_name = item.rig_name
        item_key = item.rig_key

        if landmarks_running():
            self.report({'ERROR'},
                        "Finish landmark mode first (Esc in the split view)")
            return {'CANCELLED'}

        entry = character_for_item(item)
        if entry is not None:
            skel, root = entry.skel, entry.root
            cage, target = entry.cage, entry.target
        else:
            skel = root = cage = target = None
            if item.is_new:            # in-Setup pair, never rooted
                cage, target = mh.cage, mh.target

        if skel is None and root is None and not item.is_new:
            self.report({'ERROR'}, "That rig no longer exists - use Refresh")
            return {'CANCELLED'}
        # A linked character lives in another .blend.  Nothing here can be
        # deleted, and trying threw a raw traceback at the artist.
        linked = [d for d in (skel, root, cage, target)
                  if d is not None and getattr(d, "library", None) is not None]
        if linked:
            self.report({'ERROR'},
                        f"'{rig_name}' is linked from another file - remove it "
                        "in the file it came from, or unlink the collection "
                        "here first")
            return {'CANCELLED'}
        if skel is not None and skel.get(rest_tuning.TONGUE_SESSION_PROP):
            self.report({'ERROR'}, "Finish or Cancel Tongue Edit first")
            return {'CANCELLED'}

        stop_running()
        was_active = (
            (skel is not None and skel == mh.skeleton)
            or (skel is None and mh.skeleton is None
                and cage is not None and cage == mh.cage)
            or item.is_new)

        # Purge handles every case, including an in-Setup pair (skeleton/root
        # None): it still deletes the add-on-appended cage and keeps the head.
        from ..core import registry as _registry
        record = _registry.find(context.scene, item_key)
        kept, restored = _purge_rig(context, skel, root, cage, target,
                                    rig_name, record=record)

        if was_active:
            _reset_active_pair(context, mh)

        from ..core import registry
        saved_file = str(getattr(record, "file_path", "") or "")
        registry.remove(context.scene, item_key)
        bump_rig_topology()
        _rescan(full=True)
        _evaluate_guarded(list(_caches.values()))
        sync_rig_ui_state(mh)
        message = (f"Removed rig '{rig_name}' - kept {kept} of your "
                   "object(s)")
        if restored:
            message += "; " + ", ".join(restored)
        if saved_file and os.path.exists(saved_file):
            message += (f"; the character is still saved in "
                        f"{os.path.basename(saved_file)}")
        self.report({'INFO'}, message)
        return {'FINISHED'}


_classes = (MHFRT_OT_save_character, MHFRT_OT_load_character,
            MHFRT_OT_rebuild_board, MHFRT_OT_board_layout,
            MHFRT_OT_eye_target, MHFRT_OT_follow_head,
            MHFRT_OT_edit_board, MHFRT_OT_activate_character,
            MHFRT_OT_new_character, MHFRT_OT_remove_rig,
            MHFRT_OT_dismiss_repair_notes)


# every event that rebuilds or swaps datablocks must flush the caches
_INVALIDATE_EVENTS = ("load_pre", "load_post", "undo_pre", "undo_post",
                      "redo_pre", "redo_post")


def _restore_auto_muted_shape_keys():
    """Leave every ShapeKey mute flag as it was before runtime optimization."""
    # Live pointer ownership survives KeyBlock renames, unlike the persistent
    # name marker.  Restore it first, then sweep markers left by an older
    # session/add-on version.
    for cache in _caches.values():
        for record in cache.get("shape_key_auto_mute_records", {}).values():
            try:
                shape_keys = record["shape_keys"]
                current = {
                    key.as_pointer(): key for key in shape_keys.key_blocks
                }
                for pointer in tuple(record.get("owned", {})):
                    key = current.get(pointer)
                    if key is not None and key.mute:
                        key.mute = False
                record["owned"].clear()
                _store_auto_mute_names(record)
            except ReferenceError:
                pass
    for shape_keys in bpy.data.shape_keys:
        names = {
            name for name in shape_keys.get(AUTO_MUTE_PROP, "").split("\n")
            if name
        }
        for key in shape_keys.key_blocks:
            if key.name in names and key.mute:
                key.mute = False
        if AUTO_MUTE_PROP in shape_keys:
            del shape_keys[AUTO_MUTE_PROP]


@persistent
def _on_save_pre(*_args):
    """Do not serialize runtime mute flags/ownership into the artist's file."""
    # Also the last moment before a save to get the interface lock into the
    # file itself, so it is already set the next time this .blend is opened -
    # and because "set the output path, save, render" is exactly the sequence
    # that catches a session where no depsgraph tick has happened yet.
    ensure_render_safety(deep=True)
    _restore_auto_muted_shape_keys()
    # Board clips are OBJECT-level actions; an unguarded one gets replayed on
    # the character itself by FBX export (see op_anim.guard_board_actions).
    # Doing it here also repairs files made with an older add-on version and
    # board animation the artist keyed by hand.
    try:
        from .op_anim import ensure_export_safety
        for skel in _rig_skeletons():
            ensure_export_safety(skel)
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    # Every character's file goes with the save: the manifest and the ledger of
    # what we changed always (kilobytes), and the character payload itself the
    # first time - after that it is rewritten on demand, from Save Character,
    # rather than adding a multi-megabyte write to every Ctrl+S.
    try:
        from ..core import sidecar, registry
        scene = getattr(bpy.context, "scene", None)
        for record in registry.records(scene) if scene else ():
            sidecar.touch(scene, record,
                          include_blend=not sidecar.has_payload(record))
    except Exception:                        # noqa: BLE001 - never block a save
        pass


@persistent
def _on_save_post(*_args):
    """Rebuild runtime-only mutes after Blender has written the clean file."""
    invalidate()
    # Character files written while the .blend had no path of its own now
    # follow it into the project folder.
    try:
        from ..core import sidecar
        sidecar.relocate(getattr(bpy.context, "scene", None))
    except Exception:                        # noqa: BLE001
        pass


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    handlers = bpy.app.handlers
    if _rig_handler not in handlers.depsgraph_update_post:
        handlers.depsgraph_update_post.append(_rig_handler)
    if _viewport_click_sync_handler not in handlers.depsgraph_update_post:
        handlers.depsgraph_update_post.append(_viewport_click_sync_handler)
    if _frame_change_handler not in handlers.frame_change_post:
        handlers.frame_change_post.append(_frame_change_handler)
    if _on_save_pre not in handlers.save_pre:
        handlers.save_pre.append(_on_save_pre)
    if _on_save_post not in handlers.save_post:
        handlers.save_post.append(_on_save_post)
    for event in _INVALIDATE_EVENTS:
        hlist = getattr(handlers, event)
        if _on_invalidate not in hlist:
            hlist.append(_on_invalidate)
    invalidate()


def unregister():
    handlers = bpy.app.handlers
    if _rig_handler in handlers.depsgraph_update_post:
        handlers.depsgraph_update_post.remove(_rig_handler)
    if _viewport_click_sync_handler in handlers.depsgraph_update_post:
        handlers.depsgraph_update_post.remove(_viewport_click_sync_handler)
    if _frame_change_handler in handlers.frame_change_post:
        handlers.frame_change_post.remove(_frame_change_handler)
    if _on_save_pre in handlers.save_pre:
        handlers.save_pre.remove(_on_save_pre)
    if _on_save_post in handlers.save_post:
        handlers.save_post.remove(_on_save_post)
    for event in _INVALIDATE_EVENTS:
        hlist = getattr(handlers, event)
        if _on_invalidate in hlist:
            hlist.remove(_on_invalidate)
    _restore_auto_muted_shape_keys()
    _caches.clear()
    _control_map.clear()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
