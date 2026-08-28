"""Facial animation import - UE5 performance -> the live control board.

One button: pick a file, the clip lands as keyframes on the character's own
control board and plays through the always-on RigLogic runtime (bones +
morphs), so scrubbing, playback and manual polish all keep working.

Supported sources (auto-detected):

* .fbx - an Unreal LEVEL SEQUENCE export of the MetaHuman face board
  (Face_ControlBoard_CtrlRig track -> Export... FBX).  Every control comes
  in as an animated node whose local x/y ARE the board's tx/ty values;
  MetaHuman Animator performances baked to the control rig export this way.
  AnimSequence exports that carry CTRL_expressions_* custom-attribute
  curves are harvested too (bone-only exports contain no facial curves -
  the importer says so instead of guessing).
* .csv - a Live Link Face (ARKit) take, or a generic "Frame,curve,..."
  dump.  ARKit blend shapes map onto RigLogic raw controls, then the raw
  values are inverted back to board values.
* .json - {"fps": 30, "curves": {name: [[frame, value], ...]}} (or just
  the curves dict).  Names may be board controls ("CTRL_C_jaw.ty"), raw
  controls ("CTRL_expressions_jawOpen") or ARKit shapes ("jawOpen").

Raw/ARKit curves go through riglogic.invert_raw_frames - exact for values
produced by the DNA's own GUI->raw map (no raw control is fed by two GUI
channels in the bundled DNA).
"""

import os
import re
import csv
import json
import math

import numpy as np
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper

from ..props import RIG_ID_PROP
from ..core import riglogic
from ..core import board
from ..ui import flow

# ID properties on the skeleton: the imported clip (ID pointer -> action)
# and a short human-readable summary for the panel row.
ANIM_ACTION_PROP = "mhfrt_face_anim_action"
ANIM_INFO_PROP = "mhfrt_face_anim_info"

# --- Exporter safety, see guard_board_actions() and _ensure_bake_action() ---
# The board's clip is an OBJECT-level action (it keyframes control locations).
# Blender's FBX exporter, with "Bake Animation > All Actions" (its default),
# force-assigns EVERY action in the file to each exported object whose data
# paths all resolve there - and "location" resolves on any object, so the
# character's armature inherits a control's motion as ROOT animation: the whole
# rig slides around exactly like the face clip.  A guard channel on a custom
# property that only board controls carry makes the clip resolve on the board
# and nowhere else, and a tiny marker action on the skeleton gives the exporter
# the right thing to bake instead: the live rig, over the clip's frame range.
BOARD_GUARD_PROP = "mhfrt_board_control"
BAKE_MARK_PROP = "mhfrt_face_anim_bake"
ANIM_BAKE_ACTION_PROP = "mhfrt_face_anim_bake_action"

# Native export format.  A unique extension plus a self-describing header, so
# an exported clip is recognisably THIS add-on's file rather than an anonymous
# .json.  It is still JSON underneath (human-readable, robust): the importer
# loads .mhfa and .json through the very same parser, and the extra header keys
# are ignored on load.
FACE_ANIM_EXT = ".mhfa"                 # MetaHuman Facial Animation clip
FACE_ANIM_FORMAT_ID = "MHFRT_FACE_ANIM"
FACE_ANIM_FORMAT_VERSION = 1

_DUP_SUFFIX = re.compile(r"\.\d{3,}$")          # Blender's .001 rename suffix
_RAW_PREFIXES = ("CTRL_expressions.", "CTRL_expressions_")

# Live Link Face columns that are head/eye rigid pose, not blend shapes.
_ARKIT_POSE = {"headYaw", "headPitch", "headRoll",
               "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
               "rightEyeYaw", "rightEyePitch", "rightEyeRoll"}

# ARKit blend shape -> RigLogic raw control label(s) (CTRL_expressions.<label>).
# One ARKit value drives every listed raw at full weight (quadrant splits).
ARKIT_TO_RAW = {
    "eyeBlinkLeft": ("eyeBlinkL",),
    "eyeBlinkRight": ("eyeBlinkR",),
    "eyeLookDownLeft": ("eyeLookDownL",),
    "eyeLookDownRight": ("eyeLookDownR",),
    "eyeLookInLeft": ("eyeLookRightL",),      # "in" = toward the nose
    "eyeLookInRight": ("eyeLookLeftR",),
    "eyeLookOutLeft": ("eyeLookLeftL",),
    "eyeLookOutRight": ("eyeLookRightR",),
    "eyeLookUpLeft": ("eyeLookUpL",),
    "eyeLookUpRight": ("eyeLookUpR",),
    "eyeSquintLeft": ("eyeSquintInnerL",),
    "eyeSquintRight": ("eyeSquintInnerR",),
    "eyeWideLeft": ("eyeWidenL",),
    "eyeWideRight": ("eyeWidenR",),
    "browDownLeft": ("browDownL",),
    "browDownRight": ("browDownR",),
    "browInnerUp": ("browRaiseInL", "browRaiseInR"),
    "browOuterUpLeft": ("browRaiseOuterL",),
    "browOuterUpRight": ("browRaiseOuterR",),
    "cheekPuff": ("mouthCheekBlowL", "mouthCheekBlowR"),
    "cheekSquintLeft": ("eyeCheekRaiseL",),
    "cheekSquintRight": ("eyeCheekRaiseR",),
    "noseSneerLeft": ("noseWrinkleL",),
    "noseSneerRight": ("noseWrinkleR",),
    "jawForward": ("jawFwd",),
    "jawLeft": ("jawLeft",),
    "jawRight": ("jawRight",),
    "jawOpen": ("jawOpen",),
    "mouthClose": ("mouthLipsTogetherUL", "mouthLipsTogetherUR",
                   "mouthLipsTogetherDL", "mouthLipsTogetherDR"),
    "mouthFunnel": ("mouthFunnelUL", "mouthFunnelUR",
                    "mouthFunnelDL", "mouthFunnelDR"),
    "mouthPucker": ("mouthLipsPurseUL", "mouthLipsPurseUR",
                    "mouthLipsPurseDL", "mouthLipsPurseDR"),
    "mouthLeft": ("mouthLeft",),
    "mouthRight": ("mouthRight",),
    "mouthSmileLeft": ("mouthCornerPullL",),
    "mouthSmileRight": ("mouthCornerPullR",),
    "mouthFrownLeft": ("mouthCornerDepressL",),
    "mouthFrownRight": ("mouthCornerDepressR",),
    "mouthDimpleLeft": ("mouthDimpleL",),
    "mouthDimpleRight": ("mouthDimpleR",),
    "mouthStretchLeft": ("mouthStretchL",),
    "mouthStretchRight": ("mouthStretchR",),
    "mouthRollLower": ("mouthLowerLipRollInL", "mouthLowerLipRollInR"),
    "mouthRollUpper": ("mouthUpperLipRollInL", "mouthUpperLipRollInR"),
    "mouthShrugLower": ("jawChinRaiseDL", "jawChinRaiseDR"),
    "mouthShrugUpper": ("mouthLipsPushUL", "mouthLipsPushUR"),
    "mouthPressLeft": ("mouthPressUL", "mouthPressDL"),
    "mouthPressRight": ("mouthPressUR", "mouthPressDR"),
    "mouthLowerDownLeft": ("mouthLowerLipDepressL",),
    "mouthLowerDownRight": ("mouthLowerLipDepressR",),
    "mouthUpperUpLeft": ("mouthUpperLipRaiseL",),
    "mouthUpperUpRight": ("mouthUpperLipRaiseR",),
    "tongueOut": ("tongueOut",),
}


class AnimImportError(Exception):
    """User-facing import failure (message is shown as the error report)."""


# --------------------------------------------------------------- fcurves ---

def _action_fcurves(action):
    """Every fcurve of an action, slotted (4.4+) or legacy layout."""
    if action is None:
        return
    seen = False
    try:
        from bpy_extras import anim_utils
        for slot in action.slots:
            bag = anim_utils.action_get_channelbag_for_slot(action, slot)
            if bag is not None:
                for fc in bag.fcurves:
                    seen = True
                    yield fc
    except (ImportError, AttributeError):
        pass
    if not seen:
        for fc in getattr(action, "fcurves", ()):
            yield fc


def _fcurve_keys(fc):
    """[(frame, value), ...] of an fcurve, fast."""
    n = len(fc.keyframe_points)
    co = np.empty(n * 2)
    fc.keyframe_points.foreach_get("co", co)
    return co.reshape(n, 2)


# ------------------------------------------------------------ FBX reading ---

_DATA_COLLECTIONS = ("objects", "actions", "meshes", "armatures", "cameras",
                     "lights", "images", "materials", "collections")


def _load_fbx_curves(context, filepath):
    """Import the FBX into a holding pen, harvest curves, remove every trace.

    Returns (gui_curves, raw_curves, src_fps, found_names):
      gui_curves: {"CTRL_x.tx": (n,2) [frame, value] array} - board controls
      raw_curves: {"CTRL_expressions.x": (n,2) array} - raw control curves
      found_names: animated node names seen (for the error message when
      nothing maps).
    """
    scene = context.scene
    before = {name: set(getattr(bpy.data, name).values())
              for name in _DATA_COLLECTIONS}
    saved = (scene.render.fps, scene.render.fps_base, scene.frame_start,
             scene.frame_end, scene.frame_current)
    gui_curves, raw_curves, found = {}, {}, []
    try:
        try:
            if hasattr(bpy.ops.import_scene, "fbx"):
                bpy.ops.import_scene.fbx(filepath=filepath)
            else:
                bpy.ops.wm.fbx_import(filepath=filepath)
        except Exception as exc:
            raise AnimImportError(f"Could not read FBX: {exc}") from exc

        src_fps = scene.render.fps / (scene.render.fps_base or 1.0)

        def harvest(name, path, fc):
            base = _DUP_SUFFIX.sub("", name)
            if path == "location" and fc.array_index in (0, 1) \
                    and base.startswith("CTRL_"):
                channel = "tx" if fc.array_index == 0 else "ty"
                gui_curves[f"{base}.{channel}"] = _fcurve_keys(fc)
            elif path.startswith('["'):        # animated custom attribute
                prop = path[2:].split('"')[0]
                label = _raw_label(prop)
                if label:
                    raw_curves[label] = _fcurve_keys(fc)

        new_objects = [o for o in bpy.data.objects.values()
                       if o not in before["objects"]]
        for obj in new_objects:
            action = obj.animation_data.action if obj.animation_data else None
            if action is None:
                continue
            found.append(_DUP_SUFFIX.sub("", obj.name))
            for fc in _action_fcurves(action):
                if fc.data_path.startswith('pose.bones["'):
                    bone = fc.data_path.split('"')[1]
                    sub = fc.data_path.rsplit(".", 1)[-1]
                    harvest(bone, sub, fc)
                else:
                    harvest(obj.name, fc.data_path, fc)
    finally:
        _purge_new_datablocks(before)
        (scene.render.fps, scene.render.fps_base, scene.frame_start,
         scene.frame_end, scene.frame_current) = saved
    return gui_curves, raw_curves, src_fps, found


def _purge_new_datablocks(before):
    """Remove everything the FBX import created (objects first, then data)."""
    for name in _DATA_COLLECTIONS:
        coll = getattr(bpy.data, name)
        for block in [b for b in coll.values() if b not in before[name]]:
            try:
                coll.remove(block)
            except (ReferenceError, RuntimeError):
                pass


def _raw_label(name):
    """'CTRL_expressions_jawOpen' / '...expressions.jawOpen' -> 'jawOpen'."""
    for prefix in _RAW_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


# ------------------------------------------------------------ CSV reading ---

_STD_RATES = (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 120.0)


def _load_csv_curves(filepath):
    """Live Link Face take, or generic Frame,<curve>,... dump.

    Returns (named_series, times, src_fps): named_series maps source curve
    name -> (n,) value array sampled at `times` (seconds from clip start).
    """
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise AnimImportError("The CSV file is empty") from None
        rows = [r for r in reader if len(r) >= len(header)]
    if not rows:
        raise AnimImportError("The CSV file has a header but no rows")
    header = [h.strip() for h in header]

    if header[0].lower() == "timecode":
        first_curve = 2 if header[1].lower() == "blendshapecount" else 1
        frames = []
        for r in rows:
            parts = r[0].split(":")
            try:
                h, m, s, fr = (float(p) for p in parts[:4])
            except ValueError:
                raise AnimImportError(
                    f"Bad timecode '{r[0]}' - expected HH:MM:SS:FF") from None
            frames.append((h * 3600 + m * 60 + s, fr))
        max_frame = max(fr for _s, fr in frames)
        src_fps = next((r for r in _STD_RATES if r > max_frame), 60.0)
        times = np.array([s + fr / src_fps for s, fr in frames])
        times -= times[0]
        if not np.all(np.diff(times) >= 0):        # midnight wrap etc.
            times = np.arange(len(rows)) / src_fps
    else:
        # generic: first column is the frame number
        first_curve = 1
        src_fps = 30.0
        try:
            times = (np.array([float(r[0]) for r in rows])
                     - float(rows[0][0])) / src_fps
        except ValueError:
            raise AnimImportError(
                "Unrecognised CSV - expected a Live Link Face take "
                "(Timecode,BlendShapeCount,...) or Frame,<curve>,... "
                "columns") from None

    series = {}
    for c, name in enumerate(header[first_curve:], start=first_curve):
        if not name:
            continue
        try:
            series[name] = np.array([float(r[c]) for r in rows])
        except ValueError:
            continue                                # non-numeric column
    return series, times, src_fps


# ----------------------------------------------------------- JSON reading ---

def _load_json_curves(filepath):
    """{"fps": 30, "curves": {name: [[frame, v], ...]}} or the bare dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise AnimImportError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AnimImportError("JSON must be an object of curves")
    src_fps = float(data.get("fps", 30.0)) or 30.0
    curves = data.get("curves", data)
    if not isinstance(curves, dict):
        raise AnimImportError("JSON 'curves' must map names to key lists")
    out = {}
    for name, keys in curves.items():
        if name == "fps" or not isinstance(keys, (list, tuple)) or not keys:
            continue
        try:
            arr = np.asarray(keys, float).reshape(len(keys), 2)
        except (ValueError, TypeError):
            continue
        out[name] = arr
    if not out:
        raise AnimImportError("No curves found in the JSON file")
    return out, src_fps


# ------------------------------------------------------- name resolution ---

def _resolve_named_curves(named, meta):
    """Split arbitrary source names into board / raw / arkit buckets.

    `named` maps name -> curve payload; returns (gui, raw, arkit, unknown)
    with the same payloads keyed by canonical names.
    """
    gui_names = set(meta["gui_names"])
    raw_labels = {n.rpartition(".")[2] for n in meta["raw_names"]}
    gui, raw, arkit, unknown = {}, {}, {}, []
    for name, payload in named.items():
        base = name.strip()
        label = _raw_label(base)
        if label is not None:
            if label in raw_labels:
                raw[label] = payload
            else:
                unknown.append(base)
            continue
        if base in gui_names:
            gui[base] = payload
            continue
        cam = base[:1].lower() + base[1:]
        if cam in _ARKIT_POSE:
            continue                          # rigid head/eye pose: skipped
        if cam in ARKIT_TO_RAW:
            arkit[cam] = payload
            continue
        if base in raw_labels:
            raw[base] = payload
            continue
        unknown.append(base)
    return gui, raw, arkit, unknown


# ------------------------------------------------------ curve conversion ---

def _series_to_gui(raw_series, arkit_series, times, meta):
    """Synchronised raw/ARKit value arrays -> {gui_name: (n,2) [t, value]}.

    `raw_series`/`arkit_series` map names to (n,) arrays over `times`.
    """
    raw_index = {n.rpartition(".")[2]: i
                 for i, n in enumerate(meta["raw_names"])}
    n = len(times)
    raw = np.zeros((n, meta["raw_count"]))
    for label, values in raw_series.items():
        raw[:, raw_index[label]] = values
    for cam, values in arkit_series.items():
        for label in ARKIT_TO_RAW[cam]:
            i = raw_index.get(label)
            if i is not None:
                raw[:, i] = np.maximum(raw[:, i], values)
    gui = riglogic.invert_raw_frames(raw)
    out = {}
    for g, name in enumerate(meta["gui_names"]):
        col = gui[:, g]
        if np.abs(col).max() > 1e-5:
            out[name] = np.column_stack([times, col])
    return out


def _keys_to_series(curves, src_fps, start=None):
    """Per-curve [frame, value] keys -> synchronised arrays on a common grid.

    Returns (series dict name -> (n,) values, times seconds).  `start`
    overrides the grid's first source frame so several curve groups from
    one file share a single clip-relative time base.
    """
    if start is None:
        start = min(float(arr[:, 0].min()) for arr in curves.values())
    end = max(float(arr[:, 0].max()) for arr in curves.values())
    count = int(round((end - start))) + 1
    if count > 200000:
        raise AnimImportError("Clip is too long (over 200k source frames)")
    grid = np.arange(count, dtype=float) + start
    series = {}
    for name, arr in curves.items():
        order = np.argsort(arr[:, 0], kind="stable")
        series[name] = np.interp(grid, arr[order, 0], arr[order, 1])
    return series, (grid - start) / src_fps


# ---------------------------------------------------------- board writing ---

def _ensure_action_channelbag(anim_utils, action, slot):
    """Blender 4.5-compatible channelbag creation for a fresh layered action."""
    ensure = getattr(anim_utils, "action_ensure_channelbag_for_slot", None)
    if ensure is not None:
        return ensure(action, slot)
    existing = anim_utils.action_get_channelbag_for_slot(action, slot)
    if existing is not None:
        return existing
    layer = action.layers[0] if action.layers else action.layers.new("Face Animation")
    strip = layer.strips[0] if layer.strips else layer.strips.new(type='KEYFRAME')
    return strip.channelbags.new(slot)


def _board_controls(skel):
    return board.controls_for_rig(RIG_ID_PROP, skel.get(RIG_ID_PROP), skel=skel)


def _board_owners(skel):
    """The Objects that hold this board's animation data, de-duplicated.

    Animation lives on objects, never on pose bones, so every read or write of
    board animation goes through here: one entry per loose-object control, and
    a single entry for a whole bone board.
    """
    owners = {}
    for control in _board_controls(skel).values():
        owner = board.owner_object(control)
        if owner is not None:
            owners[owner.name] = owner
    return owners


# ------------------------------------------------------- exporter safety ---

def _guard_data_path():
    return f'["{BOARD_GUARD_PROP}"]'


def _channelbag(action, slot):
    if action is None or slot is None:
        return None
    try:
        from bpy_extras import anim_utils
        return anim_utils.action_get_channelbag_for_slot(action, slot)
    except (ImportError, AttributeError):
        return None


def _add_guard_curve(bag, frame_start, frame_end):
    """One constant channel that only resolves on a board control."""
    path = _guard_data_path()
    for fc in bag.fcurves:
        if fc.data_path == path:
            return False
    fc = bag.fcurves.new(data_path=path, index=0)
    fc.keyframe_points.add(2)
    fc.keyframe_points.foreach_set(
        "co", [float(frame_start), 1.0, float(frame_end), 1.0])
    fc.keyframe_points.foreach_set("interpolation", [0, 0])   # CONSTANT
    fc.update()
    return True


def guard_board_actions(skel):
    """Stop exporters from replaying the board's clip on the character itself.

    An action is only offered to an object when EVERY one of its channels
    resolves there (see the module header).  Adding one channel on
    ``mhfrt_board_control`` - a property only board controls have - keeps the
    clip on the board, so a merged rig never inherits it as root motion.
    Idempotent, and safe to run on hand-keyed board animation too.

    A bone board is already safe on its own - its channels are
    ``pose.bones["CTRL_*"].location`` and no character armature has bones by
    those names - but the guard is kept for both layouts so the two behave
    identically and a board merged into another armature stays covered.  The
    property goes on the OWNER object, which is what the action resolves
    against: the board armature for a bone board, the control itself for a
    loose object.
    """
    if skel is None:
        return 0
    guarded = 0
    for obj in _board_owners(skel).values():
        ad = obj.animation_data
        action = ad.action if ad else None
        if action is None:
            continue
        if obj.get(BOARD_GUARD_PROP) is None:
            obj[BOARD_GUARD_PROP] = 1.0
        bag = _channelbag(action, getattr(ad, "action_slot", None))
        if bag is None:
            continue
        start, end = action.frame_range
        if _add_guard_curve(bag, start, end):
            guarded += 1
    return guarded


def _ensure_bake_action(skel, clip_name, frame_start, frame_end):
    """Give exporters a clip to bake for the character itself.

    With the board clip guarded, "All Actions" would find nothing to export for
    the armature and the face would be missing from the FBX.  This action
    carries a single constant channel on the skeleton's own property, so the
    exporter samples the LIVE rig (the runtime drives the facial bones) across
    the clip's frame range and writes a proper bone animation - with the
    character's own transform left exactly where it is.
    """
    from bpy_extras import anim_utils

    _remove_bake_action(skel)
    skel[BAKE_MARK_PROP] = 0.0
    action = bpy.data.actions.new(f"FaceAnim_{clip_name}_Bake")
    slot = action.slots.new('OBJECT', skel.name)
    bag = _ensure_action_channelbag(anim_utils, action, slot)
    fc = bag.fcurves.new(data_path=f'["{BAKE_MARK_PROP}"]', index=0)
    fc.keyframe_points.add(2)
    fc.keyframe_points.foreach_set(
        "co", [float(frame_start), 0.0, float(frame_end), 0.0])
    fc.keyframe_points.foreach_set("interpolation", [0, 0])   # CONSTANT
    fc.update()
    # Nothing in the scene uses it - it exists purely so exporters bake this
    # character over the clip's range, so it needs a fake user to survive.
    action.use_fake_user = True
    skel[ANIM_BAKE_ACTION_PROP] = action
    return action


def face_bake_action(skel):
    action = skel.get(ANIM_BAKE_ACTION_PROP) if skel else None
    return action if isinstance(action, bpy.types.Action) else None


def _board_frame_range(skel):
    """Frame span of everything keyed on this character's board, or None."""
    start = end = None
    for obj in _board_owners(skel).values():
        ad = obj.animation_data
        action = ad.action if ad else None
        if action is None:
            continue
        lo, hi = action.frame_range
        start = lo if start is None else min(start, lo)
        end = hi if end is None else max(end, hi)
    if start is None or end <= start:
        return None
    return start, end


def _clip_label(skel):
    action = face_animation_action(skel)
    if action is not None:
        name = action.name
        prefix = "FaceAnim_"
        return name[len(prefix):] if name.startswith(prefix) else name
    return "Board"


def ensure_export_safety(skel):
    """Make this character safe AND complete to export at any moment.

    Guards the board clips (so no exporter replays them on the character) and
    makes sure the character itself has a clip to bake, including for board
    animation the artist keyed by hand and for files animated by an older
    add-on version.  Idempotent.
    """
    if skel is None:
        return 0
    guarded = guard_board_actions(skel)
    span = _board_frame_range(skel)
    if span is None:
        if face_animation_action(skel) is None:
            _remove_bake_action(skel)      # nothing left on the board
        return guarded
    if face_bake_action(skel) is None:
        _ensure_bake_action(skel, _clip_label(skel), span[0], span[1])
    return guarded


def _remove_bake_action(skel):
    action = face_bake_action(skel)
    if action is not None:
        try:
            bpy.data.actions.remove(action)
        except (ReferenceError, RuntimeError):
            pass
    for prop in (ANIM_BAKE_ACTION_PROP, BAKE_MARK_PROP):
        if skel is not None and prop in skel:
            del skel[prop]


def _channel_axis(control, channel):
    """(location index, sign) a control uses for a GUI channel, or None.

    A board bone reads tx/ty off local X/Y; a legacy loose object carries the
    Maya Y-up swap, so its ty is +Z (see core/board).
    """
    return board.channel_axis(control, channel)


def _write_board_animation(context, skel, controls, gui_curves,
                           clip_name, src_fps):
    """{gui_name: (n,2) [t_seconds or frame, value]} -> keyframes on the board.

    Curves whose payload is [frame, value] at src_fps must be pre-converted:
    here the first column is SECONDS from clip start.  Returns
    (control_count, key_count, duration_seconds, missing_names).

    ``control_count`` counts CONTROLS, not action owners: a bone board keys
    every control into the one armature's slot, so counting owners would report
    "1 control" for a whole performance.
    """
    from bpy_extras import anim_utils

    scene = context.scene
    scene_fps = scene.render.fps / (scene.render.fps_base or 1.0)
    divisor = board.gui_scale(skel)
    gui_index = {name: i for i, name in enumerate(riglogic.meta()["gui_names"])}

    remove_face_animation(skel)                      # replace, never stack

    action = bpy.data.actions.new(f"FaceAnim_{clip_name}")
    slots, bags = {}, {}
    n_keys = 0
    duration = 0.0
    animated = {}
    keyed = set()
    missing = []
    for gui_name, keys in gui_curves.items():
        template, _, channel = gui_name.rpartition(".")
        control = controls.get(template)
        axis = _channel_axis(control, channel)
        if control is None or axis is None:
            missing.append(gui_name)
            continue
        g = gui_index.get(gui_name)
        lo, hi = riglogic.gui_channel_range(g) if g is not None else (-1.0, 1.0)
        times = keys[:, 0]
        values = np.clip(keys[:, 1], lo, hi)
        if np.abs(values).max() <= 1e-6:
            continue                                  # never leaves rest
        loc_index, sign = axis
        rest = board.rest_location(control)[loc_index]
        frames = 1.0 + times * scene_fps
        co = np.column_stack([frames, rest + values * divisor * sign])
        # One slot per OWNER object: a bone board keeps every control in the
        # single armature slot, separated by its pose-bone data path.
        owner = board.owner_object(control)
        if owner is None:
            missing.append(gui_name)
            continue
        if owner.name not in slots:
            slot = action.slots.new('OBJECT', owner.name)
            slots[owner.name] = slot
            bags[owner.name] = _ensure_action_channelbag(
                anim_utils, action, slot)
            animated[owner.name] = owner
        fc = bags[owner.name].fcurves.new(
            data_path=board.location_data_path(control), index=loc_index)
        n = len(co)
        fc.keyframe_points.add(n)
        fc.keyframe_points.foreach_set("co", co.reshape(-1).tolist())
        fc.keyframe_points.foreach_set("interpolation", [1] * n)  # LINEAR
        fc.update()
        n_keys += n
        keyed.add(template)
        duration = max(duration, float(times[-1]))

    if not animated:
        bpy.data.actions.remove(action)
        raise AnimImportError(
            "The clip's controls did not match this character's board "
            "(missing: " + ", ".join(missing[:4]) + "...)" if missing else
            "The clip contains no motion (every curve stays at rest)")

    for name, obj in animated.items():
        if obj.animation_data is None:
            obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slots[name]

    action.use_fake_user = False
    skel[ANIM_ACTION_PROP] = action
    skel[ANIM_INFO_PROP] = (f"{clip_name}  ·  {len(keyed)} controls · "
                            f"{duration:.1f}s")

    scene.frame_start = 1
    scene.frame_end = max(scene.frame_start + 1,
                          int(math.ceil(1.0 + duration * scene_fps)))
    # Keep the clip on the board and hand exporters the character's own clip
    # to bake (see the module header): without this an FBX export replays a
    # board control's motion on the whole character.
    guard_board_actions(skel)
    _ensure_bake_action(skel, clip_name, scene.frame_start, scene.frame_end)
    scene.frame_set(scene.frame_start)     # evaluates the rig at frame 1
    return len(keyed), n_keys, duration, missing


def face_animation_action(skel):
    action = skel.get(ANIM_ACTION_PROP) if skel else None
    return action if isinstance(action, bpy.types.Action) else None


def remove_face_animation(skel):
    """Clear a previously imported clip and rest the board. Returns count."""
    action = face_animation_action(skel)
    cleared = 0
    _remove_bake_action(skel)
    if action is not None:
        controls = _board_controls(skel)
        for owner in _board_owners(skel).values():
            ad = owner.animation_data
            if ad is not None and ad.action == action:
                owner.animation_data_clear()
        for control in controls.values():
            cleared += int(board.rest_pose(control))
        bpy.data.actions.remove(action)
    if ANIM_ACTION_PROP in skel:
        del skel[ANIM_ACTION_PROP]
    if ANIM_INFO_PROP in skel:
        del skel[ANIM_INFO_PROP]
    return cleared


# ---------------------------------------------------------- board reading ---

def _object_slot_fcurves(obj):
    """Every fcurve for this object's own action slot (4.4+) or legacy action.

    Board controls imported from one clip share a single action but each has
    its own slot, so reading by slot keeps one control's curves from bleeding
    into another's.
    """
    ad = obj.animation_data
    action = ad.action if ad else None
    if action is None:
        return
    slot = getattr(ad, "action_slot", None)
    if slot is not None:
        try:
            from bpy_extras import anim_utils
            bag = anim_utils.action_get_channelbag_for_slot(action, slot)
        except (ImportError, AttributeError):
            bag = None
        if bag is not None:
            for fc in bag.fcurves:
                yield fc
            return
    for fc in getattr(action, "fcurves", ()):
        yield fc


def board_has_animation(skel):
    """True if any board control carries keyframes (imported or hand-keyed)."""
    for owner in _board_owners(skel).values():
        ad = owner.animation_data
        if ad is not None and ad.action is not None:
            return True
    return False


def _read_board_animation(skel, controls):
    """Board control fcurves -> {gui_name: (n,2) [frame, board value]}.

    The exact inverse of `_write_board_animation`'s per-key math: strips the
    control's rest offset, sign and the DNA gui-scale divisor so the numbers
    match what the importer expects to read straight back in.
    """
    divisor = board.gui_scale(skel)
    # (owner, data path, location index) -> (template, control, channel, sign)
    wanted = {}
    owners = {}
    for template, control in controls.items():
        owner = board.owner_object(control)
        if owner is None:
            continue
        owners[owner.name] = owner
        data_path = board.location_data_path(control)
        for channel in ("tx", "ty"):
            axis = board.channel_axis(control, channel)
            if axis is None:
                continue
            index, sign = axis
            wanted[(owner.name, data_path, index)] = (
                template, control, channel, index, sign)

    curves = {}
    for owner in owners.values():
        for fc in _object_slot_fcurves(owner):
            entry = wanted.get((owner.name, fc.data_path, int(fc.array_index)))
            if entry is None:
                continue
            template, control, channel, index, sign = entry
            keys = _fcurve_keys(fc)
            if len(keys) == 0:
                continue
            rest = board.rest_location(control)[index]
            values = (keys[:, 1] - rest) * sign / divisor
            curves[f"{template}.{channel}"] = \
                np.column_stack([keys[:, 0], values])
    return curves


def _addon_version():
    """Add-on version string from the manifest, best-effort ('unknown')."""
    try:
        from .. import paths
        manifest = os.path.join(paths.ADDON_DIR, "blender_manifest.toml")
        with open(manifest, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
    except (OSError, IndexError):
        pass
    return "unknown"


# -------------------------------------------------------------- operators ---

def _active_rig(context):
    mh = getattr(context.scene, "mhfrt", None)
    skel = mh.skeleton if mh else None
    if skel is None or not skel.get(RIG_ID_PROP):
        return None
    return skel


class MHFRT_OT_import_face_anim(bpy.types.Operator, ImportHelper):
    bl_idname = "mhfrt.import_face_anim"
    bl_label = "Import Face Animation"
    bl_description = ("Load a facial performance exported from Unreal Engine "
                      "and play it on this character's control board.\n"
                      "FBX: Level Sequence export of the MetaHuman face "
                      "board (Face_ControlBoard_CtrlRig track)\n"
                      "CSV: Live Link Face (ARKit) take\n"
                      "JSON: curve dump {name: [[frame, value], ...]}")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: bpy.props.StringProperty(
        default="*.fbx;*.csv;*.json;*.mhfa", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _active_rig(context) is not None

    def execute(self, context):
        from .op_live import stop_running
        stop_running()

        skel = _active_rig(context)
        controls = _board_controls(skel)
        if not controls:
            self.report({'ERROR'},
                        "This character has no control board - press "
                        "Build Rig first")
            return {'CANCELLED'}
        if not riglogic.available():
            self.report({'ERROR'}, "Rig logic data missing from the add-on")
            return {'CANCELLED'}

        path = self.filepath
        ext = os.path.splitext(path)[1].lower()
        clip = os.path.splitext(os.path.basename(path))[0]
        meta = riglogic.meta()
        try:
            if ext == ".fbx":
                gui_keys, raw_keys, src_fps, found = \
                    _load_fbx_curves(context, path)
                unknown = []
                gui_names = set(meta["gui_names"])
                gui_keys = {n: k for n, k in gui_keys.items()
                            if n in gui_names}
                # one shared clip start for every curve group in the file
                all_keys = list(gui_keys.values()) + list(raw_keys.values())
                f0 = min(float(a[:, 0].min()) for a in all_keys) \
                    if all_keys else 0.0
                gui_curves = {
                    name: np.column_stack([(arr[:, 0] - f0) / src_fps,
                                           arr[:, 1]])
                    for name, arr in gui_keys.items()}
                if raw_keys:
                    series, times = _keys_to_series(raw_keys, src_fps,
                                                    start=f0)
                    gui_curves.update(_series_to_gui(series, {}, times, meta))
                if not gui_curves:
                    hint = (f"Animated nodes found: "
                            f"{', '.join(sorted(set(found))[:5])}..."
                            if found else
                            "No animated face controls in this FBX.")
                    raise AnimImportError(
                        "No MetaHuman face-board curves in this file. "
                        "In Unreal, export the LEVEL SEQUENCE with the "
                        "Face_ControlBoard_CtrlRig track (not the "
                        "skeletal-mesh animation). " + hint)
            elif ext == ".csv":
                named, times, src_fps = _load_csv_curves(path)
                gui_direct, raw_s, arkit_s, unknown = \
                    _resolve_named_curves(named, meta)
                if not raw_s and not arkit_s and not gui_direct:
                    raise AnimImportError(
                        "No recognisable facial curves in this CSV "
                        "(expected Live Link Face blend shapes or "
                        "CTRL_ control columns)")
                gui_curves = _series_to_gui(raw_s, arkit_s, times, meta)
                for name, values in gui_direct.items():
                    gui_curves[name] = np.column_stack([times, values])
            elif ext in (".json", FACE_ANIM_EXT):
                keyed, src_fps = _load_json_curves(path)
                gui_direct, raw_k, arkit_k, unknown = \
                    _resolve_named_curves(keyed, meta)
                all_keys = (list(gui_direct.values()) + list(raw_k.values())
                            + list(arkit_k.values()))
                f0 = min(float(a[:, 0].min()) for a in all_keys) \
                    if all_keys else 0.0
                gui_curves = {}
                if raw_k or arkit_k:
                    series, times = _keys_to_series(
                        {**raw_k, **{f"@{k}": v for k, v in arkit_k.items()}},
                        src_fps, start=f0)
                    raw_series = {k: v for k, v in series.items()
                                  if not k.startswith("@")}
                    arkit_series = {k[1:]: v for k, v in series.items()
                                    if k.startswith("@")}
                    gui_curves.update(_series_to_gui(
                        raw_series, arkit_series, times, meta))
                for name, arr in gui_direct.items():
                    gui_curves[name] = np.column_stack(
                        [(arr[:, 0] - f0) / src_fps, arr[:, 1]])
                if not gui_curves:
                    raise AnimImportError(
                        "No recognisable facial curves in this JSON "
                        "(names must be board controls, CTRL_expressions "
                        "raws, or ARKit shapes)")
            else:
                raise AnimImportError(
                    "Unsupported file type - use .mhfa (this add-on's own "
                    "clip), .fbx (Unreal Sequence export), .csv (Live Link "
                    "Face) or .json")

            n_ctrl, n_keys, duration, missing = _write_board_animation(
                context, skel, controls, gui_curves, clip, src_fps)
        except AnimImportError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        parts = [f"{n_ctrl} controls", f"{duration:.1f}s @ {src_fps:g} fps"]
        if missing:
            parts.append(f"{len(missing)} unmatched")
        if unknown:
            parts.append(f"{len(unknown)} unknown curves skipped")
        self.report({'INFO'}, f"Face animation: {', '.join(parts)} - press "
                              "Space to play")
        flow._redraw(context)
        return {'FINISHED'}


class MHFRT_OT_test_face_anim(bpy.types.Operator):
    bl_idname = "mhfrt.test_face_anim"
    bl_label = "Test Face Animation"
    bl_description = ("Load the bundled Face Board Range-Of-Motion clip onto "
                      "this character's board - every control cycles through "
                      "its range, so you can verify the rig end-to-end and "
                      "mark the Animation step done without hunting for a "
                      "real take")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_rig(context) is not None

    def execute(self, context):
        from .. import paths
        path = os.path.join(paths.DATA_DIR, "MHC_FaceBoardROM.fbx")
        if not os.path.exists(path):
            self.report({'ERROR'},
                        f"Bundled test animation missing at {path}")
            return {'CANCELLED'}
        # Delegate to the real importer with EXEC_DEFAULT so its file-browser
        # invoke path is bypassed and the bundled file is used verbatim.
        return bpy.ops.mhfrt.import_face_anim('EXEC_DEFAULT', filepath=path)


class MHFRT_OT_remove_face_anim(bpy.types.Operator):
    bl_idname = "mhfrt.remove_face_anim"
    bl_label = "Remove Face Animation"
    bl_description = ("Delete the imported facial animation and return the "
                      "control board to rest")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        skel = _active_rig(context)
        return skel is not None and face_animation_action(skel) is not None

    def execute(self, context):
        skel = _active_rig(context)
        cleared = remove_face_animation(skel)
        from . import op_rig
        op_rig._rescan()
        op_rig._evaluate_guarded(op_rig._caches.values())
        self.report({'INFO'}, f"Animation removed ({cleared} controls reset)")
        flow._redraw(context)
        return {'FINISHED'}


class MHFRT_OT_export_face_anim(bpy.types.Operator, ExportHelper):
    bl_idname = "mhfrt.export_face_anim"
    bl_label = "Export Face Animation"
    bl_description = ("Save this character's control-board animation to a "
                      "native .mhfa clip file.\n"
                      "Captures every keyed board control - an imported clip "
                      "or your own hand-keyed polish - at the scene frame "
                      "rate. Re-imports here; readable JSON inside")
    bl_options = {'REGISTER'}

    filename_ext = FACE_ANIM_EXT
    filter_glob: bpy.props.StringProperty(
        default="*" + FACE_ANIM_EXT, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        skel = _active_rig(context)
        return skel is not None and board_has_animation(skel)

    def invoke(self, context, event):
        skel = _active_rig(context)
        action = face_animation_action(skel) if skel else None
        base = action.name if action is not None else "FaceAnimation"
        self.filepath = base + self.filename_ext
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        skel = _active_rig(context)
        if skel is None:
            self.report({'ERROR'}, "No active character")
            return {'CANCELLED'}
        curves = _read_board_animation(skel, _board_controls(skel))
        if not curves:
            self.report({'ERROR'},
                        "No animation on the board to export - import a clip "
                        "or key some controls first")
            return {'CANCELLED'}
        scene = context.scene
        fps = scene.render.fps / (scene.render.fps_base or 1.0)
        first = min(float(arr[:, 0].min()) for arr in curves.values())
        last = max(float(arr[:, 0].max()) for arr in curves.values())
        n_keys = sum(len(a) for a in curves.values())
        # Self-describing header identifies the file as this add-on's own
        # format; `fps` and `curves` are what the importer actually reads, so
        # the header can grow without ever breaking round-trip loading.
        data = {
            "format": "MetaHuman Facial Rig Transfer - Face Animation",
            "format_id": FACE_ANIM_FORMAT_ID,
            "format_version": FACE_ANIM_FORMAT_VERSION,
            "addon_version": _addon_version(),
            "rig": skel.name,
            "controls": len(curves),
            "keyframes": n_keys,
            "duration_seconds": round((last - first) / (fps or 1.0), 4),
            "fps": round(fps, 6),
            "curves": {
                name: [[round(float(f), 4), round(float(v), 6)]
                       for f, v in arr]
                for name, arr in curves.items()
            },
        }
        try:
            with open(self.filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1)
        except OSError as exc:
            self.report({'ERROR'}, f"Could not write file: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'},
                    f"Exported {len(curves)} controls · {n_keys} keys -> "
                    f"{os.path.basename(self.filepath)}")
        return {'FINISHED'}


_classes = (MHFRT_OT_import_face_anim, MHFRT_OT_test_face_anim,
            MHFRT_OT_remove_face_anim, MHFRT_OT_export_face_anim)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
