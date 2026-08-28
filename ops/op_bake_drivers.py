"""Bake the scripted rig into a standalone character driven by real drivers.

The live rig is Python: a depsgraph handler reads the control board, runs
RigLogic in numpy and writes 838 pose bones every tick.  Hand that file to a
client without the add-on and the face is frozen.  This module rewrites the
same character as something Blender evaluates on its own, and writes it to a
.blend of the artist's choosing.

What survives verbatim, as a native driver network
--------------------------------------------------
RigLogic's front half is exactly expressible as drivers, so it is not
approximated at all:

* every board control's location is turned back into its DNA GUI value with
  the same affine the runtime uses (``board.channel_value``), including the
  handle's Limit Location frame;
* GUI -> raw is the DNA's own piecewise-linear segment table.  The bundled DNA
  feeds each raw control from exactly ONE GUI channel (see
  ``riglogic.gui_segments``), which is what makes a per-raw expression exact;
* raw -> PSD is a product of 2..6 factors clamped to [0, 1], written as one
  expression per corrective.  Nested PSDs (48 references) get their own driven
  property so the graph stays acyclic and shallow.

Each raw and each referenced PSD becomes a driven ``rl_*`` custom property on
the delivered skeleton, so the client can read the rig's whole internal state
in the N-panel and key or drive it further.

What is converted, and why it has to be
---------------------------------------
RigLogic's back half - inputs -> joint deltas -> skinning - cannot be drivers.
The baked joint-group tables carry 957,244 non-zero coefficients over 5,016
output DOFs (a mean of 191 input terms per DOF, and thresholding barely dents
it: at 0.005 degrees it is still 181).  One driver variable per term is most of
a million RNA path resolves per update; Blender does not survive that, and no
sparsification exists to find because the tables are genuinely dense.

``Face Motion`` picks which of the two ways out is delivered.

**Bones (the default).**  Each RigLogic input is solved ALONE -
``inputs_to_joint_deltas`` is linear in the input vector, so a unit vector
isolates exactly that input's joint contribution - and the resulting pose is
written into one baked Action.  Every bone that input moves gets an Action
constraint reading a hidden per-input bone, which is where the driver network
parks that input's value; a constraint reads a BONE natively, so the ~163,000
(bone, input) terms cost no drivers at all, only the ~800 that feed the input
bones.  The mesh then carries nothing but the DNA's and the artist's own
CORRECTIVE shape keys, copied over verbatim and re-driven.  See the baked
joint actions section below for the mix-mode and NLA findings this rests on.

Bones are what the rig really does, so this route keeps rotations rigid and
moves anything skinned or bone-parented to the face.  What it approximates is
only the rotation: stacked constraints compose their rotations where RigLogic
adds the euler deltas first, a second-order error that grows with how far the
face is pushed.

Measured end to end against the live rig (Trevor2, 4,050-vertex head, 838
facial joints, worst vertex error):

    pose                       travel     bones      shape keys
    neutral                    0 mm       0.04 mm    0.0003 mm
    12 channels, moderate      3.8 mm     0.04 mm    0.07 mm
    25 channels, full range    9.5 mm     1.5 mm     3.6 mm
    108 channels, all at 1.0   48 mm      4.6 mm     52 mm

The last row is the point: the shape-key route is LINEAR in the inputs, so a
pose that fires a hundred channels at once sums a hundred vertex deltas and
comes apart - 52 mm of error on 48 mm of motion.  Bones degrade gently instead,
because each contribution stays a rotation.

The trade is evaluation cost.  On that character the bone rig runs ~77 ms per
update against ~6 ms for shape keys, since ~137,000 constraints are solved
every frame and shape keys skip whatever sits at zero.  ``Detail`` is the dial:
Controls Only drops the joint correctives to ~31,000 constraints.  File size
goes the other way and does not depend on the mesh: the bone rig is ~28 MB
whatever the head's resolution, while the shape route was 13 MB on a
4,050-vertex head and scales with every vertex.

**Shape keys.**  The older route, kept for pipelines that would rather have
geometry than a constraint stack.  Because nothing drives the facial bones on
this route, they are not delivered at all: all 838 are removed and every facial
vertex group is summed onto the head deform bone chosen during the Merge step,
so the client gets an ordinary character - head bone, head weights, blend
shapes on top - instead of a skeleton that looks riggable and is not.  Each input's mesh difference is stored as
one shape key driven by that input's expression, so playback is ``Basis +
sum(value_i * delta_i)`` - RigLogic's own decomposition, except that skinning
is not linear in joint rotation, so a rotated vertex travels a chord instead
of an arc.  The DNA's PSD correctives are baked as their own shapes, so control
COMBINATIONS are carried too; this is not a naive one-shape-per-slider rig.
Shape keys evaluate before modifiers, and the armature is identity at rest, so
the deltas are captured in pre-armature space: the delivered meshes keep their
vertex groups and Armature modifier and still follow body/neck animation.  The
facial bones stay at rest and nothing drives them, which is exactly why nothing
parented to them moves either.

The look-at comes along too
---------------------------
``CTRL_C_eyesAim`` is not a DNA channel - op_rig solves it in trigonometry
against the live eye joints - but it is rebuilt here out of constraints rather
than dropped, because it feeds the eye GUI channels and therefore moves the
eyelids and their correctives, not just the eyeball.  Three helper bones and
two Damped Track constraints per eye reproduce the runtime's yaw/pitch to 1e-4
degrees; drivers then invert the same measured 2x2 response the runtime
inverts, blend on CTRL_lookAtSwitch, and hand the result to the ordinary raw
network.  The centre-eye master (CTRL_C_eye) rides the same path.  See the eye
aim section below for the construction.

Known limits, reported to the artist after the bake:
* on the shape-key route eye look is a bone rotation of up to 42 degrees, and
  as a shape key the eyeball travels a chord (the same caveat op_arkit
  documents).  The bone route has no such problem - the eye simply turns.
  ``Bake Eye Look`` still drops those inputs for pipelines that re-rig the eyes
  by hand, which also drops the look-at, since it drives those same channels;
* per-channel Bone Intensity is a live viewport control rather than part of the
  DNA, and the delivered rig runs those channels at full strength either way.
  The bake names the dialled-down channels in its report.

The artist's scene is restored exactly, the way op_arkit does it: one
``_Restore`` log, everything inside try/finally, sources read and never
written.
"""

import os
import subprocess
import tempfile

import numpy as np
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras import anim_utils
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector

from ..core import board
from ..core import deliver
from ..props import RIG_DESTINATION_PROP, RIG_MERGED_PROP
from ..core import dna_apply
from ..core import organization
from ..core import riglogic
from ..ui import flow
from . import op_anim, op_arkit, op_export, op_live, op_merge, op_rig
from .op_arkit import _Restore, _prune
from .op_weights import ARM_MOD_NAME

# Custom-property namespace on the delivered rig.  Deliberately NOT "mhfrt": a
# delivered character must carry none of this add-on's tags, or a client who
# DOES have it installed gets a second live rig fighting the drivers.
PROP_PREFIX = "rl_"
BAKED_MARK_PROP = "baked_facial_rig"

# Property name prefixes stripped from every delivered datablock.
_ADDON_PREFIXES = ("mhfrt", "dna_", "mhfa")

# Worn by every copy so it can coexist with the artist's own objects while the
# bake runs; stripped again on the way into the delivered file.
BAKE_SUFFIX = "_Baked"

_RAW_PREFIX = "CTRL_expressions."

# ChannelDriver.expression is char[256]; leave room for the terminator and a
# little slack rather than writing right up to the edge.
_EXPRESSION_LIMIT = 240

# Blender ID names cap at 63 bytes; PSD names are built from their factors and
# run long, so they are truncated and then made unique.
_NAME_LIMIT = 56

_EPS = 1e-12


# --------------------------------------------------------------- expressions ---

def _num(value):
    """Compact literal for a driver expression."""
    text = f"{float(value):.9g}"
    return "0" if text in ("-0", "-0.0") else text


def _signed(value):
    return f"+{_num(value)}" if value >= 0.0 else f"-{_num(abs(value))}"


def _clamped(expression, low, high):
    if low is not None and high is not None:
        return f"min(max({expression},{_num(low)}),{_num(high)})"
    if low is not None:
        return f"max({expression},{_num(low)})"
    if high is not None:
        return f"min({expression},{_num(high)})"
    return expression


def _gui_expression(source, divisor, variable):
    """One board channel's DNA GUI value, as an expression over its location.

    The runtime reads ``(location[axis] - rest) * sign / divisor`` after
    clamping into the handle's Limit Location frame (``board.channel_value``),
    so the inverse-free form is that same affine written out.  Returns None for
    a channel this board has no control for.
    """
    control, _index, rest, sign, limits = source
    if control is None or sign == 0.0:
        return None
    expression = variable
    if limits is not None:
        expression = _clamped(expression, limits[0], limits[1])
    scale = sign / divisor
    offset = -rest * sign / divisor
    if abs(scale - 1.0) > _EPS:
        expression = f"{_num(scale)}*{expression}"
    if abs(offset) > _EPS:
        expression = f"{expression}{_signed(offset)}"
    return f"({expression})"


def _scaled_source(source, scale):
    """One channel description, restated in the merged armature's units.

    Joining the board into the skeleton reinterprets every bone-LOCAL length -
    ``location``, and the Limit Location frame around it - in the skeleton
    object's scale instead of the board's. The affine that turns a location
    into a DNA GUI value is written in those same units, so rest, frame and
    divisor all move by the same factor and the GUI value comes out unchanged.
    """
    if abs(scale - 1.0) <= _EPS:
        return source
    control, index, rest, sign, limits = source
    return (
        control,
        index,
        rest * scale,
        sign,
        None if limits is None else (limits[0] * scale, limits[1] * scale),
    )


def _relative_scale(skel_copy, board_copy):
    """The uniform scale the join would force on `board_copy`'s local channels.

    None when the relation is not a uniform scale - a sheared or non-uniformly
    scaled board cannot be corrected by one factor, and a board is never worth
    delivering wrong, so the caller leaves it as its own object instead.
    """
    relative = skel_copy.matrix_world.inverted_safe() @ board_copy.matrix_world
    scale = relative.to_scale()
    if min(abs(v) for v in scale) <= 1e-9:
        return None
    if max(abs(scale[i] - scale[0]) for i in (1, 2)) > 1e-5 * abs(scale[0]):
        return None                      # non-uniform: no single correction
    return float(scale[0])


def _compensate_board_channels(board_copy, scale):
    """Restate every bone-local length on the board in `scale`-times units.

    Blender rescales bone REST positions when armatures are joined but copies
    pose channels verbatim - measured: a board at scale 2 keeps location 0.05
    and limits +/-0.1 while its bone doubles, so the handle travels half as far
    and its frame shrinks to half. Multiplying the channels by the same factor
    is what puts the handle back on the pixel it was on.
    """
    if abs(scale - 1.0) <= _EPS:
        return
    for pose_bone in board_copy.pose.bones:
        pose_bone.location = pose_bone.location * scale
        for constraint in pose_bone.constraints:
            if constraint.type != 'LIMIT_LOCATION':
                continue
            for axis in "xyz":
                for edge in ("min", "max"):
                    name = f"{edge}_{axis}"
                    setattr(constraint, name,
                            getattr(constraint, name) * scale)


def _raw_expression(gui_expression, segments, channel_range):
    """One raw control from its GUI channel, as the DNA's segment table.

    ``gui_to_raw`` gives a segment ``slope * clamp(g, lo, hi) + cut`` while g
    lies inside it and exactly zero outside.  Clamping alone reproduces that
    only where the segment already evaluates to zero at the boundary it faces -
    true for the plain [-1, 0] / [0, 1] split that most channels use, and false
    for the three-phase sliders (``mouthLipsSticky*``, split at 0.33 / 0.66):
    the first phase rises to 1.0 at 0.33 and the clamp then HOLDS 1.0 for the
    rest of the channel, firing a corrective the rig never fires.  So a
    boundary is gated whenever the segment is non-zero there and the channel
    actually extends past it - measured, not assumed.

    Comparisons are widened by the same ``_SEG_EPS`` the evaluator snaps with,
    so a value sitting exactly on a phase joint cannot fall between two gates
    and leave the control momentarily undriven.
    """
    epsilon = riglogic._SEG_EPS
    channel_low, channel_high = channel_range
    parts = []
    for low, high, slope, cut in segments:
        body = _clamped(gui_expression, low, high)
        if abs(slope - 1.0) > _EPS:
            body = f"{_num(slope)}*{body}"
        if abs(cut) > _EPS:
            body = f"({body}{_signed(cut)})"
        gates = []
        if low > channel_low + epsilon and abs(slope * low + cut) > _EPS:
            gates.append(f"({gui_expression}>={_num(low - epsilon)})")
        if high < channel_high - epsilon and abs(slope * high + cut) > _EPS:
            gates.append(f"({gui_expression}<={_num(high + epsilon)})")
        if gates:
            body = f"{'*'.join(gates)}*({body})"
        parts.append(body)
    return "+".join(parts) if parts else "0"


def _psd_expression(factors, names):
    """A PSD corrective: the clamped product of its factors' properties."""
    constant = 1.0
    for _column, weight in factors:
        constant *= weight
    body = "*".join(names)
    if abs(constant - 1.0) > _EPS:
        body = f"{_num(constant)}*{body}"
    return f"min(1,max(0,{body}))"


# ------------------------------------------------------------- input table ---

def _raw_segment_table():
    """raw index -> (gui channel, [(lo, hi, slope, cut), ...]).

    The DNA feeds each raw from exactly one GUI channel, which is what lets a
    raw be inverted into a single-variable expression.  Twelve of the 263 raws
    have no segment at all - nothing can ever move them - so they are simply
    absent here.
    """
    table = {}
    for channel, segments in riglogic.gui_segments().items():
        for raw_out, low, high, slope, cut in segments:
            entry = table.setdefault(raw_out, (channel, []))
            entry[1].append((low, high, slope, cut))
    return table


def _reachable_inputs(entries, cache):
    """Inputs this board can actually drive.

    Baking an input the controls cannot reach would spend a shape key and a
    megabyte on geometry that can never fire: a raw with no GUI segment, a raw
    whose handle this board layout does not carry, or any corrective standing
    on one of those.  Feeding RigLogic directly during the bake is what makes
    them reachable at all - the live rig never sees them either.
    """
    segments = _raw_segment_table()
    sources = cache["sources"]
    reachable = set()
    for index, entry in enumerate(entries):
        if entry is None:
            continue
        if entry["kind"] == 'RAW':
            found = segments.get(index)
            if found is None:
                continue
            source = sources[found[0]] if found[0] < len(sources) else None
            if source is None or source[0] is None or source[3] == 0.0:
                continue
            reachable.add(index)
        elif all(column in reachable for column, _weight in entry["factors"]):
            reachable.add(index)      # index order guarantees factors resolved
    return reachable


def _unique(name, used):
    name = name[:_NAME_LIMIT]
    candidate, counter = name, 1
    while candidate in used:
        candidate = f"{name}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _input_table():
    """One entry per RigLogic input: how to name it and how to compute it.

    ``kind`` is 'RAW' or 'PSD'; a PSD's ``factors`` are (input index, weight)
    pairs in the combined input space.  Entries are built in index order, which
    the PSD table guarantees resolves nested factors first, so a PSD can be
    named after factors that already have names.
    """
    meta = riglogic.meta()
    raw_count = int(meta["raw_count"])
    total = raw_count + int(meta["psd_count"])
    factor_table = riglogic.psd_factors()

    entries = [None] * total
    used = set()
    for index in range(raw_count):
        name = meta["raw_names"][index]
        if name.startswith(_RAW_PREFIX):
            name = name[len(_RAW_PREFIX):]
        entries[index] = {
            "index": index,
            "kind": 'RAW',
            "name": _unique(name, used),
            "factors": (),
        }
    for index in range(raw_count, total):
        factors = factor_table.get(index)
        if not factors:
            continue          # a PSD slot the DNA never defines
        labels = []
        for column, _weight in factors:
            factor = entries[column] if column < total else None
            labels.append(factor["name"] if factor else f"in{column}")
        entries[index] = {
            "index": index,
            "kind": 'PSD',
            "name": _unique("_".join(labels), used),
            "factors": factors,
        }
    return entries


# ----------------------------------------------------------------- geometry ---

def _local_coords(obj, depsgraph):
    """Evaluated vertex positions of `obj` in its OWN local space.

    Local, not world (op_arkit's choice), because the delivered mesh keeps its
    object transform and its Armature modifier: a shape key is stored in the
    same pre-modifier space these coordinates are read from.  Returns None when
    a modifier changed the vertex count, so the caller can abort instead of
    guessing a correspondence.
    """
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.data
    count = len(mesh.vertices)
    if count != len(obj.data.vertices):
        return None
    flat = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(count, 3).astype(np.float64)


def _strip_addon_props(datablock):
    try:
        names = list(datablock.keys())
    except TypeError:
        return              # not an ID-property owner (a KeyBlock, say)
    for key in [k for k in names if str(k).startswith(_ADDON_PREFIXES)]:
        try:
            del datablock[key]
        except (KeyError, TypeError):
            pass


def _copy_worn(source):
    """A body/clothing mesh, delivered as it is.

    No shape keys and no bake: these are skinned to the character's bones and
    carry no facial rig, so they only need to arrive with their weights intact
    and their modifier pointed at the delivered skeleton.  Leaving them behind
    is what made the earlier bake a floating head.
    """
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = f"{source.name}{BAKE_SUFFIX}"
    copy.data.name = copy.name
    copy.hide_viewport = False
    copy.hide_select = False
    copy.hide_render = False
    _strip_addon_props(copy)
    _strip_addon_props(copy.data)
    return copy


def _copy_with_correctives(source):
    """A head mesh delivered with its CORRECTIVE shape keys still on it.

    The bone route bakes no geometry at all, so there is nothing to rebuild
    here: the DNA's blend shapes and the artist's sculpted correctives are
    already shape keys on this mesh, in the pre-modifier space a shape key
    lives in, and copying them verbatim is both exact and free.  Only their
    drivers change - from the add-on's runtime to the delivered network.

    The Weight-Cleanup pose keys are the one exception: they are a binding aid
    (see op_arkit), and left at a value they would bake an open mouth into the
    neutral, so they are parked at zero on the way out.
    """
    copy = _copy_worn(source)
    shape_keys = copy.data.shape_keys
    if shape_keys is None:
        return copy
    # Only the Key datablock carries ID properties; a KeyBlock does not.
    _strip_addon_props(shape_keys)
    for key in shape_keys.key_blocks:
        key.mute = False
    for name in (op_arkit.MOUTH_OPEN_KEY, op_arkit.CLOSE_EYES_KEY):
        key = shape_keys.key_blocks.get(name)
        if key is not None:
            key.value = 0.0
    return copy


def _make_mesh_copy(source, rest_coords, reenable):
    """A delivered twin of `source`: same weights, modifiers and materials,
    a clean Basis on the rig's rest pose, and none of the add-on's tags."""
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = f"{source.name}{BAKE_SUFFIX}"
    copy.data.name = copy.name
    copy.animation_data_clear()
    copy.shape_key_clear()
    copy.hide_viewport = False
    copy.hide_select = False
    copy.hide_render = False
    # The bake needed generative modifiers off to keep the vertex count; the
    # delivered mesh gets the artist's stack back exactly as authored.
    for modifier in copy.modifiers:
        if modifier.name in reenable:
            modifier.show_viewport = True
    _strip_addon_props(copy)
    _strip_addon_props(copy.data)

    copy.data.vertices.foreach_set(
        "co", np.ascontiguousarray(rest_coords, dtype=np.float32).ravel())
    copy.data.update()
    basis = copy.shape_key_add(name="Basis", from_mix=False)
    basis.interpolation = 'KEY_LINEAR'
    _strip_addon_props(copy.data.shape_keys)
    return copy


def _add_shape(copy, name, coords):
    key = copy.shape_key_add(name=name, from_mix=False)
    key.slider_min, key.slider_max, key.value = 0.0, 1.0, 0.0
    key.data.foreach_set(
        "co", np.ascontiguousarray(coords, dtype=np.float32).ravel())
    return key


# ------------------------------------------------------------------ eye aim ---
#
# CTRL_C_eyesAim is not a DNA channel: op_rig solves it in trigonometry against
# the live eye joints and writes the answer into the eye GUI channels, which is
# why the look-at moves the EYELIDS and their correctives too and not just the
# eyeball.  That rules out delivering it as a Damped Track on the eye - the lids
# would stay behind - so the solve itself is rebuilt out of constraints, and its
# result re-enters the ordinary driver network as a GUI channel value.
#
# Three helper bones per eye reproduce ``_eye_aim_gui`` exactly (measured
# against it to 1e-4 degrees, with the armature at a non-identity transform):
#
#   Ref     rest +Z along the handle's REST direction, parented to the eye's
#           parent, Damped Track -> the aim handle.  Its pose is therefore R,
#           the shortest rotation carrying rest_dir onto the live direction -
#           precisely ``rest.rotation_difference(local)``.  Because the
#           baseline is the handle's own rest, an untouched board still solves
#           to zero on any character, which is the invariant the runtime went
#           to some trouble to get (see op_rig's look-at notes).
#   Target  child of Ref, resting on the eye's forward ray.  A child's head
#           rides its parent's rotation about their shared head, so it lands at
#           eye + R @ (d * FORWARD): the runtime's `gaze`, as a point.
#   Gaze    rest = the EYE's rest, parented to the eye's parent, Damped Track
#           -> Target.  A TRACKED bone's own local delta is the rotation from
#           its rest +Z to the target, so reading it as a YXZ euler gives
#           ROT_Y = yaw and ROT_X = -pitch in the eye's frame - the exact pair
#           ``_eye_aim_channels`` inverts.  (The delta has to come off the
#           tracked bone itself; a bone merely riding a rotated parent has an
#           identity local transform and reads zero.)
#
# Everything downstream - the 2x2 quadrant inverse, the lookAtSwitch blend, the
# centre-eye master - is arithmetic, so it goes into the driver expressions.
#
# One measured divergence, worth about 1.5e-3 on a channel that travels [-1, 1]
# (0.8 mm of geometry at a hard glance, zero at rest): the runtime measures from
# the POSED eye head, which RigLogic has already translated a little by the time
# the solve runs - the solve therefore reads a position it partly caused.  The
# helper bones sit at the eye's REST head instead, so the delivered rig has no
# such feedback term.  The error grows with deflection and vanishes at rest,
# which is the signature of exactly that difference.

AIM_BONE_PREFIX = "MHFR_"
_FORWARD = Vector((0.0, 0.0, 1.0))
# The helper bones are pure measurement; length only affects how they draw.
_AIM_BONE_LENGTH = 0.02
_AIM_COLLECTION = "MHFR Look-At"


def _aim_plan(cache, control_owner):
    """What the native aim rig needs, or None when this rig has no chain."""
    aim = cache.get("eye_aim")
    if aim is None:
        return None
    board_copy = control_owner.get(aim["board"].as_pointer())
    switch_owner = board.owner_object(aim["switch"])
    switch_copy = (control_owner.get(switch_owner.as_pointer())
                   if switch_owner else None)
    if board_copy is None or switch_copy is None:
        return None
    eyes = []
    for eye in aim["eyes"]:
        # Probing costs four RigLogic evaluations per eye; the delivered rig
        # needs the same numbers the live one inverts, so ask for them here.
        _columns, quadrants = op_rig._eye_response(aim, eye)
        if len(quadrants) != 4:
            continue          # a singular column: better no aim than a wrong one
        eyes.append({
            "joint": eye["joint"],
            "handle": eye["handle"],
            "rest": Vector(eye["rest"]),
            "x": eye["x"],
            "y": eye["y"],
            "quadrants": quadrants,
        })
    if not eyes:
        return None
    return {
        "eyes": eyes,
        "board": board_copy,
        "switch": switch_copy,
        "switch_bone": aim["switch"].name,
    }


def _merge_boards(context, skel_copy, board_copies, control_owner):
    """Join the delivered board armatures into the delivered skeleton.

    One armature is what an animator wants to be handed: select the character,
    see the panel and the rig together, one object to key. The join is only
    safe once the bone-local channels have been restated in the skeleton's
    units (see ``_compensate_board_channels``), and only for armature boards -
    a legacy object board is a mesh and stays its own object.

    Returns {board owner pointer: scale} so the driver affine can be written in
    the units the merged handles now use. Board owners that were merged are
    repointed at the skeleton in ``control_owner``: their bones live there now,
    and every driver and constraint addresses them by bone name.
    """
    scales = {}
    if not board_copies:
        return scales
    mergeable = []
    for pointer, copy in list(control_owner.items()):
        if copy not in board_copies or copy.type != 'ARMATURE':
            continue
        scale = _relative_scale(skel_copy, copy)
        if scale is None:
            continue                     # left as its own object, still works
        mergeable.append((pointer, copy, scale))
    if not mergeable:
        return scales

    # The panel hangs off the head so it travels with the character; inside one
    # armature that same inheritance has to come from the bone hierarchy, the
    # way parent_facial_roots does it for the facial joints after a merge.
    anchor = None
    for _pointer, copy, _scale in mergeable:
        if copy.parent is skel_copy and copy.parent_type == 'BONE':
            anchor = copy.parent_bone
            break

    for _pointer, copy, scale in mergeable:
        _compensate_board_channels(copy, scale)

    op_arkit._mode_object(context)
    for obj in context.selected_objects:
        obj.select_set(False)
    before = {bone.name for bone in skel_copy.data.bones}
    for _pointer, copy, _scale in mergeable:
        copy.select_set(True)
    skel_copy.select_set(True)
    context.view_layer.objects.active = skel_copy
    result = bpy.ops.object.join()
    if 'FINISHED' not in result:
        return scales

    arrived = [name for name in (bone.name for bone in skel_copy.data.bones)
               if name not in before]
    if anchor and skel_copy.data.bones.get(anchor) is not None:
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            edit_bones = skel_copy.data.edit_bones
            parent = edit_bones.get(anchor)
            for name in arrived:
                bone = edit_bones.get(name)
                if bone is None or bone.parent is not None:
                    continue
                bone.use_connect = False   # never snap a handle onto the head
                bone.parent = parent
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')

    for pointer, _copy, scale in mergeable:
        control_owner[pointer] = skel_copy
        scales[pointer] = scale
    return scales


def _live(objects):
    """The objects Blender has not freed under us.

    ``bpy.ops.object.join`` REMOVES what it merges, so a bone board joined into
    the delivered skeleton leaves every list built before the merge holding
    dead pointers - and touching one raises ReferenceError rather than
    returning None.  A loose-object board never merges, which is why this only
    ever bites on a bone board.
    """
    alive = []
    for obj in objects:
        try:
            obj.type
        except ReferenceError:
            continue
        alive.append(obj)
    return alive


def _protect_board_transforms(cache, board_copies, control_owner):
    """Stop a client's Alt+S or Alt+R from wrecking the delivered panel.

    A GUI handle's SIZE is part of how the panel is drawn: frames and the
    handles inside them carry compensating scales (measured on a delivered
    board: 172 of 376 objects are off unit scale, one frame at 5x), so a Clear
    Scale over a selection that happens to include the panel does not tidy
    anything - it explodes it.  The artist's own file survives that only by
    habit, not by protection.

    Nothing in the rig ever reads scale or rotation off a control; every driver
    reads ``location`` and nothing else.  So both are locked on the way out and
    the panel can only be slid, which is all it is for.  Location is left alone
    deliberately - that IS the animation channel.

    The artist's own file is protected the same way now (``board.set_board_locked``,
    v4.12.0), so this is no longer the only place it happens - but it still has
    to happen here: the delivered board carries no add-on tags and nothing in the
    client's file would ever put the locks back.
    """
    locked = 0
    for copy in board_copies:
        try:
            copy.lock_scale = (True, True, True)
            copy.lock_rotation = (True, True, True)
            copy.lock_rotation_w = True
            locked += 1
        except (AttributeError, ReferenceError):
            pass
    for control in cache["controls"].values():
        if not board.is_pose_bone(control):
            continue
        owner = board.owner_object(control)
        delivered = (control_owner.get(owner.as_pointer())
                     if owner is not None else None)
        pose_bone = (delivered.pose.bones.get(control.name)
                     if delivered is not None and delivered.pose else None)
        if pose_bone is None:
            continue
        pose_bone.lock_scale = (True, True, True)
        pose_bone.lock_rotation = (True, True, True)
        pose_bone.lock_rotation_w = True
        locked += 1
    return locked


def _settle_follow_head(skel_copy):
    """Keep the delivered Follow Head switch honest. Returns what happened.

    The board's two roots ride the character's head through an Armature
    constraint aimed at a head bone, and by the time the file is written that
    bone may not be there any more: the shape-key route retires every
    ``FACIAL_`` bone, and on a standalone facial rig the head IS
    ``FACIAL_C_FacialRoot``.

    A constraint whose subtarget does not exist is not inert - Blender draws it
    red and the switch silently stops meaning anything.  So it is removed along
    with its driver, and the panel simply stays parented to the armature, which
    is what a rig with no head bone can offer.  The switch handle stays on the
    board either way; it is part of the authored panel.
    """
    if skel_copy is None or getattr(skel_copy, "pose", None) is None:
        return 0
    dropped = 0
    for root in set(board.FOLLOW_HEAD_SWITCHES.values()):
        pose_bone = skel_copy.pose.bones.get(root)
        if pose_bone is None:
            continue
        for constraint in list(pose_bone.constraints):
            if constraint.name != board.FOLLOW_HEAD_CONSTRAINT:
                continue
            aimed = board.follow_head_target(constraint)
            if aimed is not None:
                target, subtarget = aimed
                if (target.type == 'ARMATURE'
                        and subtarget in target.pose.bones):
                    continue
            pose_bone.constraints.remove(constraint)
            dropped += 1
        animation = skel_copy.animation_data
        if animation is None or dropped == 0:
            continue
        path = (f'pose.bones["{bpy.utils.escape_identifier(root)}"]'
                f'.constraints["{board.FOLLOW_HEAD_CONSTRAINT}"].influence')
        fcurve = animation.drivers.find(path)
        if fcurve is not None:
            animation.drivers.remove(fcurve)
    return dropped


def _aim_bone_collection(armature):
    collections = getattr(armature, "collections", None)
    if collections is None:
        return None
    existing = collections.get(_AIM_COLLECTION)
    if existing is None:
        existing = collections.new(_AIM_COLLECTION)
    existing.is_visible = False
    return existing


def _add_helper_bone(edit_bones, name, matrix, head, parent):
    bone = edit_bones.new(name)
    bone.head = head
    bone.tail = head + Vector((0.0, _AIM_BONE_LENGTH, 0.0))
    bone.parent = parent
    bone.use_connect = False
    bone.matrix = Matrix.Translation(head) @ matrix.to_3x3().to_4x4()
    bone.length = _AIM_BONE_LENGTH
    bone.use_deform = False
    return bone


def _build_aim_bones(context, skel_copy, plan):
    """Create the helper chain. Returns {joint name: gaze bone name}.

    Edit Mode is the only way to add a bone, and it is picky: the armature has
    to be the view layer's ACTIVE object, not merely active in an override -
    with only the override, ``mode_set`` quietly does nothing, ``edit_bones``
    comes back empty, and the look-at silently goes missing.  So the active
    object is set for real and put back afterwards, and the mode is verified
    rather than assumed.
    """
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = (previous_active.mode
                     if previous_active is not None else 'OBJECT')
    names = {}
    try:
        if previous_active is not None and previous_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            skel_copy.hide_set(False)
        except RuntimeError:
            return {}                  # not in this view layer: cannot edit it
        view_layer.objects.active = skel_copy
        bpy.ops.object.mode_set(mode='EDIT')
        if skel_copy.mode != 'EDIT':
            return {}
        try:
            edit_bones = skel_copy.data.edit_bones
            for eye in plan["eyes"]:
                joint = edit_bones.get(eye["joint"])
                if joint is None:
                    continue
                rest_matrix = joint.matrix.copy()
                head = rest_matrix.translation.copy()
                parent = joint.parent
                side = eye["joint"].replace("FACIAL_", "")
                swing = _FORWARD.rotation_difference(
                    eye["rest"]).to_matrix().to_4x4()

                ref = _add_helper_bone(
                    edit_bones, f"{AIM_BONE_PREFIX}{side}_AimRef",
                    rest_matrix @ swing, head, parent)
                _add_helper_bone(
                    edit_bones, f"{AIM_BONE_PREFIX}{side}_AimTarget",
                    rest_matrix,
                    head + rest_matrix.to_3x3() @ (_FORWARD * eye["distance"]),
                    ref)
                gaze = _add_helper_bone(
                    edit_bones, f"{AIM_BONE_PREFIX}{side}_AimGaze",
                    rest_matrix, head, parent)
                names[eye["joint"]] = gaze.name
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        if previous_active is not None:
            view_layer.objects.active = previous_active

    collection = _aim_bone_collection(skel_copy.data)
    for eye in plan["eyes"]:
        side = eye["joint"].replace("FACIAL_", "")
        ref_name = f"{AIM_BONE_PREFIX}{side}_AimRef"
        target_name = f"{AIM_BONE_PREFIX}{side}_AimTarget"
        gaze_name = names.get(eye["joint"])
        if gaze_name is None:
            continue
        for name in (ref_name, target_name, gaze_name):
            bone = skel_copy.data.bones.get(name)
            if bone is not None and collection is not None:
                collection.assign(bone)
        ref_bone = skel_copy.pose.bones.get(ref_name)
        gaze_bone = skel_copy.pose.bones.get(gaze_name)
        if ref_bone is None or gaze_bone is None:
            continue
        track = ref_bone.constraints.new('DAMPED_TRACK')
        track.name = "Aim Handle"
        track.target = plan["board"]
        track.subtarget = eye["handle"]
        track.track_axis = 'TRACK_Z'
        track = gaze_bone.constraints.new('DAMPED_TRACK')
        track.name = "Aim Gaze"
        track.target = skel_copy
        track.subtarget = target_name
        track.track_axis = 'TRACK_Z'
    return names


def _aim_distance(skel, cache, eye):
    """Rest distance from the eye joint to its handle, in armature space.

    Only the DIRECTION to the Target bone matters, so any positive number would
    solve identically; using the real distance just keeps the helper sitting
    where an artist would expect to find it.
    """
    bone = skel.data.bones.get(eye["joint"])
    aim = cache.get("eye_aim")
    handle = aim["board"].pose.bones.get(eye["handle"]) if aim else None
    if bone is None or handle is None:
        return 0.1
    try:
        offset = ((aim["board"].matrix_world
                   @ op_rig._handle_rest_head(handle))
                  - (skel.matrix_world @ bone.head_local))
        return max(float(offset.length), 1e-3)
    except (AttributeError, ValueError):
        return 0.1


def _aim_expression(quadrants, row):
    """One eye channel from (yaw, pitch), inverting the measured response.

    ``_eye_aim_channels`` picks the sign quadrant its own answer lands in; the
    response is diagonally dominant (tens of degrees on the diagonal against
    about one on the cross terms), so the quadrant is selected here from the
    signs of the ANGLES instead, which agrees everywhere except a thin wedge
    around each axis where the value being chosen is itself near zero.

    The driver reads ROT_X, and pitch is its negative, so the pitch column is
    negated here rather than spending another property on the sign flip.
    """
    parts = []
    for sign_yaw, yaw_gate in ((1.0, "(y>=0)"), (-1.0, "(y<0)")):
        for sign_pitch, pitch_gate in ((1.0, "(x<=0)"), (-1.0, "(x>0)")):
            inverse = quadrants[(sign_yaw, sign_pitch)]
            gain_yaw = float(inverse[row][0])
            gain_pitch = -float(inverse[row][1])
            parts.append(
                f"{yaw_gate}*{pitch_gate}*({_num(gain_yaw)}*y"
                f"{_signed(gain_pitch)}*x)")
    return f"min(1,max(-1,{'+'.join(parts)}))"


# ------------------------------------------------------------ armature copies ---

def _copy_armature(source, name_suffix):
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = f"{source.name}{name_suffix}"
    copy.data.name = copy.name
    copy.hide_viewport = False
    copy.hide_select = False
    _strip_addon_props(copy)
    _strip_addon_props(copy.data)
    for bone in copy.data.bones:
        _strip_addon_props(bone)
    for pose_bone in copy.pose.bones:
        _strip_addon_props(pose_bone)
    return copy


def _rest_facial_bones(skel_copy):
    """Park every bone the runtime was driving.

    The pose is part of the object, so the copy arrives carrying whatever
    RigLogic last wrote.  Nothing drives these bones in the delivered file -
    the motion is in the shape keys - so they belong on their rest pose.
    """
    for pose_bone in skel_copy.pose.bones:
        if not pose_bone.name.startswith(dna_apply.FACIAL_PREFIX):
            continue
        pose_bone.matrix_basis.identity()


def _repoint_modifiers(mesh_copies, skel, skel_copy):
    for copy in mesh_copies:
        for modifier in copy.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object == skel:
                modifier.object = skel_copy
                if modifier.name == ARM_MOD_NAME:
                    modifier.name = "Armature"


def _remap_relations(copies, remap):
    """Point copied drivers and constraints at the delivered twins.

    A character that arrived on someone else's rig brings that rig's own
    machinery - Rigify puts drivers and constraints all over its controls - and
    ``Object.copy()`` duplicates them still aimed at the ORIGINAL objects.
    Left alone the delivered rig reaches back into the artist's scene, which
    both drags those objects into the written file and breaks the moment they
    are not there.  Anything without a twin is left as it was; it travels as an
    ordinary dependency.
    """
    def retarget(obj):
        replacement = remap.get(obj.as_pointer()) if obj is not None else None
        return replacement

    for copy in copies:
        # Shape keys carry their own animation data, and a mesh delivered with
        # its correctives intact brings any legacy driver already on them; left
        # alone those keep pointing into the artist's scene.
        owners = [copy, copy.data]
        shape_keys = getattr(copy.data, "shape_keys", None)
        if shape_keys is not None:
            owners.append(shape_keys)
        for owner in owners:
            animation = getattr(owner, "animation_data", None)
            if animation is None:
                continue
            for fcurve in animation.drivers:
                for variable in fcurve.driver.variables:
                    for target in variable.targets:
                        replacement = retarget(target.id)
                        if replacement is not None:
                            target.id = replacement
        stacks = [copy.constraints]
        if copy.type == 'ARMATURE' and copy.pose is not None:
            stacks.extend(bone.constraints for bone in copy.pose.bones)
        for stack in stacks:
            for constraint in stack:
                replacement = retarget(getattr(constraint, "target", None))
                if replacement is not None:
                    constraint.target = replacement
                # An Armature constraint - the panel's own Follow Head, and
                # anything a Rigify-style rig brought with it - has no `.target`
                # at all: its targets live in a collection, one entry per bone.
                # Skipping those left the delivered board reaching back into the
                # artist's scene for the head it rides.
                for entry in getattr(constraint, "targets", ()) or ():
                    replacement = retarget(getattr(entry, "target", None))
                    if replacement is not None:
                        entry.target = replacement


def _reparent(copies, remap):
    """Keep parenting that stays inside the delivered set; drop the rest.

    A head parented to something we are not shipping would land at the origin
    with no parent, so the world matrix is re-asserted after the parent goes.
    """
    for copy in copies:
        parent = copy.parent
        if parent is None:
            continue
        replacement = remap.get(parent.as_pointer())
        if replacement is not None:
            # Assigning .parent resets parent_type to OBJECT, which silently
            # unhooks anything bone-parented - a hat, a prop, a weapon. Put the
            # bone relationship back before restoring the transform.
            parent_type = copy.parent_type
            parent_bone = copy.parent_bone
            matrix = copy.matrix_world.copy()
            copy.parent = replacement
            copy.parent_type = parent_type
            if parent_type == 'BONE':
                copy.parent_bone = parent_bone
            copy.matrix_world = matrix
        else:
            matrix = copy.matrix_world.copy()
            copy.parent = None
            copy.matrix_world = matrix


# ------------------------------------------------------------------ drivers ---

def _new_driver(owner, data_path, index=-1):
    fcurve = (owner.driver_add(data_path) if index < 0
              else owner.driver_add(data_path, index))
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    while driver.variables:
        driver.variables.remove(driver.variables[0])
    return driver


def _add_variable(driver, name, target_id, data_path):
    variable = driver.variables.new()
    variable.name = name
    variable.type = 'SINGLE_PROP'
    target = variable.targets[0]
    target.id_type = 'OBJECT'
    target.id = target_id
    target.data_path = data_path
    return variable


def _declare(owner, name, description, low=0.0, high=1.0):
    """A driven float property, hard-limited to the range its layer lives in.

    The limit is not decoration.  A raw control is [0, 1] everywhere - verified
    over 4000 randomised GUI poses - EXCEPT exactly on one of the DNA's phase
    boundaries (0.2, 0.33, 0.4, 0.6, 0.66, 0.8), where ``gui_to_raw``'s
    _SEG_EPS snapping makes both adjacent segments active at once and their sum
    spikes to 2.0 for the width of a float.  The reference runtime carries that
    spike; a delivered rig should not, because a client scrubbing through the
    boundary would see the corrective double for one frame.  An IDProperty
    min/max is enforced on write, drivers included, so the layer simply cannot
    leave its design range.
    """
    owner[name] = 0.0
    try:
        owner.id_properties_ui(name).update(
            min=low, max=high, soft_min=low, soft_max=high,
            description=description)
    except (AttributeError, TypeError):
        pass          # older RNA: an undecorated float property still drives
    return f'["{bpy.utils.escape_identifier(name)}"]'


class _Network:
    """Builds the delivered driver graph and remembers what it named things.

    Three layers, each a real Blender driver: an optional GUI-channel property
    (only where an inline affine would overrun the 256-byte expression field),
    one property per raw control, and one per PSD that another PSD references.
    Shape keys then read a single property, or the short product expression of
    a PSD that nothing else needs.
    """

    def __init__(self, skel_copy, cache, control_owner, entries,
                 control_scale=None):
        self.skel = skel_copy
        self.cache = cache
        self.control_owner = control_owner      # original control -> its copy
        self.control_scale = control_scale or {}
        self.entries = entries
        self.paths = {}                         # input index -> property path
        self.gui_paths = {}                     # gui channel -> property path
        self.raw_segments = _raw_segment_table()
        self.count = 0

    # -- board access -------------------------------------------------------

    def _control_target(self, channel):
        """(delivered owner object, data path) for one GUI channel's location."""
        source = self.cache["sources"][channel]
        control = source[0]
        if control is None or source[3] == 0.0:
            return None, None
        owner = board.owner_object(control)
        delivered = self.control_owner.get(owner.as_pointer()) if owner else None
        if delivered is None:
            return None, None
        return delivered, f"{board.location_data_path(control)}[{source[1]}]"

    def _channel_scale(self, channel):
        """The unit change the board merge forced on this channel, or 1.0."""
        source = self.cache["sources"][channel]
        control = source[0]
        if control is None:
            return 1.0
        owner = board.owner_object(control)
        if owner is None:
            return 1.0
        return self.control_scale.get(owner.as_pointer(), 1.0)

    def _affine(self, channel):
        """(source, divisor) for this channel in the delivered rig's units."""
        scale = self._channel_scale(channel)
        return (_scaled_source(self.cache["sources"][channel], scale),
                self.cache["divisor"] * scale)

    def _gui_property(self, channel):
        """Give one GUI channel its own driven property, and reuse it after."""
        path = self.gui_paths.get(channel)
        if path is not None:
            return path
        owner, data_path = self._control_target(channel)
        if owner is None:
            return None
        expression = _gui_expression(*self._affine(channel), "v")
        if expression is None:
            return None
        name = f"{PROP_PREFIX}gui{channel:03d}"
        # A GUI channel is signed - most of them run [-1, 1] - so this layer
        # gets the DNA's own range, not the [0, 1] the raw layer lives in.
        low, high = riglogic.gui_channel_range(channel)
        path = _declare(self.skel, name,
                        "DNA GUI channel value read off the control board",
                        low=min(low, 0.0), high=max(high, 0.0))
        driver = _new_driver(self.skel, path)
        _add_variable(driver, "v", owner, data_path)
        driver.expression = expression
        self.gui_paths[channel] = path
        self.count += 1
        return path

    # -- the three layers ---------------------------------------------------

    def _build_raw(self, entry):
        channel_segments = self.raw_segments.get(entry["index"])
        if channel_segments is None:
            return None                     # a raw the DNA never feeds
        channel, segments = channel_segments
        owner, data_path = self._control_target(channel)
        if owner is None:
            return None                     # this board has no such handle

        source = self.cache["sources"][channel]
        channel_range = riglogic.gui_channel_range(channel)
        existing = self.gui_paths.get(channel)
        if existing is not None:
            # An eye channel: its value is already assembled elsewhere (centre
            # -eye master, then the look-at blend), so read it, do not re-derive
            # it from the handle and lose both.
            path = _declare(self.skel, f"{PROP_PREFIX}{entry['name']}",
                            f"RigLogic raw control {entry['name']}")
            driver = _new_driver(self.skel, path)
            _add_variable(driver, "g", self.skel, existing)
            driver.expression = _raw_expression("g", segments, channel_range)
            self.count += 1
            return path

        inline = _gui_expression(*self._affine(channel), "v")
        expression = (_raw_expression(inline, segments, channel_range)
                      if inline else None)
        variables = [("v", owner, data_path)]
        if expression is None or len(expression) > _EXPRESSION_LIMIT:
            # Rare: a legacy centimetre board with a gated multi-segment
            # channel repeats the affine six times.  Hand the GUI value its own
            # property and the raw expression collapses to a few characters.
            gui_path = self._gui_property(channel)
            if gui_path is None:
                return None
            expression = _raw_expression("g", segments, channel_range)
            variables = [("g", self.skel, gui_path)]

        path = _declare(self.skel, f"{PROP_PREFIX}{entry['name']}",
                        f"RigLogic raw control {entry['name']}")
        driver = _new_driver(self.skel, path)
        for name, target_id, target_path in variables:
            _add_variable(driver, name, target_id, target_path)
        driver.expression = expression
        self.count += 1
        return path

    def _psd_terms(self, entry):
        """(expression, [(variable, path)]) for one PSD, or None."""
        names, variables = [], []
        for position, (column, _weight) in enumerate(entry["factors"]):
            path = self.paths.get(column)
            if path is None:
                return None                 # a factor we could not build
            name = f"v{position}"
            names.append(name)
            variables.append((name, path))
        return _psd_expression(entry["factors"], names), variables

    def _build_psd_property(self, entry):
        terms = self._psd_terms(entry)
        if terms is None:
            return None
        expression, variables = terms
        path = _declare(self.skel, f"{PROP_PREFIX}{entry['name']}",
                        f"RigLogic corrective {entry['name']}")
        driver = _new_driver(self.skel, path)
        for name, target_path in variables:
            _add_variable(driver, name, self.skel, target_path)
        driver.expression = expression
        self.count += 1
        return path

    # -- the eye channels ---------------------------------------------------

    def _centre_eye_source(self, channel):
        """The centre-eye master's source for the channel it offsets, if any.

        CTRL_C_eye is an additive master over both eyes' tx / ty
        (op_rig.CENTER_EYE_LINKS).  It is a plain GUI read like any other, so
        it just adds a second variable - but it has to be in the expression or
        the delivered board grows a handle that does nothing.
        """
        for source, indices in self.cache.get("center_eye_links", ()):
            if channel in indices:
                return source
        return None

    def build_eye_channels(self, plan, gaze_names):
        """Assemble the four eye GUI channels, look-at and all.

        ``_read_gui`` layers these in a fixed order - own handle, plus the
        centre-eye master clamped to [-1, 1], then blended toward the aim
        solve by however far CTRL_lookAtSwitch is up - and the order matters,
        so it is reproduced rather than rearranged.  Called before the raws, so
        ``_build_raw`` finds these paths already registered and reads them.
        """
        switch_path = None
        if plan is not None:
            switch_path = _declare(
                self.skel, f"{PROP_PREFIX}lookAt",
                "How far CTRL_lookAtSwitch is up: 0 ignores the aim handles, "
                "1 points the eyes exactly at them")
            driver = _new_driver(self.skel, switch_path)
            _add_variable(
                driver, "v", plan["switch"],
                f'pose.bones["{bpy.utils.escape_identifier(plan["switch_bone"])}"]'
                ".location[1]")
            # The switch is read as a bare location, so BOTH unit changes apply
            # to it exactly as they do to every other handle: the board's own
            # placement scale (``cache["divisor"]`` - the head-size factor baked
            # into the rest, the same number ``_affine`` divides every GUI
            # channel by) and, on top of it, whatever the merge did to this
            # board's object matrix. Only the merge one was divided out here,
            # so on a board placed onto a head of any other size the delivered
            # rig's look-at saturated part-way up the handle - the same size bug
            # ``board.switch_unit`` documents on the live side.
            switch_owner = board.owner_object(self.cache["eye_aim"]["switch"])
            switch_scale = (
                self.control_scale.get(switch_owner.as_pointer(), 1.0)
                if switch_owner is not None else 1.0)
            unit = self.cache["divisor"] * switch_scale
            raw = ("v" if abs(unit - 1.0) <= _EPS
                   else f"{_num(1.0 / unit)}*v")
            driver.expression = f"min(1,max(0,{raw}))"
            self.count += 1

        solved = {}
        for eye in (plan["eyes"] if plan else ()):
            gaze = gaze_names.get(eye["joint"])
            if gaze is None:
                continue
            side = eye["joint"].replace("FACIAL_", "")
            for row, (channel, axis) in enumerate(
                    ((eye["x"], "x"), (eye["y"], "y"))):
                path = _declare(
                    self.skel, f"{PROP_PREFIX}aim{side}{axis.upper()}",
                    f"Eye channel the aim handle asks for ({axis} axis)",
                    low=-1.0, high=1.0)
                driver = _new_driver(self.skel, path)
                for name, transform in (("y", 'ROT_Y'), ("x", 'ROT_X')):
                    variable = driver.variables.new()
                    variable.name = name
                    variable.type = 'TRANSFORMS'
                    target = variable.targets[0]
                    target.id = self.skel
                    target.bone_target = gaze
                    target.transform_type = transform
                    target.transform_space = 'LOCAL_SPACE'
                    target.rotation_mode = 'YXZ'
                driver.expression = _aim_expression(eye["quadrants"], row)
                self.count += 1
                solved[channel] = path

        # Every channel the centre-eye master touches needs assembling, even
        # the ones with no aim solve behind them.
        channels = set(solved)
        for _source, indices in self.cache.get("center_eye_links", ()):
            channels.update(indices)
        for channel in sorted(channels):
            self._build_eye_channel(channel, solved.get(channel), switch_path)

    def _build_eye_channel(self, channel, aim_path, switch_path):
        owner, data_path = self._control_target(channel)
        if owner is None:
            return
        source, divisor = self._affine(channel)
        own = _gui_expression(source, divisor, "o")
        if own is None:
            return
        variables = [("o", owner, data_path)]

        base = own
        centre = self._centre_eye_source(channel)
        centre_owner = (board.owner_object(centre[0])
                        if centre is not None and centre[0] is not None
                        else None)
        delivered = (self.control_owner.get(centre_owner.as_pointer())
                     if centre_owner is not None else None)
        if delivered is not None and centre[3] != 0.0:
            # The master can sit on a different board object, so it carries its
            # own unit change.
            centre_scale = self.control_scale.get(
                centre_owner.as_pointer(), 1.0)
            offset = _gui_expression(
                _scaled_source(centre, centre_scale),
                self.cache["divisor"] * centre_scale, "c")
            if offset is not None:
                base = f"min(1,max(-1,{own}+{offset}))"
                variables.append(
                    ("c", delivered,
                     f"{board.location_data_path(centre[0])}[{centre[1]}]"))

        expression = base
        if aim_path is not None and switch_path is not None:
            expression = f"{base}*(1-w)+a*w"

        low, high = riglogic.gui_channel_range(channel)
        path = _declare(self.skel, f"{PROP_PREFIX}gui{channel:03d}",
                        "DNA GUI channel value read off the control board",
                        low=min(low, 0.0), high=max(high, 0.0))
        driver = _new_driver(self.skel, path)
        for name, target_id, target_path in variables:
            _add_variable(driver, name, target_id, target_path)
        if aim_path is not None and switch_path is not None:
            _add_variable(driver, "a", self.skel, aim_path)
            _add_variable(driver, "w", self.skel, switch_path)
        driver.expression = expression
        self.gui_paths[channel] = path
        self.count += 1

    def build_inputs(self, wanted):
        """Create every property `wanted` needs, factors included.

        A PSD is only given a property of its own when another PSD reads it -
        otherwise its product goes straight into the shape key's own driver,
        which keeps the delivered graph one layer shallower.
        """
        needed = set()
        pending = list(wanted)
        while pending:
            index = pending.pop()
            if index in needed:
                continue
            entry = self.entries[index] if index < len(self.entries) else None
            if entry is None:
                continue
            needed.add(index)
            pending.extend(column for column, _weight in entry["factors"])

        referenced = {column
                      for index in needed
                      for column, _weight in self.entries[index]["factors"]
                      if self.entries[index]["kind"] == 'PSD'}

        for index in sorted(needed):        # raws first, then PSDs in order
            entry = self.entries[index]
            if entry["kind"] == 'RAW':
                path = self._build_raw(entry)
            elif index in referenced:
                path = self._build_psd_property(entry)
            else:
                continue                    # inlined into its shape key driver
            if path is not None:
                self.paths[index] = path
        return needed

    def drive_from_input(self, owner, data_path, array_index, index):
        """Drive one channel from RigLogic input `index`. True if wired.

        A PSD that no other PSD references never got a property of its own, so
        its product expression is inlined here instead, which keeps the
        delivered graph one layer shallower.
        """
        path = self.paths.get(index)
        if path is not None:
            expression, variables = "v", [("v", path)]
        else:
            entry = self.entries[index] if index < len(self.entries) else None
            if entry is None or entry["kind"] != 'PSD':
                return False
            terms = self._psd_terms(entry)
            if terms is None:
                return False
            expression, variables = terms
        driver = _new_driver(owner, data_path, array_index)
        for name, target_path in variables:
            _add_variable(driver, name, self.skel, target_path)
        driver.expression = expression
        self.count += 1
        return True

    def drive_shape_key(self, key, index):
        """Point one delivered shape key at its RigLogic input. True if wired."""
        return self.drive_from_input(
            key.id_data, key.path_from_id("value"), -1, index)

    def drive_input_bone(self, bone_name, index):
        """Put one input's value on the bone its Action constraints read."""
        path = (f'pose.bones["{bpy.utils.escape_identifier(bone_name)}"]'
                ".location")
        return self.drive_from_input(self.skel, path, 1, index)


# -------------------------------------------------- stripping the face bones ---
#
# On the shape-key route NOTHING drives the facial bones: the motion is in the
# geometry and the 838 joints sit at rest forever.  Delivering them anyway
# hands the client a skeleton that looks riggable and is not, and leaves the
# head's weight spread over bones that will never move - so the face silently
# stops following the neck the moment anything renormalises.
#
# So they go, and their weight goes with them: every facial vertex group is
# folded into the ONE head deform bone the artist picked during the Merge step
# (RIG_DEFORM_BONE_PROP), which is the bone that carries the head in the body
# rig.  The result is a normal character - head bone, head weights, blend
# shapes on top.

def _is_standalone(skel):
    """True when this facial rig is the character's ONLY armature.

    A standalone rig was deliberately detached from any body (see
    op_merge.confirm_standalone_rig), so there is no head deform bone anywhere
    to look for, and every question phrased as "which body bone..." has no answer
    rather than a missing one.
    """
    if skel is None or skel.get(RIG_MERGED_PROP):
        return False
    return str(skel.get(RIG_DESTINATION_PROP, "")) == 'STANDALONE'


def _head_deform_bone(skel, skel_copy):
    """The bone the face's weight belongs to once the facial bones are gone.

    The artist's Merge choice is the authority.  The fallback walks up out of
    the facial hierarchy instead of guessing a name, so a rig merged by an
    older version (or under different naming) still resolves.
    """
    for name in (op_merge.merged_deform_bone(skel),
                 op_merge.merged_parent_bone(skel)):
        if name and skel_copy.data.bones.get(name) is not None:
            return name
    for bone in skel_copy.data.bones:
        if not bone.name.startswith(dna_apply.FACIAL_PREFIX):
            continue
        parent = bone.parent
        while (parent is not None
               and parent.name.startswith(dna_apply.FACIAL_PREFIX)):
            parent = parent.parent
        if parent is not None and parent.use_deform:
            return parent.name
    return ""


def _fold_facial_weights(mesh_copies, skel_copy, deform_bone):
    """Move every facial vertex group's weight onto `deform_bone`.

    Summed, not averaged: a vertex the face owned 100% has to stay owned 100%,
    just by the head bone now.  ``'ADD'`` both creates the entry and accumulates
    into an existing one, so a mesh already weighted to the head keeps that
    share and gains the face's on top.
    """
    facial = {bone.name for bone in skel_copy.data.bones
              if bone.name.startswith(dna_apply.FACIAL_PREFIX)}
    folded = 0
    for copy in mesh_copies:
        groups = copy.vertex_groups
        names = [group.name for group in groups if group.name in facial]
        if not names:
            continue
        carrier = groups.get(deform_bone) or groups.new(name=deform_bone)
        indices = {groups[name].index for name in names}
        for vertex in copy.data.vertices:
            share = sum(element.weight for element in vertex.groups
                        if element.group in indices and element.weight > 0.0)
            if share > 0.0:
                carrier.add([vertex.index], share, 'ADD')
        for name in names:
            groups.remove(groups[name])
        folded += len(names)
    return folded


def _reparent_off_facial_bones(delivered, skel_copy, deform_bone):
    """Move anything bone-parented to a facial bone onto the deform bone.

    A prop or a hair mesh parented to a jaw bone would otherwise lose its
    parent with the bone and jump to the origin.
    """
    moved = 0
    for copy in delivered:
        if (copy.parent is not skel_copy or copy.parent_type != 'BONE'
                or not copy.parent_bone.startswith(dna_apply.FACIAL_PREFIX)):
            continue
        matrix = copy.matrix_world.copy()
        copy.parent_bone = deform_bone
        copy.matrix_world = matrix
        moved += 1
    return moved


def _remove_facial_bones(context, skel_copy):
    """Delete the facial bones, keeping every other bone where it was.

    Children that are NOT ours - the look-at helpers, an artist's own bone
    hanging off the jaw - are re-parented up to the first surviving ancestor
    first, so removing a bone never silently drags something else out of the
    hierarchy with it.  Bone collections left with nothing in them go too.
    """
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = (previous_active.mode
                     if previous_active is not None else 'OBJECT')
    removed = 0
    try:
        if previous_active is not None and previous_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            skel_copy.hide_set(False)
        except RuntimeError:
            return 0
        view_layer.objects.active = skel_copy
        bpy.ops.object.mode_set(mode='EDIT')
        if skel_copy.mode != 'EDIT':
            return 0
        try:
            edit_bones = skel_copy.data.edit_bones
            doomed = [bone for bone in edit_bones
                      if bone.name.startswith(dna_apply.FACIAL_PREFIX)]
            names = {bone.name for bone in doomed}
            for bone in edit_bones:
                if bone.name in names:
                    continue
                parent = bone.parent
                while parent is not None and parent.name in names:
                    parent = parent.parent
                if bone.parent is not None and bone.parent.name in names:
                    bone.use_connect = False
                    bone.parent = parent
            for bone in doomed:
                edit_bones.remove(bone)
                removed += 1
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        if previous_active is not None:
            view_layer.objects.active = previous_active

    collections = getattr(skel_copy.data, "collections", None)
    if collections is not None:
        for collection in list(getattr(skel_copy.data, "collections_all",
                                       collections)):
            try:
                if not collection.bones:
                    collections.remove(collection)
            except (RuntimeError, ReferenceError, TypeError):
                pass
    return removed


# ------------------------------------------------------- baked joint actions ---
#
# The joint half of RigLogic delivered as BONE motion instead of geometry.
#
# The obstacle is the same one the module header describes: a bone's pose is a
# sum over ~198 inputs (163,277 (bone, input) terms over the whole rig, and
# thresholding barely dents it because the tables are genuinely dense), so one
# driver per term is out.  What makes it tractable is that an Action constraint
# does not need a driver at all - it reads a BONE's transform natively.  So the
# driver network's answer is parked on one hidden bone per input, and every
# constraint standing on that input reads it for free: ~800 drivers rather than
# 160,000, with the per-term cost reduced to a constraint Blender evaluates in
# C.
#
# Each input gets ONE Action holding every bone it moves, with each bone's
# channels in a group named after that bone, because that group is what the
# constraint extracts for its own owner.  The Action spans two frames - rest at
# 0 and the input at full value at 1 - so the constraint's [0, 1] range maps
# onto it and linear interpolation reproduces `value * delta`, which is exactly
# how RigLogic scales one input's contribution.
#
# Two properties of ``AFTER_SPLIT`` make it the only usable mix mode here, both
# measured rather than assumed:
#
# * it keeps location, rotation and scale on separate accumulators, so stacked
#   contributions ADD their translations.  Under the ordinary ``AFTER`` mode
#   each translation is turned by the rotation of every constraint before it,
#   which is wrong: RigLogic's translation is a linear map of the input vector
#   (see dna_apply.joint_pose_arrays) and really is a plain sum.  Measured on a
#   three-bone rig, AFTER_SPLIT reproduces the intended pose to 1e-9 where
#   AFTER drifts by the full rotated offset;
# * a constraint whose input sits at 0 contributes the Action's frame 0, which
#   is the rest pose, so an inactive input is exactly identity.
#
# What is left approximate is the rotation: the constraints compose their
# rotations (R1.R2....) where RigLogic adds the euler deltas before building
# one matrix.  Over 24 randomised full-face poses that costs a median of 0.0
# degrees, a 99th percentile of 0.015, and at most 0.32 mm on a point 3 cm out
# - against the shape-key route's chord error on the same rotations, which is
# what this replaces.  The one large angular outlier is FACIAL_*_Pupil spinning
# about its own gaze axis, which moves no geometry at all (0.000 mm).
#
# NLA tracks would have been the tidier mechanism and cannot be used: a strip's
# influence is not drivable.  Enabling ``use_animated_influence`` creates a
# control F-Curve on the strip, and that curve is evaluated AFTER any driver on
# the same property and overwrites it - verified in both orders, with the
# driver reading a custom property, a bone, and a separate object.

INPUT_BONE_PREFIX = "RL_"
_INPUT_COLLECTION = "MHFR RigLogic Inputs"
_INPUT_BONE_LENGTH = 0.008
_INPUT_BONE_SPACING = 0.012
_INPUT_BONE_COLUMNS = 32

# Rest at frame 0, the input at full value at frame 1.
_ACTION_START = 0
_ACTION_END = 1

# A rotation is judged by the arc it sweeps at the bone's own length; a bone
# shorter than this is still credited with a little skin around it, so a real
# rotation on a stub joint is not pruned as if it moved nothing.
_MIN_LEVER = 0.005


def _decompose_bases(basis):
    """(euler XYZ, scale) for a stack of pose bases.

    ``dna_apply.pose_bases`` applies the joint's scale to the basis COLUMNS, so
    the column lengths are that scale and dividing them out leaves the rotation
    a bone's euler channels can carry.
    """
    scale = np.linalg.norm(basis, axis=1)
    flipped = np.linalg.det(basis) < 0.0
    scale[flipped, 0] = -scale[flipped, 0]
    safe = np.where(np.abs(scale) < _EPS, 1.0, scale)
    euler = riglogic.euler_xyz_from_matrices(basis / safe[:, None, :])
    return euler, scale


def _input_pose(constants, index, scale):
    """(euler, translation, scale) per target for one input held at 1.0."""
    basis, translation = dna_apply.joint_pose_arrays(
        constants, riglogic.joint_delta_rows(index), scale)
    euler, bone_scale = _decompose_bases(basis)
    return euler, translation, bone_scale


def _bone_levers(skel_copy, targets):
    """(bone name or None, rotation lever) per target, resolved once.

    Resolved once and not per input: the same 838 pose-bone lookups repeated
    for each of ~800 inputs is most of a million RNA queries, and none of the
    answers change.
    """
    names, levers = [], []
    for target in targets:
        pose_bone = skel_copy.pose.bones.get(target["name"])
        names.append(None if pose_bone is None else target["name"])
        levers.append(0.0 if pose_bone is None
                      else max(float(pose_bone.bone.length), _MIN_LEVER))
    return names, np.asarray(levers, dtype=float)


def _bake_input_action(skel_copy, name, bone_names, levers, pose, threshold):
    """One RigLogic input's whole joint pose as a two-frame Action.

    Which channels are worth keying is judged in metres travelled: a hundredth
    of a degree on a 5 mm eyelash bone is the micron it really is and goes,
    while the same angle on the jaw stays.  Dropping those costs nothing
    visible and buys back both file size and per-frame constraint work.

    Returns (action, slot, bone names) - or (None, None, ()) when this input
    moves nothing far enough to be worth an Action at all.
    """
    euler, translation, bone_scale = pose
    arm = levers[:, None]
    masks = (np.abs(translation) > threshold,
             np.abs(euler) * arm > threshold,
             np.abs(bone_scale - 1.0) * arm > threshold)
    rows = np.flatnonzero(masks[0].any(1) | masks[1].any(1) | masks[2].any(1))
    if not rows.size:
        return None, None, ()

    action = bpy.data.actions.new(name)
    slot = action.slots.new('OBJECT', skel_copy.name)
    bag = op_anim._ensure_action_channelbag(anim_utils, action, slot)
    channels = (("location", translation, 0.0),
                ("rotation_euler", euler, 0.0),
                ("scale", bone_scale, 1.0))
    used = []
    for row in rows.tolist():
        bone_name = bone_names[row]
        if bone_name is None:
            continue
        group = bag.groups.new(bone_name)
        path = f'pose.bones["{bpy.utils.escape_identifier(bone_name)}"]'
        for (channel, values, rest), mask in zip(channels, masks):
            for axis in np.flatnonzero(mask[row]).tolist():
                fcurve = bag.fcurves.new(
                    data_path=f"{path}.{channel}", index=axis)
                fcurve.keyframe_points.add(2)
                fcurve.keyframe_points.foreach_set(
                    "co", [float(_ACTION_START), rest,
                           float(_ACTION_END), float(values[row][axis])])
                fcurve.keyframe_points.foreach_set("interpolation", [1, 1])
                fcurve.update()
                fcurve.group = group
        used.append(bone_name)
    if not used:
        bpy.data.actions.remove(action)
        return None, None, ()
    return action, slot, tuple(used)


def _add_input_constraints(skel_copy, label, action, slot, bone_names,
                           input_bone):
    """Give every bone this input moves its Action constraint.

    Built once and then COPIED, which is not a micro-optimisation: every RNA
    setter on a constraint runs an update that walks every constraint on the
    armature, so setting ten properties on each of ~163,000 constraints pays
    that walk ten times over a list that is itself growing.  Measured, copying
    a prepared template is 4-5x faster at 12,000 constraints and pulls further
    ahead from there, for a bit-identical result.

    The copies keep the template's name because a bone never carries two
    constraints from the SAME input, so there is nothing for Blender to
    uniquify - the delivered rig reads "RL jawOpen" on every bone the jaw
    moves rather than a column of "Action.037".
    """
    bones = [skel_copy.pose.bones.get(name) for name in bone_names]
    bones = [bone for bone in bones if bone is not None]
    if not bones:
        return 0

    template = bones[0].constraints.new('ACTION')
    template.name = f"RL {label}"[:63]
    template.target = skel_copy
    template.subtarget = input_bone
    template.transform_channel = 'LOCATION_Y'
    template.target_space = 'LOCAL'
    template.owner_space = 'LOCAL'
    template.min = 0.0
    template.max = 1.0
    template.frame_start = _ACTION_START
    template.frame_end = _ACTION_END
    template.action = action
    template.action_slot = slot
    template.mix_mode = 'AFTER_SPLIT'
    for pose_bone in bones[1:]:
        pose_bone.constraints.copy(template)
    return len(bones)


def _input_bone_origin(skel_copy):
    """Where to park the hidden input bones: clear of the character."""
    bones = skel_copy.data.bones
    if not bones:
        return Vector((0.0, 0.0, 0.0))
    far = max(float(bone.head_local.x) for bone in bones)
    high = min(float(bone.head_local.z) for bone in bones)
    return Vector((far + 4.0 * _INPUT_BONE_SPACING, 0.0, high))


def _build_input_bones(context, skel_copy, wanted, entries):
    """One hidden bone per RigLogic input, to carry its value.

    An Action constraint reads a bone but cannot read a custom property, so the
    driver network's answer has to land somewhere a constraint can see it.
    These deform nothing and sit in a hidden collection in a tidy grid beside
    the character, so an artist who unhides them can still read the rig's whole
    internal state off them.

    Edit Mode is as picky here as it is for the look-at bones: the armature has
    to be the view layer's ACTIVE object or ``edit_bones`` comes back empty and
    the rig silently ships with no motion at all, so the mode is verified
    rather than assumed.
    """
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = (previous_active.mode
                     if previous_active is not None else 'OBJECT')
    names = {}
    origin = _input_bone_origin(skel_copy)
    try:
        if previous_active is not None and previous_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            skel_copy.hide_set(False)
        except RuntimeError:
            return {}                  # not in this view layer: cannot edit it
        view_layer.objects.active = skel_copy
        bpy.ops.object.mode_set(mode='EDIT')
        if skel_copy.mode != 'EDIT':
            return {}
        try:
            edit_bones = skel_copy.data.edit_bones
            for position, index in enumerate(wanted):
                column = position % _INPUT_BONE_COLUMNS
                depth = position // _INPUT_BONE_COLUMNS
                head = origin + Vector((column * _INPUT_BONE_SPACING, 0.0,
                                        depth * _INPUT_BONE_SPACING))
                bone = edit_bones.new(
                    f"{INPUT_BONE_PREFIX}{entries[index]['name']}"[:63])
                bone.head = head
                bone.tail = head + Vector((0.0, _INPUT_BONE_LENGTH, 0.0))
                bone.use_deform = False
                bone.use_connect = False
                names[index] = bone.name
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        if previous_active is not None:
            view_layer.objects.active = previous_active

    collections = getattr(skel_copy.data, "collections", None)
    if collections is not None:
        collection = (collections.get(_INPUT_COLLECTION)
                      or collections.new(_INPUT_COLLECTION))
        collection.is_visible = False
        for name in names.values():
            bone = skel_copy.data.bones.get(name)
            if bone is not None:
                collection.assign(bone)
    return names


def _lock_input_bones(skel_copy, names):
    """The input bones are readouts, not handles: nothing should drag them."""
    for name in names.values():
        pose_bone = skel_copy.pose.bones.get(name)
        if pose_bone is None:
            continue
        pose_bone.lock_location = (True, False, True)
        pose_bone.lock_rotation = (True, True, True)
        pose_bone.lock_rotation_w = True
        pose_bone.lock_scale = (True, True, True)
        bone = skel_copy.data.bones.get(name)
        if bone is not None:
            bone.hide_select = True


# ----------------------------------------------------------------- operator ---

class MHFRT_OT_bake_drivers(bpy.types.Operator, ExportHelper):
    """Bake this character into a standalone .blend driven by ordinary Blender \
drivers, so it animates from the same control board with the add-on uninstalled"""
    bl_idname = "mhfrt.bake_drivers"
    bl_label = "Bake to Drivers"
    bl_options = {'REGISTER'}

    filename_ext = ".blend"
    filter_glob: StringProperty(default="*.blend", options={'HIDDEN'})

    motion: EnumProperty(
        name="Face Motion",
        items=(
            ('BONES', "Bones + Correctives",
             "Deliver the face as real bone motion: one baked Action per "
             "RigLogic input, stacked onto the bones by Action constraints. "
             "Only the DNA and sculpted CORRECTIVES stay as shape keys. "
             "Far more faithful at strong expressions, moves anything skinned "
             "or parented to a facial bone, and its size does not grow with "
             "the mesh - but it is heavier to play back (lower Detail helps)"),
            ('SHAPES', "Shape Keys Only",
             "Deliver every RigLogic input as a shape key. The facial bones "
             "are removed entirely and their weights folded into the head "
             "Deform Bone from the Merge step. Much faster to play back, but "
             "a rotating joint travels a chord instead of an arc and strong "
             "combined poses drift badly"),
        ),
        default='BONES',
    )
    detail: EnumProperty(
        name="Detail",
        items=(
            ('FULL', "Full (with correctives)",
             "Every RigLogic input, the DNA's combination correctives "
             "included. Control combinations keep their shape - this is the "
             "faithful bake, and the heaviest one to build and to play back"),
            ('CONTROLS', "Controls Only",
             "Only the raw controls, no combination correctives on the "
             "joints. Roughly a third of the work, and much lighter to play "
             "back, but combined controls lose their corrective shape. "
             "Sculpted corrective SHAPE KEYS are delivered either way"),
        ),
        default='FULL',
    )
    bake_eye_look: BoolProperty(
        name="Bake Eye Look",
        default=True,
        description=("Bake the eye-look inputs. Eye look is a bone rotation of "
                     "up to 42 degrees, so as a shape key the eyeball travels "
                     "a chord instead of an arc. Turn off to leave the eyes "
                     "for hand-rigging"),
    )
    prune_threshold: FloatProperty(
        name="Prune Below",
        default=1e-5, min=0.0, max=1.0, precision=6, step=0.01,
        subtype='DISTANCE',
        description=("Vertices moving less than this are snapped back to rest, "
                     "and a shape that moves nothing at all is dropped "
                     "entirely. Keeps the delivered file far smaller"),
    )
    keep_in_scene: BoolProperty(
        name="Keep Copy In This File",
        default=False,
        description=("Also leave the baked character in the current file so "
                     "you can inspect it. Off by default: the .blend on disk "
                     "is the deliverable"),
    )

    _sources = ()
    _worn = ()
    _joints = (0, 0, 0, 0)      # inputs, actions, constraints, bones moved
    _strip = (0, 0)             # facial bones removed, weight groups folded
    _strip_reason = ""
    _strip_bone = ""
    _standalone_kept = False

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        if mh is None or mh.skeleton is None or not riglogic.available():
            return False
        return flow.bind_done(mh)

    def invoke(self, context, event):
        mh = context.scene.mhfrt
        self._sources = op_arkit._source_objects(mh)
        if not self.filepath:
            name = op_rig.character_label(mh.skeleton) or "Character"
            base = bpy.data.filepath
            folder = os.path.dirname(base) if base else ""
            self.filepath = os.path.join(folder, f"{name}_Standalone.blend")
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        layout = self.layout
        # Face Motion decides what the client is handed, so it is shown
        # expanded and unsplit: the pick has to be readable and changeable
        # here, at the moment the path is chosen.
        column = layout.column(align=True)
        column.label(text="Face Motion")
        column.prop(self, "motion", expand=True)
        layout.separator()
        layout.use_property_split = True
        layout.prop(self, "detail")
        layout.prop(self, "bake_eye_look")
        layout.prop(self, "prune_threshold")
        layout.prop(self, "keep_in_scene")

        box = layout.box()
        box.label(text=f"Baking {len(self._sources)} mesh(es)",
                  icon='OUTLINER_OB_MESH')
        column = box.column(align=True)
        column.enabled = False
        for obj in self._sources[:6]:
            column.label(text=obj.name)
        if len(self._sources) > 6:
            column.label(text=f"... and {len(self._sources) - 6} more")

    # -- lifecycle ----------------------------------------------------------

    def execute(self, context):
        mh = context.scene.mhfrt
        skel = mh.skeleton
        if skel is None:
            self.report({'ERROR'}, "No active character rig")
            return {'CANCELLED'}
        sources = op_arkit._source_objects(mh)
        if not sources:
            self.report({'ERROR'}, "No head mesh to bake")
            return {'CANCELLED'}
        if not self.filepath:
            self.report({'ERROR'}, "Choose where to write the .blend")
            return {'CANCELLED'}

        op_live.stop_running()
        op_arkit._mode_object(context)

        self._created = []          # every datablock we made, for rollback
        restore = _Restore()
        window_manager = context.window_manager
        window_manager.progress_begin(0, 100)
        try:
            result = self._run(context, skel, sources, restore)
        except Exception:
            self._discard()
            raise
        finally:
            restore.run()
            window_manager.progress_end()
            op_rig.invalidate()
        if result != {'FINISHED'} or not self.keep_in_scene:
            self._discard()
        return result

    def _discard(self):
        """Remove everything this bake created.

        Containers go first (collection, objects); the data they used to own is
        only removed once it really is unused, so a mesh the artist's file also
        points at is never taken with it.
        """
        created = getattr(self, "_created", ())
        containers = (bpy.types.Scene, bpy.types.Collection, bpy.types.Object)
        for pick_containers in (True, False):
            for collection, datablock in reversed(created):
                if isinstance(datablock, containers) != pick_containers:
                    continue
                try:
                    if pick_containers or datablock.users == 0:
                        collection.remove(datablock)
                except (ReferenceError, RuntimeError, TypeError):
                    pass
        self._created = []

    def _track(self, collection, datablock):
        self._created.append((collection, datablock))
        return datablock

    # -- the bake -----------------------------------------------------------

    def _run(self, context, skel, sources, restore):
        window_manager = context.window_manager

        visible = set(context.view_layer.objects)
        missing = [obj.name for obj in sources if obj not in visible]
        if missing:
            self.report(
                {'ERROR'},
                "Not in the current view layer (excluded collection?): "
                + ", ".join(missing[:4]))
            return {'CANCELLED'}

        # 1. The runtime handler must not fight the poses we are about to
        #    write, so it is silenced for the whole bake and released last.
        previously_updating = op_rig._updating
        restore.add(lambda v=previously_updating:
                    setattr(op_rig, "_updating", v))
        op_rig._updating = True

        # 2. Freeze everything that could move under us, exactly as the ARKit
        #    bake does: evaluated animation is flushed back to the datablock,
        #    so a keyed board would overwrite the poses we are about to write.
        op_arkit._freeze_animation(skel, restore)
        if skel.parent is not None:
            op_arkit._freeze_animation(skel.parent, restore)
        reenable = {}
        for obj in sources:
            op_arkit._freeze_animation(obj, restore)
            op_arkit._force_visible(obj, restore)
            op_arkit._zero_cleanup_keys(obj, restore)
            # Only the shape route reads evaluated geometry, and only it needs
            # the vertex count held still to do so.
            reenable[obj.as_pointer()] = set(
                op_arkit._disable_generative_modifiers(obj, restore)
                if self.motion == 'SHAPES' else ())

        try:
            cache = op_export._rig_cache(skel)
        except RuntimeError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        for owner in {board.owner_object(control)
                      for control in cache["controls"].values()}:
            op_arkit._freeze_animation(owner, restore)
        op_arkit._snapshot_controls(cache, restore)
        for key, _index in cache["shape_key_targets"]:
            restore.add(lambda k=key, v=float(key.value):
                        setattr(k, "value", v))

        # The WHOLE skeleton goes to rest, not just the facial joints. A shape
        # key is stored before the modifier stack, so the delta has to be
        # measured against the bind pose: read it against a posed body instead
        # and the delivered mesh would carry that pose in its Basis and take
        # the armature's version of it again on top.
        for pose_bone in skel.pose.bones:
            restore.add(lambda b=pose_bone, m=pose_bone.matrix_basis.copy():
                        setattr(b, "matrix_basis", m))
            pose_bone.matrix_basis.identity()

        entries = _input_table()
        wanted = self._wanted_inputs(entries, cache)
        if not wanted:
            self.report({'ERROR'}, "Nothing to bake from this DNA")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        op_arkit._controls_to_rest(cache)

        shapes = {}                 # input index -> [(copy, key), ...]
        empty = []
        if self.motion == 'BONES':
            # The bone route reads no geometry at all: the joint deltas are
            # numbers, and the correctives are already shape keys on the source
            # mesh.  So the meshes are simply copied, correctives and all, and
            # the whole per-input evaluation loop below is skipped.
            mesh_copies = [
                self._track(bpy.data.objects, _copy_with_correctives(obj))
                for obj in sources
            ]
            for copy in mesh_copies:
                self._track(bpy.data.meshes, copy.data)
        else:
            # 3. Rest pose -> the Basis of every delivered mesh.  The armature
            #    is identity at rest, so these local coordinates double as the
            #    pre-modifier space the shape keys are stored in.
            self._pose(cache, None)
            depsgraph.update()

            rest = {}
            for obj in sources:
                coords = _local_coords(obj, depsgraph)
                if coords is None:
                    self.report(
                        {'ERROR'},
                        f"{obj.name}: a modifier changes the vertex count. "
                        "Apply it (or disable it) and bake again.")
                    return {'CANCELLED'}
                rest[obj.as_pointer()] = coords

            mesh_copies = [
                self._track(bpy.data.objects,
                            _make_mesh_copy(obj, rest[obj.as_pointer()],
                                            reenable[obj.as_pointer()]))
                for obj in sources
            ]
            for copy in mesh_copies:
                self._track(bpy.data.meshes, copy.data)

            # 4. One pose per RigLogic input.  inputs_to_joint_deltas is linear
            #    in the input vector, so a unit vector isolates exactly this
            #    input's joint contribution - no subtraction of neighbours.
            total = len(wanted)
            for step, index in enumerate(wanted):
                self._pose(cache, index)
                depsgraph.update()
                created = []
                for obj, copy in zip(sources, mesh_copies):
                    pointer = obj.as_pointer()
                    coords = _local_coords(obj, depsgraph)
                    if coords is None:
                        self.report(
                            {'ERROR'},
                            f"{obj.name}: vertex count changed mid-bake")
                        return {'CANCELLED'}
                    coords, peak = _prune(rest[pointer], coords,
                                          self.prune_threshold)
                    if peak < self.prune_threshold:
                        continue    # this input does not touch this mesh
                    created.append(
                        (copy, _add_shape(copy, entries[index]["name"],
                                          coords)))
                if created:
                    shapes[index] = created
                else:
                    empty.append(entries[index]["name"])
                if step % 8 == 0:
                    window_manager.progress_update(int(80 * step / total))

        # 5. The artist's scene comes back before anything else happens, so the
        #    armature copies below carry the real pose and the real animation.
        restore.run()
        op_rig.invalidate()
        window_manager.progress_update(85)

        # 6. Delivered rig: skeleton, board and the driver network.
        skel_copy = self._track(bpy.data.objects,
                                _copy_armature(skel, BAKE_SUFFIX))
        self._track(bpy.data.armatures, skel_copy.data)
        _rest_facial_bones(skel_copy)
        skel_copy[BAKED_MARK_PROP] = (
            "Driven by Blender drivers - no add-on required")
                # --------------------------------------------------------
        # PRESERVE EXTERNAL ARMATURE DEPENDENCIES
        #
        # The MHFRT/DAZ deform rig can have constraints targeting
        # another armature. In the Diffeomorphic Keep Daz Rig setup,
        # that external armature is the MHX controller.
        #
        # We keep the ORIGINAL MHX object in the delivery collection.
        # We do NOT copy it, reweight anything, or change mesh
        # modifiers.
        # --------------------------------------------------------

        external_armatures = {}

        for pose_bone in skel.pose.bones:
            for constraint in pose_bone.constraints:

                target = getattr(
                    constraint,
                    "target",
                    None
                )

                if target is None:
                    continue

                if target.type != 'ARMATURE':
                    continue

                if target == skel:
                    continue

                external_armatures[
                    target.as_pointer()
                ] = target

        print()
        print("=" * 78)
        print("MHFRT DELIVERY EXTERNAL ARMATURES")
        print("=" * 78)

        for armature in external_armatures.values():
            print(
                "  PRESERVING:",
                armature.name
            )

        print(
            "  TOTAL:",
            len(external_armatures)
        )
        print("=" * 78)

        # Body, clothing, hair - everything the character arrived rigged with.
        worn_sources = [
            obj for obj in organization.character_meshes(
                [skel], cage=skel.get("mhfrt_cage"),
                target=skel.get("mhfrt_target"), scene=context.scene)
            if obj not in sources
        ]
        self._worn = worn_sources
        self._body_rigs = op_export.body_rigs(context, skel)
        worn_copies = []
        for obj in worn_sources:
            copy = self._track(bpy.data.objects, _copy_worn(obj))
            self._track(bpy.data.meshes, copy.data)
            worn_copies.append(copy)

        control_owner = {}          # original board owner -> delivered copy
        board_copies = []
        for owner in cache.get("board_owners", ()):
            copy = self._track(bpy.data.objects, _copy_armature(owner, BAKE_SUFFIX)
                               if owner.type == 'ARMATURE'
                               else self._plain_copy(owner))
            if owner.type == 'ARMATURE':
                self._track(bpy.data.armatures, copy.data)
            control_owner[owner.as_pointer()] = copy
            board_copies.append(copy)

        remap = {skel.as_pointer(): skel_copy}
        for original, copy in zip(cache.get("board_owners", ()), board_copies):
            remap[original.as_pointer()] = copy
        for obj, copy in zip(sources + worn_sources, mesh_copies + worn_copies):
            remap[obj.as_pointer()] = copy
        _repoint_modifiers(mesh_copies + worn_copies, skel, skel_copy)
        delivered = mesh_copies + worn_copies + board_copies + [skel_copy]
        _reparent(delivered, remap)
        # Before the look-at constraints go on, so ours are not retargeted.
        _remap_relations(delivered, remap)

        # 7. Collection and scene now rather than at the end: creating the
        #    look-at helper bones needs Edit Mode, and Edit Mode needs the
        #    armature to be in the active view layer.
        label = op_rig.character_label(skel) or "Character"
        collection = self._track(
            bpy.data.collections,
            bpy.data.collections.new(f"{label}_Standalone")
        )

        for copy in (
            mesh_copies
            + worn_copies
            + board_copies
            + [skel_copy]
        ):
            collection.objects.link(copy)

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # These are ORIGINAL external armature objects.
        #
        # Do NOT put them into `delivered`, because MHFRT's
        # `_reparent()` / `_remap_relations()` routines are intended
        # for objects created by the bake.
        #
        # We simply make the external controller(s) members of the
        # collection that deliver.write_blend() writes.
        # --------------------------------------------------------

        for external in external_armatures.values():

            if external.name not in collection.objects:

                collection.objects.link(
                    external
                )

        print()
        print(
            "MHFRT DELIVERY COLLECTION ARMATURES:"
        )

        for obj in collection.objects:

            if obj.type == 'ARMATURE':

                print(
                    "  ",
                    obj.name
                )
        context.scene.collection.children.link(collection)
        linked_here = True

        # 7b. One armature to hand over: the board joins the skeleton now, before
        #     anything reads a control, so the aim constraints and every driver
        #     below are built against the bones where they finally live.
        control_scale = _merge_boards(context, skel_copy, board_copies,
                                      control_owner)
        # The merge consumed the board armatures it joined, so anything built
        # from this list afterwards has to be re-taken from the survivors.
        board_copies = _live(board_copies)
        _protect_board_transforms(cache, board_copies, control_owner)

        plan = None
        gaze_names = {}
        if self.bake_eye_look:
            plan = _aim_plan(cache, control_owner)
            if plan is not None:
                for eye in plan["eyes"]:
                    eye["distance"] = _aim_distance(skel, cache, eye)
                gaze_names = _build_aim_bones(context, skel_copy, plan)
                if not gaze_names:
                    plan = None

        network = _Network(skel_copy, cache, control_owner, entries,
                           control_scale=control_scale)
        network.build_eye_channels(plan, gaze_names)

        wired, unwired = 0, 0
        if self.motion == 'BONES':
            joints, correctives = self._bake_bone_motion(
                context, skel_copy, mesh_copies, cache, entries, wanted,
                network, window_manager)
            self._joints = joints
            wired, unwired = correctives
        else:
            network.build_inputs(shapes.keys())
            for index, created in shapes.items():
                for _copy, key in created:
                    if network.drive_shape_key(key, index):
                        wired += 1
                    else:
                        unwired += 1
            self._strip = self._strip_face_bones(
                context, skel, skel_copy,
                _live(mesh_copies + worn_copies + board_copies + [skel_copy]))
        # After every path that can remove a bone: the panel's Follow Head rides
        # one, and a delivered file must not ship a constraint aimed at nothing.
        _settle_follow_head(skel_copy)
        window_manager.progress_update(92)

        # 8. Write the deliverable.  The collection is handed over as it
        #    stands: no scene is built around it here, because the delivered
        #    file's scene is made by the second pass and a scene made only to
        #    be written is what crashes Blender 5.2 (see core/deliver.py).
        filepath = bpy.path.ensure_ext(self.filepath, ".blend")
        try:
            fallback, displaced = deliver.write_blend(
                filepath, collection, label, context.scene.frame_start,
                context.scene.frame_end, strip_suffix=BAKE_SUFFIX)
        except (RuntimeError, OSError) as error:
            self.report({'ERROR'}, f"Could not write {filepath}: {error}")
            return {'CANCELLED'}
        finally:
            if linked_here and not self.keep_in_scene:
                context.scene.collection.children.unlink(collection)
        if fallback:
            self.report(
                {'WARNING'},
                "Wrote a library file instead of an openable one "
                f"({fallback}). Everything is in it - link or append the "
                f"'{collection.name}' collection rather than opening it.")
        if displaced:
            # Something the character depends on already held a delivered name
            # - usually a still-separate body rig the clothing is skinned to.
            # It kept its data and is in the file under '<name>_original'.
            self.report(
                {'WARNING'},
                f"{len(displaced)} name(s) were already taken in the "
                f"delivered file and moved to '_original': "
                f"{', '.join(displaced[:4])}"
                + (" ..." if len(displaced) > 4 else ""))

        self._report(filepath, shapes, wired, unwired, empty, network, skel,
                     cache, plan)
        window_manager.progress_update(100)
        return {'FINISHED'}

    def _bake_bone_motion(self, context, skel_copy, mesh_copies, cache,
                          entries, wanted, network, window_manager):
        """Deliver the joint half as bones and the correctives as shape keys.

        Returns ((inputs, actions, constraints, bones), (wired, unwired)).

        The pose constants come from the SOURCE cache on purpose: they are
        derived from bone rests, which the copy shares exactly, and
        ``_copy_armature`` has already stripped the add-on properties that
        rebuilding them on the copy would need.
        """
        targets = cache["targets"]
        constants = cache["pose_constants"]
        # Reachability is the same test the joint bake uses: a corrective whose
        # control this board does not carry can never fire, so it earns neither
        # an input bone nor a driver - it simply ships parked at zero.
        reachable = _reachable_inputs(entries, cache)
        corrective_inputs = {int(index)
                             for _key, index in cache["shape_key_targets"]
                             if int(index) in reachable}
        if not targets or constants is None:
            self.report({'WARNING'},
                        "This rig drives no facial bones - nothing to bake")
            return (0, 0, 0, 0), (0, 0)

        for target in targets:
            pose_bone = skel_copy.pose.bones.get(target["name"])
            # The euler channels an Action carries are only meaningful in the
            # mode the deltas were decomposed in.
            if pose_bone is not None:
                pose_bone.rotation_mode = 'XYZ'

        # A corrective the artist sculpted is worth delivering whatever the
        # joint detail setting says: it is an existing shape key, so it costs
        # this bake nothing to carry, and dropping it would silently lose work.
        needed = list(dict.fromkeys(list(wanted) + sorted(corrective_inputs)))
        bone_names = _build_input_bones(context, skel_copy, needed, entries)
        if not bone_names:
            self.report(
                {'ERROR'},
                "Could not create the RigLogic input bones (Edit Mode was "
                "refused on the delivered armature)")
            return (0, 0, 0, 0), (0, 0)
        _lock_input_bones(skel_copy, bone_names)
        network.build_inputs(needed)

        target_names, levers = _bone_levers(skel_copy, targets)
        threshold = float(self.prune_threshold)
        actions = constraints = 0
        moved = set()
        total = max(1, len(needed))
        for step, index in enumerate(needed):
            input_bone = bone_names.get(index)
            if input_bone is None:
                continue
            if not network.drive_input_bone(input_bone, index):
                continue            # no control on this board reaches it
            pose = _input_pose(constants, index, cache["scale"])
            action, slot, used = _bake_input_action(
                skel_copy, f"RL_{entries[index]['name']}", target_names,
                levers, pose, threshold)
            if action is None:
                continue            # this input moves no bone far enough
            self._track(bpy.data.actions, action)
            actions += 1
            moved.update(used)
            constraints += _add_input_constraints(
                skel_copy, entries[index]["name"], action, slot, used,
                input_bone)
            if step % 8 == 0:
                window_manager.progress_update(int(85 + 6 * step / total))

        wired, unwired = self._drive_correctives(
            mesh_copies, cache, network, needed)
        return (len(needed), actions, constraints, len(moved)), (wired, unwired)

    def _strip_face_bones(self, context, skel, skel_copy, delivered):
        """Retire the facial bones on the shape-key route. (bones, groups).

        Refused rather than half-done when there is no head deform bone to fold
        the weight into: dropping the bones then would leave the face weighted
        to nothing at all, which looks fine at rest and falls off the neck the
        moment the body moves.

        On a STANDALONE rig there is nothing to refuse.  The facial skeleton is
        the character's only armature, so the facial bones are not a layer over
        a body's head bone - they ARE the deform rig, and folding them into a
        head bone that does not exist is not a thing that could be wanted.
        Keeping them is the correct outcome, and reporting a missing Merge-step
        setting to an artist who was never offered the Merge step is an error
        message about nothing.
        """
        if _is_standalone(skel):
            self._strip_reason = ""
            self._standalone_kept = True
            return (0, 0)
        deform_bone = _head_deform_bone(skel, skel_copy)
        if not deform_bone:
            self._strip_reason = (
                "no head deform bone is set - the facial bones were kept. "
                "Set Deform Bone in the Merge step to retire them")
            return (0, 0)
        meshes = [copy for copy in delivered if copy.type == 'MESH']
        groups = _fold_facial_weights(meshes, skel_copy, deform_bone)
        _reparent_off_facial_bones(delivered, skel_copy, deform_bone)
        bones = _remove_facial_bones(context, skel_copy)
        self._strip_reason = ""
        self._strip_bone = deform_bone
        return (bones, groups)

    def _drive_correctives(self, mesh_copies, cache, network, needed):
        """Hand the delivered correctives over to the driver network.

        The keys are the artist's own, copied verbatim, so they are matched by
        NAME on the delivered mesh rather than rebuilt.
        """
        allowed = set(needed)
        by_name = {}
        for key, index in cache["shape_key_targets"]:
            by_name.setdefault(key.name, int(index))
        wired = unwired = 0
        for copy in mesh_copies:
            shape_keys = copy.data.shape_keys
            if shape_keys is None:
                continue
            for key in shape_keys.key_blocks[1:]:
                index = by_name.get(key.name)
                if index is None:
                    continue        # the artist's own key: leave it as authored
                # Every runtime-driven key is parked, wired or not. The live
                # runtime is what used to hold these at a value, and a key it
                # no longer owns would otherwise ship frozen mid-expression.
                key.value = 0.0
                if index not in allowed:
                    continue
                if network.drive_shape_key(key, index):
                    wired += 1
                else:
                    unwired += 1
        return wired, unwired

    def _plain_copy(self, obj):
        """A legacy object-board control, delivered as-is minus our tags.

        Its data is shared rather than duplicated: a board handle's mesh is a
        widget nobody edits, and sharing keeps the delivered file smaller.
        """
        copy = obj.copy()
        copy.name = f"{obj.name}{BAKE_SUFFIX}"
        _strip_addon_props(copy)
        return copy

    # -- helpers ------------------------------------------------------------

    def _wanted_inputs(self, entries, cache):
        """Which RigLogic inputs get a shape, in evaluation order."""
        reachable = _reachable_inputs(entries, cache)
        skip = set()
        if not self.bake_eye_look:
            skip = {index for index, entry in enumerate(entries)
                    if entry is not None and entry["kind"] == 'RAW'
                    and "eyeLook" in entry["name"]}
            # A corrective built on a skipped control goes with it.
            for index, entry in enumerate(entries):
                if entry is None or entry["kind"] != 'PSD':
                    continue
                if any(column in skip for column, _w in entry["factors"]):
                    skip.add(index)
        wanted = []
        for index, entry in enumerate(entries):
            if entry is None or index in skip or index not in reachable:
                continue
            if self.detail == 'CONTROLS' and entry["kind"] == 'PSD':
                continue
            wanted.append(index)
        return wanted

    def _pose(self, cache, index):
        """Pose the live rig for exactly one RigLogic input (None = rest).

        Written straight through dna_apply rather than through the board: the
        board can only reach an input by way of its GUI channel, which would
        fire that channel's other raws and every PSD standing on them.  The
        input vector is the layer where isolation is exact.
        """
        meta = riglogic.meta()
        total = int(meta["raw_count"]) + int(meta["psd_count"])
        inputs = np.zeros(total)
        if index is not None:
            inputs[index] = 1.0
        deltas = riglogic.inputs_to_joint_deltas(inputs)
        if cache.get("pose_constants") is not None:
            cache["pose_constants"]["previous"] = None   # force a full write
        dna_apply.apply_joint_outputs(
            cache["skel"], cache["targets"], cache["pose_constants"],
            deltas, cache["scale"])
        for key, input_index in cache["shape_key_targets"]:
            value = 1.0 if input_index == index else 0.0
            if abs(float(key.value) - value) > 1e-9:
                key.value = value

    def _report(self, filepath, shapes, wired, unwired, empty, network, skel,
                cache, plan):
        aim = " + native look-at" if plan is not None else ""
        worn = f", {len(self._worn)} body/clothing mesh(es)" if self._worn else ""
        if self.motion == 'BONES':
            inputs, actions, constraints, bones = getattr(
                self, "_joints", (0, 0, 0, 0))
            self.report(
                {'INFO'},
                f"{actions} baked Action(s) over {bones} bone(s), "
                f"{constraints} Action constraint(s), {wired} corrective "
                f"shape key(s), {network.count} drivers{aim}{worn} "
                f"-> {os.path.basename(filepath)}")
            if inputs and actions < inputs:
                self.report(
                    {'INFO'},
                    f"{inputs - actions} input(s) moved no bone further than "
                    "Prune Below and were dropped")
            # Every constraint is solved on every frame, so the count IS the
            # playback cost and the artist should hear it now rather than
            # discover it in the delivered file.
            if constraints > 60000 and self.detail == 'FULL':
                self.report(
                    {'INFO'},
                    f"{constraints} constraints evaluate every frame. If the "
                    "delivered rig feels heavy, bake again with Detail = "
                    "Controls Only, or raise Prune Below")
        else:
            meshes = sum(len(created) for created in shapes.values())
            bones, groups = getattr(self, "_strip", (0, 0))
            retired = (f", {bones} facial bone(s) retired into "
                       f"'{self._strip_bone}'" if bones else "")
            self.report(
                {'INFO'},
                f"{len(shapes)} shapes ({meshes} across the meshes), "
                f"{network.count} drivers{aim}{worn}{retired} "
                f"-> {os.path.basename(filepath)}")
            if bones:
                self.report(
                    {'INFO'},
                    f"{groups} facial vertex group(s) were summed into "
                    f"'{self._strip_bone}', so the face still follows the "
                    "head and neck with no facial bones left")
            elif getattr(self, "_standalone_kept", False):
                self.report(
                    {'INFO'},
                    "Standalone rig: the facial bones ARE this character's "
                    "skeleton, so they are delivered with it and carry the "
                    "head's weight themselves")
            elif getattr(self, "_strip_reason", ""):
                self.report({'WARNING'}, self._strip_reason)

        if unwired:
            self.report({'WARNING'},
                        f"{unwired} shape(s) got no driver - their control is "
                        "missing from this board")
        if empty:
            self.report({'INFO'},
                        f"{len(empty)} input(s) moved nothing and were dropped")
        if skel.get(op_rig.CHANNEL_GUI_INDICES_PROP):
            if self.motion == 'BONES':
                self.report(
                    {'WARNING'},
                    "Bone Intensity is dialled down on some channels. It is a "
                    "live viewport control, not part of the DNA, so the "
                    "delivered rig runs those channels at full strength")
            else:
                self.report(
                    {'WARNING'},
                    "Bone Intensity is dialled down on some channels. It "
                    "scales bones but not sculpted shapes, and the bake merges "
                    "the two, so the delivered rig runs at full strength")
        others = getattr(self, "_body_rigs", ())
        if others:
            self.report(
                {'WARNING'},
                f"'{others[0].name}' still drives this character's body and "
                "clothing, and a separate rig cannot be baked with the face. "
                "Merge Into One Armature first to deliver the whole character")
        if plan is None and op_rig.look_at_enabled(cache):
            self.report(
                {'WARNING'},
                "The eyesAim look-at could not be rebuilt for this rig - the "
                "delivered eyes follow their direct channels only")


def register():
    bpy.utils.register_class(MHFRT_OT_bake_drivers)


def unregister():
    bpy.utils.unregister_class(MHFRT_OT_bake_drivers)
