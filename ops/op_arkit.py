"""Bake the live facial rig into a portable ARKit blend-shape head.

RigLogic and the sculpted correctives are Python; neither survives an FBX.
What DOES survive is geometry, so this module poses the character through the
very same path the Live Link Face importer uses - ARKit shape -> raw controls
(``op_anim.ARKIT_TO_RAW``) -> GUI channels -> the real control board - reads the
depsgraph-evaluated mesh at each pose, and stores the difference from rest as
one shape key on a fresh, unrigged copy.  Head, eyes, teeth, tongue and lashes
are baked individually and then joined, so the result is ONE mesh carrying the
whole 52-shape set.

Driving the actual board (rather than feeding RigLogic directly) is what makes
this what-you-see-is-what-you-bake: per-channel Bone Intensity, channel mutes
and rig intensity all apply exactly as they do in the viewport.  It costs
nothing in accuracy - every one of the 52 ARKit poses round-trips through
``riglogic.invert_raw_frames`` with zero error, because the bundled DNA feeds
each raw control from exactly one GUI channel.

Sculpted correctives ride along for free: their shape keys are driven off the
same RigLogic input vector, so the evaluated mesh already contains them.  Only
the 69 channels a SINGLE ARKit shape can reach are captured, which is inherent
to a 52-shape set rather than to this bake - the remaining channels fire only
on control combinations that no linear blend-shape rig can represent.

The artist's scene is never written to except for three restorable things
(board control locations, non-armature modifier visibility, the Weight-Cleanup
key values), all inside one try/finally.  Source meshes are read, never
touched; every write goes to the copies.
"""

import os

import numpy as np
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

from ..core import deliver
from ..core import organization
from ..core import riglogic
from ..core import board
from ..ui import flow
from . import op_attach, op_export, op_live, op_morphs, op_rig
from .op_anim import ARKIT_TO_RAW
from .op_wrap import CLOSE_EYES_KEY, MOUTH_OPEN_KEY

_RAW_PREFIX = "CTRL_expressions."

# ``mouthClose`` is the one ARKit shape RigLogic cannot pose on its own: the
# four mouthLipsTogether raws are pure PSD correctives and move NOTHING while
# the jaw is shut (measured: max joint delta 0.0).  ARKit's own definition is
# relative too - "close the lips while the jaw is open" - so the shape is posed
# WITH jawOpen and the jawOpen delta is subtracted afterwards.  Playing
# jawOpen + mouthClose together in an engine then closes the lips, exactly as
# ARKit intends.
MOUTH_CLOSE = "mouthClose"
JAW_OPEN = "jawOpen"

# Apple's canonical ARKit order (blend shape index 0..51).  Order matters: some
# importers read by index rather than by name.
_ARKIT_ORDER = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthRight", "mouthLeft",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownRight", "mouthFrownLeft",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
)

# Epic / Unreal ARKit order - identical names, right side first.
_UNREAL_ORDER = (
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "jawForward", "jawRight", "jawLeft", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthRight", "mouthLeft",
    "mouthSmileRight", "mouthSmileLeft", "mouthFrownRight", "mouthFrownLeft",
    "mouthDimpleRight", "mouthDimpleLeft", "mouthStretchRight",
    "mouthStretchLeft", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthPressRight", "mouthPressLeft",
    "mouthLowerDownRight", "mouthLowerDownLeft", "mouthUpperUpRight",
    "mouthUpperUpLeft",
    "browDownRight", "browDownLeft", "browInnerUp", "browOuterUpRight",
    "browOuterUpLeft",
    "cheekPuff", "cheekSquintRight", "cheekSquintLeft",
    "noseSneerRight", "noseSneerLeft", "tongueOut",
)

# FaceCap / generic "_L / _R" naming, in FaceCap's own order.
_FACECAP = (
    ("browInnerUp", "browInnerUp"),
    ("browDownLeft", "browDown_L"), ("browDownRight", "browDown_R"),
    ("browOuterUpLeft", "browOuterUp_L"),
    ("browOuterUpRight", "browOuterUp_R"),
    ("eyeLookUpLeft", "eyeLookUp_L"), ("eyeLookUpRight", "eyeLookUp_R"),
    ("eyeLookDownLeft", "eyeLookDown_L"),
    ("eyeLookDownRight", "eyeLookDown_R"),
    ("eyeLookInLeft", "eyeLookIn_L"), ("eyeLookInRight", "eyeLookIn_R"),
    ("eyeLookOutLeft", "eyeLookOut_L"), ("eyeLookOutRight", "eyeLookOut_R"),
    ("eyeBlinkLeft", "eyeBlink_L"), ("eyeBlinkRight", "eyeBlink_R"),
    ("eyeSquintLeft", "eyeSquint_L"), ("eyeSquintRight", "eyeSquint_R"),
    ("eyeWideLeft", "eyeWide_L"), ("eyeWideRight", "eyeWide_R"),
    ("cheekPuff", "cheekPuff"),
    ("cheekSquintLeft", "cheekSquint_L"),
    ("cheekSquintRight", "cheekSquint_R"),
    ("noseSneerLeft", "noseSneer_L"), ("noseSneerRight", "noseSneer_R"),
    ("jawOpen", "jawOpen"), ("jawForward", "jawForward"),
    ("jawLeft", "jawLeft"), ("jawRight", "jawRight"),
    ("mouthFunnel", "mouthFunnel"), ("mouthPucker", "mouthPucker"),
    ("mouthLeft", "mouthLeft"), ("mouthRight", "mouthRight"),
    ("mouthRollUpper", "mouthRollUpper"), ("mouthRollLower", "mouthRollLower"),
    ("mouthShrugUpper", "mouthShrugUpper"),
    ("mouthShrugLower", "mouthShrugLower"),
    ("mouthClose", "mouthClose"),
    ("mouthSmileLeft", "mouthSmile_L"), ("mouthSmileRight", "mouthSmile_R"),
    ("mouthFrownLeft", "mouthFrown_L"), ("mouthFrownRight", "mouthFrown_R"),
    ("mouthDimpleLeft", "mouthDimple_L"),
    ("mouthDimpleRight", "mouthDimple_R"),
    ("mouthUpperUpLeft", "mouthUpperUp_L"),
    ("mouthUpperUpRight", "mouthUpperUp_R"),
    ("mouthLowerDownLeft", "mouthLowerDown_L"),
    ("mouthLowerDownRight", "mouthLowerDown_R"),
    ("mouthPressLeft", "mouthPress_L"), ("mouthPressRight", "mouthPress_R"),
    ("mouthStretchLeft", "mouthStretch_L"),
    ("mouthStretchRight", "mouthStretch_R"),
    ("tongueOut", "tongueOut"),
)

SHAPE_SETS = {
    'ARKIT': tuple((name, name) for name in _ARKIT_ORDER),
    'UNREAL': tuple((name, name) for name in _UNREAL_ORDER),
    'FACECAP': _FACECAP,
}

# Eye look is bone ROTATION of the eyeball.  As a shape key the vertices travel
# a straight chord instead of an arc, so a fully rotated eye flattens slightly
# and diagonal looks land a little off.  Every ARKit head has this; the option
# to skip them exists for pipelines that keep the eyes bone-driven.
_EYE_LOOK = frozenset(
    name for name in _ARKIT_ORDER if name.startswith("eyeLook")
)

_ARKIT_MOD_PROP = "mhfrt_arkit_baked"   # marks the generated mesh
_ADDON_PREFIXES = ("mhfrt", "dna_", "mhfa")
FACIAL_PREFIX = "FACIAL_"


# ------------------------------------------------------------ scene state ---

class _Restore:
    """Undo log for the handful of things the bake has to touch.

    Callables run newest-first in the ``finally`` block, so a failure halfway
    through still puts the artist's scene back exactly as it was.
    """

    def __init__(self):
        self._steps = []

    def add(self, func):
        self._steps.append(func)

    def run(self):
        while self._steps:
            step = self._steps.pop()
            try:
                step()
            except (ReferenceError, RuntimeError, AttributeError):
                pass          # a restore must never mask the real error


def _mode_object(context):
    if context.object is not None and context.object.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass


def _freeze_animation(obj, restore):
    """Silence actions and NLA so our own writes are not overwritten.

    Blender flushes evaluated (animated) object transforms back to the original
    datablock, so a keyframed control board would clobber the GUI values this
    module writes between poses.
    """
    if obj is None or obj.animation_data is None:
        return
    state = op_export._snapshot_animation(obj)
    restore.add(lambda o=obj, s=state: op_export._restore_animation(o, s))
    obj.animation_data.action = None
    for track in obj.animation_data.nla_tracks:
        track.mute = True


def _force_visible(obj, restore):
    """A hide_viewport object is dropped from the depsgraph - its evaluated
    mesh would be stale or missing, so unhide for the duration."""
    if obj.hide_viewport:
        restore.add(lambda o=obj: setattr(o, "hide_viewport", True))
        obj.hide_viewport = False
    try:
        if obj.hide_get():
            restore.add(lambda o=obj: o.hide_set(True))
            obj.hide_set(False)
    except RuntimeError:
        pass          # not in the current view layer; caller reports it


def _disable_generative_modifiers(obj, restore):
    """Turn off everything except Armature.

    A Subsurf or Mirror changes the evaluated vertex count, which would make
    the shape-key write impossible.  Returns the names that were disabled so
    the operator can tell the artist what it did.
    """
    disabled = []
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            continue
        if not modifier.show_viewport:
            continue
        restore.add(
            lambda m=modifier: setattr(m, "show_viewport", True))
        modifier.show_viewport = False
        disabled.append(modifier.name)
    return disabled


def _zero_cleanup_keys(obj, restore):
    """Weight-Cleanup pose keys are a binding aid, not part of the character.

    Left at a non-zero value they would bake an open mouth into the neutral.
    """
    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return
    for name in (MOUTH_OPEN_KEY, CLOSE_EYES_KEY):
        key = shape_keys.key_blocks.get(name)
        if key is None or key.value == 0.0:
            continue
        restore.add(lambda k=key, v=key.value: setattr(k, "value", v))
        key.value = 0.0


# --------------------------------------------------------- control board ---

def _snapshot_controls(cache, restore):
    for control in cache["controls"].values():
        # Both defaults are evaluated in THIS scope: read the location from
        # `control`, never from the parameter being defined.
        restore.add(lambda c=control, loc=tuple(control.location):
                    setattr(c, "location", loc))


def _controls_to_rest(cache):
    """Every board control back to its DNA rest position = GUI all zero.

    This also parks the center-eye master, whose offset would otherwise be
    added on top of the per-eye channels we write (see op_rig.CENTER_EYE_LINKS).
    """
    for control in cache["controls"].values():
        board.rest_pose(control)


def _write_gui(cache, gui):
    """Push a GUI channel vector onto the board - inverse of op_rig's reader.

    board.channel_value computes ``(location[axis] - rest) * sign / div``
    with sign in {+1, -1}, so the inverse is ``rest + value * div * sign``.
    """
    divisor = cache["divisor"]
    for value, source in zip(gui, cache["sources"]):
        control, axis, rest, sign = source[0], source[1], source[2], source[3]
        if control is None or sign == 0.0:
            continue
        control.location[axis] = rest + float(value) * divisor * sign


def _raw_vector(shape_name, raw_index, raw_count):
    """ARKit shape -> raw control vector, using the Live Link mapping."""
    raw = np.zeros(raw_count)
    labels = list(ARKIT_TO_RAW.get(shape_name, ()))
    if shape_name == MOUTH_CLOSE:
        labels.append(JAW_OPEN)      # see MOUTH_CLOSE note at the top
    for label in labels:
        index = raw_index.get(label)
        if index is not None:
            raw[index] = 1.0
    return raw


# ------------------------------------------------------------- geometry ---

def _world_coords(obj, depsgraph):
    """Evaluated vertex positions of `obj`, in world space.

    Returns None when the evaluated vertex count no longer matches the base
    mesh - the caller aborts rather than guessing a correspondence.
    """
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.data
    count = len(mesh.vertices)
    if count != len(obj.data.vertices):
        return None
    flat = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    local = flat.reshape(count, 3).astype(np.float64)
    matrix = evaluated.matrix_world
    rotation = np.array(
        [[matrix[0][0], matrix[0][1], matrix[0][2]],
         [matrix[1][0], matrix[1][1], matrix[1][2]],
         [matrix[2][0], matrix[2][1], matrix[2][2]]], dtype=np.float64)
    translation = np.array(
        [matrix[0][3], matrix[1][3], matrix[2][3]], dtype=np.float64)
    return local @ rotation.T + translation


def _make_copy(context, source, collection, rest_coords, keep_groups):
    """An unrigged twin of `source`, sitting at the world-space rest pose."""
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = f"{source.name}_ARKit"
    copy.data.name = copy.name
    collection.objects.link(copy)

    copy.animation_data_clear()
    copy.constraints.clear()
    copy.modifiers.clear()
    copy.shape_key_clear()
    copy.parent = None
    copy.matrix_world = Matrix.Identity(4)
    copy.hide_viewport = False
    copy.hide_select = False
    if not keep_groups:
        copy.vertex_groups.clear()

    flat = np.ascontiguousarray(rest_coords, dtype=np.float32).ravel()
    copy.data.vertices.foreach_set("co", flat)
    copy.data.update()
    basis = copy.shape_key_add(name="Basis", from_mix=False)
    basis.interpolation = 'KEY_LINEAR'
    return copy


def _add_key(copy, name, coords):
    key = copy.shape_key_add(name=name, from_mix=False)
    key.slider_min, key.slider_max, key.value = 0.0, 1.0, 0.0
    key.data.foreach_set(
        "co", np.ascontiguousarray(coords, dtype=np.float32).ravel())
    return key


def _prune(rest, coords, threshold):
    """Snap sub-threshold vertices back onto rest; report the peak movement.

    Pruning is per VERTEX (not per axis) so a tiny motion is either kept whole
    or removed whole, never flattened onto one axis.
    """
    delta = coords - rest
    distance = np.linalg.norm(delta, axis=1)
    peak = float(distance.max()) if len(distance) else 0.0
    if peak == 0.0:
        return rest, 0.0
    delta[distance < threshold] = 0.0
    return rest + delta, peak


def _match_uv_names(head_copy, copies):
    """Give every copy the head's UV layer names, positionally.

    ``object.join`` merges UV layers BY NAME: a mesh whose layer is called
    "UV0" instead of "UVMap" would land in a second, half-empty layer and the
    eyes would silently lose their texture coordinates.
    """
    head_names = [layer.name for layer in head_copy.data.uv_layers]
    if not head_names:
        return
    for copy in copies:
        if copy is head_copy:
            continue
        for index, layer in enumerate(copy.data.uv_layers):
            if index < len(head_names) and layer.name != head_names[index]:
                layer.name = head_names[index]


def _source_objects(mh):
    """Head + attached parts + extra morph meshes, deduplicated, in scene.

    Deliberately NOT op_export.export_objects(): on a rig merged into a body
    armature that would sweep the whole torso into the head bake.
    """
    scene_objects = set(bpy.context.scene.objects)
    out, seen = [], set()
    for obj in [mh.target, *op_attach.attached_objects(mh),
                *op_morphs.extra_morph_objects(mh)]:
        if obj is None or obj.type != 'MESH' or obj not in scene_objects:
            continue
        key = obj.as_pointer()
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out


def _target_collection(context, name):
    """A collection in the artist's scene to assemble the delivery in.

    It is linked into the scene because the join and the depsgraph reads below
    need these objects in the view layer; it is taken out again - with
    everything in it - once the file is written.
    """
    collection = bpy.data.collections.new(name)
    context.scene.collection.children.link(collection)
    return collection


# Worn only so the delivered copies can coexist with the artist's own objects
# while the bake runs; stripped again on the way into the file.
DELIVER_SUFFIX = "_MHFRDeliver"


def _deliver_skeleton(skel, collection):
    """A clean copy of the character's rig for the delivered file.

    The face is shape keys now, so every facial joint is parked on its rest:
    nothing drives them there, and a copy carries whatever RigLogic last wrote.
    The board is deliberately not copied - the delivered character is animated
    by the ARKit shapes, and a panel that no longer drives anything would only
    mislead whoever opens the file.
    """
    copy = skel.copy()
    copy.data = skel.data.copy()
    copy.name = f"{skel.name}{DELIVER_SUFFIX}"
    copy.data.name = copy.name
    copy.animation_data_clear()
    copy.hide_viewport = False
    copy.hide_select = False
    copy.parent = None
    copy.matrix_world = skel.matrix_world.copy()
    for prop in [k for k in copy.keys() if str(k).startswith(_ADDON_PREFIXES)]:
        del copy[prop]
    for prop in [k for k in copy.data.keys()
                 if str(k).startswith(_ADDON_PREFIXES)]:
        del copy.data[prop]
    for pose_bone in copy.pose.bones:
        for prop in [k for k in pose_bone.keys()
                     if str(k).startswith(_ADDON_PREFIXES)]:
            del pose_bone[prop]
        if pose_bone.name.startswith(FACIAL_PREFIX):
            pose_bone.matrix_basis = Matrix.Identity(4)
    collection.objects.link(copy)
    return copy


def _deliver_worn(obj, skel, skel_copy, collection):
    """A body or clothing mesh, delivered bound to the copied rig."""
    copy = obj.copy()
    copy.data = obj.data.copy()
    copy.name = f"{obj.name}{DELIVER_SUFFIX}"
    copy.data.name = copy.name
    copy.animation_data_clear()
    copy.hide_viewport = False
    copy.hide_select = False
    copy.hide_render = False
    copy.parent = None
    copy.matrix_world = obj.matrix_world.copy()
    for modifier in list(copy.modifiers):
        if modifier.type != 'ARMATURE':
            continue
        if skel_copy is not None and modifier.object is skel:
            modifier.object = skel_copy
        elif skel_copy is None:
            # No rig in the delivery: a modifier still aimed at the artist's
            # armature would drag that whole rig into the written file.
            copy.modifiers.remove(modifier)
    collection.objects.link(copy)
    return copy


# ------------------------------------------------------------- operator ---

class MHFRT_OT_generate_arkit_head(bpy.types.Operator, ExportHelper):
    """Bake the facial rig and sculpted correctives into a character carrying \
the 52 ARKit blend shapes, written to its own .blend file"""
    bl_idname = "mhfrt.generate_arkit_head"
    bl_label = "Export ARKit Character"
    bl_options = {'REGISTER'}

    filename_ext = ".blend"
    filter_glob: StringProperty(default="*.blend", options={'HIDDEN'})

    include_rig: BoolProperty(
        name="Include Rig",
        default=True,
        description=("Deliver the character's skeleton and keep the meshes "
                     "skinned to it, so the body still animates. The face is "
                     "the ARKit shapes; the facial joints ride along at rest. "
                     "The control board is never included"),
    )
    include_body: BoolProperty(
        name="Include Body & Clothing",
        default=True,
        description=("Deliver the rest of the character alongside the baked "
                     "head. Off writes the head and its parts only"),
    )

    shape_set: EnumProperty(
        name="Naming",
        items=(
            ('ARKIT', "ARKit (Apple)",
             "Apple's canonical names and blend shape order 0-51"),
            ('UNREAL', "Unreal / Epic",
             "Same names, Epic's ARKit order (right side first)"),
            ('FACECAP', "FaceCap (_L / _R)",
             "eyeBlink_L style names, FaceCap order"),
        ),
        default='ARKIT',
    )
    merge_into_one: BoolProperty(
        name="Merge Into One Mesh",
        default=True,
        description=("Join head, eyes, teeth, tongue and lashes into a single "
                     "mesh. Turn off to keep one baked copy per object"),
    )
    bake_eye_look: BoolProperty(
        name="Bake Eye Look Shapes",
        default=True,
        description=("Bake the 8 eyeLook shapes. Eye look is a bone rotation, "
                     "so as a blend shape the eyeball travels a chord instead "
                     "of an arc. Turn off if the eyes stay bone-driven"),
    )
    keep_vertex_groups: BoolProperty(
        name="Keep Vertex Groups",
        default=False,
        description=("Keep the deform weights on the generated mesh. Off by "
                     "default: the result is not rigged"),
    )
    prune_threshold: FloatProperty(
        name="Prune Below",
        default=1e-5, min=0.0, max=1.0, precision=6, step=0.01,
        subtype='DISTANCE',
        description=("Vertices that move less than this in a shape are snapped "
                     "back to rest, which keeps the file smaller"),
    )

    _sources = ()
    _modifier_warning = ""

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        if mh is None or mh.skeleton is None or not riglogic.available():
            return False
        return flow.bind_done(mh)

    def invoke(self, context, event):
        mh = context.scene.mhfrt
        self._sources = _source_objects(mh)
        pending = []
        for obj in self._sources:
            names = [m.name for m in obj.modifiers
                     if m.type != 'ARMATURE' and m.show_viewport]
            if names:
                pending.append(f"{obj.name}: {', '.join(names)}")
        self._modifier_warning = " | ".join(pending)
        if not self.filepath:
            label = (mh.target.name if mh.target is not None
                     else (mh.skeleton.name if mh.skeleton else "Character"))
            self.filepath = f"{label}_ARKit.blend"
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        sources = getattr(self, "_sources", ())
        box = layout.box()
        box.label(text=f"Baking {len(sources)} mesh(es)",
                  icon='OUTLINER_OB_MESH')
        column = box.column(align=True)
        column.enabled = False
        for obj in sources[:6]:
            column.label(text=obj.name)
        if len(sources) > 6:
            column.label(text=f"... and {len(sources) - 6} more")

        layout.prop(self, "shape_set")
        layout.prop(self, "merge_into_one")
        layout.prop(self, "bake_eye_look")
        layout.prop(self, "include_rig")
        layout.prop(self, "include_body")
        rig_off = layout.row()
        rig_off.enabled = not self.include_rig
        rig_off.prop(self, "keep_vertex_groups")
        layout.prop(self, "prune_threshold")

        if self._modifier_warning:
            warning = layout.box()
            warning.label(text="Modifiers disabled while baking",
                          icon='MODIFIER')
            hint = warning.column(align=True)
            hint.enabled = False
            hint.label(text=self._modifier_warning[:120])
            hint.label(text="Apply generative modifiers first if you need "
                            "them baked in")

    def execute(self, context):
        mh = context.scene.mhfrt
        skel = mh.skeleton
        if skel is None:
            self.report({'ERROR'}, "No active character rig")
            return {'CANCELLED'}

        sources = _source_objects(mh)
        if not sources:
            self.report({'ERROR'}, "No head mesh to bake")
            return {'CANCELLED'}

        shapes = [(source, name) for source, name in SHAPE_SETS[self.shape_set]
                  if self.bake_eye_look or source not in _EYE_LOOK]

        op_live.stop_running()
        _mode_object(context)

        restore = _Restore()
        self._created = []          # (collection, copies) - dropped on failure
        window_manager = context.window_manager
        window_manager.progress_begin(0, len(shapes) + 1)
        try:
            result = self._bake(context, skel, sources, shapes, restore)
        except Exception:
            self._discard_created()
            raise
        finally:
            restore.run()
            window_manager.progress_end()
            op_rig.invalidate()
        if result != {'FINISHED'}:
            # A half-built pile of copies is worse than nothing: an aborted
            # bake must leave the file exactly as it found it.
            self._discard_created()
        return result

    def _discard_created(self):
        for collection, copies in getattr(self, "_created", ()):
            for copy in copies:
                try:
                    data = copy.data
                    bpy.data.objects.remove(copy)
                    if data is not None and data.users == 0:
                        # the delivery creates armatures as well as meshes
                        group = (bpy.data.armatures
                                 if isinstance(data, bpy.types.Armature)
                                 else bpy.data.meshes)
                        group.remove(data)
                except (ReferenceError, RuntimeError, TypeError):
                    pass
            if collection is None:
                continue
            try:
                bpy.data.collections.remove(collection)
            except (ReferenceError, RuntimeError, TypeError):
                pass
        self._created = []

    # -- the bake itself ----------------------------------------------------

    def _bake(self, context, skel, sources, shapes, restore):
        # 0. An object outside the view layer is not in the depsgraph at all:
        #    evaluated_get would hand back undeformed base geometry and the
        #    bake would look like it worked.  Refuse instead.
        visible = set(context.view_layer.objects)
        missing = [obj.name for obj in sources if obj not in visible]
        if missing:
            self.report(
                {'ERROR'},
                "Not in the current view layer (excluded collection?): "
                + ", ".join(missing[:4]))
            return {'CANCELLED'}

        # 1. Freeze everything that could move under us.
        _freeze_animation(skel, restore)
        if skel.parent is not None:
            _freeze_animation(skel.parent, restore)
        disabled = []
        for obj in sources:
            _freeze_animation(obj, restore)
            _force_visible(obj, restore)
            _zero_cleanup_keys(obj, restore)
            disabled.extend(_disable_generative_modifiers(obj, restore))

        # 2. A clean rig cache, built AFTER the scene is frozen.
        try:
            cache = op_export._rig_cache(skel)
        except RuntimeError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        # Animation data lives on the OBJECT: for a bone board that is the one
        # board armature, so the set keeps this to a single freeze.
        for owner in {board.owner_object(control)
                      for control in cache["controls"].values()}:
            _freeze_animation(owner, restore)
        _snapshot_controls(cache, restore)

        meta = riglogic.meta()
        raw_count = int(meta["raw_count"])
        raw_index = {
            name[len(_RAW_PREFIX):] if name.startswith(_RAW_PREFIX) else name:
            index
            for index, name in enumerate(meta["raw_names"])
        }

        depsgraph = context.evaluated_depsgraph_get()

        # 3. Rest pose -> the Basis of every copy.
        _controls_to_rest(cache)
        op_rig._evaluate_guarded([cache])
        depsgraph.update()
        rest = {}
        for obj in sources:
            coords = _world_coords(obj, depsgraph)
            if coords is None:
                self.report(
                    {'ERROR'},
                    f"{obj.name}: a modifier changes the vertex count. "
                    "Apply it (or disable it) and generate again.")
                return {'CANCELLED'}
            rest[obj.as_pointer()] = coords

        name = f"{context.scene.mhfrt.target.name}_ARKit"
        collection = _target_collection(context, name)
        # A delivered rig needs the weights: without them the meshes arrive
        # skinned to nothing and the body cannot move.
        keep_groups = self.keep_vertex_groups or self.include_rig
        copies = [
            _make_copy(context, obj, collection, rest[obj.as_pointer()],
                       keep_groups)
            for obj in sources
        ]
        self._created.append((collection, copies))
        context.window_manager.progress_update(1)

        # 4. One pose per ARKit shape.
        jaw_open = {}
        empty = []
        for step, (source_name, key_name) in enumerate(shapes):
            raw = _raw_vector(source_name, raw_index, raw_count)
            gui = riglogic.invert_raw_frames(raw)[0]
            _controls_to_rest(cache)
            _write_gui(cache, gui)
            op_rig._evaluate_guarded([cache])
            depsgraph.update()

            for obj, copy in zip(sources, copies):
                pointer = obj.as_pointer()
                coords = _world_coords(obj, depsgraph)
                if coords is None:
                    self.report(
                        {'ERROR'},
                        f"{obj.name}: vertex count changed mid-bake")
                    return {'CANCELLED'}
                if source_name == JAW_OPEN:
                    jaw_open[pointer] = coords.copy()
                coords, peak = _prune(rest[pointer], coords,
                                      self.prune_threshold)
                _add_key(copy, key_name, coords)
                if peak < self.prune_threshold:
                    empty.append((copy, key_name))
            context.window_manager.progress_update(step + 2)

        # 5. mouthClose is jawOpen-relative (see the note at the top).
        self._resolve_mouth_close(sources, copies, rest, jaw_open, shapes,
                                  empty)

        # 6. Back to the artist's pose before anything else happens.
        restore.run()
        op_rig.invalidate()

        merged = self._finish(context, copies, name)
        merged[_ARKIT_MOD_PROP] = self.shape_set
        # After a merge the other copies are gone - touching one would be a
        # dangling pointer, so the survivor is named outright.
        delivered = ([merged] if self.merge_into_one and len(copies) > 1
                     else list(copies))

        # 7. The character around the head: its rig, and everything else it
        #    wears. The board is never part of this - the face is the shapes.
        skel_copy = None
        if self.include_rig:
            skel_copy = _deliver_skeleton(skel, collection)
            self._created.append((None, [skel_copy]))
            for copy in delivered:
                modifier = copy.modifiers.new("Armature", 'ARMATURE')
                modifier.object = skel_copy
        if self.include_body:
            worn = [
                obj for obj in organization.character_meshes(
                    [skel], cage=skel.get("mhfrt_cage"),
                    target=skel.get("mhfrt_target"), scene=context.scene)
                if obj not in sources
            ]
            worn_copies = [_deliver_worn(obj, skel, skel_copy, collection)
                           for obj in worn]
            if worn_copies:
                self._created.append((None, worn_copies))

        # 8. Write it, then take every copy back out of the artist's file.  The
        #    collection goes over on its own - the delivered scene is built by
        #    the second pass, and a scene made only to be written is what
        #    crashes Blender 5.2 (see core/deliver.py).
        filepath = bpy.path.ensure_ext(self.filepath, ".blend")
        try:
            fallback, displaced = deliver.write_blend(
                filepath, collection, name, context.scene.frame_start,
                context.scene.frame_end, strip_suffix=DELIVER_SUFFIX)
        except (RuntimeError, OSError) as error:
            self.report({'ERROR'}, f"Could not write {filepath}: {error}")
            return {'CANCELLED'}
        finally:
            context.scene.collection.children.unlink(collection)
        self._discard_created()

        dead = sorted({key_name for _copy, key_name in empty})
        message = (f"{len(shapes)} shapes written to "
                   f"{os.path.basename(filepath)}")
        if self.merge_into_one:
            message += f" ({len(copies)} meshes merged)"
        if self.include_rig:
            message += " | rigged"
        if disabled:
            message += f" | {len(disabled)} modifier(s) bypassed"
        self.report({'INFO'}, message)
        if fallback:
            self.report(
                {'WARNING'},
                "Wrote a library file instead of an openable one "
                f"({fallback}). Everything is in it - link or append the "
                f"'{collection.name}' collection rather than opening it.")
        if displaced:
            self.report(
                {'WARNING'},
                f"{len(displaced)} name(s) were already taken in the "
                f"delivered file and moved to '_original': "
                f"{', '.join(displaced[:4])}")
        if dead:
            self.report(
                {'WARNING'},
                f"{len(dead)} shape(s) produced no movement: "
                f"{', '.join(dead[:6])}"
                + (" ..." if len(dead) > 6 else ""))
        return {'FINISHED'}

    def _resolve_mouth_close(self, sources, copies, rest, jaw_open, shapes,
                             empty):
        """mouthClose := Basis + (jawOpen + lipsTogether) - jawOpen.

        The shape was posed WITH the jaw open because the four
        mouthLipsTogether raws are PSD correctives that move nothing on their
        own.  Subtracting the jawOpen pose leaves exactly the lip closure, so
        an engine playing jawOpen + mouthClose lands on closed lips.
        """
        key_name = next((out for src, out in shapes if src == MOUTH_CLOSE),
                        None)
        if key_name is None or not jaw_open:
            return
        for obj, copy in zip(sources, copies):
            pointer = obj.as_pointer()
            base = rest.get(pointer)
            open_pose = jaw_open.get(pointer)
            keys = copy.data.shape_keys
            key = keys.key_blocks.get(key_name) if keys else None
            if base is None or open_pose is None or key is None:
                continue
            count = len(key.data)
            posed = np.empty(count * 3, dtype=np.float32)
            key.data.foreach_get("co", posed)
            corrected = (base + posed.reshape(count, 3).astype(np.float64)
                         - open_pose)
            corrected, peak = _prune(base, corrected, self.prune_threshold)
            key.data.foreach_set(
                "co", np.ascontiguousarray(corrected, dtype=np.float32).ravel())
            if peak < self.prune_threshold:
                empty.append((copy, key_name))

    def _finish(self, context, copies, name):
        """Join the copies (or not), then hand the result to the artist."""
        head_copy = copies[0]
        if self.merge_into_one and len(copies) > 1:
            _match_uv_names(head_copy, copies)
            for obj in context.selected_objects:
                obj.select_set(False)
            for copy in copies:
                copy.select_set(True)
            context.view_layer.objects.active = head_copy
            with context.temp_override(active_object=head_copy,
                                       object=head_copy,
                                       selected_objects=copies,
                                       selected_editable_objects=copies):
                bpy.ops.object.join()
        head_copy.name = name
        head_copy.data.name = name
        for obj in context.selected_objects:
            obj.select_set(False)
        head_copy.select_set(True)
        context.view_layer.objects.active = head_copy
        return head_copy


def register():
    bpy.utils.register_class(MHFRT_OT_generate_arkit_head)


def unregister():
    bpy.utils.unregister_class(MHFRT_OT_generate_arkit_head)
