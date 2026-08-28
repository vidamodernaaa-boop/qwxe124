"""The character's facial control board.

Two layouts exist and both are read here, so nothing else in the add-on has to
care which one a character has:

* **BONES** - one ``face_gui`` armature whose pose bones ARE the controls, with
  authored widgets, per-bone colours, bone collections and Limit Location
  frames.  Every rig built from v3.0 gets this board.  It is the MetaHuman DNA
  importer's own board, bundled under GPL-3.0 (``data/LICENSE-OpenRigLogic.txt``).
* **OBJECTS** - the legacy loose-object MH4 board: ~390 empties and meshes, one
  per control.  Scenes built before v3.0 keep theirs and keep working.

A "control" is therefore either a ``PoseBone`` or an ``Object``.  Both expose
``.location``, ``.name`` and custom properties, which is what makes a single set
of accessors enough.

The two layouts differ in three ways, all handled by :func:`channel_axis`:

* axes - a board bone reads tx/ty/tz straight off local X/Y/Z, while a loose
  object carries the Maya Y-up swap (ty is +Z, tz is -Y),
* scale - the bone board is authored 1:1 with the DNA's +/-1 control range
  (``dna_gui_scale`` 1.0); the object board is at centimetre scale (0.01),
* limits - 169 board bones carry a Limit Location constraint drawn as the frame
  around the handle.  Dragging past it leaves ``location`` outside the frame
  while the constraint holds the handle still on screen, and a GUI value outside
  every one of its segments contributes nothing - so the driven bones would snap
  to neutral at exactly the moment the control hits its limit.  Saturating the
  value at the frame edge is what the frame means (see :func:`channel_limits`).

Only ONE number on a board control is animation: its ``location``, and only on
the 161 templates the DNA actually drives.  Everything else - the other 274
bones, and every bone's rotation, scale and widget size - is the panel's DESIGN,
and this module treats it as such throughout:

* it is written down (:func:`stamp_design`), so a Clear Scale is undoable;
* it is locked when nobody is redesigning (:func:`set_board_locked`), so a
  shortcut aimed at the face cannot reach it;
* it is remembered whole (:func:`capture_layout`), so an artist's rearrangement
  survives a rebuild and travels in the .mhfrt;
* and the two follow-head switches are wired to real constraints
  (:func:`install_follow_head`), so the panel's own behaviour belongs to the rig
  rather than to this add-on being installed.
"""

import json

import bpy
from mathutils import Matrix, Quaternion

# Tags this add-on writes on the controls it owns.
#
# Before 3.2 these used the DNA importer's names, shared on purpose so a board
# built by either add-on was understood by the other. That add-on is gone and
# the rule now is that a board this one builds belongs to this one - nothing
# else should be able to find it by scanning for a tag, and this add-on must
# never claim a rig another tool imported. So the names are namespaced. The old
# ones are still READ, because boards in scenes built before 3.2 carry them.
TEMPLATE_PROP = "mhfrt_control_name"
REST_PROP = "mhfrt_control_rest"
SOURCE_PROP = "mhfrt_board_source"
BOARD_ARMATURE_PROP = "mhfrt_face_board"

_LEGACY_PROPS = {
    TEMPLATE_PROP: "dna_gui_template_name",
    REST_PROP: "dna_gui_rest_location",
    SOURCE_PROP: "dna_gui_source",
    BOARD_ARMATURE_PROP: "dna_face_board_armature",
}


def tag(owner, prop):
    """Read one of our tags, falling back to the pre-3.2 name.

    A tag with no pre-3.2 spelling simply has no fallback - looking one up
    unconditionally would raise KeyError on every property added since.
    """
    value = owner.get(prop)
    if value is None:
        legacy = _LEGACY_PROPS.get(prop)
        return owner.get(legacy) if legacy else None
    return value

# A board handle is drawn by a custom-shape object referenced by the bone and
# linked to no collection at all, so nothing that walks the scene ever sees it.
# That also means nothing would ever clean the ~460 of them up: the tag below is
# how a rebuilt or removed board finds its own widgets again.  It is
# deliberately NOT the rig id property - a widget is a mesh, and the merge path
# collects rig-tagged meshes to rebind them to the character's armature.
WIDGET_OWNER_PROP = "mhfrt_board_widget_rig"

BONES_SOURCE = "MH4_BONES"
OBJECTS_SOURCE = "MH4"

# The board armature inside the source character blend.
FACE_BOARD_ARMATURE = "Ada_Head_face_gui"
FACE_BOARD_ANCHOR_BONE = "CTRL_faceGUI"
# The eye-aim chain is a second root, floating in front of the face rather than
# sitting on the panel (see op_rig._place_eye_aim).
FACE_BOARD_ANCHOR_EYES = "CTRL_C_eyesAim"

# Where the board's shapeless helper bones are parked (see stow_helper_bones).
# Lower case to sit naturally beside the asset's own 'controls', 'text',
# 'eye_aim' and 'other' collections.
HELPER_COLLECTION = "helpers"

# ---------------------------------------------------- the panel handles ------
#
# The three flat bars an artist grabs to slide a GUI panel around the viewport.
# Each is the root of ONE ``FRM_`` frame and drags that frame and nothing else:
#
#   CTRL_faceGUI                      the expressions / rigLogic panel (all of
#                                     it - every other handle hangs off it)
#   CTRL_faceTweakersGUI              the TWEAKERS panel
#   CTRL_faceAndEyesAimFollowHeadGUI  the two follow-head switches
#
# They are LAYOUT, not animation.  None of them appears in the DNA's
# ``gui_names`` (161 driving templates on the bundled DNA; none of these three),
# so a handle contributes nothing to the face however far it is dragged - and
# that cuts both ways.  Sending one to rest does not neutralise an expression,
# it throws the artist's panel back across the head, which is why every
# "reset the pose" path has to leave them alone (see :func:`rest_pose`).
LAYOUT_HANDLES = (
    "CTRL_faceGUI",
    "CTRL_faceTweakersGUI",
    "CTRL_faceAndEyesAimFollowHeadGUI",
)
LAYOUT_LABELS = {
    "CTRL_faceGUI": "Main panel",
    "CTRL_faceTweakersGUI": "Tweakers panel",
    "CTRL_faceAndEyesAimFollowHeadGUI": "Follow-head switches",
}

# ------------------------------------------------- the flattened placement ---
#
# The board used to be placed beside the head by transforming the OBJECT:
# ``matrix_world = Translation(offset) @ Scale(head-size ratio)``.  That works,
# but it leaves the artist looking at an armature whose N-panel reads a random
# offset and a scale of, say, 1.037 - and it makes Alt+S in Object Mode a
# destructive operation, because "clear scale" throws away a measurement the
# add-on took and nothing puts back.
#
# The placement lives in the REST now (``op_rig._place_bone_board`` bakes it
# into the edit bones) and the object sits at location 0, rotation 0, scale 1.
# Clear Transform on it is then a no-op by construction rather than by
# prohibition, which is the only kind of safe there is.
#
# The one thing that has to move with it is what a bone-local unit MEANS.  A
# handle's ``location`` is in armature-local units, so scaling the rest by the
# head ratio scales the travel of every handle and every Limit Location frame by
# the same amount - and ``channel_value`` divides by ``dna_gui_scale``, which is
# exactly the field that exists to say how many armature units one GUI unit is.
# Scale the rest, the locations, the rests and the limits together, set the
# divisor to match, and every GUI channel reads bit for bit what it read before
# (verified: the whole 174-channel vector, unchanged).
#
# The matrix that went into the rest is kept here so re-placing is a DELTA
# rather than a fresh bake on top of the last one.
BOARD_PLACEMENT_PROP = "mhfrt_board_placement"

# A saved layout, as JSON, on the character's SKELETON - deliberately not on
# the board.  A board is replaceable (Rebuild Control Board deletes the whole
# armature and appends a fresh one); the skeleton is the character.
LAYOUT_PROP = "mhfrt_board_layout"

# The GUI channel suffix -> (local location axis, sign) for each layout.
_BONE_AXES = {"tx": (0, 1.0), "ty": (1, 1.0), "tz": (2, 1.0)}
_OBJECT_AXES = {"tx": (0, 1.0), "ty": (2, 1.0), "tz": (1, -1.0)}


def is_pose_bone(control):
    return isinstance(control, bpy.types.PoseBone)


def control_armature(control):
    """The armature object a board bone belongs to, or None for an object."""
    if not is_pose_bone(control):
        return None
    owner = getattr(control, "id_data", None)
    if isinstance(owner, bpy.types.Object) and owner.type == 'ARMATURE':
        return owner
    # Some Blender builds hand back the Armature datablock instead.
    for obj in bpy.data.objects:
        if (obj.type == 'ARMATURE' and obj.pose is not None
                and obj.pose.bones.get(control.name) == control):
            return obj
    return None


def owner_object(control):
    """The Object that owns this control - itself, or its board armature.

    Animation data, depsgraph updates and collection membership all live on the
    object, never on the pose bone.
    """
    return control_armature(control) if is_pose_bone(control) else control


def is_board_armature(obj):
    return bool(obj is not None and obj.type == 'ARMATURE'
                and tag(obj, BOARD_ARMATURE_PROP))


def is_rig_skeleton(obj, rig_id_prop):
    """True when `obj` is a character's DEFORMING skeleton.

    The bone control board is an armature carrying the SAME rig id - that is
    how its controls are found - so every "find this character's skeleton"
    lookup has to exclude it.  A board has no cage or target pointer either, so
    any lookup that treats "no pointers" as "matches any character" will pick
    it: that is how pressing New Character used to leave the panel's skeleton
    pointing at the PREVIOUS character's board, after which Update Rig fitted
    an armature with zero FACIAL_ bones and no second character could ever be
    built.
    """
    return bool(obj is not None and obj.type == 'ARMATURE'
                and obj.get(rig_id_prop) and not is_board_armature(obj))


def board_armature_for_rig(rig_id_prop, rig_id, skel=None):
    """The control-board armature carrying `rig_id`.

    Pass `skel` whenever it is known: two armatures can carry one id - that is
    what Scene > New > **Full Copy** produces - and the tag scan alone returns
    whichever of them sorts last in ``bpy.data.objects``, which is how a
    visible board ends up driving an invisible twin.
    """
    own = own_board_armature(skel, rig_id_prop)
    if own is not None:
        return own
    if not rig_id:
        return None
    for obj in bpy.data.objects:
        if is_board_armature(obj) and str(obj.get(rig_id_prop) or "") == str(rig_id):
            return obj
    return None


# props.RIG_GUI_COLL_PROP, spelled out for the same reason the rig id property
# arrives as a parameter: this module stays free of add-on-level imports.
GUI_COLL_PROP = "mhfrt_gui_coll"


def own_board_armature(skel, rig_id_prop):
    """The board this skeleton itself claims, or None.

    Resolved through the skeleton's own links - the GUI collection it points
    at, then the board parented to it - so it is correct by construction even
    when a copy of the character is answering to the same id.
    """
    if skel is None:
        return None
    rid = str(skel.get(rig_id_prop) or "")
    if not rid:
        return None

    def mine(obj):
        return (is_board_armature(obj)
                and str(obj.get(rig_id_prop) or "") == rid)

    coll = skel.get(GUI_COLL_PROP)
    if isinstance(coll, bpy.types.Collection):
        for obj in coll.all_objects:
            if mine(obj):
                return obj
    for obj in bpy.data.objects:
        if obj.parent == skel and mine(obj):
            return obj
    return None


def set_bone_hidden(pose_bone, hidden):
    """Hide or show one bone, on either side of the Blender 5 move.

    ``hide`` lives on ``PoseBone`` from Blender 5.0 and on ``Bone`` before it,
    and writing the wrong one is silent - it sets a property nothing reads.
    That is how the whole eye-aim chain stayed invisible: the bundled asset
    carries ``PoseBone.hide`` (36 bones), while ``Bone.hide`` is True on none of
    them.  Both are written so the result is the same in either version.
    """
    for owner in (pose_bone, pose_bone.bone):
        try:
            owner.hide = hidden
        except AttributeError:
            pass


def bone_hidden(pose_bone):
    """True when this bone is hidden, whichever version's flag carries it."""
    return bool(getattr(pose_bone, "hide", False) or pose_bone.bone.hide)


def set_bone_flag(pose_bone, name, value):
    """Write one viewport flag on whichever of Bone/PoseBone still carries it.

    The Blender 5 move split them: verified on 5.1.2, ``select`` lives on
    ``PoseBone`` only and ``hide_select`` on ``Bone`` only, while ``hide`` is
    readable on both.  Writing the wrong one raises on the half that is gone
    and is a silent no-op on the half that lingers, so both are attempted and
    a missing attribute is not an error.  Returns True if either write landed.
    """
    written = False
    for owner in (pose_bone, pose_bone.bone):
        try:
            setattr(owner, name, value)
            written = True
        except AttributeError:
            continue
    return written


# The eye-aim chain, from the DNA importer's own EYE_AIM_BONES.  Its GRP_/LOC_
# members are the shapeless helpers stow_helper_bones parks; the rest are the
# real widgets an artist grabs.
EYE_AIM_BONES = (
    "LOC_R_eyeUIDriver", "LOC_L_eyeUIDriver", "LOC_C_eyeUIDriver",
    "LOC_R_eyeDriver", "LOC_L_eyeDriver", "LOC_C_eyeDriver",
    "LOC_R_eyeAimDriver", "LOC_L_eyeAimDriver",
    "LOC_R_eyeAimUp", "LOC_L_eyeAimUp",
    "GRP_convergenceGUI", "GRP_L_eyeAim", "GRP_R_eyeAim",
    "FRM_convergenceGUI", "FRM_convergenceSwitch", "TEXT_convergence",
    "CTRL_C_eyesAim", "CTRL_L_eyeAim", "CTRL_R_eyeAim",
    "CTRL_convergenceSwitch",
)
LOOK_AT_SWITCH = "CTRL_lookAtSwitch"

# The two circles inside the aim frame - one per eye.  The pair an artist reads
# as "the eye target": the chain is sized so THESE land on the character's own
# eye lines, which is the whole of what op_rig.fit_eye_aim solves for.
EYE_AIM_HANDLES = ("CTRL_L_eyeAim", "CTRL_R_eyeAim")

# How far in front of the eyes this character's aim target floats, in metres.
# On the SKELETON: it is a property of the character, and a board can be
# rebuilt.  Absent means "never set" - op_rig.eye_aim_distance then answers
# with the automatic default rather than with 0.
EYE_AIM_DISTANCE_PROP = "mhfrt_eye_aim_distance"


EYE_AIM_COLLECTION = "eye_aim"


def expose_eye_aim(arm_obj):
    """Make the look-at handles visible and findable, in their own collection.

    The bundled asset arrives with the whole chain hidden: Character DNA hides
    it whenever ``CTRL_lookAtSwitch`` is below 0.99, and the file was saved with
    that switch at 0.  Inherited, that is why the control appears to be missing
    on a character nobody ever touched.

    Deliberately NOT the reference's rule.  Its switch is an on/off that only
    acts at >= 0.99; ours is a BLEND - ``op_rig.look_at_weight`` returns 0..1 and
    the solve runs at any weight above 0 (v3.7.0, so the eyes ease onto the
    target instead of snapping).  Gating visibility at 0.99 would hide a control
    that is actively steering the eyes at 0.5, so the switch governs how much
    look-at applies and nothing else.

    Visibility belongs to the ``eye_aim`` bone collection instead - the asset
    already has one holding ``CTRL_C_eyesAim``, and the rest of the handles join
    it, exclusively, so its checkbox in Armature Data > Bone Collections really
    does toggle the look-at rig.  It also puts the name in that list, which is
    most of the reason the control was hard to find: it floats 0.3 m in FRONT of
    the face rather than sitting on the panel with everything else.

    GRP_/LOC_ members are left in the hidden ``helpers`` collection - those are
    the shapeless crosses, and the reference skips them here too.

    Returns the number of handles exposed.  Idempotent.
    """
    armature = getattr(arm_obj, "data", None)
    if arm_obj is None or arm_obj.type != 'ARMATURE' or armature is None:
        return 0
    if not hasattr(armature, "collections"):
        return 0

    coll = next((c for c in _armature_collections(armature)
                 if c.name == EYE_AIM_COLLECTION), None)
    if coll is None:
        coll = armature.collections.new(EYE_AIM_COLLECTION)
    coll.is_visible = True

    exposed = 0
    for name in EYE_AIM_BONES:
        if name.startswith(("GRP_", "LOC_")):
            continue
        pose_bone = arm_obj.pose.bones.get(name)
        if pose_bone is None:
            continue
        bone = pose_bone.bone
        for other in list(bone.collections):
            if other != coll:
                other.unassign(bone)
        if bone.name not in coll.bones:
            coll.assign(bone)
        set_bone_hidden(pose_bone, False)
        exposed += 1
    return exposed


def _armature_collections(armature):
    collections = getattr(armature, "collections_all", None)
    if collections is None:
        collections = getattr(armature, "collections", ())
    return list(collections)


def stow_helper_bones(arm_obj):
    """Park the board's shapeless bones in a hidden bone collection.

    29 of the 435 board bones carry an EMPTY as their custom shape instead of a
    mesh widget, so they draw as black PLAIN_AXES crosses that a click cannot
    select: 17 grouping nodes (``GRP_*``) scattered over the panel, and 12 in
    the eye-aim chain - nine of those sitting ON the face, at the eyeballs,
    because they are the internal driver locators the look-at solves through.

    None of them is a handle.  The look-at control an artist actually grabs is
    ``CTRL_C_eyesAim`` (the rounded frame), ``CTRL_L_eyeAim`` / ``CTRL_R_eyeAim``
    (the two circles inside it) and ``CTRL_convergenceSwitch``, and every one of
    those is drawn by a real mesh widget, so nothing that can be posed moves
    here - a bone must OWN a non-mesh shape to qualify, and a bone with no shape
    at all is never touched (see the loop).

    A hidden bone COLLECTION rather than per-bone ``hide``: it is one checkbox
    in Armature Data > Bone Collections, so the crosses stay one click from
    coming back, and Pose Mode's Alt+H (Reveal Hidden) cannot undo it the way it
    wipes ``bone.hide``.  The bones have to LEAVE their old collection as well
    as join this one - a bone is visible while ANY collection holding it is
    visible, and the asset files all 29 under 'other' next to 197 real handles.

    Returns the number of bones stowed.  Idempotent.
    """
    armature = getattr(arm_obj, "data", None)
    if arm_obj is None or arm_obj.type != 'ARMATURE' or armature is None:
        return 0
    if not hasattr(armature, "collections"):
        return 0        # pre-4.0 bone groups; the board needs 4.0+ anyway

    helpers = next((coll for coll in _armature_collections(armature)
                    if coll.name == HELPER_COLLECTION), None)
    if helpers is None:
        helpers = armature.collections.new(HELPER_COLLECTION)
    helpers.is_visible = False

    stowed = 0
    for pose_bone in arm_obj.pose.bones:
        shape = pose_bone.custom_shape
        # A bone with NO shape at all is deliberately left alone.  "Drawn by an
        # Empty" and "widget missing" are different conditions that look nothing
        # alike: an Empty draws the black cross this function exists to remove,
        # while a bone with no shape falls back to the plain octahedral bone -
        # ugly, but visible and selectable, and on a real control that fallback
        # is the only thing the artist still has to grab.
        #
        # It happens: widgets are fake-user-only and linked to no collection
        # (see WIDGET_OWNER_PROP), so a Purge that includes fake users empties
        # every custom_shape on the board.  Treating that as "helper" would
        # sweep all 435 controls into the hidden collection, the look-at control
        # among them.  Rebuild Control Board is the repair for a board in that
        # state; this function must not make it worse.
        if shape is None or shape.type == 'MESH':
            continue
        bone = pose_bone.bone
        for coll in list(bone.collections):
            if coll != helpers:
                coll.unassign(bone)
        if bone.name not in helpers.bones:
            helpers.assign(bone)
        # The collection owns visibility now; a leftover per-bone hide from an
        # older build would leave its checkbox doing nothing.
        set_bone_hidden(pose_bone, False)
        stowed += 1
    return stowed


def widgets_for_rig(rig_id):
    """The custom-shape objects this rig's board handles are drawn with."""
    if not rig_id:
        return []
    wanted = str(rig_id)
    return [obj for obj in bpy.data.objects
            if str(obj.get(WIDGET_OWNER_PROP) or "") == wanted]


def controls_for_rig(rig_id_prop, rig_id, skel=None):
    """``{template name: control}`` for one rig, whichever layout it uses.

    Bone controls win over an object of the same template name: a scene that
    was migrated keeps its old objects around only as leftovers.

    Pass `skel` where it is known.  The tag scan below dedupes by template
    name, so if two boards share an id every single control resolves to the
    same copy - the one sorting last in ``bpy.data.objects`` - and the other
    character's board drives nothing at all.  The skeleton's OWN board is
    applied last so it wins outright.
    """
    controls = {}
    if not rig_id:
        return controls
    wanted = str(rig_id)
    for obj in bpy.data.objects:
        if str(obj.get(rig_id_prop) or "") != wanted:
            continue
        template = tag(obj, TEMPLATE_PROP)
        if template:
            controls[str(template)] = obj
    for obj in bpy.data.objects:
        if not is_board_armature(obj) or obj.pose is None:
            continue
        if str(obj.get(rig_id_prop) or "") != wanted:
            continue
        for pose_bone in obj.pose.bones:
            template = tag(pose_bone, TEMPLATE_PROP)
            if template:
                controls[str(template)] = pose_bone
    own = own_board_armature(skel, rig_id_prop)
    if own is not None and own.pose is not None:
        for pose_bone in own.pose.bones:
            template = tag(pose_bone, TEMPLATE_PROP)
            if template:
                controls[str(template)] = pose_bone
    return controls


def rest_location(control):
    if control is None:
        return (0.0, 0.0, 0.0)
    raw = tag(control, REST_PROP)
    if raw is None:
        return (0.0, 0.0, 0.0)
    try:
        if len(raw) >= 3:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        pass
    return (0.0, 0.0, 0.0)


def store_rest_location(control):
    if control is None:
        return
    control[REST_PROP] = (float(control.location.x),
                          float(control.location.y),
                          float(control.location.z))


# ---------------------------------------------- the authored board design ---
#
# A handle's LOCATION is the animation channel and has always been recorded
# (:data:`REST_PROP`).  Its rotation, its scale and the size its widget is drawn
# at are not animation - they are how the panel is DRAWN, and on this asset they
# are anything but identity: 192 of the 435 bones carry a pose scale
# (``CTRL_faceGUI`` at 0.005, most ``FRM_`` frames at 4 in Y) and 125 carry a
# pose rotation.  None of that was written down anywhere.
#
# So one Alt+S in Pose Mode over a select-all - Blender's Clear Pose Scale,
# aimed at the face and landing on the panel - flattened all 192 to 1.0 with
# nothing in the file able to say what they used to be.  Alt+R did the same to
# the 125 rotations.  The board looked shredded and no Restore could bring it
# back, because "restore" only ever knew about three bars' locations.
#
# These two tags are that missing rest.  They are stamped once per board, at
# import, straight off the authored asset, which makes the damage recoverable
# after the fact (:func:`restore_design`) as well as preventable (the locks in
# :func:`set_board_locked`).
SHAPE_PROP = "mhfrt_control_shape"       # (qw,qx,qy,qz, sx,sy,sz) of the pose
WIDGET_PROP = "mhfrt_control_widget"     # the custom shape's draw transform

# Bumped when the stamps below change shape, so an older board re-stamps itself
# on the next rescan instead of needing a rebuild.
DESIGN_VERSION_PROP = "mhfrt_board_design"
DESIGN_VERSION = 1


def _basis_shape(control):
    """(quaternion, scale) of this control's own basis, whatever rotation mode.

    Read off ``matrix_basis`` rather than off ``rotation_quaternion`` so a
    control left in Euler or axis-angle mode is captured correctly; both
    PoseBone and Object carry the matrix.
    """
    basis = getattr(control, "matrix_basis", None)
    if basis is None:
        return ((1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    _loc, quat, scale = basis.decompose()
    return (tuple(float(v) for v in quat), tuple(float(v) for v in scale))


def _write_rotation(control, quat):
    """Set a control's rotation in every representation at once.

    Which one Blender reads depends on ``rotation_mode``, and a board that was
    authored in quaternions can be switched to Euler by an artist at any time -
    writing only the one we happened to author in would silently do nothing.
    """
    rotation = Quaternion(quat)
    if rotation.magnitude < 1e-9:
        rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    control.rotation_quaternion = rotation
    control.rotation_euler = rotation.to_euler()
    axis, angle = rotation.to_axis_angle()
    control.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)


def _widget_draw(pose_bone):
    """The custom shape's draw transform, or None when there is nothing to say.

    Blender 4.0 split the old single ``custom_shape_scale`` into a vector plus
    an explicit translation and rotation; all three are part of how the panel
    looks and none of them is reachable from a bone rest.
    """
    scale = getattr(pose_bone, "custom_shape_scale_xyz", None)
    if scale is None:
        return None
    translation = getattr(pose_bone, "custom_shape_translation", (0.0, 0.0, 0.0))
    rotation = getattr(pose_bone, "custom_shape_rotation_euler",
                       (0.0, 0.0, 0.0))
    return tuple(float(v) for v in
                 (*scale, *translation, *rotation))


def store_design(control):
    """Write down how this control is drawn: its rotation, scale and widget."""
    if control is None:
        return
    quat, scale = _basis_shape(control)
    control[SHAPE_PROP] = (*quat, *scale)
    if is_pose_bone(control):
        widget = _widget_draw(control)
        if widget is not None:
            control[WIDGET_PROP] = widget


def store_widget(pose_bone):
    """Re-record just the widget draw transform, leaving the pose stamps alone.

    Used where the widget is deliberately rescaled and the rotation/scale stamps
    are being written by hand (see ``op_rig.bake_panel_bars``).
    """
    if pose_bone is None:
        return
    widget = _widget_draw(pose_bone)
    if widget is not None:
        pose_bone[WIDGET_PROP] = widget


def design_rest(control):
    """((qw,qx,qy,qz), (sx,sy,sz)) this control was authored with, or None."""
    raw = control.get(SHAPE_PROP) if control is not None else None
    try:
        if raw is not None and len(raw) >= 7:
            return (tuple(float(v) for v in raw[:4]),
                    tuple(float(v) for v in raw[4:7]))
    except (TypeError, ValueError):
        pass
    return None


def restore_design(control):
    """Put one control's authored rotation, scale and widget size back.

    Returns True when something actually moved, so a repair can report how much
    it found to fix rather than claiming success on an untouched board.
    """
    rest = design_rest(control)
    if rest is None:
        return False
    quat, scale = rest
    changed = False
    current_quat, current_scale = _basis_shape(control)
    if any(abs(a - b) > 1.0e-6 for a, b in zip(current_quat, quat)):
        _write_rotation(control, quat)
        changed = True
    if any(abs(a - b) > 1.0e-6 for a, b in zip(current_scale, scale)):
        control.scale = scale
        changed = True
    widget = control.get(WIDGET_PROP) if is_pose_bone(control) else None
    if widget is not None and len(widget) >= 9:
        try:
            values = [float(v) for v in widget]
        except (TypeError, ValueError):
            return changed
        for attr, wanted in (("custom_shape_scale_xyz", values[0:3]),
                             ("custom_shape_translation", values[3:6]),
                             ("custom_shape_rotation_euler", values[6:9])):
            current = getattr(control, attr, None)
            if current is None:
                continue
            if any(abs(a - b) > 1.0e-6 for a, b in zip(current, wanted)):
                setattr(control, attr, wanted)
                changed = True
    return changed


# Authored float dust below this reads as a control faintly off rest.
_DUST = 1.0e-5


def snap_dust(control):
    """The control's authored location with float dust snapped to zero.

    The frame correction derives its bounds from where the handle rests, so the
    dust has to go first - otherwise a bound lands a hair off rest and the
    neutral pose reads a few microns instead of exactly zero.
    """
    location = [float(v) for v in control.location]
    for index in range(3):
        if abs(location[index]) < _DUST:
            location[index] = 0.0
    return tuple(location)


def neutral_location(control):
    """The control's authored rest, cleaned so it evaluates to exactly zero.

    Two things would otherwise leave a channel faintly live in the neutral
    pose: float dust around 1e-8 on most handles, and dust just outside a
    handle's own Limit Location frame - the read saturates at the frame edge
    while the stored rest keeps the out-of-range value, so the difference
    between them never reaches zero. Snapping the dust and clamping into the
    frame makes rest and read agree.
    """
    location = [float(v) for v in control.location]
    for channel, (index, _sign) in (
            (name, axis) for name, axis in
            ((c, channel_axis(control, c)) for c in ("tx", "ty", "tz"))
            if axis is not None):
        if abs(location[index]) < _DUST:
            location[index] = 0.0
        limits = channel_limits(control, channel)
        if limits is None:
            continue
        low, high = limits
        if low is not None and location[index] < low:
            location[index] = low
        elif high is not None and location[index] > high:
            location[index] = high
    for index in range(3):
        if abs(location[index]) < _DUST:
            location[index] = 0.0
    return tuple(location)


def channel_axis(control, channel):
    """(location index, sign) this control uses for a GUI channel, or None."""
    table = _BONE_AXES if is_pose_bone(control) else _OBJECT_AXES
    return table.get(channel)


_LIMIT_AXIS_ATTR = "xyz"


def channel_limits(control, channel):
    """(min, max) the control's own Limit Location allows on this channel.

    Returned in the control's LOCAL axis, so the caller saturates before the
    sign/scale conversion.  ``None`` when the control has no limit.
    """
    axis = channel_axis(control, channel)
    if control is None or axis is None:
        return None
    try:
        constraints = control.constraints
    except (AttributeError, ReferenceError):
        return None
    attr = _LIMIT_AXIS_ATTR[axis[0]]
    for constraint in constraints:
        if (constraint.type != 'LIMIT_LOCATION'
                or constraint.owner_space != 'LOCAL'):
            continue
        low = (getattr(constraint, f"min_{attr}")
               if getattr(constraint, f"use_min_{attr}", False) else None)
        high = (getattr(constraint, f"max_{attr}")
                if getattr(constraint, f"use_max_{attr}", False) else None)
        if low is None and high is None:
            return None
        return (low, high)
    return None


def align_limits_to_dna(control, channel_ranges):
    """Make a handle's frame span exactly the range of the DNA channel it draws.

    ``channel_ranges`` maps this control's channel -> (lo, hi) from the DNA.

    The frame is where the read saturates (see :func:`channel_limits`), so it has
    to agree with the DNA in BOTH directions:

    * a frame NARROWER than its channel silently deletes range - the handle
      cannot reach expressions the DNA defines, and neither can an imported clip;
    * a frame WIDER than its channel is worse than it looks. The value saturates
      at the frame edge, which is now past the channel's last segment, and a value
      outside every segment contributes nothing - so the expression collapses to
      neutral at exactly the moment the artist drags to the end of the frame.
      ``CTRL_C_tongue_move`` does this: 23.5 mm of tongue motion at the range
      end, 0.0 mm a little further into its own frame.

    Both are authoring drift between the board template and the DNA revision, and
    the DNA is the authority on how far a channel goes. Only bounds that disagree
    are touched, and only on axes the DNA actually drives - a control whose other
    axis is deliberately pinned keeps that pin.

    Returns the channels whose frame was corrected.
    """
    limit = None
    for constraint in getattr(control, "constraints", ()):
        if (constraint.type == 'LIMIT_LOCATION'
                and constraint.owner_space == 'LOCAL'):
            limit = constraint
            break
    if limit is None:
        return []

    corrected = []
    for channel, (dna_low, dna_high) in channel_ranges.items():
        axis = channel_axis(control, channel)
        if axis is None:
            continue
        index, sign = axis
        rest = float(control.location[index])
        low, high = sorted((rest + dna_low * sign, rest + dna_high * sign))
        attr = _LIMIT_AXIS_ATTR[index]
        changed = False
        for use, name, wanted in (("use_min_", f"min_{attr}", low),
                                  ("use_max_", f"max_{attr}", high)):
            if not getattr(limit, use + attr, False):
                continue
            if abs(getattr(limit, name) - wanted) > 1e-6:
                setattr(limit, name, wanted)
                changed = True
        if changed:
            corrected.append(channel)
    return corrected


def clamp_into_frame(control):
    """Put a handle dragged past its frame back on the frame edge.

    Blender's transform writes wherever the mouse went; the Limit Location
    constraint only holds the DRAWN bone still, so ``location`` keeps the
    out-of-range value.  The rig reads the same saturated value either way,
    but the stored one is what gets keyframed and exported - so it is written
    back, exactly as the DNA importer does.  Returns True if it moved.
    """
    limit = None
    for constraint in getattr(control, "constraints", ()):
        if (constraint.type == 'LIMIT_LOCATION'
                and constraint.owner_space == 'LOCAL'):
            limit = constraint
            break
    if limit is None:
        return False
    location = [float(v) for v in control.location]
    clamped = list(location)
    for index, attr in enumerate(_LIMIT_AXIS_ATTR):
        if getattr(limit, f"use_min_{attr}", False):
            clamped[index] = max(clamped[index], getattr(limit, f"min_{attr}"))
        if getattr(limit, f"use_max_{attr}", False):
            clamped[index] = min(clamped[index], getattr(limit, f"max_{attr}"))
    if clamped == location:
        return False
    control.location = clamped
    return True


def channel_source(control, channel):
    """Everything the per-frame read needs, resolved once per rig cache.

    ``(control, location index, rest value, sign, limits)``; a control with no
    such channel gets a zero-sign entry so the read still costs one tuple
    unpack and returns 0.0.
    """
    if control is None:
        return (None, 0, 0.0, 0.0, None)
    axis = channel_axis(control, channel)
    if axis is None:
        return (control, 0, 0.0, 0.0, None)
    index, sign = axis
    rest = rest_location(control)[index]
    return (control, index, rest, sign, channel_limits(control, channel))


def channel_value(source, divisor, evaluated=None):
    """`evaluated` is {bone name: evaluated pose bone}, or None for originals.

    The originals are the right thing to read in the viewport - Blender copies
    each evaluated pose back onto them, which is how the panel and this read
    see an animated board for free.  Nothing is copied back from a RENDER
    depsgraph, though, so during a render the originals still hold whatever the
    viewport last left there and the board has to be read from the render's own
    copy instead. See op_rig._evaluated_controls.
    """
    control, index, rest, sign, limits = source
    if control is None or sign == 0.0:
        return 0.0
    if evaluated is not None:
        control = evaluated.get(control.name, control)
    value = control.location[index]
    if limits is not None:
        low, high = limits
        if low is not None and value < low:
            value = low
        elif high is not None and value > high:
            value = high
    return (value - rest) * sign / divisor


def set_channel_value(control, channel, value, divisor):
    """Move a control so its GUI channel reads ``value``. Returns True if set."""
    axis = channel_axis(control, channel)
    if control is None or axis is None:
        return False
    index, sign = axis
    rest = rest_location(control)[index]
    location = list(control.location)
    location[index] = rest + value * divisor * sign
    control.location = location
    return True


def rest_pose(control, orientation=False, layout=False):
    """Put one control back on its rest location. Returns True if it moved.

    ``orientation`` also sends rotation and scale back to their AUTHORED values
    (:func:`restore_design`), falling back to identity on a board built before
    those were stamped.  Only the eye-aim handles need it: a panel slider is a
    translation and nothing else, but an aim handle is grabbed in the viewport
    like any object, and rotating the frame swings the two circles inside it -
    so a location-only reset would leave the gaze off-centre with every handle
    apparently at rest.

    Authored, not identity, because identity is a lie on this board: 192 of its
    bones carry a pose scale and 125 a pose rotation, and clearing those is
    exactly the damage the design stamps exist to undo.  The aim handles happen
    to be authored at identity, so this changes nothing for today's callers and
    stops being wrong the moment one of them is not.

    A LAYOUT control is skipped unless ``layout`` says otherwise.  Every caller
    here means "give me the neutral face": clearing an imported clip, baking
    ARKit shapes, lining a clip up for export.  Layout carries no expression, so
    resetting it buys that caller nothing and costs the artist the arrangement
    they made - ``remove_face_animation`` in particular used to fling all three
    panels back beside the head.  :func:`reset_layout` is the one place that
    passes ``layout=True``, because there it IS the job.

    "Layout" means every bone the DNA does not drive, not just the three bars
    (see :func:`is_layout_control`).  Narrower than that and Export or Remove
    Face Animation would quietly undo a panel redesign and switch Follow Head
    back off, neither of which is anything to do with a neutral face.
    """
    if control is None or (not layout and is_layout_control(control)):
        return False
    rest = tag(control, REST_PROP)
    if rest is None:
        return False
    try:
        control.location = tuple(float(v) for v in rest)
    except (TypeError, ValueError):
        return False
    if orientation and not restore_design(control):
        if design_rest(control) is None:
            _write_rotation(control, (1.0, 0.0, 0.0, 0.0))
            control.scale = (1.0, 1.0, 1.0)
    return True


# ------------------------------------------------------------ panel layout ---

def control_template(control):
    """This control's board template name - its bone name on a bone board."""
    if control is None:
        return ""
    try:
        name = tag(control, TEMPLATE_PROP)
        return str(name) if name else str(control.name)
    except (AttributeError, ReferenceError):
        return ""


def is_layout_handle(control):
    """True for one of the three panel-position bars (:data:`LAYOUT_HANDLES`)."""
    return control_template(control) in LAYOUT_HANDLES


_DRIVEN_TEMPLATES = None


def driven_templates():
    """The board templates the DNA actually drives - the ANIMATION controls.

    Everything else on the board (197 ``FRM_`` frames, 31 ``TEXT_`` labels, 20
    ``GRP_`` nodes, the three panel bars and the handful of behaviour switches)
    is LAYOUT: moving it changes how the panel looks and cannot change the face,
    because ``channel_value`` only ever reads a template that appears in the
    DNA's ``gui_names``.

    That split is the whole design of the redesign feature.  Layout may be
    dragged, rotated, scaled and remembered wholesale; an animation control's
    location is the artist's pose and is never restored behind their back.

    Empty when the rig-logic data is missing, which makes every caller fall back
    to "treat the board as all layout" - conservative in the right direction:
    the redesign tools then simply do less, rather than moving a pose.
    """
    global _DRIVEN_TEMPLATES
    if _DRIVEN_TEMPLATES is None:
        try:
            from . import riglogic
            names = riglogic.meta()["gui_names"]
        except Exception:                       # noqa: BLE001 - data may be gone
            return frozenset()
        _DRIVEN_TEMPLATES = frozenset(
            name.rpartition(".")[0] for name in names if "." in name)
    return _DRIVEN_TEMPLATES


def is_driven_control(control):
    """True when moving this control moves the FACE rather than the panel."""
    return control_template(control) in driven_templates()


# The look-at handles.  Not DNA templates - the runtime turns where they END UP
# into ordinary eye channel values (op_rig._eye_aim_gui) - but they steer the
# face all the same, so every "neutral face" path has to reset them.
AIM_CONTROLS = ("CTRL_C_eyesAim", "CTRL_L_eyeAim", "CTRL_R_eyeAim")


# What the artist calls a "controller": every CTRL_ bone on the panel. There
# are 178 of them and the DNA drives only 161, so a rule built on
# :func:`driven_templates` alone left 17 dots on screen during a redesign -
# CTRL_rigLogicSwitch, CTRL_lookAtSwitch, the two follow-head switches,
# CTRL_C_eye (the centre-eye master, which the runtime adds on top of the
# per-eye channels rather than reading as a channel of its own), the eye-aim
# handles, and a handful of mouth controls the bundled DNA has no channel for.
CONTROL_PREFIX = "CTRL_"


def is_solo_hidden(control):
    """True for a bone the redesign mode puts away.

    Every controller, plus anything the DNA drives whatever it is called.  The
    three panel bars are the exception: they are named ``CTRL_`` too, and they
    are the thing being placed.
    """
    template = control_template(control)
    if template in LAYOUT_HANDLES:
        return False
    return (template.startswith(CONTROL_PREFIX)
            or template in driven_templates())


def is_layout_control(control):
    """True when this control is the PANEL rather than the face.

    Falls back to "only the three bars" when the rig-logic data is missing and
    :func:`driven_templates` therefore knows nothing: without that fallback
    every control would read as layout and the neutral-pose paths would quietly
    stop resetting anything at all.
    """
    template = control_template(control)
    if template in AIM_CONTROLS:
        return False
    driven = driven_templates()
    if not driven:
        return template in LAYOUT_HANDLES
    return template not in driven


def stamp_design(arm_obj, force=False):
    """Write the authored rotation/scale/widget of every bone on this board.

    Idempotent and versioned: a board already carrying the current stamps is
    left alone, so this can sit on the rescan path and cost nothing.  ``force``
    re-stamps from the CURRENT pose, which is what a rebuild wants and what a
    repair must never do.

    Returns the number of bones stamped.

    The one thing that must not happen here is stamping a board that is already
    broken - that would enshrine the damage as the design.  Hence the version
    gate: the stamps are written once, at import, off the authored asset, and a
    board that missed that (built before v4.12.0) is stamped on its first rescan
    from whatever it looks like then.  That is the best available answer for an
    existing scene, and it is why Rebuild Control Board stays the repair of last
    resort for a board that was already wrecked before it was ever stamped.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return 0
    if not force and int(arm_obj.get(DESIGN_VERSION_PROP, 0) or 0) >= DESIGN_VERSION:
        return 0
    stamped = 0
    for pose_bone in arm_obj.pose.bones:
        store_design(pose_bone)
        stamped += 1
    if arm_obj.library is None:
        arm_obj[DESIGN_VERSION_PROP] = DESIGN_VERSION
    return stamped


# Where the folded bar transform is remembered, so a layout saved before the
# fold can still be read back (its numbers are in the OLD frame).
BAR_BAKE_PROP = "mhfrt_board_bars_baked"


def bar_bake(arm_obj):
    """{template: the basis that was folded into that bar's rest}."""
    raw = arm_obj.get(BAR_BAKE_PROP) if arm_obj is not None else None
    if not raw:
        return {}
    try:
        stored = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    out = {}
    for template, values in stored.items():
        matrix = _matrix_from(values)
        if matrix is not None:
            out[str(template)] = matrix
    return out


def bars_are_baked(arm_obj):
    return bool(bar_bake(arm_obj))


def plan_bar_bake(arm_obj):
    """What to fold out of each panel bar, or {} when there is nothing to do.

    The three bars are the only bones on the board an artist ever grabs on
    purpose, and two of them are authored at a pose scale that makes no sense
    to look at: ``CTRL_faceGUI`` at 0.005 and ``CTRL_faceAndEyesAimFollowHeadGUI``
    at 0.5, one of them carrying a location of (34, 139, 3) as well.  Selecting
    a bar and reading 0.005 in the N-panel tells the artist nothing, and it is
    the reason Clear Scale on a bar was destructive at all: 1.0 was never the
    right answer for it.

    Folding that into the REST makes 1.0 the right answer.  It is only possible
    because of two facts about these three bones specifically:

    * their scale is UNIFORM (0.005, 0.5, 1.0).  153 bones elsewhere on the
      board carry a non-uniform one - (1, 4, 1), (0.5, 2, 0.2) - and an edit
      bone has head, tail and roll and no way to hold that;
    * each bar has exactly ONE child, and it is a ``FRM_`` frame.  Folding a
      scale out of a bone pushes it into its direct children, and a frame is
      neither a DNA channel nor a Limit Location owner - so not one GUI value
      moves.  The whole rest of the subtree is untouched.

    The AUTHORED basis is folded, not the current one, so a board whose bars the
    artist has already dragged keeps its arrangement: the bar's live pose is
    restated in the new frame afterwards.
    """
    plan = {}
    for template, pose_bone in layout_handle_bones(arm_obj):
        rest = design_rest(pose_bone)
        if rest is None:
            continue
        quat, scale = rest
        if (max(scale) - min(scale)) > 1.0e-6:
            continue                    # non-uniform: not representable
        location = tag(pose_bone, REST_PROP) or (0.0, 0.0, 0.0)
        try:
            translation = tuple(float(v) for v in location[:3])
        except (TypeError, ValueError):
            translation = (0.0, 0.0, 0.0)
        basis = (Matrix.Translation(translation)
                 @ Quaternion(quat).to_matrix().to_4x4()
                 @ Matrix.Diagonal((*scale, 1.0)))
        if all(abs(basis[row][col] - (1.0 if row == col else 0.0)) < 1.0e-9
               for row in range(4) for col in range(4)):
            continue                    # already identity; nothing to fold
        plan[template] = basis
    return plan


def note_bar_bake(arm_obj, plan):
    """Remember what was folded, merging with anything folded before."""
    if arm_obj is None or arm_obj.library is not None or not plan:
        return
    merged = bar_bake(arm_obj)
    for template, basis in plan.items():
        previous = merged.get(template)
        merged[template] = basis if previous is None else previous @ basis
    arm_obj[BAR_BAKE_PROP] = json.dumps(
        {template: [float(v) for row in matrix for v in row]
         for template, matrix in merged.items()},
        separators=(",", ":"))


def restore_board_design(arm_obj):
    """Undo an Alt+S / Alt+R over the whole board. Returns how many bones moved.

    Rotation, scale and widget size only - every bone, animation controls
    included, because none of those three is ever read as a pose value.  A
    control's LOCATION is deliberately untouched: that is the artist's
    expression, and a repair that cleared it would trade one disaster for
    another.  Panel POSITIONS come back through :func:`apply_layout` instead.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return 0
    return sum(int(restore_design(pose_bone))
               for pose_bone in arm_obj.pose.bones)


def layout_handle_bones(arm_obj):
    """[(template, pose bone)] for the panel handles this board actually has.

    In :data:`LAYOUT_HANDLES` order, so the UI and the selection helper always
    lead with the main panel rather than with whatever ``pose.bones`` hands
    back first.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return []
    found = {}
    for pose_bone in arm_obj.pose.bones:
        template = control_template(pose_bone)
        if template in LAYOUT_HANDLES and template not in found:
            found[template] = pose_bone
    return [(name, found[name]) for name in LAYOUT_HANDLES if name in found]


def capture_layout(arm_obj):
    """Where this board's panels sit now, as a plain JSON-able dict, or None.

    Two parts, because the artist has two ways to move a panel and both matter:

    * ``handles`` - each bar's pose location, which is BONE-local, so it is
      exact and means the same thing whatever the board object was moved,
      scaled or parented to since;
    * ``object`` - the armature's own ``matrix_basis``, which is what
      ``_place_bone_board`` computes when it sits the board beside the head.
      Dragging the whole panel across the scene is done by moving the object,
      not a bone, and a layout that forgot that would restore half the change.

    A third part since v4.12.0, ``bones`` and ``shape``, because three bars were
    never the whole layout - an artist redesigning the panel drags frames,
    labels and groups around, and none of that was being remembered by anything.
    Only what DIFFERS from the authored design is written (see
    :func:`stamp_design`), so a board nobody has redesigned still saves the same
    handful of numbers it always did, and the file only grows with real work:

    * ``bones``  - the full ``matrix_basis`` of each LAYOUT bone that has moved
      (frames, labels, groups, the bars).  Layout carries no expression, so its
      location is safe to remember and safe to put back.
    * ``shape``  - rotation and scale of each ANIMATION control that has been
      changed.  Never its location: that is the artist's pose, and a Restore
      Layout that quietly re-posed the face would be the worst kind of bug.
    """
    if (arm_obj is None or arm_obj.type != 'ARMATURE'
            or getattr(arm_obj, "pose", None) is None):
        return None
    bones = layout_handle_bones(arm_obj)
    handles = {template: [float(v) for v in pose_bone.location]
               for template, pose_bone in bones}
    if not handles:
        return None
    driven = driven_templates()
    layout_bones = {}
    shapes = {}
    for pose_bone in arm_obj.pose.bones:
        template = control_template(pose_bone)
        if template in driven:
            shape = _changed_shape(pose_bone)
            if shape is not None:
                shapes[template] = shape
        elif _moved_from_design(pose_bone):
            layout_bones[template] = [
                round(float(v), 7)
                for row in pose_bone.matrix_basis for v in row]
    return {
        "version": 2,
        # What one stored length means, so a layout saved on a big head lands
        # correctly on a small one. Bone-local units scale with the board's
        # placement (see BOARD_PLACEMENT_PROP), and a .mhfrt is meant to be
        # carried between characters - without this, moving a panel 30 units on
        # a 1.2-scale board would move it 30 units on a 0.8-scale one, which is
        # a different place on the face.
        "unit": placement_scale(arm_obj),
        # Whether the bars' authored transform has been folded into their rest.
        # A layout written before that fold describes them in the OLD frame, and
        # applying it verbatim would put the main panel 200x away from the head.
        "bars_baked": bars_are_baked(arm_obj),
        "handles": handles,
        # The full basis as well, so a panel that was SCALED or ROTATED comes
        # back that way too. Location is still written on its own because an
        # older add-on version reads only that, and half a layout beats none.
        "handles_full": {
            template: [float(v) for row in pose_bone.matrix_basis for v in row]
            for template, pose_bone in bones
        },
        "bones": layout_bones,
        "shape": shapes,
        "object": [float(v) for row in arm_obj.matrix_basis for v in row],
    }


def _design_matrix(control):
    """The ``matrix_basis`` this control was authored with, or None."""
    rest = design_rest(control)
    if rest is None:
        return None
    quat, scale = rest
    location = tag(control, REST_PROP)
    try:
        translation = (tuple(float(v) for v in location[:3])
                       if location is not None else (0.0, 0.0, 0.0))
    except (TypeError, ValueError):
        translation = (0.0, 0.0, 0.0)
    return (Matrix.Translation(translation)
            @ Quaternion(quat).to_matrix().to_4x4()
            @ Matrix.Diagonal((*scale, 1.0)))


def _moved_from_design(control):
    """True when this layout bone is not where the asset put it."""
    authored = _design_matrix(control)
    if authored is None:
        # Nothing to compare against: an unstamped board has to save everything
        # rather than silently save nothing.
        return True
    current = control.matrix_basis
    return any(abs(current[row][col] - authored[row][col]) > 1.0e-6
               for row in range(4) for col in range(4))


def _changed_shape(control):
    """(qw,qx,qy,qz,sx,sy,sz) when it differs from the authored one, else None."""
    rest = design_rest(control)
    quat, scale = _basis_shape(control)
    if rest is not None:
        authored_quat, authored_scale = rest
        if (all(abs(a - b) <= 1.0e-6 for a, b in zip(quat, authored_quat))
                and all(abs(a - b) <= 1.0e-6
                        for a, b in zip(scale, authored_scale))):
            return None
    elif (all(abs(a - b) <= 1.0e-6
              for a, b in zip(quat, (1.0, 0.0, 0.0, 0.0)))
          and all(abs(v - 1.0) <= 1.0e-6 for v in scale)):
        return None
    return [round(float(v), 7) for v in (*quat, *scale)]


def _matrix_from(values):
    """A 4x4 from 16 row-major numbers, or None.

    Deliberately duck-typed rather than ``isinstance(values, (list, tuple))``:
    the same matrices come back out of a custom property as an
    ``IDPropertyArray``, which is neither - reading a stored placement as "not a
    matrix" silently meant "this board was never placed", and every re-place
    then baked a second placement on top of the first.
    """
    try:
        if len(values) != 16:
            return None
        return Matrix([[float(values[row * 4 + col]) for col in range(4)]
                       for row in range(4)])
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def placement_matrix(arm_obj):
    """The transform baked into this board's rest, or the identity.

    The identity also means "this board predates the flattening and still
    carries its placement on the object", which is exactly how the callers below
    want to treat it: nothing to undo, and its bone-local units are the authored
    ones.
    """
    if arm_obj is None:
        return Matrix.Identity(4)
    return _matrix_from(arm_obj.get(BOARD_PLACEMENT_PROP)) or Matrix.Identity(4)


def store_placement(arm_obj, matrix):
    if arm_obj is None or arm_obj.library is not None:
        return
    arm_obj[BOARD_PLACEMENT_PROP] = [float(v) for row in matrix for v in row]


def placement_scale(arm_obj):
    """How many armature units one AUTHORED board unit is on this board.

    1.0 on a board that was never flattened, and the head-size ratio on one that
    was.  This is the number ``dna_gui_scale`` has to agree with, and the number
    a saved layout is converted through when it lands on another character.
    """
    scale = placement_matrix(arm_obj).to_scale()
    value = (abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3.0
    return value if value > 1.0e-9 else 1.0


def is_flattened(arm_obj):
    """True when the placement lives in the rest and the object is at identity."""
    return arm_obj is not None and BOARD_PLACEMENT_PROP in arm_obj.keys()


def rescale_channels(arm_obj, factor, names=None):
    """Restate every bone-local length on this board in `factor`-times units.

    Three things measure a length in armature units and all three have to move
    together, or the board stops agreeing with itself:

    * each handle's ``location`` - the live channel value;
    * its stored rest (:data:`REST_PROP`), which the value is measured FROM;
    * its Limit Location frame, which is where the value saturates.

    Rotation, scale and widget size are ratios, not lengths, so they are not
    touched.  Returns the number of bones restated.

    ``names`` restricts it to part of the board - the eye-aim chain is resized
    on its own, to put its circles on the character's eyes, and the panel it
    shares an armature with must not move with it.

    The caller is responsible for moving ``dna_gui_scale`` by the same factor;
    the two together are what make this invisible to the rig.  A partial
    rescale needs no such adjustment as long as the bones it touches own no DNA
    channel, which is exactly why only the aim chain may be passed here.
    """
    if (arm_obj is None or getattr(arm_obj, "pose", None) is None
            or abs(factor - 1.0) < 1.0e-12):
        return 0
    if names is None:
        bones = arm_obj.pose.bones
    else:
        bones = [pb for pb in (arm_obj.pose.bones.get(n) for n in names)
                 if pb is not None]
    changed = 0
    for pose_bone in bones:
        pose_bone.location = [v * factor for v in pose_bone.location]
        rest = tag(pose_bone, REST_PROP)
        if rest is not None:
            try:
                pose_bone[REST_PROP] = tuple(float(v) * factor
                                             for v in rest[:3])
            except (TypeError, ValueError):
                pass
        for constraint in pose_bone.constraints:
            if (constraint.type != 'LIMIT_LOCATION'
                    or constraint.owner_space != 'LOCAL'):
                continue
            for axis in _LIMIT_AXIS_ATTR:
                for name in (f"min_{axis}", f"max_{axis}"):
                    setattr(constraint, name,
                            getattr(constraint, name) * factor)
        changed += 1
    return changed


def apply_layout(arm_obj, data, object_transform=True):
    """Put the panels back where `data` says. Returns how many handles moved.

    Unknown handle names are ignored rather than refused: a layout saved by a
    later add-on that adds a fourth panel still restores the three this board
    has.

    A format-2 layout (``bones``/``shape``) restores the whole redesign, and
    every bone it does NOT mention is put back on the authored design first -
    otherwise "restore" would leave behind whatever the artist has moved since
    the save, which is the one thing the word cannot mean.  Animation controls
    keep their location throughout: a layout restores the panel, never the pose.
    """
    if (arm_obj is None or getattr(arm_obj, "pose", None) is None
            or not isinstance(data, dict)):
        return 0
    handles = data.get("handles")
    if not isinstance(handles, dict):
        return 0

    # Every stored length is converted into THIS board's units. A layout with no
    # "unit" was written before the placement moved into the rest, so its
    # numbers are in authored units - which is what 1.0 means here, and what
    # makes the conversion correct for those too.
    try:
        saved_unit = float(data.get("unit", 1.0)) or 1.0
    except (TypeError, ValueError):
        saved_unit = 1.0
    unit = placement_scale(arm_obj) / saved_unit
    # A bar's basis saved before the fold is expressed in the frame the fold
    # removed, so it is put through the same change of frame here. Only bars are
    # affected; nothing else on the board was folded.
    reframe = ({} if data.get("bars_baked")
               else {template: _reframe_bar(matrix)
                     for template, matrix in bar_bake(arm_obj).items()})

    def convert(matrix, template):
        matrix = _in_units(matrix, unit)
        change = reframe.get(template)
        if matrix is None or change is None:
            return matrix
        left, right = change
        return left @ matrix @ right

    bones = data.get("bones")
    shapes = data.get("shape")
    if isinstance(bones, dict) or isinstance(shapes, dict):
        bones = bones if isinstance(bones, dict) else {}
        shapes = shapes if isinstance(shapes, dict) else {}
        driven = driven_templates()
        for pose_bone in arm_obj.pose.bones:
            template = control_template(pose_bone)
            if template in driven:
                shape = shapes.get(template)
                if isinstance(shape, (list, tuple)) and len(shape) >= 7:
                    try:
                        _write_rotation(pose_bone,
                                        [float(v) for v in shape[:4]])
                        pose_bone.scale = [float(v) for v in shape[4:7]]
                    except (TypeError, ValueError):
                        pass
                else:
                    restore_design(pose_bone)
                continue
            matrix = convert(_matrix_from(bones.get(template)), template)
            if matrix is None:
                matrix = _design_matrix(pose_bone)
            if matrix is not None:
                pose_bone.matrix_basis = matrix

    full = data.get("handles_full")
    full = full if isinstance(full, dict) else {}
    applied = 0
    for template, pose_bone in layout_handle_bones(arm_obj):
        # Prefer the whole basis (carries scale and rotation); fall back to the
        # location a layout saved before those were remembered.
        matrix = convert(_matrix_from(full.get(template)), template)
        if matrix is not None:
            pose_bone.matrix_basis = matrix
            applied += 1
            continue
        location = handles.get(template)
        try:
            matrix = convert(
                Matrix.Translation((float(location[0]), float(location[1]),
                                    float(location[2]))), template)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        pose_bone.location = matrix.translation
        applied += 1

    # A flattened board's object transform is not the layout's business: the
    # placement lives in its rest and the object is deliberately at identity, so
    # writing a matrix saved before the flattening would move the whole panel
    # off the head by the amount that is already baked into the bones.
    matrix = _matrix_from(data.get("object"))
    if object_transform and matrix is not None and not is_flattened(arm_obj):
        arm_obj.matrix_basis = matrix
    return applied


def _reframe_bar(folded):
    """(left, right) that restate a pre-fold bar basis in the folded frame.

    The fold is not one change of basis but two, and they act on opposite sides:

    * its RIGID part - the authored translation and rotation - moved into the
      bone's rest, so it comes off the LEFT: whatever the artist did is now
      measured from the new rest instead of the old one;
    * its SCALE moved down into the child frame, so it comes off the RIGHT: the
      bar's own transform simply stops carrying it.

    Doing both on the left (the obvious first guess) divides the artist's
    translation by the authored scale - for ``CTRL_faceGUI`` that is a factor of
    200, and their panel lands somewhere outside the scene.
    """
    scale = folded.to_scale().x or 1.0
    rigid = (Matrix.Translation(folded.translation)
             @ folded.to_quaternion().to_matrix().to_4x4())
    return (rigid.inverted_safe(),
            Matrix.Diagonal((1.0 / scale, 1.0 / scale, 1.0 / scale, 1.0)))


def _in_units(matrix, unit):
    """The same basis with its TRANSLATION restated in this board's units.

    Rotation and scale are ratios and carry over as they are; only the offset
    is a length.
    """
    if matrix is None or abs(unit - 1.0) < 1.0e-12:
        return matrix
    out = matrix.copy()
    out.translation = matrix.translation * unit
    return out


# Set on the board armature while the artist is placing panels. The locks are
# the enforcement; this is the MODE, and it has to be stored rather than
# inferred so a rescan (or reopening the file mid-placement) can tell "the
# artist is dragging right now" from "this board predates the locks".
LAYOUT_MODE_PROP = "mhfrt_layout_placing"


def _set_locks(owner, location=None, rotation=None, scale=None):
    """Write the lock flags that differ. Returns True if any did.

    ``None`` means "leave that one as it is" - a control's LOCATION lock is the
    difference between a panel bar and an animation handle, so it is set
    explicitly per bone rather than swept along with the rest.
    """
    changed = False
    if location is not None:
        state = (location,) * 3
        if tuple(owner.lock_location) != state:
            owner.lock_location = state
            changed = True
    if rotation is not None:
        state = (rotation,) * 3
        if (tuple(owner.lock_rotation) != state
                or bool(owner.lock_rotation_w) != bool(rotation)):
            owner.lock_rotation = state
            owner.lock_rotation_w = bool(rotation)
            changed = True
    if scale is not None:
        state = (scale,) * 3
        if tuple(owner.lock_scale) != state:
            owner.lock_scale = state
            changed = True
    return changed


def set_board_locked(arm_obj, locked):
    """Lock (or free) everything on this board that is not animation.

    This is what makes a saved layout STAY put: a locked channel is refused by
    Blender's own clear operators, so Alt+G, Alt+R, Alt+S and Clear Transform
    over a whole selection all leave the panel exactly where the artist put it
    - verified on all four. It doubles as the mode itself, since an unlocked
    handle is precisely one the artist is allowed to drag.

    Three tiers, because the board has three kinds of thing on it:

    * the three panel bars - location, rotation AND scale, since none of the
      three is animation;
    * every other bone - rotation and scale only.  Location is left free
      because on an animation control that IS the expression, and on a frame or
      a label leaving it free costs nothing that Restore Layout cannot undo.
      This tier is the one that matters: 192 bones carry an authored pose scale
      and 125 an authored rotation, and a single Alt+S over a select-all used to
      flatten all of them at once;
    * the eye-aim chain - untouched.  Its handles are a 3D control, not a flat
      panel: the look-at solve reads where ``CTRL_L_eyeAim`` and
      ``CTRL_R_eyeAim`` END UP, so rotating or scaling their parent frame is a
      legitimate way to aim and converge the gaze, and locking it would take
      away a control the rig genuinely uses.

    The board OBJECT is locked too, and it has to be: the bone locks say nothing
    about the armature itself, so Alt+S in OBJECT mode threw away the
    head-size placement scale that ``_place_bone_board`` computed and left the
    panel the wrong size beside the head with no way back.

    Returns how many owners were changed.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return 0
    bars = {template for template, _ in layout_handle_bones(arm_obj)}
    changed = 0
    for pose_bone in arm_obj.pose.bones:
        template = control_template(pose_bone)
        if template in EYE_AIM_BONES:
            continue
        location = locked if template in bars else None
        changed += int(_set_locks(pose_bone, location=location,
                                  rotation=locked, scale=locked))
    changed += int(_set_locks(arm_obj, location=locked, rotation=locked,
                              scale=locked))
    if arm_obj.library is None:
        if locked:
            if LAYOUT_MODE_PROP in arm_obj.keys():
                del arm_obj[LAYOUT_MODE_PROP]
        else:
            arm_obj[LAYOUT_MODE_PROP] = True
    return changed


def layout_unlocked(arm_obj):
    """True while the artist is placing panels - i.e. the bars can be dragged."""
    return bool(arm_obj is not None and arm_obj.get(LAYOUT_MODE_PROP))


def ensure_board_locked(arm_obj):
    """Lock a board that is not being redesigned right now.

    Called on every full rescan so a board built before the locks existed -
    or one an older add-on version left free - is protected without waiting
    for a rebuild. A board mid-placement is left alone: re-locking under the
    artist's cursor would cancel the very thing they are doing.
    """
    if arm_obj is None or layout_unlocked(arm_obj):
        return 0
    changed = set_board_locked(arm_obj, True)
    return changed + ensure_bars_unselectable(arm_obj)


# Where the solo remembers what it hid, so leaving the mode reveals exactly the
# bones it took away and nothing the artist had hidden themselves.
LAYOUT_SOLO_PROP = "mhfrt_layout_solo"


def _solo_state(arm_obj):
    """What the solo recorded on the way in, as a dict. ``{}`` when not soloed.

    An empty dict means "this board is not in the redesign mode", which is why
    every caller can tell the two apart: an entry that IS in the mode always
    wrote at least the two keys.
    """
    raw = arm_obj.get(LAYOUT_SOLO_PROP) if arm_obj is not None else None
    if not raw:
        return {}
    try:
        state = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    # A solo recorded before the hide_select flags rode along is a bare list.
    if isinstance(state, list):
        return {"hidden": [str(n) for n in state], "unselectable": []}
    return state if isinstance(state, dict) else {}


def solo_hidden_bones(arm_obj):
    """Every control this board is currently hiding for a redesign.

    Read off the BONES, not off the record - it is what an escape hatch needs:
    the artist is looking at a panel with no controllers on it and wants them
    back, whatever the record does or does not say.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return []
    return [pose_bone for pose_bone in arm_obj.pose.bones
            if is_solo_hidden(pose_bone) and bone_hidden(pose_bone)]


def reveal_controls(arm_obj):
    """Show every controller this board is hiding. Returns how many appeared.

    The way out of a panel whose handles will not come back, whatever put them
    away.  Deliberately blunt: it ignores the solo record entirely, because the
    one situation it exists for is the record being wrong.
    """
    shown = 0
    for pose_bone in solo_hidden_bones(arm_obj):
        set_bone_hidden(pose_bone, False)
        shown += 1
    return shown


def set_layout_solo(arm_obj, on):
    """Show only the panel's own structure while it is being redesigned.

    Placing a panel means looking at where its FRAMES and its grab bars are, and
    435 bones of handles on top of that is 161 unrelated dots covering the very
    edges being lined up.  So the animation controls (:func:`driven_templates`)
    step out of the way for the duration - the frames, the labels and the three
    bars stay, which is the panel's skeleton.

    It also carries the ``hide_select`` flags of the panel bars.  The asset
    ships 263 of its bones unselectable - every frame, every group and all
    three bars - which is what made them safe from a stray Alt+S in the first
    place: a select-all cannot reach a bone a click cannot reach.  Grab Panel
    Handles has to clear that flag to hand the bars to the artist, and until
    v4.12.0 it never put it back, so from the first time the panel was ever
    moved, ``CTRL_faceGUI`` was permanently select-able - and one Clear Scale
    over the whole board then threw away its authored 0.005 and blew the panel
    up 200-fold.  Recorded on the way in, restored on the way out.

    What was hidden BEFORE is recorded too, so leaving the mode restores that
    state rather than revealing whatever the artist had deliberately put away.
    Returns how many bones changed visibility.

    What the record stores is what was ALREADY away when the mode was entered -
    ``kept`` - and leaving reveals every hidden controller that is not on that
    list.  The obvious alternative, listing what we hid and revealing exactly
    that, is what shipped until v4.18.0 and it had one fatal property: the list
    could go missing while the bones stayed hidden, and then nothing in the
    add-on could bring them back.

    That is not hypothetical.  Grab Panel Handles is a button an artist presses
    AGAIN mid-redesign - to get the bars re-selected after clicking something
    else - and the second press found all 161 controls already hidden, built an
    empty "what I hid" list, and overwrote the real one with it.  Save Layout
    then dutifully restored nothing: the panel came back locked with its
    controllers gone for good.  The same press recorded an empty
    ``unselectable``, which would also have left the bars permanently
    select-able - the exact Alt+S hazard this function exists to close.

    Stated the other way round, re-entering is harmless: ``kept`` is written
    once, on the entry that found the board untouched, and a board already in
    the mode keeps the record it has.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return 0
    if on:
        previous = _solo_state(arm_obj)
        # Only the FIRST entry sees the board as the artist left it; after that
        # what is hidden is our doing and would be recorded as theirs.
        kept = ([str(name) for name in (previous.get("kept") or ())]
                if previous
                else [pose_bone.name
                      for pose_bone in solo_hidden_bones(arm_obj)])
        unselectable = (
            [str(name) for name in (previous.get("unselectable") or ())]
            if previous
            else [pose_bone.name
                  for _template, pose_bone in layout_handle_bones(arm_obj)
                  if getattr(pose_bone.bone, "hide_select", False)])
        newly = 0
        for pose_bone in arm_obj.pose.bones:
            if not is_solo_hidden(pose_bone) or bone_hidden(pose_bone):
                continue
            set_bone_hidden(pose_bone, True)
            newly += 1
        if arm_obj.library is None:
            arm_obj[LAYOUT_SOLO_PROP] = json.dumps(
                {"kept": kept, "unselectable": unselectable},
                separators=(",", ":"))
        return newly

    state = _solo_state(arm_obj)
    if arm_obj.library is None and LAYOUT_SOLO_PROP in arm_obj.keys():
        del arm_obj[LAYOUT_SOLO_PROP]
    # A record written before v4.18.0 has no `kept`, so nothing is held back and
    # every hidden controller comes out. On a board that is in this mode that is
    # the right answer anyway - they were hidden by the mode - and it is what
    # un-sticks a file whose old record was already lost.
    kept = {str(name) for name in (state.get("kept") or ())}
    shown = 0
    for pose_bone in solo_hidden_bones(arm_obj):
        if pose_bone.name in kept:
            continue
        set_bone_hidden(pose_bone, False)
        shown += 1
    for name in state.get("unselectable") or ():
        pose_bone = arm_obj.pose.bones.get(str(name))
        if pose_bone is not None:
            set_bone_flag(pose_bone, "hide_select", True)
    return shown


def ensure_bars_unselectable(arm_obj):
    """Put the panel bars back out of a select-all's reach. Returns the count.

    The repair for a board that went through Grab Panel Handles before the flag
    was restored on the way out: it is still carrying selectable bars, and one
    Clear Scale reaches them.  Only applied to a board that is NOT being
    redesigned right now - inside the mode, selectable is the whole point.
    """
    if (arm_obj is None or getattr(arm_obj, "pose", None) is None
            or layout_unlocked(arm_obj)):
        return 0
    changed = 0
    for _template, pose_bone in layout_handle_bones(arm_obj):
        if getattr(pose_bone.bone, "hide_select", False):
            continue
        if set_bone_flag(pose_bone, "hide_select", True):
            changed += 1
    return changed


def reset_layout(arm_obj):
    """The whole panel back onto its authored design. Returns the count.

    Every LAYOUT bone, not just the three bars - a redesign moves frames and
    labels too, and a reset that put back only the bars would leave the artist
    staring at half the old arrangement.  Animation controls keep their
    location and get their authored rotation/scale back, which is the same
    split :func:`apply_layout` uses.

    The board OBJECT's placement beside the head is the rig's to recompute
    (``op_rig._place_bone_board``), not something a rest on a bone can describe.
    """
    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return 0
    driven = driven_templates()
    reset = 0
    for pose_bone in arm_obj.pose.bones:
        if control_template(pose_bone) in driven:
            reset += int(restore_design(pose_bone))
            continue
        matrix = _design_matrix(pose_bone)
        if matrix is None:
            reset += int(rest_pose(pose_bone, layout=True))
            continue
        if _moved_from_design(pose_bone):
            pose_bone.matrix_basis = matrix
            reset += 1
    return reset


# ------------------------------------------------------------- follow head ---
#
# The board ships with two switches nothing was reading.  They are real bones -
# ``CTRL_faceGUIfollowHead`` and ``CTRL_eyesAimFollowHead``, each a 0..1 slider
# held in its own Limit Location frame - and in the source asset each of the
# board's two roots carries a CHILD_OF constraint aimed at that character's
# ``head`` bone, sitting at influence 0 waiting for its switch to be wired up.
#
# The import stripped those constraints (they pointed at the SOURCE character's
# head, which would have dragged this rig's panel onto Ada's face) and never put
# ours back, so both switches slid up and down driving nothing at all.
#
# Rebuilt here as CONSTRAINT + DRIVER rather than as anything this add-on
# evaluates per frame:
#
# * it works with the add-on disabled, in a linked or appended file, and in the
#   standalone .blend the driver bake writes out - the bake joins the board into
#   the delivered skeleton, and a constraint and a driver both survive that join
#   by bone name (``op_bake_drivers._remap_relations`` retargets them first);
# * it costs the depsgraph one constraint each, against a Python handler that
#   would have to run on every frame of playback;
# * and a bone-local ``location`` is exactly what the rest of the board already
#   means by "a switch", so the same handle reads the same way everywhere.
#
# The constraint is ARMATURE, not CHILD OF, and that is the whole difference
# between this working and this being a nuisance.
#
# Child Of parents through a ``inverse_matrix`` that has to be CAPTURED - the
# Set Inverse button - and everything after that is measured against the moment
# of capture.  v4.12.0 used it and it was wrong in every way a captured bind can
# be wrong: bind while the head happened to be posed and the panel sat at a
# permanent offset; re-fit the skeleton (Update Rig moves every bone's rest) and
# the old inverse described a head that no longer exists, so the panel flew off;
# and every path that re-touched the board had to decide whether to re-take the
# bind, which is a decision no code can make correctly - re-taking it on a turned
# head is exactly as wrong as not re-taking it after a re-fit.
#
# The Armature constraint has no bind.  It reads the target bone's pose RELATIVE
# TO ITS OWN REST (Blender's own docs call it "a more flexible replacement for
# the Child Of constraint" that "does not need the Set Inverse operation"), so:
#
# * at rest the delta is the identity and the switch is a no-op - always, not
#   just when someone remembered to bind at rest;
# * the panel's own position, rotation and scale are applied after the delta, so
#   moving, rotating, scaling or re-placing the board keeps working with nothing
#   to re-bind;
# * re-fitting the skeleton moves the rest and the pose together, so the delta
#   stays the identity and the panel does not move at all.
FOLLOW_HEAD_CONSTRAINT = "MHFRT Follow Head"
FOLLOW_HEAD_CONSTRAINT_TYPE = 'ARMATURE'

# switch bone -> the board root it parents to the head.
FOLLOW_HEAD_SWITCHES = {
    "CTRL_faceGUIfollowHead": FACE_BOARD_ANCHOR_BONE,
    "CTRL_eyesAimFollowHead": FACE_BOARD_ANCHOR_EYES,
}
FOLLOW_HEAD_LABELS = {
    "CTRL_faceGUIfollowHead": "Panel follows head",
    "CTRL_eyesAimFollowHead": "Eye target follows head",
}

# The switch's travel is its Limit Location frame: y from 0 to 1.
_FOLLOW_AXIS = 1


def switch_unit(arm_obj):
    """What ONE on a 0..1 board switch is worth in this board's bone units.

    Every switch handle on the panel - the two follow-head sliders, the look-at
    blend, rigLogic - travels from its rest to ``1.0 * unit``, because
    ``_place_bone_board`` bakes the head-size ratio into the rest and
    ``rescale_channels`` moves the Limit Location frame with it.  The expression
    channels have always divided it back out (:func:`channel_value`); the
    SWITCHES did not, and read their raw travel as if it were already 0..1.

    Measured on a 12x head (board unit 11.89): the look-at blend saturated after
    8.4% of the handle's travel and the remaining 91.6% did nothing, so a blend
    the artist is meant to ease into behaved as an on/off. On a board unit of
    0.99 the opposite: dragging the handle to the very top of its frame asked
    for 0.99, so full look-at - and a panel that follows the head all the way -
    were simply not reachable. Both are size bugs; neither shows on a character
    whose ratio happens to be 1.
    """
    return placement_scale(arm_obj)


def follow_head_value(arm_obj, switch):
    """How far one follow-head switch is up, 0..1 whatever the board's size."""
    pose = getattr(arm_obj, "pose", None)
    pose_bone = pose.bones.get(switch) if pose is not None else None
    if pose_bone is None:
        return 0.0
    try:
        unit = switch_unit(arm_obj)
        return min(1.0, max(0.0,
                            float(pose_bone.location[_FOLLOW_AXIS]) / unit))
    except (ReferenceError, AttributeError, TypeError, ValueError,
            ZeroDivisionError):
        return 0.0


def set_follow_head_value(arm_obj, switch, value):
    """Slide one follow-head switch. Returns True when the bone was there."""
    pose = getattr(arm_obj, "pose", None)
    pose_bone = pose.bones.get(switch) if pose is not None else None
    if pose_bone is None:
        return False
    location = list(pose_bone.location)
    location[_FOLLOW_AXIS] = (min(1.0, max(0.0, float(value)))
                              * switch_unit(arm_obj))
    pose_bone.location = location
    return True


def _follow_constraint(pose_bone):
    for constraint in pose_bone.constraints:
        if constraint.name == FOLLOW_HEAD_CONSTRAINT:
            return constraint
    return None


def follow_head_target(constraint):
    """(object, bone name) an installed follow-head constraint rides, or None.

    An Armature constraint keeps its targets in a COLLECTION and has no
    ``.target`` at all, so anything that walks constraints looking for one - the
    delivery pass, this module's own checks - has to ask here.
    """
    if constraint is None or constraint.type != FOLLOW_HEAD_CONSTRAINT_TYPE:
        return None
    for target in constraint.targets:
        if target.target is not None and target.subtarget:
            return (target.target, target.subtarget)
    return None


def install_follow_head(arm_obj, skel, bone_name):
    """Wire both follow-head switches to real constraints. Idempotent.

    ``bone_name`` is the head bone on `skel` the panel should ride - the body's
    own head on a merged rig, the facial root on a standalone one.  Returns how
    many switches are now live.

    Safe to call at ANY time, from any pose: there is nothing captured here, so
    unlike the Child Of version this replaced there is no such thing as calling
    it at a bad moment.  That is why every path that touches the board calls it
    unconditionally.

    A board whose switch bone is missing (a pre-3.0 loose-object board, or one a
    Purge blanked) simply gets nothing wired, rather than a constraint no handle
    can reach.
    """
    if (arm_obj is None or getattr(arm_obj, "pose", None) is None
            or skel is None or not bone_name
            or skel.pose is None or bone_name not in skel.pose.bones):
        return 0
    live = 0
    for switch, root in FOLLOW_HEAD_SWITCHES.items():
        if switch not in arm_obj.pose.bones:
            continue
        pose_bone = arm_obj.pose.bones.get(root)
        if pose_bone is None:
            continue
        constraint = _follow_constraint(pose_bone)
        # A v4.12.0 board carries the Child Of this replaced. Removing it by
        # type rather than leaving it alongside is the migration: two parenting
        # constraints on one bone would fight, and the stale one would win the
        # moment its captured inverse disagreed.
        if constraint is not None \
                and constraint.type != FOLLOW_HEAD_CONSTRAINT_TYPE:
            pose_bone.constraints.remove(constraint)
            constraint = None
        if constraint is None:
            constraint = pose_bone.constraints.new(
                FOLLOW_HEAD_CONSTRAINT_TYPE)
            constraint.name = FOLLOW_HEAD_CONSTRAINT
        while len(constraint.targets) > 1:
            constraint.targets.remove(constraint.targets[-1])
        target = (constraint.targets[0] if constraint.targets
                  else constraint.targets.new())
        target.target = skel
        target.subtarget = bone_name
        # One target, so the weight is normalised to 1 whatever it says; it is
        # the constraint's INFLUENCE that the switch drives, and that is the one
        # Blender blends properly (slerp on rotation, lerp on the rest).
        target.weight = 1.0
        constraint.use_bone_envelopes = False
        constraint.use_deform_preserve_volume = False
        _migrate_switch_unit(arm_obj, root, switch)
        _drive_follow_influence(arm_obj, root, switch)
        live += 1
    return live


def _migrate_switch_unit(arm_obj, root, switch):
    """Restate a switch written in the OLD raw units, once.

    Before the unit was divided out, "on" was written as a literal 1.0 whatever
    the board's size.  Read the new way that is ``1.0 / unit`` - 8% of the way
    up on a 12x board - so a character whose panel was following its head would
    quietly stop.  The conversion is the same clamp the old reader applied,
    times the unit, and it runs only while the driver still carries the old
    expression, which is what keeps it from firing twice.
    """
    unit = switch_unit(arm_obj)
    if abs(unit - 1.0) <= 1.0e-9:
        return False
    animation = getattr(arm_obj, "animation_data", None)
    if animation is None:
        return False                    # never wired; nothing to convert
    fcurve = animation.drivers.find(_follow_influence_path(root))
    if fcurve is None or fcurve.driver is None:
        return False
    if fcurve.driver.expression != _follow_expression(1.0):
        return False                    # already in board units
    pose_bone = arm_obj.pose.bones.get(switch)
    if pose_bone is None:
        return False
    old = min(1.0, max(0.0, float(pose_bone.location[_FOLLOW_AXIS])))
    location = list(pose_bone.location)
    location[_FOLLOW_AXIS] = old * unit
    pose_bone.location = location
    return True


def _follow_influence_path(root):
    return (f'pose.bones["{bpy.utils.escape_identifier(root)}"]'
            f'.constraints["{FOLLOW_HEAD_CONSTRAINT}"].influence')


def _follow_expression(unit):
    """The influence driver's expression for a board of this unit."""
    if abs(unit - 1.0) <= 1.0e-9:
        return "min(1.0, max(0.0, follow))"
    return f"min(1.0, max(0.0, follow/{unit!r}))"


def _drive_follow_influence(arm_obj, root, switch):
    """influence = the switch handle's own travel, as an ordinary driver.

    Clamped in the expression rather than trusted from the handle: the Limit
    Location constraint holds the drawn bone inside its frame but ``location``
    keeps whatever the mouse wrote (see :func:`clamp_into_frame`), so a handle
    flicked past the end of its travel would otherwise ask for an influence of 4.

    The board's unit is DIVIDED OUT in the expression, as a literal constant.
    It has to be a constant: a driver evaluates in Blender's own namespace and
    cannot call back into the add-on, and this driver is meant to keep working
    in a delivered .blend with no add-on at all.  That makes it a stale number
    the moment the board is re-placed onto a different-sized head - which is
    why this is rewritten, not skipped, every time ``install_follow_head``
    runs, and why every path that re-places a board calls it unconditionally.
    """
    path = _follow_influence_path(root)
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    existing = arm_obj.animation_data.drivers.find(path)
    if existing is not None:
        arm_obj.animation_data.drivers.remove(existing)
    fcurve = arm_obj.driver_add(path)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    for variable in list(driver.variables):
        driver.variables.remove(variable)
    variable = driver.variables.new()
    variable.name = "follow"
    variable.type = 'SINGLE_PROP'
    target = variable.targets[0]
    target.id_type = 'OBJECT'
    target.id = arm_obj
    target.data_path = (
        f'pose.bones["{bpy.utils.escape_identifier(switch)}"]'
        f'.location[{_FOLLOW_AXIS}]')
    driver.expression = _follow_expression(switch_unit(arm_obj))
    # A driver with no keyframes evaluates its expression; one with a modifier
    # (which driver_add adds by default) evaluates the modifier instead and the
    # expression is ignored - the single most common reason a hand-built driver
    # reads a constant.
    for modifier in list(fcurve.modifiers):
        fcurve.modifiers.remove(modifier)
    return fcurve


def follow_head_installed(arm_obj):
    """True when this board's follow-head constraints are wired up CURRENTLY.

    A v4.12.0 board carrying the old Child Of reads as NOT installed, which is
    what makes the rescan replace it without anyone having to rebuild a board.

    EVERY root has to be wired, not just one: a board caught half-way through a
    migration - the panel converted, the eye aim still on the old constraint -
    would otherwise report itself finished and keep the stale half forever.

    The influence driver has to agree with the board's CURRENT unit too.  Its
    expression carries that unit as a literal (see :func:`_drive_follow_influence`),
    so a board placed onto a head of a different size - or one built before the
    unit was divided out at all - is wired but wired wrong: the switch would
    saturate part-way up its travel, or never reach the top.  Reporting that as
    "not installed" is what makes the next rescan rewrite it, with no rebuild
    and nothing for the artist to do.
    """
    pose = getattr(arm_obj, "pose", None)
    if pose is None:
        return False
    roots = [pose.bones[root] for root in set(FOLLOW_HEAD_SWITCHES.values())
             if root in pose.bones]
    if not roots:
        return False
    if not all(follow_head_target(_follow_constraint(pose_bone)) is not None
               for pose_bone in roots):
        return False
    wanted = _follow_expression(switch_unit(arm_obj))
    animation = getattr(arm_obj, "animation_data", None)
    if animation is None:
        return False
    for pose_bone in roots:
        fcurve = animation.drivers.find(_follow_influence_path(pose_bone.name))
        if fcurve is None or fcurve.driver is None:
            return False
        if fcurve.driver.expression != wanted:
            return False
    return True


def remove_follow_head(arm_obj):
    """Take the follow-head rig back off. Returns how many roots were freed."""
    pose = getattr(arm_obj, "pose", None)
    if pose is None:
        return 0
    removed = 0
    for root in set(FOLLOW_HEAD_SWITCHES.values()):
        pose_bone = pose.bones.get(root)
        if pose_bone is None:
            continue
        constraint = _follow_constraint(pose_bone)
        if constraint is not None:
            pose_bone.constraints.remove(constraint)
            removed += 1
        animation = arm_obj.animation_data
        if animation is not None:
            fcurve = animation.drivers.find(_follow_influence_path(root))
            if fcurve is not None:
                animation.drivers.remove(fcurve)
    return removed


def location_data_path(control):
    """The fcurve data path for this control's location, on its owner object."""
    if is_pose_bone(control):
        return f'pose.bones["{bpy.utils.escape_identifier(control.name)}"].location'
    return "location"


def bone_name_from_data_path(data_path):
    """``pose.bones["CTRL_C_jaw"].location`` -> ``CTRL_C_jaw``; else None."""
    if not data_path.startswith('pose.bones["') or not data_path.endswith(".location"):
        return None
    try:
        return data_path.split('"')[1]
    except IndexError:
        return None


# The DNA only ever drives tx and ty (155 ty channels, 13 tx, no tz), so these
# are the location indices an exported or imported board curve can use.
BONE_AXES = frozenset((_BONE_AXES["tx"][0], _BONE_AXES["ty"][0]))
OBJECT_AXES = frozenset((_OBJECT_AXES["tx"][0], _OBJECT_AXES["ty"][0]))


def animated_axes(control):
    return BONE_AXES if is_pose_bone(control) else OBJECT_AXES


def curve_control(control_or_none, data_path, array_index, by_bone_name):
    """Resolve one fcurve to a control, or None when it is not a board curve.

    ``control_or_none`` is the Object-board control the curve's slot resolved
    to; a bone board instead resolves per curve, because every control shares
    the one armature slot.
    """
    bone_name = bone_name_from_data_path(data_path)
    if bone_name is not None:
        control = by_bone_name.get(bone_name)
        if control is None or array_index not in BONE_AXES:
            return None
        return control
    if (control_or_none is None or data_path != "location"
            or array_index not in OBJECT_AXES):
        return None
    return control_or_none


def gui_scale(skel, default=0.01):
    """The board's units-per-GUI-unit divisor, stored on the skeleton.

    1.0 for the bone board (authored 1:1 with the +/-1 control range), 0.01 for
    the legacy centimetre-scale object board.
    """
    try:
        value = float(skel.get("dna_gui_scale", default))
    except (TypeError, ValueError):
        return default
    return value or default
