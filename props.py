"""Scene-level property groups for the transfer workflow.

Also home of the shared RIG ID-property names. Each fitted skeleton carries
these as custom properties: they tag it as a live rig, link it to its cage /
target / control board (ID pointers, so appending any piece of a character
pulls the whole chain into the new file), and store its per-character
settings. Defined here because props is the one module everything can import
without cycles.
"""

import bpy
from bpy.app.handlers import persistent

from .core import registry

# ID-property names on the skeleton, character meshes, and board objects.
RIG_ID_PROP = "mhfrt_rig_id"            # unique id -> this armature IS a live rig
RIG_CAGE_PROP = "mhfrt_cage"            # ID pointer: the character's cage
RIG_TARGET_PROP = "mhfrt_target"        # ID pointer: the character's head
RIG_GUI_COLL_PROP = "mhfrt_gui_coll"    # ID pointer: the character's control board
RIG_INTENSITY_PROP = "mhfrt_scale_mul"  # per-character expression intensity
RIG_UI_TAB_PROP = "mhfrt_ui_tab"        # per-character last workflow step
# One-armature rigs (game export): the skeleton IS the character's body
# armature, with the FACIAL_* bones joined in under RIG_MERGE_BONE_PROP.
RIG_MERGED_PROP = "mhfrt_merged_body_rig"
RIG_MERGE_BONE_PROP = "mhfrt_merged_parent_bone"
RIG_DESTINATION_PROP = "mhfrt_rig_destination"
RIG_DEFORM_BONE_PROP = "mhfrt_head_deform_bone"
RIG_BODY_ARMATURE_PROP = "mhfrt_body_armature"
# Armature-space unit ratio between the DNA source alignment and the skeleton
# the bones actually live in (1.0 unless the rig was merged into an armature
# with a different object scale). Scales RigLogic joint translations.
RIG_SOURCE_SCALE_PROP = "mhfrt_rest_source_to_rig_scale"
# Matching 3x3 basis: the DNA source armature's axes expressed in this rig's
# data space. Folded into the per-bone RigLogic change of basis (core/dna_apply).
RIG_SOURCE_BASIS_PROP = "mhfrt_rest_source_to_rig_basis"
# Per-bone: the joint's head in the SOURCE (Ada) space, stored once at import.
# The control board is authored in that same space, so it is also what places
# the board beside a character of any size (see ops/op_rig).
RIG_REST_HEAD_PROP = "mhfrt_rest_head"


def _is_mesh(self, obj):
    return obj.type == 'MESH'


def _is_free_mesh(self, obj):
    """A mesh that is not already another character's Head Target.

    One head, one character.  Enforced in the dropdown itself so the clash is
    impossible to make rather than reported after the fact - picking the same
    head twice is what used to fold two characters into one collection.
    """
    if obj.type != 'MESH':
        return False
    try:
        from .core import registry
        scene = bpy.context.scene
        rec = registry.active(scene)
        return registry.target_owner(scene, obj, skip=rec) is None
    except (AttributeError, ReferenceError, KeyError):
        return True


def _is_armature(self, obj):
    return obj.type == 'ARMATURE'


def _merge_armature_update(self, context):
    """Show safe, editable bone suggestions when an armature is picked.

    Parent and deform bones are separate because arbitrary rigs often expose a
    control bone for parenting while painting the mesh to another bone. The
    operator validates both choices and never relies on a rig-name convention.
    """
    body = self.merge_body_armature
    if body is None or body.type != 'ARMATURE':
        return
    current = self.merge_head_bone
    if not current or current not in body.data.bones:
        try:
            from .ops.op_merge import suggest_parent_bone
            suggestion = suggest_parent_bone(body)
        except (ImportError, AttributeError):
            suggestion = ""
        if suggestion != current:
            self.merge_head_bone = suggestion
    _suggest_deform_bone(self)


def _suggest_deform_bone(self):
    """Keep or suggest one unambiguously painted deform carrier."""
    body = self.merge_body_armature
    target = self.target
    current = self.merge_deform_bone
    if body is None or body.type != 'ARMATURE':
        return
    try:
        from .ops.op_merge import suggest_deform_bone
        suggestion = suggest_deform_bone(
            body,
            target,
            self.merge_head_bone,
            current,
        )
    except (ImportError, AttributeError, RuntimeError):
        suggestion = ""
    if suggestion != current:
        self.merge_deform_bone = suggestion


def _merge_head_bone_update(self, context):
    _suggest_deform_bone(self)


def _view_mode_update(self, context):
    from .ops.op_pairs import apply_view_mode
    apply_view_mode(context)


def _riglogic_scale_update(self, context):
    from .ops.op_rig import apply_scale_mul
    apply_scale_mul(context)


# The per-channel Bone Intensity has no storage of its own: it reads and writes
# the weight the skeleton keeps for the morph row the Morphs list is parked on.
# A live get/set pair means the slider can never drift out of sync with the rig
# - switching character, row, or file simply shows that channel's own value.
def _morph_channel_intensity_get(self):
    try:
        from .ops import op_morphs
        return op_morphs.selected_channel_weight(self)
    except (AttributeError, ReferenceError, RuntimeError):
        return 1.0


def _morph_channel_intensity_set(self, value):
    try:
        from .ops import op_morphs
        op_morphs.set_selected_channel_weight(self, value)
    except (AttributeError, ReferenceError, RuntimeError):
        pass


# The eye target's distance is the only part of it an artist sets: its size
# and its position on the eye line are measured from the character (see
# op_rig._place_eye_aim). Live get/set on the skeleton for the same reason as
# the channel weight above - switching character shows that character's value,
# never a stale mirror of the last one's.
def _eye_aim_distance_get(self):
    try:
        from .ops import op_rig
        return op_rig.eye_aim_distance(self.skeleton)
    except (AttributeError, ReferenceError, RuntimeError):
        return 0.3


def _eye_aim_distance_set(self, value):
    try:
        from .ops import op_rig
        op_rig.queue_eye_aim_distance(self.skeleton, value)
    except (AttributeError, ReferenceError, RuntimeError):
        pass


def _ui_tab_update(self, context):
    """Each character remembers the workflow step it was left on.

    The bookmark lives on the skeleton once the rig exists; before that it
    lives on the character's cage, so half-built characters keep their place
    in the workflow too.  The character's collections are created as soon as
    it has a cage or a head (see :func:`_obj_update`), so nothing has to be
    materialized here."""
    if self.switching_character:
        return
    skel = self.skeleton
    if skel is not None and skel.get(RIG_ID_PROP):
        skel[RIG_UI_TAB_PROP] = self.ui_tab
    elif self.cage is not None:
        self.cage[RIG_UI_TAB_PROP] = self.ui_tab


def _restore_saved_tab(mh, *holders):
    """Jump back to the first workflow bookmark found on the given IDs."""
    for holder in holders:
        if holder is None:
            continue
        saved = str(holder.get(RIG_UI_TAB_PROP, ""))
        if not saved:
            continue
        from .ui.flow import STEP_IDS
        if saved in STEP_IDS and mh.ui_tab != saved:
            mh.ui_tab = saved
        return


def _tint_update(self, context):
    from .ops.op_shading import tint_objects
    try:
        tint_objects(context)
    except Exception:
        pass


def _morph_ui_sync(self):
    """Refresh derived template-list rows after character object changes."""
    try:
        from .ops import op_morphs
        op_morphs.sync_morph_ui_state(self)
    except (AttributeError, RuntimeError):
        # During registration/load the op module may not be registered yet;
        # its persistent load handler performs the same sync immediately after.
        pass


def bind_active_record(scene, mh):
    """Attach the panel's cage / head / skeleton to the character they belong to.

    The RECORD is the character.  If no record is active, the objects
    themselves name one; if they name none, a character is started here.  A
    character is never conjured out of the outliner any more, and - just as
    important - never inferred out of existence: a record with nothing in it
    yet still exists and still has a row.
    """
    from .core import registry
    record = registry.active(scene)
    if record is None:
        record = (registry.for_object(scene, mh.skeleton)
                  or registry.for_object(scene, mh.cage)
                  or registry.for_object(scene, mh.target))
    if record is None:
        record = registry.new(scene)
    registry.set_active(scene, record.uid)

    if mh.cage is not None and registry.cage_owner(
            scene, mh.cage, skip=record) is None:
        record.cage = mh.cage
        registry.claim(record, mh.cage)
    if mh.target is not None and registry.target_owner(
            scene, mh.target, skip=record) is None:
        # BEFORE anything binds: remember the head exactly as the artist
        # handed it over, so Remove can hand it back that way.
        registry.capture_baseline(record, mh.target)
        record.target = mh.target
        registry.claim(record, mh.target)
        # A character restored from a .mhfrt parked its landmark curves until
        # its head turned up.  It just did.
        if registry.apply_pending_landmarks(record):
            from .core import landmarks as lmdata
            lmdata.load_active(mh)
    if mh.skeleton is not None:
        record.skeleton = mh.skeleton
        registry.claim(record, mh.skeleton)
    # Keep the character's own file in step with what it now consists of.
    from .core import sidecar
    sidecar.touch(scene, record)
    return record


def _obj_update(self, context):
    """Cage/target changed: retint, FILE the character, and once both slots are
    set the Setup step is done - slide the panel forward to Landmarks."""
    from .ops.op_live import stop_running
    stop_running()          # the live session's mesh just changed under it
    if self.switching_character:
        _tint_update(self, context)
        _morph_ui_sync(self)
        return
    from .core import rest_tuning
    skel = self.skeleton
    if skel is not None and skel.get(rest_tuning.TONGUE_SESSION_PROP):
        saved_cage = skel.get(rest_tuning.TONGUE_SESSION_CAGE_PROP)
        saved_target = skel.get(rest_tuning.TONGUE_SESSION_TARGET_PROP)
        self.switching_character = True
        try:
            if isinstance(saved_cage, bpy.types.Object) and self.cage != saved_cage:
                self.cage = saved_cage
            if (isinstance(saved_target, bpy.types.Object)
                    and self.target != saved_target):
                self.target = saved_target
        finally:
            self.switching_character = False
        _tint_update(self, context)
        _morph_ui_sync(self)
        return
    from .core import landmarks as lmdata
    lmdata.save_for_pair(self, self.active_cage, self.active_target)
    lmdata.load_active(self)
    _sync_skeleton_to_pair(self)
    _tint_update(self, context)
    _morph_ui_sync(self)
    # File the character straight away.  This used to wait until the artist
    # left Setup, which left a brand-new rig's objects loose in the scene
    # collection while its character collection sat empty - and anything
    # created in between (the cage, the head) landed outside it.  A cage or a
    # head on its own is enough: the root is created, named after whatever
    # identifies the character best so far, and renamed once the head arrives.
    if self.cage or self.target:
        from .core import organization
        scene = context.scene if context is not None else bpy.context.scene
        record = bind_active_record(scene, self)
        organization.organize_current(context, record=record)
        from .ops import op_rig
        op_rig.bump_rig_topology()
    if self.cage and self.target and self.cage != self.target:
        if self.ui_tab == 'SETUP':
            self.ui_tab = 'LANDMARKS'


def _in_front_update(self, context):
    if self.cage:
        self.cage.show_in_front = self.cage_in_front


def _show_bones_update(self, context):
    skel = self.skeleton
    if skel is not None:
        try:
            skel.hide_set(not self.show_bones)
        except RuntimeError:
            pass


def _studio_shading_update(self, context):
    from .ops.op_shading import apply_studio_state
    apply_studio_state(context, self.studio_shading)


def _skeleton_update(self, context):
    """Switching the active character: load its cage / target / landmarks."""
    if not self.switching_character:
        from .core import rest_tuning
        active_session = next(
            (obj for obj in bpy.data.objects
             if obj.type == 'ARMATURE'
             and obj.get(rest_tuning.TONGUE_SESSION_PROP)),
            None,
        )
        if active_session is not None and self.skeleton != active_session:
            self.switching_character = True
            try:
                self.skeleton = active_session
            finally:
                self.switching_character = False
            _show_bones_update(self, context)
            _sync_attachment_from_skeleton(self)
            _sync_intensity_from_skeleton(self)
            return
    # Keep the always-visible viewport toggle authoritative when a newly
    # imported or different character skeleton becomes active.
    _show_bones_update(self, context)
    _sync_attachment_from_skeleton(self)
    if self.switching_character:
        _sync_intensity_from_skeleton(self)
        return
    skel = self.skeleton
    if skel is not None and skel.get(RIG_ID_PROP):
        activate_rig(context, skel)
        return
    _sync_intensity_from_skeleton(self)


def _sync_intensity_from_skeleton(self):
    skel = self.skeleton
    if skel is not None and RIG_INTENSITY_PROP in skel:
        val = float(skel[RIG_INTENSITY_PROP])
        if abs(self.riglogic_scale_mul - val) > 1e-6:
            self.riglogic_scale_mul = val


def _sync_attachment_from_skeleton(self):
    """Keep the scene picker aligned with this facial rig's real parent.

    A merged rig has no parent to read: it IS the character's armature, so the
    picker shows the skeleton itself and the bone the face was joined under."""
    skel = self.skeleton
    if skel is not None and skel.get(RIG_MERGED_PROP):
        if self.merge_body_armature != skel:
            self.merge_body_armature = skel
        bone = str(skel.get(RIG_MERGE_BONE_PROP, ""))
        if self.merge_head_bone != bone:
            self.merge_head_bone = bone
        deform = str(skel.get(RIG_DEFORM_BONE_PROP, ""))
        if self.merge_deform_bone != deform:
            self.merge_deform_bone = deform
        self.rig_destination = 'EXISTING'
        return
    attached = bool(
        skel is not None
        and skel.parent is not None
        and skel.parent.type == 'ARMATURE'
        and skel.parent_type == 'BONE'
        and skel.parent_bone
    )
    armature = skel.parent if attached else None
    bone = skel.parent_bone if attached else ""
    if self.merge_body_armature != armature:
        self.merge_body_armature = armature
    if self.merge_head_bone != bone:
        self.merge_head_bone = bone
    deform = str(skel.get(RIG_DEFORM_BONE_PROP, "")) if skel else ""
    if self.merge_deform_bone != deform:
        self.merge_deform_bone = deform
    holder = skel or self.cage
    choice = str(holder.get(RIG_DESTINATION_PROP, "")) if holder else ""
    if attached:
        choice = 'EXISTING'
    if choice not in {'STANDALONE', 'EXISTING'}:
        choice = 'UNDECIDED'
    self.rig_destination = choice


def _linked_obj(skel, prop):
    obj = skel.get(prop) if skel else None
    return obj if isinstance(obj, bpy.types.Object) else None


def _skeleton_matches(skel, cage, target, strict=False):
    # A skeleton with no links is still accepted for any pair - that is how a
    # legacy untagged rig picked in the panel gets adopted.  Which is exactly
    # why the control board has to be excluded first: it carries the rig id and
    # NO cage/target pointers, so it matches every character in the file (see
    # board.is_rig_skeleton).
    #
    # `strict` withdraws that courtesy for AUTOMATIC discovery: a pointerless
    # rig must then prove the pair is its own by the rig id stamped on the cage
    # (setup_rig writes it).  Without that, one pointerless rig in the file
    # matched every cage in existence - and a pointerless rig is not exotic,
    # it is exactly what the duplicate-id repair leaves behind after a Shift+D
    # (see op_rig._drop_foreign_pointers).  Loading a cage for a BRAND NEW
    # character then silently adopted that orphan as the new character's rig.
    from .core import board
    if not board.is_rig_skeleton(skel, RIG_ID_PROP):
        return False
    owner = _linked_obj(skel, RIG_CAGE_PROP)
    head = _linked_obj(skel, RIG_TARGET_PROP)
    if owner is None:
        if strict:
            rid = str(skel.get(RIG_ID_PROP) or "")
            if not rid or cage is None \
                    or str(cage.get(RIG_ID_PROP) or "") != rid:
                return False
    elif owner != cage:
        return False
    return head is None or target is None or head == target


def _find_pair_skeleton(cage, target):
    """The rig that OWNS this cage/head pair, or None.

    Strict on purpose - this is discovery, not the artist's own choice.
    """
    if cage is None:
        return None
    for obj in bpy.data.objects:
        if _skeleton_matches(obj, cage, target, strict=True):
            return obj
    return None


def _sync_skeleton_to_pair(self):
    if _skeleton_matches(self.skeleton, self.cage, self.target):
        return
    skel = _find_pair_skeleton(self.cage, self.target)
    self.switching_character = True
    try:
        self.skeleton = skel
    finally:
        self.switching_character = False
    _sync_attachment_from_skeleton(self)
    _sync_intensity_from_skeleton(self)


def activate_rig(context, skel):
    """Make one already-built character the active authoring target."""
    mh = context.scene.mhfrt
    mh.starting_new_character = False   # a real character is taking over
    current = mh.skeleton
    if current is not None and current != skel:
        from .core import rest_tuning
        if current.get(rest_tuning.TONGUE_SESSION_PROP):
            return {'CANCELLED'}
    from .core import landmarks as lmdata
    lmdata.save_for_pair(mh, mh.active_cage, mh.active_target)
    _save_display_toggles(mh)
    cage = _linked_obj(skel, RIG_CAGE_PROP)
    target = _linked_obj(skel, RIG_TARGET_PROP)
    mh.switching_character = True
    try:
        if cage is not None:
            mh.cage = cage
        if target is not None:
            mh.target = target
        mh.skeleton = skel
    finally:
        mh.switching_character = False
    lmdata.load_active(mh)
    _sync_attachment_from_skeleton(mh)
    _sync_intensity_from_skeleton(mh)
    _tint_update(mh, context)
    from .core import organization
    organization.organize_current(context)
    _restore_display_toggles(mh)
    # jump back to the step this character was left on (its own bookmark)
    _restore_saved_tab(mh, skel, _linked_obj(skel, RIG_CAGE_PROP))
    from .core import registry
    _rec = registry.for_object(context.scene, skel)
    if _rec is not None:
        registry.set_active(context.scene, _rec.uid)
    return {'FINISHED'}


def activate_pair(context, cage, target):
    """Make a character that has no rig yet (cage + head only) the active
    authoring pair - its landmarks, wrap state and bookmark come back."""
    mh = context.scene.mhfrt
    mh.starting_new_character = False   # a real pair is taking over
    current = mh.skeleton
    if current is not None:
        from .core import rest_tuning
        if current.get(rest_tuning.TONGUE_SESSION_PROP):
            return {'CANCELLED'}
    from .core import landmarks as lmdata
    lmdata.save_for_pair(mh, mh.active_cage, mh.active_target)
    _save_display_toggles(mh)
    mh.switching_character = True
    try:
        mh.skeleton = None
        mh.cage = cage
        mh.target = target
    finally:
        mh.switching_character = False
    lmdata.load_active(mh)
    _sync_attachment_from_skeleton(mh)
    _tint_update(mh, context)
    from .core import organization
    organization.organize_current(context)
    _restore_display_toggles(mh)
    _restore_saved_tab(mh, cage)
    from .core import registry
    _rec = registry.for_object(context.scene, cage)
    if _rec is not None:
        registry.set_active(context.scene, _rec.uid)
    return {'FINISHED'}


# The panel's display bar (view mode, cage in front, show bones) must reflect
# the CURRENT character.  Object-level flags (hide_get / show_in_front) already
# live on their own datablocks, so switching does not silently mutate the other
# characters - but the SCENE-level toggle state needs to be re-derived from
# whichever character just became active.
DISPLAY_TOGGLE_PROP = "mhfrt_display_toggles"  # dict on the skeleton or cage


def _display_toggle_owner(mh):
    """Prefer the skeleton so display state travels with the fitted rig; fall
    back to the cage while the character is still in setup."""
    return mh.skeleton or mh.cage


# What a rig that has never been looked at shows: everything on, nothing
# hidden.  A new rig starts here rather than inheriting whatever the artist had
# set up on the rig they were working on a moment ago.
DISPLAY_DEFAULTS = {
    'view_mode': 'BOTH',
    'show_bones': True,
    'cage_in_front': False,
    'show_pairs_overlay': True,
}


def reset_display_toggles(mh, context=None):
    """Put the display bar back to defaults for a rig with no history.

    Called by New Rig: bones, landmark overlay, cage-in-front and the
    cage/head view mode are per-rig state, and carrying the previous rig's
    settings into a fresh one made the new rig look half-hidden before it even
    had a cage.
    """
    mh.switching_character = True
    try:
        for name, value in DISPLAY_DEFAULTS.items():
            if getattr(mh, name) != value:
                setattr(mh, name, value)
    finally:
        mh.switching_character = False
    from .ops.op_pairs import apply_view_mode
    apply_view_mode(context or bpy.context)


def _save_display_toggles(mh):
    owner = _display_toggle_owner(mh)
    if owner is None:
        return
    owner[DISPLAY_TOGGLE_PROP] = {
        'view_mode': str(mh.view_mode),
        'show_bones': bool(mh.show_bones),
        'cage_in_front': bool(mh.cage_in_front),
        'show_pairs_overlay': bool(mh.show_pairs_overlay),
    }


def _restore_display_toggles(mh):
    """Bring the panel toggles back to the values the incoming character had
    when the artist last worked on it.  Fills in blanks by reading the objects'
    current visibility so a first-ever switch is still coherent."""
    owner = _display_toggle_owner(mh)
    stored = owner.get(DISPLAY_TOGGLE_PROP) if owner is not None else None
    # A newly loaded character has no stored toggles yet; derive defaults from
    # the objects' own state so the panel matches what the artist actually sees.
    cage = mh.cage
    target = mh.target
    skel = mh.skeleton
    if stored is None:
        cage_hidden = bool(cage and cage.hide_get())
        target_hidden = bool(target and target.hide_get())
        if cage_hidden and not target_hidden:
            derived_mode = 'TARGET'
        elif target_hidden and not cage_hidden:
            derived_mode = 'CAGE'
        else:
            derived_mode = 'BOTH'
        stored = {
            'view_mode': derived_mode,
            'show_bones': not (skel is not None and skel.hide_get()),
            'cage_in_front': bool(cage and cage.show_in_front),
            'show_pairs_overlay': bool(mh.show_pairs_overlay),
        }

    mh.switching_character = True
    try:
        want_mode = str(stored.get('view_mode', mh.view_mode))
        if want_mode in {'CAGE', 'TARGET', 'BOTH'} and mh.view_mode != want_mode:
            mh.view_mode = want_mode
        for name in ('show_bones', 'cage_in_front', 'show_pairs_overlay'):
            if name not in stored:
                continue
            value = bool(stored[name])
            if bool(getattr(mh, name)) != value:
                setattr(mh, name, value)
    finally:
        mh.switching_character = False

    # Re-apply the (possibly changed) mode to reconcile the OBJECTS' hide flags
    # with the panel state - a per-character toggle would otherwise be a lie:
    # 'TARGET' would show the previous character's hide state.
    from .ops.op_pairs import apply_view_mode
    apply_view_mode(context=bpy.context)
    if skel is not None:
        try:
            skel.hide_set(not bool(mh.show_bones))
        except RuntimeError:
            pass
    if cage is not None:
        cage.show_in_front = bool(mh.cage_in_front)


class MHFRT_Landmark(bpy.types.PropertyGroup):
    """One landmark pair, stored as data (never as scene objects).

    src lives on the cage in local space on the Basis shape; tgt lives on the
    target head in local space. src_vidx is the nearest cage vertex, used so
    source markers visually follow the Wrapped shape key."""
    src_co: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    tgt_co: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    has_src: bpy.props.BoolProperty(default=False)
    has_tgt: bpy.props.BoolProperty(default=False)
    src_vidx: bpy.props.IntProperty(default=-1)
    label: bpy.props.StringProperty(default="")  # named guide point (auto landmarks)
    # symmetry-created source waiting for its auto-mirrored target point
    mirror_pending: bpy.props.BoolProperty(default=False)
    # surface-curve grouping: all points drawn as one stroke share a curve id
    # and sit in stroke order in the collection (-1 = legacy single point)
    curve_id: bpy.props.IntProperty(default=-1)
    # the stroke came back onto its own start: this curve is a LOOP, so the
    # last point links back to the first (eye rims, lips, nostrils)
    curve_closed: bpy.props.BoolProperty(default=False)
    # on auto-mirrored twin points: the curve id this curve mirrors (-1 = not
    # a twin).  Lets the overlay join twin curves at centre-merged endpoints.
    mirror_of: bpy.props.IntProperty(default=-1)
    # this point sat on the centre line when its curve was confirmed, so it
    # has NO mirrored twin (centre merge).  Only such a point may be joined
    # to the twin curve; a corner point far from the centre never is.
    center_merged: bpy.props.BoolProperty(default=False)


class MHFRT_MorphTarget(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Additional Object",
        description="Extra mesh that receives separate driven morph shape keys",
        poll=_is_mesh,
    )


class MHFRT_MorphListItem(bpy.types.PropertyGroup):
    """One immutable DNA channel row backing the custom Morph UIList."""
    key_name: bpy.props.StringProperty(name="Morph")


class MHFRT_RigListItem(bpy.types.PropertyGroup):
    """One row of the Rig list (Morphs-style template_list backing).

    Runtime mirror only (SKIP_SAVE): the source of truth is the tagged
    collections / armatures in the file, rebuilt by op_rig.sync_rig_ui_state.
    ``rig_name`` is the editable display name; it is persisted on the rig's
    root collection (or skeleton) as ``mhfrt_rig_name``.

    ``rig_key`` is the row's STABLE identity - the rig id for a built rig, the
    root collection's uid for one still in setup.  Names are labels: renaming a
    skeleton or a collection used to leave every row pointing at nothing, and
    clicking one either failed or (after a name got reused) switched to the
    wrong character."""
    skel_name: bpy.props.StringProperty()
    root_name: bpy.props.StringProperty()
    rig_key: bpy.props.StringProperty()
    rig_name: bpy.props.StringProperty()
    is_setup: bpy.props.BoolProperty(default=False)   # cage/head only, no rig yet
    is_new: bpy.props.BoolProperty(default=False)     # in-Setup pair, no root yet
    has_anim: bpy.props.BoolProperty(default=False)


def _morph_object_row_update(self, context):
    from .ops import op_morphs
    op_morphs.morph_object_row_changed(self, context)


def _get_active_rig_name(self):
    idx = self.rig_ui_active_index
    if 0 <= idx < len(self.rig_ui_items):
        return self.rig_ui_items[idx].rig_name
    return ""


def _set_active_rig_name(self, value):
    idx = self.rig_ui_active_index
    if not (0 <= idx < len(self.rig_ui_items)):
        return
    value = value.strip()
    if not value:
        return
    item = self.rig_ui_items[idx]
    item.rig_name = value
    from .ops import op_rig
    op_rig.set_rig_name_for_item(item, value)


class MHFRT_Props(bpy.types.PropertyGroup):
    # Which cage LOD to load (all share the same skeleton + rig logic)
    cage_lod: bpy.props.EnumProperty(
        name="Cage LOD",
        description="Which Ada head LOD to use as the wrap cage. Lower LOD = faster "
                    "wrap; higher LOD = denser cage",
        items=[
            ('5', "LOD5 - 564 verts (lightest)", "Fastest cage"),
            ('4', "LOD4 - 1,291 verts", ""),
            ('3', "LOD3 - 2,548 verts", ""),
            ('2', "LOD2 - 5,999 verts", ""),
            ('1', "LOD1 - 11,977 verts", ""),
            ('0', "LOD0 - 24,049 verts (heaviest)", "Densest cage; slow wrap"),
        ],
        default='2',
    )

    # Core objects
    cage: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Head Cage",
        description="The low-poly head cage (wrap source)",
        poll=_is_mesh, update=_obj_update,
    )
    target: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Head Target",
        description="The user's head mesh to wrap onto (single skin mesh). "
                    "A head already used by another rig in the list is not "
                    "offered - remove that rig first to free it",
        poll=_is_free_mesh, update=_obj_update,
    )
    skeleton: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Skeleton",
        description="The active character's fitted facial skeleton (each character "
                    "keeps its own; switch here to work on another one)",
        poll=lambda self, o: o.type == 'ARMATURE',
        update=_skeleton_update,
    )
    # The non-standalone choice is always the Merge: one armature, the facial
    # bones joined into the character's own (game engines want one skeleton, and
    # two armatures were only ever a stop on the way there).  RNA identifiers
    # are retained from 0.4.0 so an armature/bone already picked in a saved
    # .blend migrates cleanly - including files left bone-parented by the old
    # Attach step, which merge takes over from as it is.
    merge_body_armature: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Character Armature",
        description=("The character's own armature - the facial bones are "
                     "joined into it. Leave blank to use the armature that "
                     "already deforms the Head Target"),
        poll=_is_armature,
        update=_merge_armature_update,
    )
    merge_head_bone: bpy.props.StringProperty(
        name="Parent Bone",
        description=("The bone the whole face hangs under - normally the head "
                     "bone. Required: check it before merging, because every "
                     "facial bone will follow it"),
        default="",
        update=_merge_head_bone_update,
    )
    merge_deform_bone: bpy.props.StringProperty(
        name="Head Deform Bone",
        description=("The deform bone whose painted head weight is shared with "
                     "the facial bones. This may differ from Parent Bone on "
                     "control rigs; it must have a vertex group on Head Target"),
        default="",
    )
    rig_destination: bpy.props.EnumProperty(
        name="Rig Connection",
        description="Choose whether this facial rig stays standalone or is "
                    "merged into the character's existing armature",
        items=[
            ('UNDECIDED', "Choose...", "A connection choice is required"),
            ('STANDALONE', "Standalone", "Keep the facial rig independent"),
            ('EXISTING', "Merge Into Character",
             "Join the facial bones INTO the character's armature, so the "
             "character stays ONE skeleton (never two)"),
        ],
        default='UNDECIDED',
    )
    active_cage: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Active Cage Cache",
        poll=_is_mesh,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    active_target: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Active Target Cache",
        poll=_is_mesh,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    switching_character: bpy.props.BoolProperty(
        name="Switching Character",
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    # Set by New Character and cleared the moment a cage or head is picked (or
    # another character is activated).  Without it the panel sits in an empty
    # Setup while the LIST still highlights the character you just left, which
    # reads as "the new character was never created".
    starting_new_character: bpy.props.BoolProperty(
        name="Starting New Character",
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    riglogic_scale_mul: bpy.props.FloatProperty(
        name="Expressions Intensity",
        description="Manual multiplier on RigLogic joint translation, on top of the "
                    "automatic head-size scale set when the skeleton is fitted. "
                    "1.0 = proportional to the source; lower = subtler, higher = stronger",
        default=1.0, min=0.0, max=3.0, soft_min=0.0, soft_max=1.5,
        subtype='FACTOR', update=_riglogic_scale_update,
    )
    morph_channel_intensity: bpy.props.FloatProperty(
        name="Channels Intensity",
        description="How strongly the morph channel selected in the list above "
                    "drives the facial BONES. 1.0 is the untouched rig; 0 "
                    "stops its bone movement while its shape key keeps "
                    "deforming, so that channel runs on its sculpted shape "
                    "alone. The board controllers never move. Stored per "
                    "character on the skeleton",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        get=_morph_channel_intensity_get, set=_morph_channel_intensity_set,
    )
    eye_aim_distance: bpy.props.FloatProperty(
        name="Target Distance",
        description="How far in front of the eyes the look-at target floats. "
                    "The only part of it to set by hand: its size and its "
                    "place on the eye line are measured from this character's "
                    "own eye joints. Stored per character on the skeleton",
        default=0.3, min=0.01, max=100.0, soft_min=0.05, soft_max=2.0,
        subtype='DISTANCE', unit='LENGTH', precision=3,
        get=_eye_aim_distance_get, set=_eye_aim_distance_set,
    )
    # No update callback on purpose: the pose is a switched-wrap solve, far too
    # slow to run on every mouse move. Dragging these only stores a number -
    # Re-Bind applies it behind the scenes (see op_mouth.sync_cleanup_pose).
    mouth_open_amount: bpy.props.FloatProperty(
        name="Mouth Open",
        description="How far the mouth opens on the cage AND your head for the "
                    "bind. The head's open pose is generated automatically "
                    "from the wrapped cage - no sculpting needed. Dragging "
                    "this changes nothing in the viewport: Re-Bind poses "
                    "behind the scenes so the separated lips get clean weights",
        default=0.0, min=0.0, max=1.0,
        subtype='FACTOR',
    )
    eyes_close_amount: bpy.props.FloatProperty(
        name="Eyes Closed",
        description="How far the eyes close on the cage AND your head for the "
                    "bind. The head's closed pose is generated automatically "
                    "from the wrapped cage - no sculpting needed. Dragging "
                    "this changes nothing in the viewport: Re-Bind poses "
                    "behind the scenes so the separated eyelids get clean "
                    "weights",
        default=0.0, min=0.0, max=1.0,
        subtype='FACTOR',
    )

    view_mode: bpy.props.EnumProperty(
        name="Show",
        description="Which mesh to display, each with only its own landmarks. "
                    "Linking lines appear only when both are shown",
        items=[
            ('CAGE', "Head Cage", "Show the head cage and its landmarks only", 'MESH_CIRCLE', 0),
            ('TARGET', "Head Target", "Show the head target and its landmarks only", 'USER', 1),
            ('BOTH', "Both", "Show head cage + head target + linking lines", 'OVERLAY', 2),
        ],
        default='BOTH',
        update=_view_mode_update,
    )

    # --- Landmark pairs (data, drawn by the monochrome GPU overlay) ---
    landmarks: bpy.props.CollectionProperty(type=MHFRT_Landmark)
    landmark_active: bpy.props.IntProperty(default=-1)

    show_pairs_overlay: bpy.props.BoolProperty(
        name="Show Landmark Overlay",
        description="Draw the landmark markers, numbers and linking lines",
        default=True,
    )
    symmetry: bpy.props.BoolProperty(
        name="Symmetry (X)",
        description="Auto-mirror each new landmark curve across X. Draw one side; "
                    "the mirrored curve is created for you. Points near the "
                    "centre line stay single. Works best when the head is "
                    "centred on X",
        default=True,
    )
    symmetry_center_threshold: bpy.props.FloatProperty(
        name="Center Snap Distance",
        description="How close a point must be to the face centre line before "
                    "it snaps and merges with its mirrored side",
        default=0.005, min=0.0, max=0.5, soft_max=0.05,
        precision=3, subtype='FACTOR',
    )
    landmark_loop_merge_threshold: bpy.props.FloatProperty(
        name="Loop Merge Distance",
        description="How close the end of a drawn curve must be to its start "
                    "before both points merge into one closed loop",
        default=0.01, min=0.0, max=0.5, soft_max=0.05,
        precision=3, subtype='FACTOR',
    )
    landmark_lazy: bpy.props.BoolProperty(
        name="Lazy Mouse",
        description="Draw with a trailing pen instead of the raw pointer, the "
                    "way sculpting apps stabilise a stroke. The curve follows "
                    "a smoothed path behind the cursor, so hand tremor, "
                    "tablet jitter and the hook at the end of a fast stroke "
                    "never reach the landmarks. Toggle with L while drawing",
        default=True,
    )
    landmark_lazy_radius: bpy.props.FloatProperty(
        name="Lazy Radius",
        description="How far the pen trails behind the pointer, in pixels. "
                    "Larger is smoother and slower to turn; 0 removes the "
                    "leash and leaves only the stabilizer",
        default=45.0, min=0.0, max=250.0, soft_max=120.0, subtype='PIXEL',
    )
    landmark_lazy_smooth: bpy.props.FloatProperty(
        name="Stabilizer",
        description="Extra easing on top of the lazy radius: 0 is a pure "
                    "trailing rope, higher values glide through corners and "
                    "lag further behind the cursor",
        default=0.35, min=0.0, max=0.95, subtype='FACTOR',
    )
    landmark_sync_view: bpy.props.BoolProperty(
        name="Sync Views",
        description="Orbit and zoom both landmark viewports together (like "
                    "ZWrap), so the cage and your head stay at the same angle. "
                    "Turn off to move the two views independently",
        default=True,
    )

    # --- Viewport comfort (the always-visible display bar) ---
    show_bones: bpy.props.BoolProperty(
        name="Show Bones",
        description="Show or hide the active character's fitted skeleton in "
                    "the viewport",
        default=True, update=_show_bones_update,
    )
    cage_in_front: bpy.props.BoolProperty(
        name="Cage In Front",
        description="Draw the cage through the target head (X-ray style)",
        default=False, update=_in_front_update,
    )
    studio_shading: bpy.props.BoolProperty(
        name="Studio Shading",
        description="Monochrome studio viewport: matcap, cavity, dark backdrop, "
                    "grey target / graphite cage. Turning it OFF restores the "
                    "shading each 3D view had before",
        default=False, update=_studio_shading_update,
    )
    cage_studio: bpy.props.BoolProperty(
        name="Cage Studio Look",
        description="Show the cage as translucent graphite with wireframe. "
                    "Turn OFF to see the cage with its normal viewport display",
        default=True, update=_tint_update,
    )

    # --- Wrap settings ---
    wrap_quality: bpy.props.EnumProperty(
        name="Quality",
        description="Wrap solver quality (coarse-to-fine surface registration)",
        items=[
            ('DRAFT', "Draft", "Fast preview wrap"),
            ('BALANCED', "Standard", "Good default quality"),
            ('HIGH', "High", "Maximum quality, slower"),
            ('CUSTOM', "Custom", "Manual control of every parameter"),
        ],
        default='BALANCED',
    )
    wrap_use_region_mask: bpy.props.BoolProperty(
        name="Use Region_Mask",
        description="Auto-freeze cage verts in the 'Region_Mask' vertex group "
                    "(e.g. inner lips) so they keep their warp shape instead of "
                    "snapping to the surface",
        default=True,
    )
    wrap_use_icp: bpy.props.BoolProperty(
        name="Tighten to Surface",
        description="After the landmark warp, register the cage onto the head "
                    "surface (coarse-to-fine)",
        default=True,
    )
    wrap_pin_landmarks: bpy.props.BoolProperty(
        name="Pin Landmarks",
        description="Hard-lock the cage points nearest each landmark so features "
                    "don't drift",
        default=False,
    )
    wrap_iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Custom quality only: total solver effort",
        default=100, min=0, max=500,
    )
    wrap_step: bpy.props.FloatProperty(
        name="Project Strength",
        description="Custom quality only: pull strength toward the surface",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )
    wrap_smooth: bpy.props.FloatProperty(
        name="Smoothing",
        description="Custom quality only: how strongly the quad flow is relaxed",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    wrap_maxdist_frac: bpy.props.FloatProperty(
        name="Max Snap Distance",
        description="Ignore surface farther than this fraction of head size",
        default=0.15, min=0.001, max=1.0, subtype='FACTOR',
    )

    # --- N-panel state (tabs + disclosure sections) ---
    ui_tab: bpy.props.EnumProperty(
        name="Step",
        description="Active workflow step",
        items=[
            ('SETUP', "Setup", "Load the cage and pick your head mesh"),
            ('LANDMARKS', "Landmarks", "Place matching points on cage and head"),
            ('WRAP', "Wrap", "Fit the cage onto the head"),
            ('RIG', "Rig", "Build the skeleton and live control board"),
            ('TUNE', "Fine-Tune", "Place eye pivots and fit the tongue bones"),
            ('BIND', "Bind", "Transfer weights and bind the head"),
            ('PARTS', "Parts", "Attach eye, teeth and tongue meshes to their bones"),
            ('MORPHS', "Morphs", "Add empty MetaHuman morph keys for sculpting"),
            ('ANIM', "Animation", "Import a facial animation onto the control board"),
            ('EXPORT', "Export", "Export the finished character to a game engine"),
        ],
        default='SETUP',
        update=_ui_tab_update,
    )
    ui_sec_refine: bpy.props.BoolProperty(
        name="Refine", default=True,
        description="Show the wrap refinement brushes",
    )
    ui_sec_cleanup: bpy.props.BoolProperty(
        name="Weight Cleanup", default=False,
        description="Show the optional weight cleanup tools",
    )
    ui_sec_board: bpy.props.BoolProperty(
        name="Panel Layout", default=True,
        description="Show the controls for where this character's control "
                    "panels sit in the viewport",
    )
    ui_adv_landmarks: bpy.props.BoolProperty(
        name="Advanced", default=False,
        description="Show advanced landmark settings",
    )
    ui_adv_wrap: bpy.props.BoolProperty(
        name="Advanced", default=False,
        description="Show advanced wrap solver settings",
    )
    # Parts step: the name being typed for a new custom slot.  The slot itself
    # lives on the attached mesh (op_attach.CUSTOM_PART_PREFIX), so this is only
    # the field's contents between typing and pressing Add.
    part_custom_name: bpy.props.StringProperty(
        name="Part Name",
        description="Name your own part slot - brows, peach fuzz, stubble "
                    "cards, a piercing. It binds the same way the eyelashes "
                    "do: skin weights sampled off the Head Target",
        default="",
        options={'SKIP_SAVE'},
    )
    morph_extra_objects: bpy.props.CollectionProperty(type=MHFRT_MorphTarget)
    morph_extra_active_index: bpy.props.IntProperty(
        name="Active Morph Object",
        default=0,
        min=0,
    )
    morph_display_threshold: bpy.props.FloatProperty(
        name="Show Above",
        description="Hide morph channels whose live value falls below this "
                    "level. 0 shows every activated morph, 1 shows only the "
                    "ones fully fired by the current pose",
        default=0.35, min=0.0, max=1.0, subtype='FACTOR',
    )
    morph_show_muted_only: bpy.props.BoolProperty(
        name="No Bones",
        description="Narrow the list to the channels you are previewing "
                    "WITHOUT their bones (the bone icon on a row), so you can "
                    "find them all and switch them back without hunting. This "
                    "is only a viewing state - it is not the same as dialling "
                    "a channel down with Channels Intensity",
        default=False,
    )
    # Runtime UIList mirrors.  The saved ownership collection above remains
    # the source of truth; these give the panel Blender-native selectable,
    # scrollable lists without serializing 1,400+ display rows into .blend.
    morph_object_ui_items: bpy.props.CollectionProperty(
        type=MHFRT_MorphTarget,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    morph_object_ui_active_index: bpy.props.IntProperty(
        name="Active Morph Object Row",
        default=0,
        min=0,
        options={'HIDDEN', 'SKIP_SAVE'},
        # Row -> viewport: picking a row selects that mesh in the 3D view, the
        # mirror of clicking the mesh in the 3D view to select its row.  On the
        # property, so arrow-key navigation syncs as well as a click (the sync
        # ignores the add-on's own writes - see op_morphs._set_object_row).
        update=_morph_object_row_update,
    )
    # The last MESH the artist clicked in the viewport, registered for morphs or
    # not.  Armatures, curves and everything else are ignored, so posing a
    # control never changes it.  Runtime only: entering the Morphs step re-reads
    # it from the viewport (op_morphs.reset_mesh_tracking).
    morph_last_mesh: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Last Clicked Mesh",
        poll=_is_mesh,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    morph_ui_items: bpy.props.CollectionProperty(
        type=MHFRT_MorphListItem,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    # -1 means NO row is selected, and that is a perfectly normal state.  The
    # Morphs list is a live view of the pose, so when the pose stops firing the
    # selected channel its row leaves the list and the selection leaves with
    # it - nothing is ever auto-selected to fill the gap (see
    # op_morphs.prune_morph_selection).
    morph_ui_active_index: bpy.props.IntProperty(
        name="Active Morph Row",
        default=-1,
        min=-1,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    # --- The character registry: the source of truth for what exists ------
    # Saved with the file on purpose.  A character is a RECORD, created when
    # the artist starts one and removed only when they remove it; nothing is
    # inferred from the outliner any more (see core/registry).
    characters: bpy.props.CollectionProperty(type=registry.MHFRT_Character)
    active_character_uid: bpy.props.StringProperty(options={'HIDDEN'})
    character_serial: bpy.props.IntProperty(default=0, options={'HIDDEN'})

    # --- Rig list (Morphs-style template_list) ---
    # Runtime mirror of every rig in the file, rebuilt by op_rig.
    rig_ui_items: bpy.props.CollectionProperty(
        type=MHFRT_RigListItem,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    rig_ui_active_index: bpy.props.IntProperty(
        name="Active Rig Row",
        default=0,
        min=0,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    active_rig_name: bpy.props.StringProperty(
        name="Rig Name",
        description="Name of the selected rig (MHFR by default). Rename it "
                    "here; the name is stored with the rig and travels with it "
                    "when the character collection is appended into another file",
        get=_get_active_rig_name,
        set=_set_active_rig_name,
        options={'SKIP_SAVE'},
    )

    # --- Live soft session (Softwrap-style interactive refinement) ---
    live_stiffness: bpy.props.FloatProperty(
        name="Elasticity",
        description="How strongly the springs defend the wrapped shape: high = "
                    "the mesh behaves like stiff cloth and follows grabs as one "
                    "piece; low = loose, local edits",
        default=0.8, min=0.0, max=1.0, subtype='FACTOR',
    )
    live_smooth: bpy.props.FloatProperty(
        name="Smooth",
        description="Continuous tangential relaxation of the quad flow while the "
                    "session runs (never shrinks the surface)",
        default=0.25, min=0.0, max=1.0, subtype='FACTOR',
    )
    live_snap: bpy.props.FloatProperty(
        name="Snap to Head",
        description="How strongly every vertex is pulled onto the target head "
                    "surface while the session runs. 0 = free softbody",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )
    live_untangle: bpy.props.FloatProperty(
        name="Untangle",
        description="Anti-overlap force: vertices that folded through their "
                    "neighbours are detected (they sit on the wrong side of "
                    "the surface) and pushed back to the correct side. Raise "
                    "it if the mesh crumples or overlaps itself",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )
    live_keep_outside: bpy.props.BoolProperty(
        name="Keep Outside",
        description="Never accept the cage sinking under the head surface: any "
                    "vertex that ends up inside the target is projected back "
                    "out onto the skin every tick",
        default=True,
    )
    live_pin_force: bpy.props.FloatProperty(
        name="Pin Force",
        description="How hard pins (Shift+click in the session) pull on the mesh",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    live_pause: bpy.props.BoolProperty(
        name="Pause",
        description="Temporarily halt the live simulation (Space in the viewport)",
        default=False,
    )

    # --- Slide brush ---
    brush_radius: bpy.props.FloatProperty(
        name="Brush Radius",
        description="Slide brush radius as a fraction of the head size",
        default=0.08, min=0.005, max=0.5, subtype='FACTOR',
    )
    brush_strength: bpy.props.FloatProperty(
        name="Brush Strength",
        description="How strongly the brushes move vertices at their center",
        default=1.0, min=0.1, max=1.0, subtype='FACTOR',
    )
    brush_pin_boundary: bpy.props.BoolProperty(
        name="Pin Borders",
        description="Keep the mesh's open borders (neck seam, eye/mouth openings) "
                    "fixed while sliding or smoothing",
        default=True,
    )


def _cage_shape_changed(mh, depsgraph):
    """True when this tick touched the cage's geometry or its shape keys.

    The landmark overlay memoizes the cage's shape-key offsets (they are asked
    for once per landmark per redraw), so the one thing that memo cannot see -
    a sculpt or a solve rewriting key DATA without changing any key value -
    is reported here instead.
    """
    cage = mh.cage
    if cage is None or depsgraph is None:
        return False
    try:
        mesh = cage.data
        keys = mesh.shape_keys if mesh is not None else None
        # as_pointer(), NOT id(): id() is the identity of the Python wrapper,
        # and while Blender happens to cache one wrapper per datablock today
        # that is an implementation detail.  as_pointer() is the datablock's
        # own address and is the documented way to compare identity.
        watched = {cage.as_pointer(), mesh.as_pointer()}
        if keys is not None:
            watched.add(keys.as_pointer())
        for update in depsgraph.updates:
            if not update.is_updated_geometry:
                continue
            if update.id.original.as_pointer() in watched:
                return True
    except (AttributeError, ReferenceError):
        return False
    return False


@persistent
def _stale_pointer_handler(scene, depsgraph=None):
    """Empty Head Cage / Head Target / Skeleton if their object was deleted.

    Blender auto-clears PointerProperty when the target ID is fully removed
    from bpy.data, but leaves it dangling when the artist merely deletes the
    object from the scene (which just unlinks it from view-layer collections).
    Detecting the latter here means the Setup slot visibly empties as soon as
    the artist presses X on the cage or target - and the panel stops claiming
    a deleted rig is the active one."""
    mh = getattr(scene, "mhfrt", None)
    if mh is None or mh.switching_character:
        return
    from .core import render_state
    # Panel bookkeeping only, and it reads ``bpy.context`` and writes scene
    # properties - neither of which a render thread may do. See
    # core.render_state for the crash that taught us this.
    if render_state.is_rendering() \
            or getattr(depsgraph, "mode", 'VIEWPORT') == 'RENDER':
        return
    from .core import landmarks as lmdata
    if _cage_shape_changed(mh, depsgraph):
        lmdata.invalidate_display_cache()
    live_names = {obj.name for obj in scene.objects}
    dirty = False
    # 'skeleton' last: clearing it re-derives the panel's attachment state, and
    # that reads the cage/target slots this loop may just have emptied.
    for prop in ("cage", "target", "skeleton"):
        obj = getattr(mh, prop, None)
        if obj is None:
            continue
        try:
            still_in_scene = obj.name in live_names
        except ReferenceError:
            still_in_scene = False
        if not still_in_scene:
            mh.switching_character = True
            try:
                setattr(mh, prop, None)
            finally:
                mh.switching_character = False
            dirty = True
    if dirty:
        from .ops import op_rig
        op_rig.bump_rig_topology()
        try:
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except (AttributeError, RuntimeError):
            pass


_HANDLER_EVENTS = ("depsgraph_update_post",)


def register():
    bpy.utils.register_class(registry.MHFRT_Character)
    bpy.utils.register_class(MHFRT_Landmark)
    bpy.utils.register_class(MHFRT_MorphTarget)
    bpy.utils.register_class(MHFRT_MorphListItem)
    bpy.utils.register_class(MHFRT_RigListItem)
    bpy.utils.register_class(MHFRT_Props)
    bpy.types.Scene.mhfrt = bpy.props.PointerProperty(type=MHFRT_Props)
    for event in _HANDLER_EVENTS:
        hlist = getattr(bpy.app.handlers, event)
        if _stale_pointer_handler not in hlist:
            hlist.append(_stale_pointer_handler)


def unregister():
    for event in _HANDLER_EVENTS:
        hlist = getattr(bpy.app.handlers, event)
        if _stale_pointer_handler in hlist:
            hlist.remove(_stale_pointer_handler)
    del bpy.types.Scene.mhfrt
    bpy.utils.unregister_class(MHFRT_Props)
    bpy.utils.unregister_class(MHFRT_RigListItem)
    bpy.utils.unregister_class(MHFRT_MorphListItem)
    bpy.utils.unregister_class(MHFRT_MorphTarget)
    bpy.utils.unregister_class(MHFRT_Landmark)
    bpy.utils.unregister_class(registry.MHFRT_Character)
