"""Collection organization for append-friendly characters."""

import re
from contextlib import contextmanager

import bpy
from mathutils import Matrix

CHARACTER_COLL_PROP = "mhfrt_character_collection"
ROOT_COLL_PROP = "mhfrt_character_root"
COLL_ROLE_PROP = "mhfrt_collection_role"
OBJECT_ROLE_PROP = "mhfrt_object_role"

BONE_COLL_PROP = "mhfrt_bone_collection"
BONE_COLL_DEFAULT_PROP = "mhfrt_bone_collection_default"
BONE_COLL_ROOT_ROLE = "facial_regions"
FACIAL_PREFIX = "FACIAL_"
MERGED_BODY_PROP = "mhfrt_merged_body_rig"

# Stamped on an armature once its 843 joints have been sorted into the region
# collections and coloured.  Re-doing that work is ~32 ms per call and it used
# to run on EVERY character switch (organize_current -> ensure_character_
# collections -> organize_skeleton), which is what made switching rigs lag:
# nothing had changed, but the armature was dirtied every time and the whole
# rig re-evaluated behind it.  Bump ORGANIZE_VERSION whenever the grouping or
# the colours change and every existing rig re-organizes itself once.
ORGANIZED_PROP = "mhfrt_organized"
ORGANIZE_VERSION = 1

# The auto-generated basis of a character root's name.  Kept so the root can be
# renamed after the head target arrives (a character that starts life as a bare
# cage would otherwise stay called 'Head_Cage_MHFRT_Character' forever) while
# still never touching a name the artist typed themselves.
ROOT_AUTONAME_PROP = "mhfrt_root_autoname"

# One deterministic colour per anatomical region.  Custom colours are used
# instead of theme slots so the visual language stays the same in every file.
# The order is also the order shown in Armature Data > Bone Collections.
BONE_REGION_SPECS = (
    ("roots", "01 Roots", (0.82, 0.86, 0.92)),
    ("neck", "02 Neck", (0.95, 0.48, 0.12)),
    ("scalp", "03 Scalp & Ears", (0.56, 0.34, 0.18)),
    ("forehead", "04 Forehead", (0.62, 0.40, 0.90)),
    ("eyes", "05 Eyes", (0.08, 0.62, 1.00)),
    ("nose", "06 Nose", (0.96, 0.78, 0.16)),
    ("cheeks", "07 Cheeks", (0.96, 0.38, 0.62)),
    ("mouth", "08 Mouth & Lips", (0.95, 0.12, 0.22)),
    ("jaw", "09 Jaw & Chin", (0.22, 0.78, 0.44)),
    ("teeth", "10 Teeth", (0.72, 0.92, 1.00)),
    ("tongue", "11 Tongue", (0.64, 0.22, 0.82)),
    ("other", "12 Other", (0.50, 0.54, 0.60)),
)

_BONE_REGION_BY_ROLE = {
    role: (label, color) for role, label, color in BONE_REGION_SPECS
}

# The two eyeball pivot bones get their own child collection under the eyes
# region so they are one click away when placing the eye rotation centers.
EYEBALL_BONES = ("FACIAL_L_Eye", "FACIAL_R_Eye")

ROLE_HEAD = "head_target"
ROLE_CAGE = "head_cage"
ROLE_RIG = "rig"
ROLE_PANEL = "panel"
ROLE_EXTRAS = "extra_morph_objects"
ROLE_PARTS = "attached_parts"
ROLE_BODY = "body"

ROLE_LABELS = {
    ROLE_HEAD: "01 Head Target",
    ROLE_CAGE: "02 Head Cage",
    ROLE_RIG: "03 Facial Rig",
    ROLE_PANEL: "04 Control Panel",
    ROLE_EXTRAS: "05 Additional Morph Objects",
    ROLE_PARTS: "06 Attached Parts",
    ROLE_BODY: "07 Body & Clothing",
}

# Property names owned by other modules.  They are spelled out rather than
# imported because those modules import THIS one; the same trick is already
# used for MERGED_BODY_PROP and the GUI collection pointer below.
BODY_ARMATURE_PROP = "mhfrt_body_armature"     # props.RIG_BODY_ARMATURE_PROP
ATTACH_PART_PROP = "mhfrt_attached_part"       # op_attach
MORPH_EXTRA_PROP = "mhfrt_extra_morph_object"  # op_morphs
CONTROL_TEMPLATE_PROP = "mhfrt_control_name"   # board.TEMPLATE_PROP
WIDGET_OWNER_PROP = "mhfrt_board_widget_rig"   # board.WIDGET_OWNER_PROP
GUI_COLL_PROP = "mhfrt_gui_coll"               # props.RIG_GUI_COLL_PROP


def _collection_chains(view_layer, objects):
    """Every LayerCollection holding one of `objects`, ancestors included.

    That chain is what decides whether the object's base is enabled at all, so
    it has to be opened up together with the object's own flags."""
    wanted = set()
    for obj in objects:
        if obj is None:
            continue
        try:
            wanted.update(obj.users_collection)
        except (AttributeError, ReferenceError):
            continue
    if view_layer is None or not wanted:
        return []
    found = []

    def walk(layer_coll, chain):
        chain = chain + [layer_coll]
        if layer_coll.collection in wanted:
            found.extend(chain)
        for child in layer_coll.children:
            walk(child, chain)

    try:
        walk(view_layer.layer_collection, [])
    except (AttributeError, ReferenceError):
        return []
    seen = set()
    unique = []
    for layer_coll in found:
        if id(layer_coll.collection) not in seen:
            seen.add(id(layer_coll.collection))
            unique.append(layer_coll)
    return unique


@contextmanager
def shown_for_edit(*objects, view_layer=None):
    """Force `objects` visible and selectable - for Edit Mode, or for an
    operator like ``object.join`` that needs a real selection - then put every
    flag back exactly as it was.

    Blender refuses Edit Mode on a hidden object, and the MONITOR icon
    (``hide_viewport``) goes further than the eye: it DISABLES the object's
    base, so the next view-layer sync drops it as the active object and the
    poll fails with "Context missing active object" - no matter that the code
    just assigned it. The same icon on any COLLECTION above the object does the
    same thing, which is why `view_layer` is worth passing: artists keep their
    character rig hidden one of those ways almost always. Callers still have to
    set the active object INSIDE the window.
    """
    saved_objects = []
    saved_colls = []
    for layer_coll in _collection_chains(view_layer, objects):
        coll = layer_coll.collection
        saved_colls.append((layer_coll, layer_coll.hide_viewport,
                            coll.hide_viewport, coll.hide_select))
        layer_coll.hide_viewport = False
        coll.hide_viewport = False
        coll.hide_select = False
    for obj in objects:
        if obj is None:
            continue
        try:
            hidden = obj.hide_get()
        except RuntimeError:            # no view layer for this object
            hidden = False
        saved_objects.append((obj, hidden, obj.hide_viewport, obj.hide_select))
        obj.hide_viewport = False
        obj.hide_select = False
        try:
            obj.hide_set(False)
        except RuntimeError:
            pass
    try:
        yield
    finally:
        # unhide order reversed: hide_set() only bites while the base is still
        # enabled, so the eye has to go back BEFORE the monitor icon does
        for obj, hidden, hide_viewport, hide_select in saved_objects:
            try:
                obj.hide_set(hidden)
            except (ReferenceError, RuntimeError):
                pass                    # joined away, or no view layer
            try:
                obj.hide_viewport = hide_viewport
                obj.hide_select = hide_select
            except ReferenceError:
                pass
        for layer_coll, lc_hidden, hide_viewport, hide_select in saved_colls:
            try:
                layer_coll.hide_viewport = lc_hidden
                layer_coll.collection.hide_viewport = hide_viewport
                layer_coll.collection.hide_select = hide_select
            except (ReferenceError, RuntimeError):
                pass


def deforming_armatures(*objects):
    """Every armature that moves these meshes: modifiers, and bone parenting.

    Both routes matter and artists use both - a head skinned by an Armature
    modifier, and a prop or a head parented straight to a bone.  Deduplicated
    by object, and the meshes' own parents are walked up so a head under a rig
    under a root still finds the rig.
    """
    found = {}
    for obj in objects:
        if obj is None:
            continue
        try:
            modifiers = list(obj.modifiers)
        except (AttributeError, ReferenceError):
            modifiers = []
        candidates = [m.object for m in modifiers if m.type == 'ARMATURE']
        parent = getattr(obj, "parent", None)
        seen = set()
        while parent is not None and parent.as_pointer() not in seen:
            seen.add(parent.as_pointer())
            candidates.append(parent)
            parent = parent.parent
        for armature in candidates:
            if (armature is not None and armature.type == 'ARMATURE'
                    and armature.data is not None):
                found[armature.as_pointer()] = armature
    return list(found.values())


@contextmanager
def meshes_at_rest(*objects, view_layer=None):
    """Hold every armature deforming `objects` in REST for the duration.

    Landmarking, wrapping and weight transfer all read the head's SURFACE and
    write something meant to be true of the character's neutral shape: a
    landmark stores a position, the wrap solves the cage onto it, the bind
    samples correspondence.  If the artist left their body rig on an animation
    frame - mid-shot, arms up, head turned - every one of those records the
    POSED surface as if it were the rest one, and the error is baked in with
    nothing on screen to say so.

    Blender's own switch is used (``Armature.pose_position``), so this costs one
    flag per armature and no geometry, and the artist's pose is untouched: it is
    still there, just not applied, and it comes back on exit even if the caller
    raises.  Armatures already in REST are left alone so nothing is "restored"
    onto a rig that was never posed.

    Deliberately includes OUR facial rig too - on a re-wrap of a live character
    the face may be posed by its own control board, and that is exactly as
    wrong to wrap onto as a posed body.
    """
    armatures = deforming_armatures(*objects)
    saved = []
    for armature in armatures:
        data = armature.data
        if data is None or not _writable_data(data):
            continue
        if data.pose_position == 'REST':
            continue
        saved.append((data, data.pose_position))
        data.pose_position = 'REST'
    if saved:
        _update_view_layer(view_layer)
    try:
        yield [armature.name for armature in armatures]
    finally:
        restored = False
        for data, position in saved:
            try:
                data.pose_position = position
                restored = True
            except (ReferenceError, AttributeError):
                pass
        if restored:
            _update_view_layer(view_layer)


def _update_view_layer(view_layer=None):
    """Flush the rest-pose switch so the next read sees the un-posed mesh."""
    try:
        if view_layer is not None:
            view_layer.update()
        else:
            bpy.context.view_layer.update()
    except (AttributeError, RuntimeError):
        pass


def reveal(obj, view_layer=None):
    """Unhide `obj` and every collection above it, and LEAVE them unhidden.

    The counterpart to :func:`shown_for_edit`, which puts every flag back: a
    mode change is not a one-shot operation, so an object the artist is about to
    work in has to stay visible afterwards.  Same hazards as that function -
    Blender refuses a mode change on a hidden object, and the monitor icon
    (``hide_viewport``) on the object OR any collection above it disables the
    object's base, so the view layer drops it as the active object before
    ``mode_set`` ever runs.

    Returns True when something was actually hidden and had to be revealed.
    """
    if obj is None:
        return False
    changed = False
    for layer_coll in _collection_chains(view_layer, (obj,)):
        coll = layer_coll.collection
        for owner, attr in ((layer_coll, "hide_viewport"),
                            (coll, "hide_viewport"),
                            (coll, "hide_select")):
            try:
                if getattr(owner, attr):
                    setattr(owner, attr, False)
                    changed = True
            except (AttributeError, ReferenceError):
                continue
    if obj.hide_viewport or obj.hide_select:
        obj.hide_viewport = False
        obj.hide_select = False
        changed = True
    try:
        if obj.hide_get():
            obj.hide_set(False)
            changed = True
    except RuntimeError:            # not in this view layer
        pass
    return changed


def match_cage_parent(cage, target):
    """Hang the cage off whatever the head target hangs off. Returns True if moved.

    The cage is a stand-in for the head: it is wrapped onto it, it previews the
    rig through the same Armature modifier, and every landmark on it is a point
    on that head.  So anything that carries the head around has to carry the
    cage too - a head parented to a body rig, to a bone, or to an empty the
    artist moves the character with, would otherwise leave the cage standing
    where the character used to be, and every measurement taken from it
    (placement, landmarks, the wrap) is then taken in the wrong place.

    Mirrors parent, parent type and parent bone, and keeps the cage's world
    transform, so nothing visibly moves the first time this runs on an existing
    scene.

    ``ARMATURE`` parenting is deliberately downgraded to ``OBJECT``: that mode
    is a DEFORM, and the cage already has its own Armature modifier pointing at
    the facial skeleton.  Copying it verbatim would deform the cage twice, by
    two different rigs.  Object parenting inherits the same motion without
    touching a vertex.
    """
    if cage is None or target is None or cage is target:
        return False
    parent = target.parent
    # The only real hazard is a CYCLE: parenting the cage to something that is
    # already below the cage. Note the direction - an earlier version asked
    # whether the cage was under the parent, which is true the moment this has
    # run once, so every follow-up call refused to update the parenting at all.
    if parent is cage or (parent is not None
                          and parent in getattr(cage, "children_recursive", ())):
        return False
    parent_type = target.parent_type if parent is not None else 'OBJECT'
    if parent_type == 'ARMATURE':
        parent_type = 'OBJECT'
    parent_bone = target.parent_bone if parent_type == 'BONE' else ""
    if (cage.parent is parent and cage.parent_type == parent_type
            and (cage.parent_bone or "") == (parent_bone or "")):
        return False
    world = cage.matrix_world.copy()
    cage.parent = parent
    cage.parent_type = parent_type
    cage.parent_bone = parent_bone
    cage.matrix_parent_inverse = Matrix.Identity(4)
    cage.matrix_world = world
    return True


def set_prop(datablock, key, value):
    """Write an ID property only when it would actually change.

    Every write to a custom property tags its datablock for a depsgraph
    update, so re-stamping unchanged values on a whole character (which is
    what organizing an already-organized rig does) is not free: it re-
    evaluates the rig, the meshes it deforms, and the panel behind them.
    """
    if datablock is None:
        return False
    try:
        current = datablock.get(key)
    except (AttributeError, ReferenceError):
        return False
    try:
        if isinstance(value, (bpy.types.ID, type(None))):
            same = current is value or (
                isinstance(current, bpy.types.ID) and current == value)
        else:
            same = current == value
    except (ReferenceError, TypeError):
        same = False            # can't compare it - write and be sure
    if same:
        return False
    try:
        datablock[key] = value
    except (TypeError, ValueError, AttributeError, ReferenceError):
        # ValueError covers an embedded ID (a scene's master collection) -
        # never something we file a character under, but this helper must not
        # be the thing that takes an operator down.
        return False
    return True


def _bone_region(name):
    """Return the anatomical region for one imported MetaHuman joint."""
    if not name.startswith(FACIAL_PREFIX):
        return None

    low = name.lower()
    if "facialroot" in low:
        return "roots"
    if "tongue" in low:
        return "tongue"
    if "teeth" in low:
        return "teeth"
    if any(token in low for token in
           ("eye", "eyelid", "eyelashes", "eyesack", "pupil")):
        return "eyes"
    if "lip" in low or "mouth" in low:
        return "mouth"
    if "nose" in low or "nostril" in low:
        return "nose"
    if "cheek" in low or "nasolabial" in low:
        return "cheeks"
    if any(token in low for token in
           ("jaw", "chin", "masseter")):
        return "jaw"
    if "neck" in low or "adams" in low:
        return "neck"
    if "forehead" in low:
        return "forehead"
    if any(token in low for token in
           ("hair", "sideburn", "temple", "ear", "skull")):
        return "scalp"
    return "other"


def _armature_collections(armature):
    collections = getattr(armature, "collections_all", None)
    if collections is None:
        collections = getattr(armature, "collections", ())
    return list(collections)


def _bone_collection_for_role(armature, role):
    return next(
        (coll for coll in _armature_collections(armature)
         if coll.get(BONE_COLL_PROP) == role),
        None,
    )


def _ensure_bone_collection(armature, role, name, *, parent=None,
                            visible=True):
    coll = _bone_collection_for_role(armature, role)
    created = coll is None
    if created:
        coll = armature.collections.new(name, parent=parent)
        coll[BONE_COLL_PROP] = role
    else:
        coll.name = name
        if parent is not None and coll.parent != parent:
            coll.parent = parent
    if not coll.get(BONE_COLL_DEFAULT_PROP):
        # The default lands exactly once. Without this a rig built by an
        # earlier version would never pick a changed default up - its
        # collections already exist, so nothing would ever set them. Once
        # stamped the artist's own eye click is the only authority.
        coll.is_visible = visible
        coll[BONE_COLL_DEFAULT_PROP] = True
    return coll, created


def _set_bone_color(bone, color):
    bone_color = getattr(bone, "color", None)
    if bone_color is None:
        return
    bone_color.palette = 'CUSTOM'
    custom = bone_color.custom
    custom.normal = color
    custom.select = tuple(min(1.0, channel * 0.65 + 0.35)
                          for channel in color)
    custom.active = (1.0, 0.82, 0.28)


def _organize_stamp(skel, armature):
    """What the last organize pass was run against.

    FLOATS, and they have to stay floats.  This lands on the armature DATA, and
    Blender's FBX exporter writes an armature as a Null node whose NodeAttribute
    carries the data-block's custom properties - unconditionally, ignoring
    ``use_custom_props`` (``export_fbx_bin.fbx_data_empty_elements``, verified in
    5.1.2).  A three-element list is written as ``p_vector``, which feeds each
    entry to ``encode_bin.add_float64`` and its ``assert isinstance(data, float)``:
    with ints in there, every Unity and Unreal character export died on that
    assertion.  Small integers are exact in float64, and Python compares
    ``[3, 843, 0] == [3.0, 843.0, 0.0]`` as equal, so a stamp written by an older
    build still matches and no rig is re-organized for this.
    """
    return [float(ORGANIZE_VERSION), float(len(armature.bones)),
            1.0 if skel.get(MERGED_BODY_PROP) else 0.0]


def _writable_data(datablock):
    """False for a library-linked datablock - nothing on it can be written."""
    return (datablock is not None
            and getattr(datablock, "library", None) is None
            and getattr(datablock, "override_library", None) is None)


def _stamp_organized(skel, armature):
    try:
        armature[ORGANIZED_PROP] = _organize_stamp(skel, armature)
    except (AttributeError, TypeError, ReferenceError):
        pass                    # linked / read-only: nothing to remember


def skeleton_is_organized(skel):
    """True when this armature's joints are already grouped and coloured."""
    if skel is None or skel.type != 'ARMATURE':
        return False
    armature = skel.data
    if not hasattr(armature, "collections"):
        return False
    stamp = armature.get(ORGANIZED_PROP)
    if stamp is None:
        return False
    try:
        return list(stamp) == _organize_stamp(skel, armature)
    except TypeError:
        return False


def organize_skeleton(skel, force=False):
    """Colour and group an imported MetaHuman armature for viewport work.

    Primary joints live directly in anatomical collections.  Dense 12IPV
    corrective joints live in a hidden child collection under their region,
    keeping the initial viewport readable while leaving every joint one click
    away.  The operation is idempotent and does not alter non-facial bones.

    Sorting 843 joints is ~32 ms and dirties the armature, so an armature that
    already carries a matching stamp is left completely alone.  Pass
    ``force=True`` (Refresh Rigs) to re-run it regardless.
    """
    if skel is None or skel.type != 'ARMATURE':
        return {}
    armature = skel.data
    if not hasattr(armature, "collections"):
        return {}
    if not force and skeleton_is_organized(skel):
        return {}
    if not _writable_data(armature):
        # A linked or overridden armature cannot have its bone collections
        # rebuilt at all - the old code got as far as collections.new() and
        # raised.  Nothing to do, and saying so costs nothing.
        return {}

    entries = []
    populated_roles = set()
    corrective_roles = set()
    has_eyeballs = False
    for bone in armature.bones:
        role = _bone_region(bone.name)
        if role is None:
            continue
        is_corrective = "12ipv" in bone.name.lower()
        is_eyeball = bone.name in EYEBALL_BONES
        entries.append((bone, role, is_corrective, is_eyeball))
        populated_roles.add(role)
        if is_corrective:
            corrective_roles.add(role)
        if is_eyeball:
            has_eyeballs = True
    if not entries:
        # Not a MetaHuman armature (a merged character rig before the facial
        # joints arrive, or the artist's own body rig).  Stamped anyway so the
        # bone scan is not repeated on every switch; the bone count is part of
        # the stamp, so joints appearing later still re-run it.
        _stamp_organized(skel, armature)
        return {}

    root, created = _ensure_bone_collection(
        armature,
        BONE_COLL_ROOT_ROLE,
        "MHFR Facial",
        visible=False,
    )
    if created:
        # Collapsed and hidden: 843 joints are not what anyone wants to look at
        # or click through, and nothing needs them visible - they still deform
        # while hidden, and the control board is a separate visible armature.
        # Only on creation, so an artist who turns them on keeps them on.
        root.is_expanded = False

    regions = {}
    correctives = {}
    for role, label, _color in BONE_REGION_SPECS:
        if role not in populated_roles:
            continue
        regions[role], _created = _ensure_bone_collection(
            armature, f"region:{role}", label, parent=root)
        if role in corrective_roles:
            correctives[role], _created = _ensure_bone_collection(
                armature,
                f"correctives:{role}",
                f"{label} - 12IPV Correctives",
                parent=regions[role],
                visible=False,
            )

    eyeballs = None
    if has_eyeballs:
        eyes_label = _BONE_REGION_BY_ROLE["eyes"][0]
        eyeballs, _created = _ensure_bone_collection(
            armature,
            "eyeballs",
            f"{eyes_label} - Eyeballs",
            parent=regions["eyes"],
        )

    managed = {
        coll for coll in _armature_collections(armature)
        if coll.get(BONE_COLL_PROP)
    }
    is_merged = bool(skel.get(MERGED_BODY_PROP))

    # The source asset ships 29 bone collections of its own (Volume, Internal,
    # the *_grp sets) and they overlap heavily - most joints belong to six at
    # once - so the append drags them into every rig. A bone is visible when
    # ANY collection holding it is visible, which is what made the add-on's own
    # eye toggle look broken: hiding our collection still left five others
    # showing the same bone. Note the ones that hold nothing but our joints;
    # the pass below empties them and they are removed at the end. A collection
    # holding even one of the character's own bones is never touched.
    inherited = [
        coll.name for coll in _armature_collections(armature)
        if coll not in managed and len(coll.bones)
        and all(b.name.startswith(FACIAL_PREFIX) for b in coll.bones)
    ]

    counts = {role: {"primary": 0, "corrective": 0}
              for role in _BONE_REGION_BY_ROLE}
    for bone, role, is_corrective, is_eyeball in entries:
        if is_eyeball:
            dest = eyeballs
        elif is_corrective:
            dest = correctives[role]
        else:
            dest = regions[role]
        # Exactly one collection per bone, ours - anything else would keep it
        # visible behind our back.
        for coll in list(bone.collections):
            if coll != dest:
                coll.unassign(bone)
        if bone.name not in dest.bones:
            dest.assign(bone)

        _set_bone_color(bone, _BONE_REGION_BY_ROLE[role][1])
        pose_bone = skel.pose.bones.get(bone.name) if skel.pose else None
        if pose_bone is not None and hasattr(pose_bone, "color"):
            # DEFAULT makes pose mode inherit the armature-bone region colour.
            pose_bone.color.palette = 'DEFAULT'
        bucket = "corrective" if is_corrective else "primary"
        counts[role][bucket] += 1

    # Now empty, the inherited collections go - including on a merged rig,
    # where leaving them behind is exactly what buried the character's own
    # collections in noise. Re-fetched by name because removing one
    # invalidates the other references.
    for name in inherited:
        coll = armature.collections_all.get(name)
        if coll is None or coll.get(BONE_COLL_PROP):
            continue
        if len(coll.bones) == 0 and len(coll.children) == 0:
            armature.collections.remove(coll)

    armature.show_bone_colors = True
    if not is_merged:
        armature.display_type = 'STICK'
        armature.show_names = False
        armature.show_axes = False
        skel.show_in_front = True

    _stamp_organized(skel, armature)
    return counts


def _as_collection(value):
    return value if isinstance(value, bpy.types.Collection) else None


def _clean_name(name):
    clean = re.sub(r"\s+", "_", name.strip()) if name else "Character"
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean)
    return clean.strip("_.") or "Character"


def _collection_parents(child):
    parents = []
    for scene in bpy.data.scenes:
        if child.name in scene.collection.children:
            parents.append(scene.collection)
    for coll in bpy.data.collections:
        if child.name in coll.children:
            parents.append(coll)
    return parents


def _ancestor_roots(coll, seen=None):
    """EVERY character root above `coll` - a linked-duplicate collection puts
    the same objects under two roots, and answering with "whichever came
    first" is how the panel used to jump to the wrong character."""
    if coll is None:
        return []
    if coll.get(ROOT_COLL_PROP):
        return [coll]
    seen = seen if seen is not None else set()
    if coll.name in seen:
        return []
    seen.add(coll.name)
    found = []
    for parent in _collection_parents(coll):
        for root in _ancestor_roots(parent, seen):
            if root not in found:
                found.append(root)
    return found


def _ancestor_root(coll, seen=None):
    roots = _ancestor_roots(coll, seen)
    return roots[0] if roots else None


def _in_scene(root, scene):
    """True when `root` is reachable from this scene's collection tree."""
    if root is None or scene is None:
        return False
    try:
        return root in scene.collection.children_recursive
    except (AttributeError, ReferenceError):
        pass
    stack = [scene.collection]
    seen = set()
    while stack:
        coll = stack.pop()
        if coll.name in seen:
            continue
        seen.add(coll.name)
        if coll == root:
            return True
        stack.extend(coll.children)
    return False


def _pick_root(candidates, scene=None):
    """One root out of several, chosen the same way every time.

    A root the CURRENT scene can reach wins - that is the character the artist
    is looking at.  Ties break on name so the answer never depends on
    ``bpy.data`` ordering, which is what made the active rig flicker between
    two duplicated collections.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    here = [c for c in candidates if _in_scene(c, scene)]
    pool = here or candidates
    return min(pool, key=lambda c: c.name)


def _root_from_objects(*objects, scene=None):
    for obj in objects:
        if obj is None:
            continue
        # The explicit pointer wins: it says which character this object was
        # actually filed under, and it survives a duplicated collection.
        root = _as_collection(obj.get(CHARACTER_COLL_PROP))
        if root is not None and root.get(ROOT_COLL_PROP):
            return root
        candidates = []
        for coll in obj.users_collection:
            for found in _ancestor_roots(coll):
                if found not in candidates:
                    candidates.append(found)
        root = _pick_root(candidates, scene)
        if root is not None:
            return root
    return None


def _target_parent_collection(context, target, *fallbacks):
    for candidate in (target, *fallbacks):
        if candidate is None:
            continue
        for coll in candidate.users_collection:
            if coll.get(ROOT_COLL_PROP) or coll.get(COLL_ROLE_PROP):
                continue
            return coll
        if candidate.users_collection:
            root = _ancestor_root(candidate.users_collection[0])
            if root is not None:
                parents = _collection_parents(root)
                if parents:
                    return parents[0]
            return candidate.users_collection[0]
    return context.scene.collection


def _link_child(parent, child):
    if child.name not in parent.children:
        parent.children.link(child)


def _unlink_child(parent, child):
    if child.name in parent.children:
        parent.children.unlink(child)


def _root_name_for(target, *objects):
    name_owner = target
    if name_owner is None:
        name_owner = next((o for o in objects if o is not None), None)
    base = _clean_name(name_owner.name if name_owner is not None else "Character")
    return f"{base}_MHFRT_Character"


def _rename_root(root, target, *objects):
    """Re-derive the root's name once a better one becomes available.

    A character now gets its collection the moment its cage is loaded, so the
    first name available is 'Head_Cage'.  When the artist picks their head
    mesh the root is renamed to match - but ONLY while the name is still the
    one we generated: the moment the artist renames the collection themselves
    it is theirs, and nothing here touches it again.
    """
    if root is None or target is None:
        return
    if not _writable_data(root):
        return                      # linked from another file: not ours
    auto = str(root.get(ROOT_AUTONAME_PROP, ""))
    if not auto or auto != root.name:
        return                      # renamed by hand - leave it alone
    wanted = _root_name_for(target, *objects)
    if wanted == root.name:
        return
    root.name = wanted
    root[ROOT_AUTONAME_PROP] = root.name


def _ensure_root(context, target, *objects, scene=None, record=None):
    root = _root_from_objects(target, *objects, scene=scene)
    if root is not None:
        _rename_root(root, target, *objects)
        return root

    parent = _target_parent_collection(context, target, *objects)
    # Half-built characters (cage but no head yet) still deserve a root so
    # they stay listed in the Characters panel while the artist finishes setup.
    root = bpy.data.collections.new(_root_name_for(target, *objects))
    root[ROOT_COLL_PROP] = True
    root[ROOT_AUTONAME_PROP] = root.name
    # We made this collection, so removal is allowed to delete it.  A
    # collection without that stamp is the artist's and is never deleted, only
    # emptied of our things.
    from . import provenance
    provenance.mark(record if record is not None else _record_for(None), root)
    _link_child(parent, root)
    return root


def _ensure_subcollections(root, record=None):
    from . import provenance
    owner = record if record is not None else _record_for(root)
    out = {}
    for role, label in ROLE_LABELS.items():
        coll = next((c for c in root.children
                     if c.get(COLL_ROLE_PROP) == role), None)
        if coll is None:
            coll = bpy.data.collections.new(label)
            coll[COLL_ROLE_PROP] = role
            provenance.mark(owner, coll)
            root.children.link(coll)
        elif not provenance.made_by(coll):
            # Made by an older version of the add-on, before creation was
            # stamped: it is plainly ours (it carries our role property), so
            # adopt it rather than leaving it undeletable forever.
            provenance.mark(owner, coll)
        out[role] = coll
    return out


def _rename_collection(coll, name):
    """Rename a role collection, unless it came from a library."""
    if coll is None or not _writable_data(coll) or coll.name == name:
        return
    coll.name = name


def _link_object(coll, obj):
    if obj and obj.name not in coll.objects:
        coll.objects.link(obj)


def owned_by_other_character(obj, root):
    """True when `obj` is already filed under a DIFFERENT character root.

    Two characters can legitimately share an armature (one body rig, two
    faces), and the mesh discovery below would then hand the same body to
    whichever character was organized last - dragging it out of the first
    character's collection every time the artist switched.  The first
    character to claim an object keeps it; Remove Rig is what releases it.
    """
    if obj is None or root is None:
        return False
    owner = _as_collection(obj.get(CHARACTER_COLL_PROP))
    if owner is None or owner == root:
        return False
    try:
        return bool(owner.get(ROOT_COLL_PROP)) and owner.users > 0
    except (AttributeError, ReferenceError):
        return False


def _record_for(root):
    """The character record that owns this root, or None (cheap: by stamp)."""
    if root is None:
        return None
    from . import registry
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return None
    try:
        uid = root.get(registry.UID_PROP)
    except (AttributeError, ReferenceError):
        return None
    return registry.find(scene, uid) if uid else registry.active(scene)


def _move_object(obj, dest, root, role, *, claim=True, record=None):
    if obj is None or not _writable_data(obj):
        return                      # linked from another file: not ours
    if not claim and owned_by_other_character(obj, root):
        return
    # Where this object lived BEFORE we filed it under the character - written
    # down now, while it is still true.  Removing the character puts it back
    # there; without this the body, the clothes and the hair were unlinked from
    # the artist's own collections and went with the root when it was deleted.
    from . import provenance
    if not provenance.made_by(obj):
        provenance.snapshot_object(record if record is not None
                                   else _record_for(root), obj, role=role)
    _link_object(dest, obj)
    # set_prop, not a plain write: re-stamping an unchanged value dirties the
    # object and re-evaluates everything hanging off it on every switch.
    set_prop(obj, CHARACTER_COLL_PROP, root)
    set_prop(obj, OBJECT_ROLE_PROP, role)

    for coll in list(obj.users_collection):
        if coll == dest:
            continue
        coll.objects.unlink(obj)


def _move_collection(coll, dest, root, role):
    if coll is None or not _writable_data(coll):
        return
    _link_child(dest, coll)
    set_prop(coll, CHARACTER_COLL_PROP, root)
    set_prop(coll, COLL_ROLE_PROP, role)
    for parent in list(_collection_parents(coll)):
        if parent == dest:
            continue
        _unlink_child(parent, coll)


# ------------------------------------------------- the rest of the character ---
#
# A character that arrived already rigged - Rigify or anything else - is body,
# clothing, hair and shoes on an armature this add-on did not build.  None of
# that is the add-on's to create, but all of it IS the character, so pretending
# it does not exist is what left an export containing a head and nothing else.
# These two functions are the one place that answers "what else is in here",
# and the organizer, every exporter and the driver bake all read them.

def _deforms_a_mesh(armature, scene=None):
    """True when this armature actually skins something.

    The test is an Armature MODIFIER, not object parenting: the bundled head
    asset arrives parented to its own source armature, and that leftover drives
    nothing at all.  Counting it as a body rig would have blocked every export
    with a "merge first" that had nothing to merge.
    """
    objects = scene.objects if scene is not None else bpy.data.objects
    pointer = armature.as_pointer()
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for modifier in obj.modifiers:
            if (modifier.type == 'ARMATURE' and modifier.object is not None
                    and modifier.object.as_pointer() == pointer):
                return True
    return False


def character_armatures(skel, target=None, scene=None):
    """Every armature that moves this character, the face rig first.

    After a merge that is one object and the list has a single entry.  Before
    one it is the facial rig plus whichever armature was already driving the
    character: an armature ancestor of the face rig, an armature bound to the
    head target, or the body the artist picked in the Rig step.  Candidates
    that skin nothing are dropped - see :func:`_deforms_a_mesh`.
    """
    candidates = []

    def add(obj):
        if (isinstance(obj, bpy.types.Object) and obj.type == 'ARMATURE'
                and obj not in candidates):
            candidates.append(obj)

    add(skel)
    node = skel.parent if skel is not None else None
    seen = set()
    while node is not None and node.as_pointer() not in seen:
        seen.add(node.as_pointer())
        add(node)
        node = node.parent
    if target is not None and target.type == 'MESH':
        add(target.parent)
        for modifier in target.modifiers:
            if modifier.type == 'ARMATURE':
                add(modifier.object)
    if skel is not None:
        add(skel.get(BODY_ARMATURE_PROP))

    if len(candidates) <= 1:
        return candidates
    return [candidates[0]] + [armature for armature in candidates[1:]
                              if _deforms_a_mesh(armature, scene)]


def _is_addon_object(obj, panel_objects):
    """True for anything the add-on made: cage, board, widgets, parts, morphs."""
    if obj.as_pointer() in panel_objects:
        return True
    if obj.get(CONTROL_TEMPLATE_PROP) or obj.get(WIDGET_OWNER_PROP):
        return True
    if obj.get(ATTACH_PART_PROP) or obj.get(MORPH_EXTRA_PROP):
        return True
    return str(obj.get(OBJECT_ROLE_PROP, "")) in {
        ROLE_CAGE, ROLE_PANEL, ROLE_HEAD, ROLE_PARTS, ROLE_EXTRAS}


def character_meshes(armatures, cage=None, target=None, scene=None):
    """The character's OTHER meshes: body, clothing, hair, shoes.

    Anything an armature of this character deforms - by modifier or by being
    parented to it - that is not the head, the cage, or a piece of add-on
    machinery.  Scanning by armature rather than by collection is deliberate:
    it is what the meshes are bound to that makes them this character, and a
    scene where the artist never tidied up would defeat anything else.
    """
    armatures = [a for a in armatures if a is not None]
    if not armatures:
        return []
    wanted = {a.as_pointer() for a in armatures}

    panel_objects = set()
    for armature in armatures:
        collection = armature.get(GUI_COLL_PROP)
        if isinstance(collection, bpy.types.Collection):
            panel_objects.update(o.as_pointer() for o in collection.all_objects)
    for collection in bpy.data.collections:
        if collection.get(COLL_ROLE_PROP) == ROLE_PANEL:
            panel_objects.update(o.as_pointer() for o in collection.all_objects)

    skip = {o.as_pointer() for o in (cage, target) if o is not None}
    objects = scene.objects if scene is not None else bpy.data.objects
    out = []
    for obj in objects:
        if obj.type != 'MESH' or obj.as_pointer() in skip:
            continue
        if _is_addon_object(obj, panel_objects):
            continue
        driven = any(m.type == 'ARMATURE' and m.object is not None
                     and m.object.as_pointer() in wanted
                     for m in obj.modifiers)
        if not driven:
            node = obj.parent
            seen = set()
            while node is not None and node.as_pointer() not in seen:
                seen.add(node.as_pointer())
                if node.as_pointer() in wanted:
                    driven = True
                    break
                node = node.parent
        if driven:
            out.append(obj)
    return out


def ensure_character_collections(context, cage=None, target=None, skel=None,
                                 gui_coll=None, extras=(), parts=(),
                                 root=None, record=None):
    """Create/update the character root and role collections.

    The root is created beside the head target's current collection so appending
    that one collection later brings the head, cage, rig, panel, and additional
    morph objects together.
    """
    # Bone organization is independent of the scene collection hierarchy and
    # should still happen when an older rig has lost its target link.
    organize_skeleton(skel)
    # Likewise the cage's parent: it belongs wherever the head belongs, and this
    # is the pass every path that rearranges a character already runs.
    match_cage_parent(cage, target)
    if context is None:
        return {}
    # Half-built characters (cage OR target only) also get a root: the
    # Characters panel needs it to keep listing the character while the
    # artist is still assembling it.  We only bail if every hook is empty.
    if target is None and cage is None and skel is None:
        return {}

    # The caller (the character RECORD) says which collection this is.  Only
    # a character that has never been filed falls back to resolving one from
    # its objects - which is what used to hand a brand-new cage to the
    # previous character's collection and collapse two characters into one.
    if record is None:
        record = _record_for(root)
    if root is None or not _writable_data(root):
        root = _ensure_root(context, target, cage, skel, scene=context.scene,
                            record=record)
    else:
        _rename_root(root, target, cage, skel)

    # A LINKED character belongs to another .blend and every datablock in it is
    # read-only.  Renaming its role collections raised on the first call and on
    # every call after it - one linked character poisoned the organizer for the
    # whole session.  Linking a rig in is a completely ordinary thing to do, so
    # it has to be a quiet no-op, not an error: the character is already
    # organized, in the file it came from.
    if not _writable_data(root):
        return {"root": root}

    set_prop(root, ROOT_COLL_PROP, True)

    # A character whose collections live in ANOTHER scene is not ours to
    # rearrange from here.  Filing its objects into that root would unlink
    # them from the scene the artist is actually looking at - they would
    # vanish from the viewport, and the panel would then clear its own
    # cage/head/skeleton slots because those objects are no longer in the
    # scene.  A root that is in NO scene at all is a different story: that is
    # an appended or orphaned character, and adopting it here is exactly right.
    if not _in_scene(root, context.scene):
        if _collection_parents(root):
            return {"root": root}
        _link_child(context.scene.collection, root)

    subs = _ensure_subcollections(root, record=record)

    _move_object(target, subs[ROLE_HEAD], root, ROLE_HEAD, record=record)
    _move_object(cage, subs[ROLE_CAGE], root, ROLE_CAGE, record=record)
    if skel is not None and skel.get(MERGED_BODY_PROP):
        # Once the artist explicitly merges for game export, the surviving
        # armature is the character rig (body + face). Keep it inside the
        # appendable character root instead of leaving it behind wherever the
        # body rig happened to live before the merge.
        _rename_collection(subs[ROLE_RIG], "03 Character Rig")
    else:
        _rename_collection(subs[ROLE_RIG], ROLE_LABELS[ROLE_RIG])
    _move_object(skel, subs[ROLE_RIG], root, ROLE_RIG, record=record)
    _move_collection(gui_coll, subs[ROLE_PANEL], root, ROLE_PANEL)
    for obj in extras or ():
        # The cage can be registered as a morph object, but it belongs in the
        # Head Cage collection (moved above), not with the extra morph meshes.
        if obj is not None and obj == cage:
            continue
        _move_object(obj, subs[ROLE_EXTRAS], root, ROLE_EXTRAS, record=record)
    for obj in parts or ():
        _move_object(obj, subs[ROLE_PARTS], root, ROLE_PARTS, record=record)

    # Body, clothing and the armature they came in on.  Discovered rather than
    # declared: the artist never told the add-on these existed.
    armatures = character_armatures(skel, target, scene=context.scene)
    for obj in character_meshes(armatures, cage=cage, target=target,
                                scene=context.scene):
        # claim=False: a body already filed under another character stays
        # there instead of being tugged back and forth on every switch.
        _move_object(obj, subs[ROLE_BODY], root, ROLE_BODY, claim=False,
                     record=record)
    for armature in armatures[1:]:
        # An unmerged body rig belongs with the character too, or appending the
        # root would bring the clothes and leave what moves them behind.
        _move_object(armature, subs[ROLE_RIG], root, ROLE_RIG, claim=False,
                     record=record)

    return {"root": root, **subs}


def organize_current(context, extras=(), parts=(), record=None):
    """File the ACTIVE character.  `record` is who that is.

    Passing the record is the whole point: the collection a character lives in
    is now a fact stored on the character, not something re-derived from
    whichever objects happen to be in the panel's slots at the time.
    """
    mh = getattr(context.scene, "mhfrt", None) if context and context.scene else None
    if mh is None:
        return {}
    from . import registry
    if record is None:
        record = registry.active(context.scene)
    skel = mh.skeleton
    gui_coll = None
    if skel is not None:
        gui_coll = _as_collection(skel.get("mhfrt_gui_coll"))
    result = ensure_character_collections(
        context,
        cage=mh.cage,
        target=mh.target,
        skel=skel,
        gui_coll=gui_coll,
        extras=extras,
        parts=parts,
        root=record.root if record is not None else None,
        record=record,
    )
    root = result.get("root") if isinstance(result, dict) else None
    if record is not None and root is not None:
        record.root = root
        registry.claim(record, root)
        for datablock in (mh.cage, mh.target, skel, gui_coll):
            registry.claim(record, datablock)
    return result
