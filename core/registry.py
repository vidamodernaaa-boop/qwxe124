"""The character registry - the one place that knows what a character IS.

Until now a character was DERIVED: scan the file for tagged collections and
armatures and work out who belonged to whom. Every multi-character failure
this add-on has had traces back to that. Two characters could be inferred into
one collection and one of them simply stopped existing. A character with no rig
yet could be inferred out of the list entirely. Which objects a character owned
depended on the order ``bpy.data`` happened to be in.

A character is now a RECORD. It is created the instant the artist starts one -
before there is a cage, let alone a rig - it carries a uid that never changes
and is never reused, and it goes away only when the artist deletes it. Objects
point AT the record (``mhfrt_character_uid``); nothing is inferred back the
other way, ever.

Two consequences worth stating, because they are the point:

* **A record is never removed automatically.** Not when its objects are
  deleted, not when a scan finds nothing, not when a file is half-built. A
  character the artist started stays in the list until the artist removes it.
* **A head mesh belongs to exactly one record.** The registry is what makes
  that answerable, and :func:`target_owner` is the check.

The record also carries the artist's ORIGINAL state - the armature that was
deforming their head before this add-on touched it, and the vertex groups the
head already had. That snapshot is what lets Remove put their character back
the way they handed it over.
"""

import json
import time
import uuid

import bpy

# Stamped on every object and collection a character owns.
UID_PROP = "mhfrt_character_uid"

DEFAULT_NAME = "MHFR"

# Bumped when the registry layout changes; the scene stores the version it was
# migrated to so an old .blend is converted exactly once.
REGISTRY_VERSION = 1
VERSION_PROP = "mhfrt_registry_version"


class MHFRT_Character(bpy.types.PropertyGroup):
    """One character. Saved in the .blend; the source of truth for the list."""

    uid: bpy.props.StringProperty()
    name: bpy.props.StringProperty()
    # Creation order. The list is sorted by this, so a row NEVER moves on its
    # own - not when a head is picked, not when anything is renamed.
    order: bpy.props.IntProperty(default=0)
    created_utc: bpy.props.StringProperty()

    cage: bpy.props.PointerProperty(type=bpy.types.Object)
    target: bpy.props.PointerProperty(type=bpy.types.Object)
    skeleton: bpy.props.PointerProperty(type=bpy.types.Object)
    root: bpy.props.PointerProperty(type=bpy.types.Collection)
    gui_coll: bpy.props.PointerProperty(type=bpy.types.Collection)

    # ---- the artist's own rig, as it was before we arrived -----------------
    # Remove Rig restores exactly this and deletes only what we added.
    orig_armature: bpy.props.PointerProperty(type=bpy.types.Object)
    orig_modifier: bpy.props.StringProperty()
    orig_parent: bpy.props.PointerProperty(type=bpy.types.Object)
    orig_parent_type: bpy.props.StringProperty()
    orig_parent_bone: bpy.props.StringProperty()
    # JSON list of the vertex groups the head had BEFORE the bind, so removal
    # can delete the ones we added and leave the artist's weights untouched.
    baseline_groups: bpy.props.StringProperty(default="")
    baseline_taken: bpy.props.BoolProperty(default=False)

    # ---- the ledger: what we added, and what was here before ---------------
    # JSON, written as it happens (see core/provenance.py).  Removal replays
    # it instead of deleting the character's collection and hoping the rules
    # about which meshes to rescue covered everything the artist owned.
    provenance: bpy.props.StringProperty(default="")
    # This character's .mhfrt on disk.  Created with the row, not on request.
    file_path: bpy.props.StringProperty(default="")
    file_error: bpy.props.StringProperty(default="")

    # ---- restored from a .mhfrt but still waiting for its head -------------
    # The cage is our own asset and is simply rebuilt.  The HEAD is the
    # artist's mesh and may not be in this file at all, so its landmark curves
    # are parked here and applied the moment they point the slot at it.
    # Losing them because a name did not match would be unforgivable.
    pending_landmarks: bpy.props.StringProperty(default="")
    pending_target_name: bpy.props.StringProperty(default="")


# ------------------------------------------------------------------ access ---

def records(scene):
    mh = getattr(scene, "mhfrt", None)
    return mh.characters if mh is not None else ()


def find(scene, uid):
    if not uid:
        return None
    for rec in records(scene):
        if rec.uid == uid:
            return rec
    return None


def sorted_records(scene):
    """Creation order. Never re-sorted by anything that can change."""
    return sorted(records(scene), key=lambda r: (r.order, r.uid))


def active(scene):
    mh = getattr(scene, "mhfrt", None)
    if mh is None:
        return None
    return find(scene, mh.active_character_uid)


def set_active(scene, uid):
    mh = getattr(scene, "mhfrt", None)
    if mh is not None:
        mh.active_character_uid = str(uid or "")


def label(rec):
    """What to call this character in a message (never used to match)."""
    if rec is None:
        return ""
    for obj in (rec.target, rec.cage, rec.skeleton):
        if obj is not None:
            try:
                return obj.name
            except ReferenceError:
                continue
    return rec.name or "(empty)"


def _same(a, b):
    """Are these the same record?

    NOT ``a is b``: Blender does not promise one Python wrapper per
    PropertyGroup, so two lookups of the same record can be different objects.
    Identity lives in the uid and nowhere else - getting this wrong told a
    character that its OWN head was already taken.
    """
    if a is None or b is None:
        return False
    try:
        return bool(a.uid) and a.uid == b.uid
    except (AttributeError, ReferenceError):
        return False


def unique_name(scene, base=DEFAULT_NAME, skip=None):
    used = {r.name for r in records(scene) if not _same(r, skip)}
    if base not in used:
        return base
    i = 1
    while f"{base}.{i:03d}" in used:
        i += 1
    return f"{base}.{i:03d}"


# ------------------------------------------------------------------ create ---

def new(scene, name=None):
    """Start a character. It exists from this moment until it is removed.

    Its ``.mhfrt`` file is written here, with the row - before there is a
    cage, a head or a rig.  Everything this add-on later does to the artist's
    scene is recorded into that file as it happens, which only works if the
    file is already there when the first thing happens.
    """
    mh = scene.mhfrt
    rec = mh.characters.add()
    rec.uid = uuid.uuid4().hex
    mh.character_serial += 1
    rec.order = mh.character_serial
    rec.name = name or unique_name(scene, DEFAULT_NAME, skip=rec)
    rec.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    from . import sidecar
    sidecar.touch(scene, rec)
    return rec


def claim(rec, datablock):
    """Stamp a datablock as belonging to this character."""
    if rec is None or datablock is None:
        return
    try:
        if datablock.library is not None:
            return
        if datablock.get(UID_PROP) != rec.uid:
            datablock[UID_PROP] = rec.uid
    except (AttributeError, ReferenceError, TypeError):
        pass


def release(datablock):
    if datablock is None:
        return
    try:
        if UID_PROP in datablock.keys():
            del datablock[UID_PROP]
    except (AttributeError, ReferenceError, TypeError):
        pass


def owned_datablocks(scene, rec):
    """Everything currently stamped with this character's uid."""
    if rec is None or not rec.uid:
        return [], []
    objects = [o for o in bpy.data.objects if o.get(UID_PROP) == rec.uid]
    colls = [c for c in bpy.data.collections if c.get(UID_PROP) == rec.uid]
    return objects, colls


def remove(scene, uid):
    """Delete the RECORD (not the objects) and unstamp everything it owned.

    The character's ``.mhfrt`` is deliberately left on disk: it is the saved
    character, and the point of removing a rig from a scene is not to destroy
    the work.  Loading that file puts the character back.
    """
    rec = find(scene, uid)
    if rec is None:
        return False
    objects, colls = owned_datablocks(scene, rec)
    for datablock in objects + colls:
        release(datablock)
    # Last write: the record is about to go, so the file has to be the one
    # thing that still knows this character existed.
    from . import sidecar, provenance
    try:
        sidecar.touch(scene, rec)
    except Exception:                        # noqa: BLE001 - never block remove
        pass
    try:
        provenance.forget(rec)
    except Exception:                        # noqa: BLE001
        pass
    mh = scene.mhfrt
    for i, other in enumerate(mh.characters):
        if other.uid == uid:
            mh.characters.remove(i)
            break
    if mh.active_character_uid == uid:
        mh.active_character_uid = ""
    return True


# ------------------------------------------------------------------ lookup ---

def for_object(scene, obj):
    """Which character owns this object? The stamp first, pointers second."""
    if obj is None:
        return None
    try:
        uid = obj.get(UID_PROP)
    except ReferenceError:
        return None
    rec = find(scene, uid) if uid else None
    if rec is not None:
        return rec
    for rec in records(scene):
        if obj in (rec.cage, rec.target, rec.skeleton, rec.orig_armature):
            return rec
    return None


def for_collection(scene, coll):
    if coll is None:
        return None
    uid = coll.get(UID_PROP)
    rec = find(scene, uid) if uid else None
    if rec is not None:
        return rec
    for rec in records(scene):
        if coll in (rec.root, rec.gui_coll):
            return rec
    return None


def target_owner(scene, obj, skip=None):
    """The character already using `obj` as its Head Target, if any.

    This is what makes "one head, one character" enforceable instead of
    hopeful.  `skip` is the record being edited, so re-confirming a character's
    own head is never treated as a clash.
    """
    if obj is None:
        return None
    for rec in records(scene):
        if _same(rec, skip):
            continue
        if rec.target is not None and rec.target == obj:
            return rec
    return None


def cage_owner(scene, obj, skip=None):
    if obj is None:
        return None
    for rec in records(scene):
        if _same(rec, skip):
            continue
        if rec.cage is not None and rec.cage == obj:
            return rec
    return None


# --------------------------------------------------- the artist's own rig ---

def capture_baseline(rec, head):
    """Remember the head exactly as the artist handed it over.

    Taken ONCE, the first time a character is given its head - before this
    add-on binds anything.  Remove Rig replays it: the artist's armature
    modifier and parenting come back, and only the vertex groups WE created are
    deleted.  Without this snapshot there is no honest way to undo a bind.
    """
    if rec is None or head is None or head.type != 'MESH':
        return
    if rec.baseline_taken:
        return
    # The structural snapshot: collections, parent, modifiers, custom props.
    # NOT the weights - copying those out of a 100k-vertex head is a second of
    # work, and picking a head must stay instant.  Nothing has touched a weight
    # yet either: the transfer only ever rewrites its own FACIAL_ groups, and
    # the one operation that rebalances the artist's own groups (a merged bind)
    # takes the weight snapshot itself, immediately before it does so.
    from . import provenance
    provenance.snapshot_object(rec, head, role=provenance.ROLE_HEAD,
                               adopt=True)
    groups = [g.name for g in head.vertex_groups]
    rec.baseline_groups = json.dumps(groups, separators=(",", ":"))
    modifier = next((m for m in head.modifiers if m.type == 'ARMATURE'
                     and m.object is not None), None)
    if modifier is not None:
        rec.orig_armature = modifier.object
        rec.orig_modifier = modifier.name
    elif head.parent is not None and head.parent.type == 'ARMATURE':
        rec.orig_armature = head.parent
    rec.orig_parent = head.parent
    rec.orig_parent_type = head.parent_type or ""
    rec.orig_parent_bone = head.parent_bone or ""
    rec.baseline_taken = True


def baseline_group_names(rec):
    if rec is None or not rec.baseline_groups:
        return set()
    try:
        data = json.loads(rec.baseline_groups)
    except (TypeError, ValueError):
        return set()
    return set(data) if isinstance(data, list) else set()


def apply_pending_landmarks(rec):
    """Attach parked landmark curves now that the character has both meshes.

    Returns True when curves were restored.
    """
    if rec is None or not rec.pending_landmarks:
        return False
    if rec.cage is None or rec.target is None:
        return False
    from . import landmarks as lmdata
    try:
        items = json.loads(rec.pending_landmarks)
    except (TypeError, ValueError):
        rec.pending_landmarks = ""
        return False
    if not isinstance(items, list) or not items:
        rec.pending_landmarks = ""
        return False
    sets = lmdata._read_sets(rec.cage)
    key = lmdata._ensure_uid(rec.target)
    sets[key] = {
        "target_uid": key,
        "target_name": rec.target.name,
        "active": -1,
        "items": items,
    }
    lmdata._write_sets(rec.cage, sets)
    rec.pending_landmarks = ""
    rec.pending_target_name = ""
    return True


def has_baseline(rec):
    """True when we actually recorded the head's original state.

    An empty ``baseline_groups`` means "unknown" - a character migrated from a
    file made before the registry existed.  Restoration then leaves the mesh
    completely alone rather than guessing which weights were ours.
    """
    return bool(rec is not None and rec.baseline_groups)


def restore_baseline(rec, head, our_bone_names=()):
    """Put the head back the way the artist handed it over.

    Deletes only vertex groups that (a) were not there before we arrived AND
    (b) name a bone of the rig we built - so a group the artist added while
    working, or one that merely shares a name with something of ours, is
    never touched.  Then the artist's own armature modifier and parenting come
    back exactly as they were.

    Returns a short description of what was restored, for the report.
    """
    notes = []
    if rec is None or head is None or head.type != 'MESH':
        return notes

    if has_baseline(rec):
        before = baseline_group_names(rec)
        ours = {n for n in our_bone_names}
        doomed = [g for g in head.vertex_groups
                  if g.name not in before and g.name in ours]
        for group in doomed:
            head.vertex_groups.remove(group)
        if doomed:
            notes.append(f"removed {len(doomed)} vertex group(s) we created")
        kept = len(before & {g.name for g in head.vertex_groups})
        if kept:
            notes.append(f"kept your {kept} original vertex group(s)")

    armature = rec.orig_armature
    if armature is not None:
        name = rec.orig_modifier or "Armature"
        existing = next((m for m in head.modifiers
                         if m.type == 'ARMATURE' and m.object == armature),
                        None)
        if existing is None:
            modifier = head.modifiers.new(name, 'ARMATURE')
            modifier.object = armature
            notes.append(f"re-bound to your armature '{armature.name}'")
        elif existing.name != name:
            existing.name = name

    if rec.orig_parent is not None and head.parent != rec.orig_parent:
        world = head.matrix_world.copy()
        head.parent = rec.orig_parent
        if rec.orig_parent_type:
            try:
                head.parent_type = rec.orig_parent_type
            except (TypeError, ValueError):
                pass
        head.parent_bone = rec.orig_parent_bone or ""
        head.matrix_world = world
        notes.append(f"re-parented to '{rec.orig_parent.name}'")
    return notes


# ----------------------------------------------------------------- migrate ---

# A copy of the record, written onto the character's root collection.  The
# records themselves live on the Scene (that is where Blender lets an add-on
# keep saved data), but a CHARACTER belongs to the FILE - so a second scene, or
# a character appended from elsewhere, rebuilds its record from this card with
# the same uid, name and position in the list.  Without it, opening a new scene
# looked exactly like every character had disappeared.
CARD_PROP = "mhfrt_record"


def write_card(rec):
    if rec is None:
        return
    card = json.dumps({"uid": rec.uid, "name": rec.name, "order": rec.order,
                       "created_utc": rec.created_utc},
                      separators=(",", ":"))
    for holder in (rec.root, rec.skeleton, rec.cage, rec.target):
        if holder is None:
            continue
        try:
            if holder.library is None and holder.get(CARD_PROP) != card:
                holder[CARD_PROP] = card
        except (AttributeError, ReferenceError, TypeError):
            continue


def _read_card(datablock):
    raw = datablock.get(CARD_PROP)
    if not raw:
        return None
    try:
        card = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return card if isinstance(card, dict) and card.get("uid") else None


def rebuild_from_cards(scene):
    """Bring back records this scene has never seen but the FILE knows about.

    Covers a second scene, a character appended from another file, and a
    registry lost to an old save.  Identity is preserved, so both scenes agree
    on which character is which.
    """
    mh = getattr(scene, "mhfrt", None)
    if mh is None:
        return 0
    known = {r.uid for r in records(scene)}
    made = 0
    for source in (bpy.data.collections, bpy.data.objects):
        for datablock in source:
            card = _read_card(datablock)
            if card is None or card["uid"] in known:
                continue
            rec = mh.characters.add()
            rec.uid = card["uid"]
            rec.name = unique_name(scene, card.get("name") or DEFAULT_NAME,
                                   skip=rec)
            rec.order = int(card.get("order", 0) or 0)
            rec.created_utc = card.get("created_utc", "")
            mh.character_serial = max(mh.character_serial, rec.order)
            known.add(rec.uid)
            made += 1
    if made:
        _relink(scene)
    return made


def _relink(scene):
    """Point every record at the datablocks stamped with its uid."""
    from . import organization as org
    by_uid = {r.uid: r for r in records(scene)}
    for obj in bpy.data.objects:
        rec = by_uid.get(obj.get(UID_PROP))
        if rec is None:
            continue
        role = str(obj.get(org.OBJECT_ROLE_PROP, ""))
        if role == org.ROLE_CAGE and rec.cage is None:
            rec.cage = obj
        elif role == org.ROLE_HEAD and rec.target is None:
            rec.target = obj
        elif obj.type == 'ARMATURE' and rec.skeleton is None and role == org.ROLE_RIG:
            rec.skeleton = obj
    for coll in bpy.data.collections:
        rec = by_uid.get(coll.get(UID_PROP))
        if rec is not None and rec.root is None \
                and coll.get(org.ROOT_COLL_PROP):
            rec.root = coll


def migrate(scene):
    """Make this scene's registry match the file.

    Two jobs: convert a .blend made before the registry existed (once, guarded
    by a version stamp), and pick up characters the FILE has that this scene's
    registry does not - a second scene, or an append.
    """
    mh = getattr(scene, "mhfrt", None)
    if mh is None:
        return 0
    made = 0
    if int(scene.get(VERSION_PROP, 0) or 0) < REGISTRY_VERSION:
        try:
            made = _migrate_from_layout(scene)
        except Exception:                    # noqa: BLE001 - never block a load
            made = 0
        scene[VERSION_PROP] = REGISTRY_VERSION
    try:
        made += rebuild_from_cards(scene)
    except Exception:                        # noqa: BLE001
        pass
    for rec in records(scene):
        write_card(rec)
    return made


def _migrate_from_layout(scene):
    """Read the old derived layout: *_MHFRT_Character roots and tagged rigs."""
    from . import organization as org
    from ..props import RIG_ID_PROP, RIG_CAGE_PROP, RIG_TARGET_PROP
    from . import board

    existing_roots = {r.root.name for r in records(scene)
                      if r.root is not None}
    existing_skels = {r.skeleton.name for r in records(scene)
                      if r.skeleton is not None}
    made = 0

    for coll in bpy.data.collections:
        if not coll.get(org.ROOT_COLL_PROP) or coll.name in existing_roots:
            continue
        cage = target = skel = None
        for child in coll.children:
            role = child.get(org.COLL_ROLE_PROP)
            objs = list(child.objects)
            if not objs:
                continue
            if role == org.ROLE_HEAD:
                target = objs[0]
            elif role == org.ROLE_CAGE:
                cage = objs[0]
            elif role == org.ROLE_RIG:
                skel = next((o for o in objs
                             if board.is_rig_skeleton(o, RIG_ID_PROP)), None)
        if cage is None and target is None and skel is None:
            continue
        rec = new(scene, name=str(coll.get("mhfrt_rig_name") or "") or None)
        rec.root, rec.cage, rec.target, rec.skeleton = coll, cage, target, skel
        for datablock in (coll, cage, target, skel):
            claim(rec, datablock)
        if target is not None:
            # An already-bound head: its baseline is whatever it has now minus
            # what we know we add, which we cannot reconstruct.  Mark it taken
            # so removal is conservative rather than guessing.
            rec.baseline_taken = True
        existing_roots.add(coll.name)
        if skel is not None:
            existing_skels.add(skel.name)
        made += 1

    # rigs that never lived in a root (merged, or hand-reorganized files)
    for obj in bpy.data.objects:
        if not board.is_rig_skeleton(obj, RIG_ID_PROP):
            continue
        if obj.name in existing_skels or obj.get(UID_PROP):
            continue
        rec = new(scene)
        rec.skeleton = obj
        linked_cage = obj.get(RIG_CAGE_PROP)
        linked_target = obj.get(RIG_TARGET_PROP)
        if isinstance(linked_cage, bpy.types.Object):
            rec.cage = linked_cage
        if isinstance(linked_target, bpy.types.Object):
            rec.target = linked_target
        rec.baseline_taken = True
        for datablock in (obj, rec.cage, rec.target):
            claim(rec, datablock)
        made += 1
    return made
