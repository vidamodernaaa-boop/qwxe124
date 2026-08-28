"""What this add-on added, and what the file looked like before it did.

Removing a character used to be a demolition: the character's root collection
and every collection under it were deleted, and a handful of hand-written rules
decided which meshes to rescue into a new collection.  Everything those rules
did not name went with the root - and the organizer had already MOVED the
artist's body, clothing, hair and shoes into ``07 Body & Clothing`` and unlinked
them from the collections they came from.  Removing the rig therefore removed
the character.

This module replaces guessing with bookkeeping.  Two facts are written down as
they happen, never reconstructed afterwards:

* **What we created.**  Every datablock the add-on brings into the file is
  stamped with the uid of the character it was made for (:data:`MADE_PROP`).
  The stamp is applied by :func:`track`, a context manager that diffs
  ``bpy.data`` around a build step, so a new object cannot be missed by
  forgetting to name it somewhere.  Removal deletes *exactly* the stamped set
  and nothing else - an object with no stamp is the artist's, full stop.

* **What we changed.**  Before the add-on touches an object that already
  existed, :func:`snapshot_object` writes down what it was: which collections
  held it, its parent, its modifiers, its vertex groups, its shape keys, its
  custom properties, its visibility.  For an armature it also records every
  bone's rest position - joining re-derives bone rolls, so "put the bones back"
  needs the numbers.  For a mesh whose weights we are about to rebalance it
  records the weights themselves, into the character's ``.mhfrt`` file (they
  are far too big to live in the .blend).

The ledger lives in two places on purpose.  The small half - collections,
parents, modifiers, bone rest data - is JSON on the character record, so it is
saved in the .blend and is there even if the ``.mhfrt`` file was moved or
deleted.  The heavy half - the pre-merge armature, the pre-bind weights - is
written into the ``.mhfrt``, which is created the moment a character row is.

Identity is by stamp, not by name: an object carries :data:`ORIGIN_KEY_PROP`
once it is snapshotted, so renaming it (or its collections) later does not lose
the entry.  A DUPLICATE carries the same key; restoration prefers the object
whose recorded name matches and leaves any rival alone rather than restoring
the wrong one.
"""

import json
import time
import uuid
from contextlib import contextmanager

import bpy

# Stamped on every datablock the add-on creates: the uid of the character it
# was created for.  This is the only thing that authorises a delete.
MADE_PROP = "mhfrt_made_by"

# Stamped on datablocks that were ALREADY here and that we then changed, so a
# rename never loses the "what it was before" entry.
ORIGIN_KEY_PROP = "mhfrt_origin_key"

LEDGER_VERSION = 1

# bpy.data collections that `track` watches.  Objects and collections are what
# removal deletes; the data blocks are tracked so their orphans can be purged
# with them instead of riding along in the file forever.
#
# Shape keys are deliberately NOT here: a Key datablock belongs to the artist's
# mesh, and stamping one would write our property onto their model to no
# purpose - nothing deletes shape keys, and the sculpted shapes are theirs.
_TRACKED = ("objects", "collections", "meshes", "armatures", "actions",
            "materials", "node_groups")

# Roles, for the report only.
ROLE_HEAD = "head"
ROLE_BODY = "body"
ROLE_ARMATURE = "armature"
ROLE_PART = "part"


# --------------------------------------------------------------- stamping ---

def _uid(record):
    try:
        return str(record.uid) if record is not None else ""
    except (AttributeError, ReferenceError):
        return ""


def _writable(datablock):
    try:
        return datablock is not None and datablock.library is None
    except (AttributeError, ReferenceError):
        return False


def mark(record, datablock):
    """Stamp `datablock` as something this add-on created for `record`.

    An existing stamp is never overwritten: a character appended from another
    file arrives with its own, and stealing it would let two records both claim
    the right to delete the same object.
    """
    uid = _uid(record)
    if not uid or not _writable(datablock):
        return False
    try:
        if datablock.get(MADE_PROP):
            return False
        datablock[MADE_PROP] = uid
    except (TypeError, AttributeError, ReferenceError):
        return False
    return True


def made_by(datablock):
    """The uid of the character this datablock was created for, or ""."""
    if datablock is None:
        return ""
    try:
        return str(datablock.get(MADE_PROP) or "")
    except (AttributeError, ReferenceError):
        return ""


def is_ours(record, datablock):
    uid = _uid(record)
    return bool(uid) and made_by(datablock) == uid


def unmark(datablock):
    for key in (MADE_PROP, ORIGIN_KEY_PROP):
        try:
            if key in datablock.keys():
                del datablock[key]
        except (AttributeError, ReferenceError, TypeError, KeyError):
            pass


def disown(datablock):
    """This is the artist's now - drop our claim on it.

    Two ways one of our stamps ends up on something that is not ours, both of
    them ordinary:

    * the artist DUPLICATES one of our objects (Blender copies custom
      properties with it) and makes the copy their head, their morph object or
      an attached part.  Using the appended cage as a starting point for a head
      is a normal thing to do;
    * a property copy carries the stamp across, the way the merge copies the
      facial rig's properties onto the artist's own armature.

    Left alone, either one hands removal permission to delete the artist's
    mesh.  So the moment something takes on a role that makes it theirs, the
    claim goes - and because it then has no made-by stamp, it gets a normal
    before-snapshot and is restored instead of deleted.
    """
    try:
        if MADE_PROP in datablock.keys():
            del datablock[MADE_PROP]
            return True
    except (AttributeError, ReferenceError, TypeError, KeyError):
        pass
    return False


def _pointers(name):
    try:
        return {d.as_pointer() for d in getattr(bpy.data, name)}
    except AttributeError:
        return set()


@contextmanager
def track(record):
    """Stamp everything that appears in ``bpy.data`` inside this block.

    Wrapped around the build steps - appending the cage, importing the board,
    fitting the skeleton, creating collections - so ownership is recorded by
    the act of creation rather than by remembering to name every new datablock
    at every call site.  The 460-odd board handle shapes, which belong to no
    collection and carry no rig id, are caught here for free.
    """
    if _uid(record) == "":
        yield
        return
    before = {name: _pointers(name) for name in _TRACKED}
    try:
        yield
    finally:
        try:
            for name in _TRACKED:
                seen = before.get(name, set())
                for datablock in getattr(bpy.data, name, ()):
                    try:
                        if datablock.as_pointer() in seen:
                            continue
                    except ReferenceError:
                        continue
                    mark(record, datablock)
        except Exception:                     # noqa: BLE001 - never break a build
            pass


def created_datablocks(record):
    """Everything currently stamped as created for this character.

    Returns (objects, collections, data) - `data` being meshes, armatures and
    actions, which are deleted only once nothing uses them any more.
    """
    uid = _uid(record)
    if not uid:
        return [], [], []
    objects = [o for o in bpy.data.objects if made_by(o) == uid]
    colls = [c for c in bpy.data.collections if made_by(c) == uid]
    data = []
    for name in ("meshes", "armatures", "actions", "node_groups"):
        data.extend(d for d in getattr(bpy.data, name, ()) if made_by(d) == uid)
    return objects, colls, data


# ----------------------------------------------------------------- ledger ---

def read(record):
    """The character's ledger as a dict (never None)."""
    if record is None:
        return {"version": LEDGER_VERSION, "objects": {}}
    try:
        raw = record.provenance
    except (AttributeError, ReferenceError):
        return {"version": LEDGER_VERSION, "objects": {}}
    if not raw:
        return {"version": LEDGER_VERSION, "objects": {}}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {"version": LEDGER_VERSION, "objects": {}}
    if not isinstance(data, dict):
        return {"version": LEDGER_VERSION, "objects": {}}
    data.setdefault("version", LEDGER_VERSION)
    data.setdefault("objects", {})
    return data


def save(record, data):
    if record is None:
        return
    try:
        record.provenance = json.dumps(data, separators=(",", ":"))
    except (TypeError, ValueError, AttributeError, ReferenceError):
        pass


def origin_key(datablock, create=True):
    """The stable ledger key for a datablock we did not create."""
    if not _writable(datablock):
        return ""
    try:
        key = str(datablock.get(ORIGIN_KEY_PROP) or "")
    except (AttributeError, ReferenceError):
        return ""
    if key or not create:
        return key
    key = uuid.uuid4().hex[:12]
    try:
        datablock[ORIGIN_KEY_PROP] = key
    except (TypeError, AttributeError, ReferenceError):
        return ""
    return key


def _matrix(matrix):
    return [round(float(v), 6) for row in matrix for v in row]


def _collection_refs(obj):
    """Where this object lives now, recorded so a rename cannot lose it."""
    masters = {}
    for scene in bpy.data.scenes:
        try:
            masters[scene.collection.as_pointer()] = scene.name
        except (AttributeError, ReferenceError):
            continue
    refs = []
    for coll in obj.users_collection:
        try:
            scene_name = masters.get(coll.as_pointer())
        except ReferenceError:
            continue
        if scene_name is not None:
            refs.append({"scene": scene_name})
            continue
        # Stamp the artist's own collection so renaming it later still resolves.
        refs.append({"coll": coll.name, "key": origin_key(coll)})
    return refs


def _modifier_refs(obj):
    out = []
    for index, mod in enumerate(obj.modifiers):
        entry = {"name": mod.name, "type": mod.type, "index": index}
        target = getattr(mod, "object", None)
        if isinstance(target, bpy.types.Object):
            entry["object"] = target.name
            entry["object_key"] = origin_key(target)
        try:
            entry["show_viewport"] = bool(mod.show_viewport)
            entry["show_render"] = bool(mod.show_render)
        except AttributeError:
            pass
        out.append(entry)
    return out


def _armature_refs(obj):
    """Every bone's rest state, plus the bone collections that were there.

    ``object.join`` re-derives bone rolls, so a merge can move the artist's own
    bones without anyone asking it to.  Recording head/tail/roll is what lets
    removal put them back to the millimetre instead of hoping.
    """
    data = getattr(obj, "data", None)
    if data is None or not isinstance(data, bpy.types.Armature):
        return None
    bones = {}
    for bone in data.bones:
        bones[bone.name] = {
            "head": [round(float(v), 6) for v in bone.head_local],
            "tail": [round(float(v), 6) for v in bone.tail_local],
            "matrix": _matrix(bone.matrix_local),
            "parent": bone.parent.name if bone.parent else "",
            "connect": bool(bone.use_connect),
            "deform": bool(bone.use_deform),
            "inherit_rotation": bool(bone.use_inherit_rotation),
        }
    colls = []
    try:
        colls = [c.name for c in data.collections_all]
    except AttributeError:
        pass
    return {
        "bones": bones,
        "collections": colls,
        "pose_position": getattr(data, "pose_position", 'POSE'),
        "data_name": data.name,
    }


def snapshot_object(record, obj, role="", weights=False, scene=None,
                    adopt=False):
    """Write down what `obj` is, once, before this add-on changes it.

    Taken the FIRST time we touch an object and never again: after that the
    object is carrying our changes and re-taking it would record them as if
    they were the artist's.  Objects we created ourselves are skipped - there
    is nothing to restore them to.

    `adopt` is for the objects the artist HANDS us - their head, their eyes,
    their morph meshes.  Those are theirs by definition, even when they made
    them by duplicating something of ours, so any claim we have on them is
    dropped first (see :func:`disown`).
    """
    if obj is None or not _writable(obj) or _uid(record) == "":
        return ""
    if adopt:
        disown(obj)
    if made_by(obj):
        return ""                       # ours: it will be deleted, not restored
    data = read(record)
    entries = data.setdefault("objects", {})
    key = origin_key(obj)
    if not key:
        return ""
    entry = entries.get(key)
    if entry is not None:
        # Already recorded.  Only ever ADD information (a role we now know, a
        # weight blob we now have) - never overwrite the original state.
        changed = False
        if role and not entry.get("role"):
            entry["role"] = role
            changed = True
        if changed:
            save(record, data)
        if weights and not entry.get("weights"):
            _store_weights(record, obj, entry)
            save(record, data)
        return key

    entry = {
        "name": obj.name,
        "type": obj.type,
        "role": role,
        "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collections": _collection_refs(obj),
        "parent": obj.parent.name if obj.parent else "",
        "parent_key": origin_key(obj.parent) if obj.parent else "",
        "parent_type": obj.parent_type or "",
        "parent_bone": obj.parent_bone or "",
        "parent_inverse": _matrix(obj.matrix_parent_inverse),
        "matrix_world": _matrix(obj.matrix_world),
        "modifiers": _modifier_refs(obj),
        "props": [k for k in obj.keys() if isinstance(k, str)],
        "hide_viewport": bool(obj.hide_viewport),
        "hide_render": bool(obj.hide_render),
    }
    if obj.type == 'MESH':
        entry["vertex_groups"] = [g.name for g in obj.vertex_groups]
        keys = obj.data.shape_keys if obj.data else None
        entry["shape_keys"] = ([k.name for k in keys.key_blocks]
                               if keys is not None else [])
        entry["data_props"] = ([k for k in obj.data.keys()
                                if isinstance(k, str)] if obj.data else [])
    elif obj.type == 'ARMATURE':
        entry["armature"] = _armature_refs(obj)
        entry["data_props"] = ([k for k in obj.data.keys()
                                if isinstance(k, str)] if obj.data else [])
    entries[key] = entry
    if weights and obj.type == 'MESH':
        _store_weights(record, obj, entry)
    save(record, data)
    return key


def snapshot_objects(record, objects, role="", weights=False):
    for obj in objects or ():
        snapshot_object(record, obj, role=role, weights=weights)


def entry_for(record, obj, data=None):
    """The ledger entry describing `obj` before we touched it, or None."""
    if obj is None:
        return None
    key = origin_key(obj, create=False)
    if not key:
        return None
    data = data if data is not None else read(record)
    return data.get("objects", {}).get(key)


def resolve(record, entry_key, entry):
    """The object an entry refers to, or None.

    By stamp first.  A duplicate carries the same stamp, so when two objects
    answer, the one whose name matches what was recorded wins and the other is
    left completely alone - restoring a copy of the artist's mesh to the
    original's state would be a silent, unfixable corruption.
    """
    matches = []
    for obj in bpy.data.objects:
        try:
            if str(obj.get(ORIGIN_KEY_PROP) or "") == entry_key:
                matches.append(obj)
        except ReferenceError:
            continue
    if not matches:
        return bpy.data.objects.get(entry.get("name", "")) if entry else None
    if len(matches) == 1:
        return matches[0]
    wanted = entry.get("name", "") if entry else ""
    exact = [o for o in matches if o.name == wanted]
    return exact[0] if exact else None


# ------------------------------------------------------ heavy state: blobs ---

def _weights_name(key):
    return f"original/weights_{key}.npz"


def _store_weights(record, obj, entry):
    """Save every vertex weight this mesh has right now, into the .mhfrt.

    The merge takes the face's share out of the artist's head-region groups.
    Handing it back by formula is lossy the moment more than one bone carried
    the head (a character with its own face rig has forty), so the honest undo
    is the numbers themselves.  They are far too big for the .blend: this goes
    into the character file, compressed.
    """
    try:
        import io
        import numpy as np
    except ImportError:                       # pragma: no cover - numpy ships
        return False
    if obj is None or obj.type != 'MESH' or obj.data is None:
        return False
    groups = list(obj.vertex_groups)
    if not groups:
        entry["weights"] = ""
        return False
    arrays = {}
    names = []
    per_group_idx = {g.index: ([], []) for g in groups}
    for vert in obj.data.vertices:
        for element in vert.groups:
            bucket = per_group_idx.get(element.group)
            if bucket is None:
                continue
            bucket[0].append(vert.index)
            bucket[1].append(element.weight)
    for group in groups:
        idx, wts = per_group_idx[group.index]
        names.append(group.name)
        arrays[f"i{len(names) - 1}"] = np.asarray(idx, dtype=np.int32)
        arrays[f"w{len(names) - 1}"] = np.asarray(wts, dtype=np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, names=np.asarray(names, dtype=object),
                        count=np.asarray([len(obj.data.vertices)]), **arrays)
    from . import sidecar
    key = origin_key(obj)
    name = _weights_name(key)
    if not sidecar.put_blob(record, name, buffer.getvalue()):
        return False
    entry["weights"] = name
    entry["weight_groups"] = names
    return True


def load_weights(record, entry):
    """(names, {name: (indices, weights)}, vertex count) or None."""
    name = (entry or {}).get("weights")
    if not name:
        return None
    from . import sidecar
    raw = sidecar.get_blob(record, name)
    if not raw:
        return None
    try:
        import io
        import numpy as np
        with np.load(io.BytesIO(raw), allow_pickle=True) as bundle:
            names = [str(n) for n in bundle["names"]]
            out = {}
            for i, group in enumerate(names):
                out[group] = (bundle[f"i{i}"], bundle[f"w{i}"])
            count = int(bundle["count"][0])
    except Exception:                         # noqa: BLE001 - corrupt blob
        return None
    return names, out, count


def _armature_name(key):
    return f"original/armature_{key}.blend"


def store_armature(record, arm):
    """Write the artist's armature into the .mhfrt exactly as it is now.

    Called before the merge joins the facial bones into it.  The ledger already
    holds every bone's rest state, which is what removal replays; this is the
    belt: if the armature is gone or unrecognisable by then, it can be brought
    back whole from here.
    """
    import os
    import tempfile
    if arm is None or not _writable(arm):
        return ""
    blocks = {arm}
    if arm.data is not None:
        blocks.add(arm.data)
    if arm.animation_data is not None and arm.animation_data.action is not None:
        blocks.add(arm.animation_data.action)
    for bone in (arm.pose.bones if arm.pose else ()):
        if bone.custom_shape is not None:
            blocks.add(bone.custom_shape)
    handle, temp = tempfile.mkstemp(suffix=".blend", prefix="mhfrt_orig_")
    os.close(handle)
    try:
        bpy.data.libraries.write(temp, blocks, fake_user=True, compress=True)
        with open(temp, "rb") as stream:
            payload = stream.read()
    except (RuntimeError, OSError):
        payload = b""
    finally:
        try:
            os.remove(temp)
        except OSError:
            pass
    if not payload:
        return ""
    from . import sidecar
    key = origin_key(arm)
    name = _armature_name(key)
    if not sidecar.put_blob(record, name, payload):
        return ""
    data = read(record)
    entry = data.get("objects", {}).get(key)
    if entry is not None:
        entry["payload"] = name
        entry["payload_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())
        save(record, data)
    return name


def recover_armature(record, entry):
    """Append the saved original armature back into the file.

    Only used when the real one cannot be repaired - it was deleted, or joined
    into something else - because an appended copy is a NEW object: anything
    that pointed at the original still points nowhere.  Reported as such.
    """
    import os
    import tempfile
    name = (entry or {}).get("payload")
    if not name:
        return None
    from . import sidecar
    raw = sidecar.get_blob(record, name)
    if not raw:
        return None
    handle, temp = tempfile.mkstemp(suffix=".blend", prefix="mhfrt_recover_")
    os.close(handle)
    try:
        with open(temp, "wb") as stream:
            stream.write(raw)
        wanted = entry.get("name", "")
        with bpy.data.libraries.load(temp, link=False) as (src, dst):
            names = [n for n in src.objects if n == wanted] or list(src.objects)
            dst.objects = names[:1]
        restored = [o for o in dst.objects if o is not None]
    except (RuntimeError, OSError):
        return None
    finally:
        try:
            os.remove(temp)
        except OSError:
            pass
    if not restored:
        return None
    obj = restored[0]
    try:
        bpy.context.scene.collection.objects.link(obj)
    except RuntimeError:
        pass
    return obj


# ---------------------------------------------------------------- restore ---

def _scene_master(name):
    scene = bpy.data.scenes.get(name)
    return scene.collection if scene is not None else None


def _collection_from_ref(ref):
    if not isinstance(ref, dict):
        return None
    if "scene" in ref:
        return _scene_master(ref["scene"])
    key = str(ref.get("key") or "")
    if key:
        for coll in bpy.data.collections:
            try:
                if str(coll.get(ORIGIN_KEY_PROP) or "") == key:
                    return coll
            except ReferenceError:
                continue
    return bpy.data.collections.get(str(ref.get("coll") or ""))


def restore_collections(record, obj, entry, fallback=None):
    """Put an object back in the collections it was in before we filed it.

    The one rule removal must never break: an object we are not deleting can
    never be left in no collection at all.  If every collection it came from is
    gone, it goes to `fallback` - a visible home beats a correct one.
    """
    wanted = []
    for ref in entry.get("collections") or ():
        coll = _collection_from_ref(ref)
        if coll is not None and coll not in wanted:
            wanted.append(coll)
    linked = 0
    for coll in wanted:
        try:
            if obj.name not in coll.objects:
                coll.objects.link(obj)
            linked += 1
        except (RuntimeError, ReferenceError):
            continue
    if linked:
        return linked
    home = fallback if fallback is not None else bpy.context.scene.collection
    try:
        if obj.name not in home.objects:
            home.objects.link(obj)
        return 1
    except (RuntimeError, ReferenceError):
        return 0


def restore_parent(obj, entry, doomed=()):
    """Give an object its original parent back (keeping where it sits)."""
    name = entry.get("parent", "")
    current = obj.parent
    wanted = bpy.data.objects.get(name) if name else None
    if wanted is None and name:
        return False
    if current == wanted:
        return False
    if current is not None and wanted is None and current not in doomed:
        # We did not park it here; leave the artist's own parenting alone.
        return False
    world = obj.matrix_world.copy()
    try:
        obj.parent = wanted
        if wanted is not None:
            obj.parent_type = entry.get("parent_type") or 'OBJECT'
            obj.parent_bone = entry.get("parent_bone") or ""
        obj.matrix_world = world
    except (TypeError, ValueError, RuntimeError, ReferenceError):
        return False
    return True


def restore_modifiers(obj, entry, doomed=()):
    """Drop the modifiers we added; put back the ones we took away.

    Only ARMATURE modifiers are ever recreated - those are the ones this add-on
    adds, retargets and dedupes.  A modifier the artist added while working is
    not in the snapshot and is not touched.
    """
    saved = entry.get("modifiers") or []
    by_name = {m.get("name"): m for m in saved}
    doomed_names = {o.name for o in doomed if o is not None}
    removed = added = 0

    for mod in list(obj.modifiers):
        if mod.type != 'ARMATURE':
            continue
        if mod.name in by_name:
            wanted = by_name[mod.name].get("object", "")
            target = getattr(mod, "object", None)
            if wanted and (target is None or target.name in doomed_names):
                replacement = bpy.data.objects.get(wanted)
                if replacement is not None:
                    mod.object = replacement
            continue
        target = getattr(mod, "object", None)
        if target is None or target.name in doomed_names:
            obj.modifiers.remove(mod)
            removed += 1

    for saved_mod in saved:
        if saved_mod.get("type") != 'ARMATURE':
            continue
        wanted = saved_mod.get("object", "")
        armature = bpy.data.objects.get(wanted) if wanted else None
        if armature is None:
            continue
        if any(m.type == 'ARMATURE' and m.object == armature
               for m in obj.modifiers):
            continue
        try:
            mod = obj.modifiers.new(saved_mod.get("name") or "Armature",
                                    'ARMATURE')
        except (RuntimeError, TypeError):
            continue
        mod.object = armature
        if "show_viewport" in saved_mod:
            mod.show_viewport = bool(saved_mod["show_viewport"])
        if "show_render" in saved_mod:
            mod.show_render = bool(saved_mod["show_render"])
        added += 1
    return removed, added


def restore_weights(record, obj, entry, our_names=()):
    """Put every vertex weight back exactly as it was before we rebalanced it.

    Groups the artist added afterwards are left alone; groups WE added (they
    name one of our bones) go.  Returns the number of groups rewritten.
    """
    loaded = load_weights(record, entry)
    if loaded is None:
        return 0
    names, arrays, count = loaded
    if count != len(obj.data.vertices):
        return 0                        # the mesh changed: do not fake it
    everything = range(len(obj.data.vertices))
    rewritten = 0
    for name in names:
        indices, weights = arrays.get(name, (None, None))
        if indices is None:
            continue
        group = obj.vertex_groups.get(name)
        if group is None:
            group = obj.vertex_groups.new(name=name)
        # Clear first: the result has to BE the snapshot, not the snapshot
        # merged with whatever is in the group now.
        group.remove(everything)
        # One `add` per distinct weight rather than one per vertex - the same
        # numbers, but a head with 24k vertices takes a moment instead of a
        # minute (painted weights repeat heavily; 1.0 alone is usually most of
        # the mesh).
        buckets = {}
        for index, weight in zip(indices.tolist(), weights.tolist()):
            buckets.setdefault(float(weight), []).append(int(index))
        for weight, members in buckets.items():
            group.add(members, weight, 'REPLACE')
        rewritten += 1
    ours = set(our_names)
    for group in list(obj.vertex_groups):
        if group.name not in names and group.name in ours:
            obj.vertex_groups.remove(group)
    return rewritten


def restore_groups(obj, entry, our_names=()):
    """No weight blob: delete only the groups that name one of our bones."""
    before = set(entry.get("vertex_groups") or ())
    ours = set(our_names)
    doomed = [g for g in obj.vertex_groups
              if g.name not in before and g.name in ours]
    for group in doomed:
        obj.vertex_groups.remove(group)
    return len(doomed)


def restore_props(obj, entry):
    """Strip every property in our namespace off an object we are handing back.

    Not "the ones that were not there before".  A property named ``mhfrt_*`` or
    ``dna_*`` is never the artist's, even when the snapshot found one already
    on the object: that happens when they built their head by DUPLICATING one
    of ours (the cage is the obvious starting point), and Blender copies custom
    properties with the object.  Honouring the snapshot there left our tags on
    their mesh forever - a character collection pointer to a collection that no
    longer exists, and a character uid nothing answers to.

    The origin key is the one exception, and only until the end: it is what
    every later step of the removal uses to find this object's entry.
    :func:`forget` takes it off last.
    """
    stripped = 0
    for key in list(obj.keys()):
        if not isinstance(key, str) or key == ORIGIN_KEY_PROP:
            continue
        if key.startswith("mhfrt_") or key.startswith("dna_"):
            try:
                del obj[key]
                stripped += 1
            except (KeyError, TypeError):
                pass
    data = getattr(obj, "data", None)
    if data is not None and _writable(data):
        for key in list(data.keys()):
            if not isinstance(key, str):
                continue
            if key.startswith("mhfrt_") or key.startswith("dna_"):
                try:
                    del data[key]
                    stripped += 1
                except (KeyError, TypeError):
                    pass
    return stripped


def restore_visibility(obj, entry):
    try:
        obj.hide_viewport = bool(entry.get("hide_viewport", False))
        obj.hide_render = bool(entry.get("hide_render", False))
        if obj.name in bpy.context.view_layer.objects and obj.hide_get():
            obj.hide_set(False)
    except (AttributeError, RuntimeError, ReferenceError):
        pass


def restore_armature_rest(context, arm, entry):
    """Put the artist's own bones back where they were before the join.

    ``object.join`` re-derives bone rolls on the armature it joins INTO, so a
    merge can quietly rotate the bases the artist's own animation is expressed
    in.  Bones that never moved are skipped, so this is a no-op on a file where
    the join behaved.  Returns the number of bones corrected.
    """
    saved = (entry or {}).get("armature") or {}
    bones = saved.get("bones") or {}
    if arm is None or arm.type != 'ARMATURE' or not bones:
        return 0
    from mathutils import Matrix
    from . import organization
    fixed = 0
    previous = context.view_layer.objects.active if context else None
    with organization.shown_for_edit(
            arm, view_layer=(context.view_layer if context else None)):
        try:
            if context:
                context.view_layer.objects.active = arm
            bpy.ops.object.mode_set(mode='EDIT')
        except RuntimeError:
            return 0
        try:
            for name, want in bones.items():
                bone = arm.data.edit_bones.get(name)
                if bone is None:
                    continue
                matrix = Matrix([want["matrix"][i:i + 4] for i in range(0, 16, 4)])
                head = tuple(want["head"])
                tail = tuple(want["tail"])
                same = (
                    max(abs(a - b) for a, b in zip(bone.head, head)) < 1e-6
                    and max(abs(a - b) for a, b in zip(bone.tail, tail)) < 1e-6
                    and max(abs(a - b)
                            for ra, rb in zip(bone.matrix, matrix)
                            for a, b in zip(ra, rb)) < 1e-6
                )
                if same:
                    continue
                bone.head = head
                bone.tail = tail
                bone.matrix = matrix
                fixed += 1
        finally:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass
            if previous is not None and context and \
                    context.view_layer.objects.get(previous.name):
                context.view_layer.objects.active = previous
    return fixed


def restore_object(context, record, obj, entry, *, our_names=(), doomed=(),
                   fallback=None, data=None):
    """Hand one object back the way the artist handed it over.

    Order matters: collections first (so it is never homeless even if a later
    step raises), then weights, then the bindings that depend on the armatures
    still existing, and the property strip last - it deletes the stamp this
    whole lookup runs on.
    """
    notes = []
    if obj is None or entry is None:
        return notes
    try:
        restore_collections(record, obj, entry, fallback=fallback)
        if obj.type == 'MESH':
            rewritten = restore_weights(record, obj, entry, our_names)
            if rewritten:
                notes.append(f"restored {rewritten} vertex group(s) on "
                             f"'{obj.name}' to their original weights")
            else:
                dropped = restore_groups(obj, entry, our_names)
                if dropped:
                    notes.append(f"removed {dropped} vertex group(s) we added "
                                 f"to '{obj.name}'")
        removed, added = restore_modifiers(obj, entry, doomed)
        if added:
            notes.append(f"re-bound '{obj.name}' to its own armature")
        restore_parent(obj, entry, doomed)
        restore_visibility(obj, entry)
        restore_props(obj, entry)
    except (RuntimeError, ReferenceError, AttributeError) as error:
        notes.append(f"could not fully restore '{entry.get('name', '?')}': "
                     f"{error}")
    return notes


def restored_names(record):
    """Names of everything the ledger knows we changed (for the report)."""
    return [e.get("name", "") for e in read(record).get("objects", {}).values()]


def forget(record):
    """Drop the ledger and every stamp that points at this character."""
    uid = _uid(record)
    data = read(record)
    for key, entry in (data.get("objects") or {}).items():
        obj = resolve(record, key, entry)
        if obj is not None:
            try:
                if ORIGIN_KEY_PROP in obj.keys():
                    del obj[ORIGIN_KEY_PROP]
            except (KeyError, TypeError, ReferenceError):
                pass
        for ref in entry.get("collections") or ():
            coll = _collection_from_ref(ref)
            if coll is None:
                continue
            try:
                if ORIGIN_KEY_PROP in coll.keys():
                    del coll[ORIGIN_KEY_PROP]
            except (KeyError, TypeError, ReferenceError):
                pass
    if uid:
        for source in (bpy.data.objects, bpy.data.collections):
            for datablock in source:
                if made_by(datablock) == uid:
                    unmark(datablock)
    try:
        record.provenance = ""
    except (AttributeError, ReferenceError):
        pass
