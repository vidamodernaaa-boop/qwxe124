"""The ``.mhfrt`` character file - a rig that outlives its .blend.

Building a MetaHuman rig is hours of work: landmark curves drawn by hand, a
wrap tuned by eye, eye pivots and tongue bones placed, morphs sculpted. All of
that lived only inside one .blend file, which meant one bad save, one crash, or
one "start again in a clean file" and it was gone.

A ``.mhfrt`` file is that work, written down. It is plain UTF-8 JSON - readable,
diffable, and recoverable by hand if it ever comes to that - holding:

* the character RECORD (its uid, name, and which objects played which part),
* every landmark curve, in the cage/head local coordinates they are authored in,
* the per-character settings (wrap quality, symmetry, intensity, ...),
* where the artist put the control panels (see ``board.LAYOUT_HANDLES``),
* the snapshot of the artist's ORIGINAL rig and vertex groups, so a character
  restored into another file can still be handed back cleanly.

A ``.mhfrt`` is a ZIP holding:

* ``character.json`` - the manifest above: readable, diffable, recoverable by
  hand if it ever comes to that;
* ``character.blend`` - the character ITSELF, written with Blender's own
  ``bpy.data.libraries.write``: the head, the cage with its wrapped shape, the
  fitted skeleton and all 843 joints, the control board and its handle shapes,
  the bind weights, the sculpted morph keys, the actions. Lossless, because
  Blender wrote it.
* ``original/`` - the artist's own things as they were BEFORE this add-on
  changed them: the body armature written out whole just before a merge joins
  the facial bones into it, and the vertex weights of every mesh whose weights
  the bind rebalances. See :mod:`.provenance`.

**The file is created the moment a character row is** - not when the artist
remembers to save one. A character that has nothing but a name already has a
file, and every step after that (a cage, a head, landmark curves, a rig, a
merge) updates it. The manifest and the restore ledger are kilobytes, so that
is cheap; the character payload is only rewritten on demand and when the
.blend itself is saved.

That timing is the point: the ``original/`` snapshots are worthless if they are
taken after the change. Removing a character reads them back, so what the
artist gets back is what they handed over - not a best guess at it.

The manifest alone was the first design and it was not good enough. It saved
the INPUTS to a rig - the landmark curves and the settings - so restoring it
into a fresh file gave you a cage and some curves and a day of work still to
do. A character file has to hold the character.

The payload can be left out (``include_blend=False``) when the meshes already
exist in the destination and only the landmarking work needs to travel; the
file is then a few KB instead of a few MB.

Versioned from the first release: ``format`` is checked on load and a file from
a newer add-on is refused with a clear message instead of being half-read.
Format 1 (plain JSON, no payload) is still read.
"""

import json
import os
import tempfile
import time
import zipfile

import bpy

FORMAT = "MHFRT-CHARACTER"
# 3 adds the restore ledger (``provenance``) and the ``original/`` snapshots.
# Both are OPTIONAL keys/entries, so a version-2 reader still opens the file.
FORMAT_VERSION = 3
EXTENSION = ".mhfrt"
MANIFEST_NAME = "character.json"
PAYLOAD_NAME = "character.blend"
# Everything under here is the artist's own state, saved before we changed it.
ORIGINAL_PREFIX = "original/"

# Per-character settings worth carrying. Names are the RNA property names on
# Scene.mhfrt; anything missing on load is simply left at its default.
SETTINGS = (
    "cage_lod", "wrap_quality", "wrap_use_region_mask", "wrap_use_icp",
    "wrap_pin_landmarks", "wrap_iterations", "wrap_step", "wrap_smooth",
    "wrap_maxdist_frac", "symmetry", "symmetry_center_threshold",
    "landmark_loop_merge_threshold", "riglogic_scale_mul",
    "mouth_open_amount", "eyes_close_amount", "rig_destination",
    "merge_head_bone", "merge_deform_bone",
)


def _obj_name(obj):
    try:
        return obj.name if obj is not None else ""
    except ReferenceError:
        return ""


def _landmarks_payload(mh):
    return [
        {
            "src": [float(v) for v in lm.src_co],
            "tgt": [float(v) for v in lm.tgt_co],
            "has_src": bool(lm.has_src),
            "has_tgt": bool(lm.has_tgt),
            "label": str(lm.label),
            "src_vidx": int(lm.src_vidx),
            "mirror_pending": bool(lm.mirror_pending),
            "curve": int(lm.curve_id),
            "curve_closed": bool(lm.curve_closed),
            "mirror_of": int(lm.mirror_of),
            "center_merged": bool(lm.center_merged),
        }
        for lm in mh.landmarks
    ]


def build(scene, record):
    """The dict that becomes a .mhfrt file."""
    mh = scene.mhfrt
    from . import landmarks as lmdata
    # The landmarks of THIS character, not whatever the panel happens to show.
    if record.cage is not None and record.target is not None:
        saved = lmdata._read_sets(record.cage).get(
            lmdata._ensure_uid(record.target))
        items = saved.get("items", []) if isinstance(saved, dict) else []
    else:
        items = []
    if not items and record.cage == mh.cage and record.target == mh.target:
        items = _landmarks_payload(mh)

    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "addon_version": _addon_version(),
        "character": {
            "uid": record.uid,
            "name": record.name,
            "created_utc": record.created_utc,
            "cage": _obj_name(record.cage),
            "target": _obj_name(record.target),
            "skeleton": _obj_name(record.skeleton),
            "root": _obj_name(record.root),
        },
        "original_rig": {
            "armature": _obj_name(record.orig_armature),
            "modifier": record.orig_modifier,
            "parent": _obj_name(record.orig_parent),
            "parent_type": record.orig_parent_type,
            "parent_bone": record.orig_parent_bone,
            "baseline_groups": record.baseline_groups,
            "baseline_taken": bool(record.baseline_taken),
        },
        "settings": {
            name: _rna_value(mh, name) for name in SETTINGS
            if hasattr(mh, name)
        },
        "board_layout": _board_layout(record),
        # What we added and what everything looked like before we did - the
        # ledger Remove replays. Mirrored here so a character restored into a
        # fresh .blend can still be taken back out cleanly.
        "provenance": _provenance(record),
        "landmarks": items,
    }


def _provenance(record):
    from . import provenance
    try:
        return provenance.read(record)
    except Exception:                        # noqa: BLE001 - never block a save
        return {}


def _board_layout(record):
    """Where this character's control panels sit, or None.

    Read off the LIVE board first, so an artist who arranged the panels and
    never pressed Save Layout still carries the arrangement into the file; the
    snapshot saved on the skeleton is the fallback for a character whose board
    is not in this .blend.

    Note this rides in the MANIFEST, not only in the payload.  A payload does
    carry the posed board, so a full save already round-trips - but a
    ``include_blend=False`` file is landmarks and settings only, and losing the
    panel arrangement there would make the light format quietly lossier than it
    looks.  It is an OPTIONAL key, so the format version stays at 2 and an older
    add-on still opens the file instead of refusing it.
    """
    from . import board
    from ..props import RIG_ID_PROP
    skel = record.skeleton
    if skel is None:
        return None
    live = board.capture_layout(
        board.board_armature_for_rig(RIG_ID_PROP, skel.get(RIG_ID_PROP),
                                     skel=skel))
    if live:
        return live
    raw = skel.get(board.LAYOUT_PROP)
    if not raw:
        return None
    try:
        saved = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return saved if isinstance(saved, dict) and saved.get("handles") else None


def _restore_board_layout(record, data):
    """Put a manifest's panel layout back on the character. Returns True if set.

    Written onto the skeleton as well as applied, so it survives a later
    Rebuild Control Board exactly like a layout the artist saved by hand.

    The board is unlocked around the apply and re-locked after.  Python
    assignment ignores a lock flag, so this is not what makes the restore work
    - it is what leaves the board in the state the rest of the add-on expects:
    a character that was saved mid-redesign carries an UNLOCKED board in its
    payload, and loading it would otherwise hand the artist a panel that any
    stray Alt+S could flatten, with no visible sign that it was unprotected.
    """
    from . import board
    from ..props import RIG_ID_PROP
    layout = data.get("board_layout")
    skel = record.skeleton
    if (skel is None or not isinstance(layout, dict)
            or not layout.get("handles")):
        return False
    skel[board.LAYOUT_PROP] = json.dumps(layout, separators=(",", ":"))
    arm_obj = board.board_armature_for_rig(RIG_ID_PROP, skel.get(RIG_ID_PROP),
                                           skel=skel)
    if arm_obj is None:
        return True                     # the board is not in this file yet
    # A board restored from a payload written before v4.12.0 has no design
    # stamps, so it has nothing to compare a layout against; stamping it here,
    # before the layout goes on, records the authored state it arrived in.
    board.stamp_design(arm_obj)
    board.set_board_locked(arm_obj, False)
    board.apply_layout(arm_obj, layout)
    board.set_board_locked(arm_obj, True)
    _restore_follow_head(skel, arm_obj)
    return True


def _restore_follow_head(skel, arm_obj):
    """Re-aim the follow-head constraints at THIS file's skeleton.

    The payload carries the board's Child Of constraints as they were, pointed
    at whichever skeleton object they were built against.  Appending renames on
    collision, so after a load into a file that already holds a character those
    constraints can be aimed at the wrong armature entirely - and the panel then
    rides a stranger's head.  Rebuilt against the record's own skeleton instead,
    with the switch positions left exactly as the artist saved them.
    """
    from ..ops import op_rig
    try:
        op_rig.install_follow_head(skel, arm_obj)
    except Exception:                    # noqa: BLE001 - never block a load
        pass


def _rna_value(mh, name):
    value = getattr(mh, name)
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [float(v) for v in value]
    except TypeError:
        return str(value)


_VERSION = None


def _addon_version():
    """The add-on version, read out of the extension manifest.

    Written into every character file: when a file will not open, the first
    question is which version wrote it, and "" was never a useful answer.
    """
    global _VERSION
    if _VERSION is not None:
        return _VERSION
    _VERSION = ""
    manifest = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "blender_manifest.toml")
    try:
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("version"):
                    _VERSION = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    return _VERSION


def character_datablocks(scene, record):
    """Everything that IS this character, for Blender to serialise.

    The root collection is the handle: it holds the head, the cage, the rig,
    the control panel, attached parts, morph objects and body.  Blender follows
    every dependency from there - meshes, armatures, shape keys, actions, and
    the ~460 board handle shapes that live in no collection and are reachable
    only through the pose bones that draw them.
    """
    from . import registry
    blocks = set()
    if record.root is not None:
        blocks.add(record.root)
    for obj in (record.cage, record.target, record.skeleton,
                record.orig_armature):
        if obj is not None:
            blocks.add(obj)
    if record.gui_coll is not None:
        blocks.add(record.gui_coll)
    objects, colls = registry.owned_datablocks(scene, record)
    blocks.update(objects)
    blocks.update(colls)
    return {b for b in blocks if b is not None and b.library is None}


def write(scene, record, filepath, include_blend=True, extra=None,
          keep_existing=False):
    """Save one character. Returns the path actually written.

    `keep_existing` carries forward everything already in the file that this
    write is not replacing - the character payload when only the manifest is
    being refreshed, and always the ``original/`` snapshots.  Without it the
    cheap manifest-only updates that run as the artist works would throw away
    the pre-merge armature, which is the one thing in there that cannot be
    recreated.
    """
    if not filepath.lower().endswith(EXTENSION):
        filepath += EXTENSION
    data = build(scene, record)
    blocks = character_datablocks(scene, record) if include_blend else set()
    data["payload"] = PAYLOAD_NAME if blocks else ""
    data["payload_objects"] = len([b for b in blocks
                                   if isinstance(b, bpy.types.Object)])

    folder = os.path.dirname(filepath)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    temp_blend = ""
    if blocks:
        handle, temp_blend = tempfile.mkstemp(suffix=".blend",
                                              prefix="mhfrt_payload_")
        os.close(handle)
        try:
            # fake_user so nothing is dropped for having no user in the
            # written file; compress because a character is mostly float data.
            bpy.data.libraries.write(temp_blend, blocks,
                                     fake_user=True, compress=True)
        except (RuntimeError, OSError) as error:
            _unlink(temp_blend)
            temp_blend = ""
            data["payload"] = ""
            data["payload_error"] = str(error)

    extra = dict(extra or {})
    written = {MANIFEST_NAME, PAYLOAD_NAME} | set(extra)
    carried = []
    if keep_existing and os.path.exists(filepath) \
            and zipfile.is_zipfile(filepath):
        try:
            with zipfile.ZipFile(filepath) as old:
                for name in old.namelist():
                    if name in written:
                        continue
                    carried.append((name, old.read(name)))
                if not temp_blend and PAYLOAD_NAME in old.namelist():
                    carried.append((PAYLOAD_NAME, old.read(PAYLOAD_NAME)))
                    data["payload"] = PAYLOAD_NAME
        except (OSError, zipfile.BadZipFile, KeyError):
            carried = []

    # Written to a temporary file and moved into place, so an interrupted save
    # can never leave a half-written character behind.
    temp_zip = filepath + ".part"
    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME,
                             json.dumps(data, indent=1, ensure_ascii=False))
            if temp_blend:
                archive.write(temp_blend, PAYLOAD_NAME)
            for name, payload in extra.items():
                archive.writestr(name, payload)
            for name, payload in carried:
                archive.writestr(name, payload)
        os.replace(temp_zip, filepath)
    finally:
        _unlink(temp_blend)
        _unlink(temp_zip)
    return filepath


# ------------------------------------------------- the file, kept up to date ---
#
# A character's file is not something the artist has to remember to make.  It
# is created with the row and updated as they work, so the record of what this
# add-on did to their scene exists BEFORE the thing that needs undoing happens.

_UNSAVED_DIRNAME = "unsaved"
# Set while a character is being restored FROM a file, so applying it does not
# immediately write over the file it came from with a half-built record.
_suspended = 0


class suspend:
    """Context manager: no auto-writes while this is open."""

    def __enter__(self):
        global _suspended
        _suspended += 1
        return self

    def __exit__(self, *_exc):
        global _suspended
        _suspended = max(0, _suspended - 1)
        return False


def enabled():
    """Is the auto-file switched on in preferences?"""
    try:
        prefs = bpy.context.preferences.addons[__package__.rsplit(".", 1)[0]]
        return bool(prefs.preferences.auto_character_files)
    except (KeyError, AttributeError, TypeError):
        return True                          # default on; prefs may not be up


def _safe_name(name):
    keep = []
    for char in str(name or ""):
        keep.append(char if (char.isalnum() or char in "-_. ") else "_")
    return ("".join(keep).strip() or "character")[:60]


def auto_dir():
    """Where auto-saved character files live.

    Beside the .blend, in a folder named after it, so a project that is copied
    or zipped takes its characters with it.  A file that has never been saved
    has nowhere to put them yet, so they go to the user's Blender data folder
    and move next to the .blend the first time it IS saved.
    """
    override = ""
    try:
        prefs = bpy.context.preferences.addons[__package__.rsplit(".", 1)[0]]
        override = bpy.path.abspath(prefs.preferences.character_dir or "")
    except (KeyError, AttributeError, TypeError):
        override = ""
    if override:
        return override
    if bpy.data.filepath:
        folder, name = os.path.split(bpy.data.filepath)
        return os.path.join(folder, os.path.splitext(name)[0] + "_MHFRT")
    try:
        base = bpy.utils.user_resource('DATAFILES', path="mhfrt_characters",
                                       create=True)
    except (TypeError, AttributeError):       # very old API
        base = tempfile.gettempdir()
    return os.path.join(base, _UNSAVED_DIRNAME)


def auto_path(record, folder=None):
    """This character's file path - the one it already has, or a fresh one."""
    existing = ""
    try:
        existing = str(record.file_path or "")
    except (AttributeError, ReferenceError):
        existing = ""
    if existing:
        return existing
    folder = folder or auto_dir()
    return os.path.join(
        folder, f"{_safe_name(record.name)}_{record.uid[:8]}{EXTENSION}")


def touch(scene, record, include_blend=False, force=False):
    """Create or refresh this character's file. Never raises.

    Returns the path on success, "" when nothing was written.  Failures are
    recorded on the record (``file_error``) instead of interrupting whatever
    the artist was doing - a scene that cannot write a sidecar must still rig.
    """
    if record is None or (_suspended and not force):
        return ""
    if not enabled():
        return ""
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
        if scene is None:
            return ""
    path = auto_path(record)
    try:
        written = write(scene, record, path, include_blend=include_blend,
                        keep_existing=True)
    except (OSError, RuntimeError, ValueError) as error:
        try:
            record.file_error = str(error)
        except (AttributeError, ReferenceError):
            pass
        return ""
    try:
        record.file_path = written
        record.file_error = ""
    except (AttributeError, ReferenceError):
        pass
    return written


def put_blob(record, name, payload, scene=None):
    """Store one raw entry (a weight snapshot, an armature) in the file.

    Creates the file first if the character has not got one yet: this is the
    call the merge makes to save the artist's armature, and "there was no file
    yet" must never be the reason it was not saved.
    """
    if record is None or not payload:
        return False
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    path = auto_path(record)
    try:
        if not os.path.exists(path):
            if not touch(scene, record, force=True):
                return False
            path = auto_path(record)
        write(scene, record, path, include_blend=False,
              extra={name: payload}, keep_existing=True)
    except (OSError, RuntimeError, ValueError) as error:
        try:
            record.file_error = str(error)
        except (AttributeError, ReferenceError):
            pass
        return False
    try:
        record.file_path = path
    except (AttributeError, ReferenceError):
        pass
    return True


def has_payload(record):
    """True when this character's file already carries the character itself."""
    path = auto_path(record)
    if not path or not os.path.exists(path) or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return PAYLOAD_NAME in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def get_blob(record, name):
    """Read one raw entry back, or None."""
    if record is None or not name:
        return None
    path = auto_path(record)
    if not path or not os.path.exists(path) or not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            if name not in archive.namelist():
                return None
            return archive.read(name)
    except (OSError, zipfile.BadZipFile, KeyError):
        return None


def relocate(scene):
    """Move files written while the .blend was unsaved next to it.

    Called after a save: the character files follow the project instead of
    staying in the user's application data where nobody would ever find them.
    """
    if scene is None or not bpy.data.filepath or not enabled():
        return 0
    from . import registry
    folder = auto_dir()
    moved = 0
    for record in registry.records(scene):
        old = str(getattr(record, "file_path", "") or "")
        if not old or not os.path.exists(old):
            continue
        if os.path.normcase(os.path.dirname(old)) == os.path.normcase(folder):
            continue
        if _UNSAVED_DIRNAME not in os.path.normpath(old).split(os.sep):
            continue                        # the artist put it there on purpose
        new = os.path.join(folder, os.path.basename(old))
        try:
            os.makedirs(folder, exist_ok=True)
            os.replace(old, new)
            record.file_path = new
            moved += 1
        except OSError:
            continue
    return moved


def _unlink(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def read(filepath):
    """Load and validate a .mhfrt.

    Returns (manifest, payload bytes or None).  Raises ValueError with a
    readable reason.  A format-1 file (plain JSON, no payload) still opens.
    """
    payload = None
    try:
        if zipfile.is_zipfile(filepath):
            with zipfile.ZipFile(filepath) as archive:
                names = set(archive.namelist())
                if MANIFEST_NAME not in names:
                    raise ValueError(
                        "not a MetaHuman Rig Transfer character file")
                data = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
                if PAYLOAD_NAME in names:
                    payload = archive.read(PAYLOAD_NAME)
        else:
            with open(filepath, "r", encoding="utf-8") as handle:
                data = json.load(handle)
    except ValueError:
        raise
    except (OSError, KeyError) as error:
        raise ValueError(f"could not be read: {error}") from error
    except Exception as error:               # noqa: BLE001 - bad JSON, bad zip
        raise ValueError(f"could not be read: {error}") from error

    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("not a MetaHuman Rig Transfer character file")
    version = int(data.get("version", 0) or 0)
    if version > FORMAT_VERSION:
        raise ValueError(
            f"was written by a newer version of the add-on (file format "
            f"{version}, this add-on reads {FORMAT_VERSION}) - update the "
            "add-on to open it")
    return data, payload


def append_payload(payload, root_name):
    """Append the character .blend carried inside a .mhfrt.

    Returns ``(root collection or None, the objects that arrived)``.  The
    object list is what the loader hands back to the caller so it can select
    exactly what it just brought in; the bytes go through a temporary file
    because Blender's loader only takes a path.
    """
    if not payload:
        return None, []
    handle, path = tempfile.mkstemp(suffix=".blend", prefix="mhfrt_restore_")
    os.close(handle)
    try:
        with open(path, "wb") as out:
            out.write(payload)
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            wanted = [c for c in src.collections
                      if c == root_name] or list(src.collections)
            dst.collections = wanted[:1]
            if not wanted:
                dst.objects = list(src.objects)
        appended = [c for c in dst.collections if c is not None]
        if appended:
            return appended[0], list(appended[0].all_objects)
        loose = [o for o in dst.objects if o is not None]
        for obj in loose:
            bpy.context.scene.collection.objects.link(obj)
        return None, loose
    finally:
        _unlink(path)


def mh_of(scene):
    return scene.mhfrt


def _adopt(record, root):
    """Point the record at the objects that just came out of the payload.

    By ROLE, not by name: an append can suffix every name it collides with, so
    matching on the saved names would miss the very objects it just restored.
    """
    from . import organization as org
    for obj in root.all_objects:
        role = str(obj.get(org.OBJECT_ROLE_PROP, ""))
        if role == org.ROLE_CAGE and record.cage is None:
            record.cage = obj
        elif role == org.ROLE_HEAD and record.target is None:
            record.target = obj
        elif role == org.ROLE_RIG and obj.type == 'ARMATURE'                 and record.skeleton is None:
            from ..props import RIG_ID_PROP
            from . import board
            if board.is_rig_skeleton(obj, RIG_ID_PROP):
                record.skeleton = obj
    for child in root.children:
        if child.get(org.COLL_ROLE_PROP) == org.ROLE_PANEL:
            record.gui_coll = child


def _apply_settings(mh, data):
    for name, value in (data.get("settings") or {}).items():
        if not hasattr(mh, name):
            continue
        try:
            setattr(mh, name, value)
        except (TypeError, ValueError, AttributeError):
            continue


def apply(scene, data, remap=None, payload=None, arrived=None):
    """Recreate the character described by `data` in this scene.

    Objects are matched BY NAME, optionally through `remap` ({saved: actual}).
    Nothing is created or renamed: whatever cannot be found is reported so the
    artist can point the slots at the right meshes themselves.

    Returns (record, missing names, cage-was-rebuilt).  `arrived`, when a list
    is passed, is filled with every object this load actually brought into the
    file, so the caller can select exactly that and nothing else.
    """
    from . import registry
    from . import landmarks as lmdata

    arrived = arrived if arrived is not None else []
    remap = remap or {}
    char = data.get("character", {})
    missing = []

    def resolve(key):
        saved = char.get(key, "")
        if not saved:
            return None
        obj = bpy.data.objects.get(remap.get(saved, saved))
        if obj is None:
            missing.append(saved)
        return obj

    record = registry.find(scene, char.get("uid", ""))
    if record is None:
        with suspend():
            record = registry.new(scene, name=char.get("name") or None)
        if char.get("uid"):
            record.uid = char["uid"]
    # The ledger of what the add-on added and what was there before it - the
    # character cannot be removed cleanly in this file without it.
    ledger = data.get("provenance")
    if isinstance(ledger, dict) and ledger.get("objects"):
        try:
            record.provenance = json.dumps(ledger, separators=(",", ":"))
        except (TypeError, ValueError):
            pass
    record.name = registry.unique_name(
        scene, char.get("name") or registry.DEFAULT_NAME, skip=record)
    record.created_utc = char.get("created_utc", record.created_utc)
    # ---- the payload: the character itself, restored whole ---------------
    adopted = False
    if payload:
        root, restored_objects = append_payload(payload, char.get("root", ""))
        arrived.extend(restored_objects)
        if root is not None:
            try:
                bpy.context.scene.collection.children.link(root)
            except RuntimeError:
                pass                        # already linked
            record.root = root
            _adopt(record, root)
            adopted = True

    if adopted:
        registry.claim(record, record.root)
        for datablock in (record.cage, record.target, record.skeleton,
                          record.gui_coll):
            registry.claim(record, datablock)
        _apply_settings(mh_of(scene), data)
        _restore_board_layout(record, data)
        items = data.get("landmarks") or []
        if items:
            record.pending_landmarks = json.dumps(items,
                                                  separators=(",", ":"))
            record.pending_target_name = char.get("target", "")
            registry.apply_pending_landmarks(record)
        return record, [], False

    # ---- no payload: re-attach to meshes already in this file -------------
    # The CAGE is our own bundled asset.  If this file has not got it, build a
    # new one rather than reporting it missing - there is nothing for the
    # artist to do about it and nothing lost by rebuilding it.
    cage = None
    saved_cage = char.get("cage", "")
    if saved_cage:
        cage = bpy.data.objects.get(remap.get(saved_cage, saved_cage))
    if cage is None:
        lod = (data.get("settings") or {}).get("cage_lod")
        try:
            from ..ops.op_load_cage import append_cage
            cage = append_cage(bpy.context, lod)
            arrived.append(cage)            # the one thing this load created
            rebuilt_cage = True
        except (RuntimeError, ImportError):
            rebuilt_cage = False
            if saved_cage:
                missing.append(saved_cage)
    else:
        rebuilt_cage = False
    record.cage = cage
    record.target = resolve("target")
    record.skeleton = resolve("skeleton")
    root_name = char.get("root", "")
    if root_name:
        record.root = bpy.data.collections.get(remap.get(root_name, root_name))

    original = data.get("original_rig", {})
    arm_name = original.get("armature", "")
    if arm_name:
        record.orig_armature = bpy.data.objects.get(
            remap.get(arm_name, arm_name))
    record.orig_modifier = original.get("modifier", "")
    parent_name = original.get("parent", "")
    if parent_name:
        record.orig_parent = bpy.data.objects.get(
            remap.get(parent_name, parent_name))
    record.orig_parent_type = original.get("parent_type", "")
    record.orig_parent_bone = original.get("parent_bone", "")
    record.baseline_groups = original.get("baseline_groups", "")
    record.baseline_taken = bool(original.get("baseline_taken", False))

    for datablock in (record.cage, record.target, record.skeleton,
                      record.root):
        registry.claim(record, datablock)

    _apply_settings(scene.mhfrt, data)
    # Harmless when the skeleton is not in this file: it simply does nothing,
    # and re-loading the .mhfrt once the character is here applies it.
    _restore_board_layout(record, data)

    # The landmark curves are the expensive, irreplaceable part.  Park them on
    # the record if the head is not here yet; they are applied the instant the
    # artist assigns one (see registry.apply_pending_landmarks).
    items = data.get("landmarks") or []
    if items:
        record.pending_landmarks = json.dumps(items, separators=(",", ":"))
        record.pending_target_name = char.get("target", "")
        registry.apply_pending_landmarks(record)
    return record, missing, rebuilt_cage
