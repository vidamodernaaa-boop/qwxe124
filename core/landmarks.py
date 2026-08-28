"""Landmark-pair data layer.

Pairs live in scene properties (``Scene.mhfrt.landmarks``) instead of scene
objects: nothing to grab by accident, a clean outliner, and full participation
in Blender's undo stack. ``src_co`` is stored in CAGE local space **on the
Basis shape** (correspondences are defined on the undeformed cage); ``tgt_co``
is stored in TARGET local space. World positions are derived on demand, so
moving either object never invalidates the landmarks.

Pairs drawn as one surface curve share a ``curve_id`` and sit in stroke order
in the collection. Points can be moved after confirmation, and its two ends
can be merged once to turn the curve into a closed loop. ``curve_id == -1``
marks a legacy single point from the click-click era.
"""

import json
import uuid

import bpy
import numpy as np
from mathutils import Vector

# Kept in sync with ops.op_wrap.WRAPPED_KEY (constant duplicated to avoid a
# core -> ops import cycle).
WRAPPED_KEY = "Wrapped"

LEGACY_COLL = "Landmarks"
UID_PROP = "mhfrt_uid"
LANDMARK_SETS_PROP = "mhfrt_landmark_sets"


# ---------------------------------------------------------- per-character ---

def _ensure_uid(obj):
    if obj is None:
        return ""
    uid = obj.get(UID_PROP)
    if not uid:
        uid = uuid.uuid4().hex[:12]
        obj[UID_PROP] = uid
    return str(uid)


def _payload_from_mh(mh, target):
    return {
        "target_uid": _ensure_uid(target),
        "target_name": target.name if target else "",
        "active": int(mh.landmark_active),
        "items": [
            {
                "src": [float(v) for v in l.src_co],
                "tgt": [float(v) for v in l.tgt_co],
                "has_src": bool(l.has_src),
                "has_tgt": bool(l.has_tgt),
                "label": str(l.label),
                "src_vidx": int(l.src_vidx),
                "mirror_pending": bool(l.mirror_pending),
                "curve": int(l.curve_id),
                "curve_closed": bool(l.curve_closed),
                "mirror_of": int(l.mirror_of),
                "center_merged": bool(l.center_merged),
            }
            for l in mh.landmarks
        ],
    }


def _read_sets(cage):
    if cage is None:
        return {}
    raw = cage.get(LANDMARK_SETS_PROP)
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sets(cage, sets):
    cage[LANDMARK_SETS_PROP] = json.dumps(sets, separators=(",", ":"))


def _apply_payload(mh, payload):
    mh.landmarks.clear()
    for item in payload.get("items", ()):
        l = mh.landmarks.add()
        l.src_co = item.get("src", (0.0, 0.0, 0.0))
        l.tgt_co = item.get("tgt", (0.0, 0.0, 0.0))
        l.has_src = bool(item.get("has_src", False))
        l.has_tgt = bool(item.get("has_tgt", False))
        l.label = str(item.get("label", ""))
        l.src_vidx = int(item.get("src_vidx", -1))
        l.mirror_pending = bool(item.get("mirror_pending", False))
        l.curve_id = int(item.get("curve", -1))
        l.curve_closed = bool(item.get("curve_closed", False))
        l.mirror_of = int(item.get("mirror_of", -1))
        l.center_merged = bool(item.get("center_merged", False))
    active = int(payload.get("active", -1))
    mh.landmark_active = min(active, len(mh.landmarks) - 1)


def save_for_pair(mh, cage, target):
    """Persist the current scene landmark collection for a cage/target pair."""
    if not cage or not target or cage == target:
        return
    _ensure_uid(cage)
    key = _ensure_uid(target)
    sets = _read_sets(cage)
    sets[key] = _payload_from_mh(mh, target)
    _write_sets(cage, sets)


def save_active(mh):
    """Persist the currently visible landmarks to the active cage/target pair."""
    save_for_pair(mh, mh.cage, mh.target)
    if hasattr(mh, "active_cage"):
        mh.active_cage = mh.cage
    if hasattr(mh, "active_target"):
        mh.active_target = mh.target


def load_for_pair(mh, cage, target):
    """Load the landmarks belonging to a cage/target pair, or show none."""
    if not cage or not target or cage == target:
        clear(mh, save=False)
        return False
    key = _ensure_uid(target)
    sets = _read_sets(cage)
    payload = sets.get(key)
    if payload is None:
        # Migration fallback for files saved before stable target UIDs existed.
        for candidate in sets.values():
            if isinstance(candidate, dict) and candidate.get("target_name") == target.name:
                payload = candidate
                break
    if payload is None:
        clear(mh, save=False)
        return False
    _apply_payload(mh, payload)
    return True


def load_active(mh):
    loaded = load_for_pair(mh, mh.cage, mh.target)
    if hasattr(mh, "active_cage"):
        mh.active_cage = mh.cage
    if hasattr(mh, "active_target"):
        mh.active_target = mh.target
    return loaded


# ------------------------------------------------------------------ queries ---

def complete_count(mh):
    return sum(1 for l in mh.landmarks if l.has_src and l.has_tgt)


# ------------------------------------------------------------------- curves ---

def curves(mh):
    """{curve_id: [(index, landmark), ...]} in stroke order, creation order."""
    out = {}
    for i, l in enumerate(mh.landmarks):
        cid = int(l.curve_id)
        if cid >= 0:
            out.setdefault(cid, []).append((i, l))
    return out


def curve_count(mh):
    return len(curves(mh))


def next_curve_id(mh):
    return max((int(l.curve_id) for l in mh.landmarks), default=-1) + 1


def pending_curve(mh, all_curves=None):
    """(curve_id, [indices]) of the cage curve awaiting its match on the head,
    or None. Mirrored twins complete automatically, so they never count.

    Pass ``all_curves`` to reuse a grouping the caller already built - the
    landmark tool's HUD asks four different questions of the same dict on
    every redraw, and regrouping every point four times is pure waste."""
    if all_curves is None:
        all_curves = curves(mh)
    for cid, pts in all_curves.items():
        if any(l.mirror_pending for _i, l in pts):
            continue
        if all(l.has_src and not l.has_tgt for _i, l in pts):
            return cid, [i for i, _l in pts]
    return None


def pending_mirror_curve(mh):
    """(curve_id, [indices]) of the auto-mirrored twin still waiting for its
    target points, or None."""
    for cid, pts in curves(mh).items():
        if all(l.mirror_pending and l.has_src and not l.has_tgt
               for _i, l in pts):
            return cid, [i for i, _l in pts]
    return None


def remove_curve(mh, cid, save=True):
    """Delete every point of one curve."""
    for i in reversed(range(len(mh.landmarks))):
        if int(mh.landmarks[i].curve_id) == cid:
            mh.landmarks.remove(i)
    if mh.landmark_active >= len(mh.landmarks):
        mh.landmark_active = len(mh.landmarks) - 1
    if save:
        save_active(mh)


def mirror_partner(mh, i):
    """Index of point i's symmetric sister - the corresponding point on the
    mirrored twin curve (or on the source curve when i sits on the twin).
    -1 when it has none: legacy points, centre-merged points (they ARE their
    own mirror), curves without a twin. The twin skips centre-merged source
    points, so the mapping pairs the k-th twin point with the k-th UNMERGED
    source point."""
    if not (0 <= i < len(mh.landmarks)):
        return -1
    lm = mh.landmarks[i]
    cid = int(lm.curve_id)
    if cid < 0:
        return -1
    all_curves = curves(mh)
    pts = all_curves.get(cid)
    if not pts:
        return -1
    k = next((j for j, (idx, _l) in enumerate(pts) if idx == i), -1)
    if k < 0:
        return -1
    src_cid = int(lm.mirror_of)
    if src_cid >= 0:
        src_pts = all_curves.get(src_cid)
        if not src_pts:
            return -1
        unmerged = [idx for idx, l in src_pts if not l.center_merged]
        return unmerged[k] if k < len(unmerged) else -1
    if lm.center_merged:
        return -1
    twin_pts = next(
        (tp for tp in all_curves.values()
         if tp and int(tp[0][1].mirror_of) == cid), None)
    if twin_pts is None:
        return -1
    rank = sum(1 for _idx, l in pts[:k] if not l.center_merged)
    return twin_pts[rank][0] if rank < len(twin_pts) else -1


def _center_threshold(mh):
    """Local-X distance under which a cage point counts as centre-line, or
    None when it cannot be determined. Only needed for curves saved before
    ``center_merged`` was stored per point."""
    cage = mh.cage
    if cage is None or cage.type != 'MESH' or not len(cage.data.vertices):
        return None
    # numpy, not a per-vertex list comprehension: this can run against a dense
    # mesh, where walking the vertices through RNA is seconds of work
    n = len(cage.data.vertices)
    co = np.empty(n * 3)
    cage.data.vertices.foreach_get("co", co)
    xs = co[0::3]
    half = max(1e-6, float(xs.max() - xs.min()) * 0.5)
    return max(float(mh.symmetry_center_threshold) * half, 1e-9)


def merge_links(mh, all_curves=None):
    """[(idx_a, idx_b), ...] joining a curve to its mirrored twin ONLY at
    endpoints that genuinely merged on the centre line.

    A merged point has no twin of its own, so joining it to the twin curve's
    matching end completes one continuous stroke across the face. An endpoint
    away from the centre - a mouth or eye corner - is never joined, even when
    the curve's OTHER end did merge: the two sides are separate strokes there.
    """
    if all_curves is None:
        all_curves = curves(mh)
    links = []
    threshold = ...             # computed lazily, only for legacy curves
    for pts in all_curves.values():
        src_cid = int(pts[0][1].mirror_of)
        if src_cid < 0:
            continue            # not a mirrored twin
        src_pts = all_curves.get(src_cid)
        if not src_pts:
            continue

        def _merged(entry):
            nonlocal threshold
            lm = entry[1]
            if lm.center_merged:
                return True
            # pre-1.9.0 curve: fall back to the geometric definition
            if threshold is ...:
                threshold = _center_threshold(mh)
            return (threshold is not None
                    and abs(lm.src_co[0]) <= threshold)

        if _merged(src_pts[0]):
            links.append((src_pts[0][0], pts[0][0]))
        if len(src_pts) > 1 and _merged(src_pts[-1]):
            links.append((src_pts[-1][0], pts[-1][0]))
    return links


def remove_incomplete(mh, save=True):
    """Drop every pair still missing a side (tool-exit hygiene: a half-drawn
    curve never outlives the tool). Returns how many points were removed."""
    removed = 0
    for i in reversed(range(len(mh.landmarks))):
        l = mh.landmarks[i]
        if not (l.has_src and l.has_tgt):
            mh.landmarks.remove(i)
            removed += 1
    if removed:
        if mh.landmark_active >= len(mh.landmarks):
            mh.landmark_active = len(mh.landmarks) - 1
        if save:
            save_active(mh)
    return removed


# ------------------------------------------------------------------ editing ---

def add_pair(mh, label=""):
    lm = mh.landmarks.add()
    lm.label = label
    return lm, len(mh.landmarks) - 1


def remove_at(mh, i, save=True):
    if 0 <= i < len(mh.landmarks):
        mh.landmarks.remove(i)
        if mh.landmark_active >= len(mh.landmarks):
            mh.landmark_active = len(mh.landmarks) - 1
        if save:
            save_active(mh)


def clear(mh, save=True):
    mh.landmarks.clear()
    mh.landmark_active = -1
    if save:
        save_active(mh)


def snapshot(mh):
    """Serializable copy of all pairs (for the in-tool undo stack)."""
    items = [(tuple(l.src_co), tuple(l.tgt_co), l.has_src, l.has_tgt,
              l.label, l.src_vidx, l.mirror_pending, int(l.curve_id),
              int(l.mirror_of), bool(l.center_merged), bool(l.curve_closed))
             for l in mh.landmarks]
    return items, mh.landmark_active


def restore(mh, snap):
    items, active = snap
    mh.landmarks.clear()
    for item in items:
        # The final field was added with editable loop closure. Accept an old
        # in-memory snapshot defensively if one survived a script reload.
        (src_co, tgt_co, has_src, has_tgt, label, vidx, mirror_pending,
         curve_id, mirror_of, center_merged) = item[:10]
        curve_closed = bool(item[10]) if len(item) > 10 else False
        l = mh.landmarks.add()
        l.src_co = src_co
        l.tgt_co = tgt_co
        l.has_src = has_src
        l.has_tgt = has_tgt
        l.label = label
        l.src_vidx = vidx
        l.mirror_pending = mirror_pending
        l.curve_id = curve_id
        l.mirror_of = mirror_of
        l.center_merged = center_merged
        l.curve_closed = curve_closed
    mh.landmark_active = min(active, len(mh.landmarks) - 1)
    save_active(mh)


# ------------------------------------------------------------ world lookups ---

# Shape-key offsets, memoized per cage and per shape-key MIX.
#
# Every landmark's display position asks what the cage's shape keys did to its
# nearest vertex, and that used to walk the whole key_blocks list per landmark,
# per marker, per viewport redraw.  On a cage carrying a few hundred sculpted
# morph keys that is tens of thousands of RNA reads a frame, which is exactly
# why landmarking got slower with every curve drawn and went back to normal the
# moment the tool closed.
#
# The mix only changes when a key's value changes or its data is edited, so the
# table is rebuilt on a cheap signature (the non-zero key values) and dropped
# outright when the depsgraph reports the cage's shape keys as updated (see
# props._stale_pointer_handler -> invalidate_display_cache).
_ZERO = Vector((0.0, 0.0, 0.0))
_ZERO.freeze()
ZERO = _ZERO            # for callers that index an offset table themselves
_wrap_cache = {"key": None, "offsets": {}}


def invalidate_display_cache(*_args):
    """Forget the memoized shape-key offsets (the cage's shape changed)."""
    _wrap_cache["key"] = None
    _wrap_cache["offsets"] = {}


def _wrap_mix_key(cage):
    """Cheap signature of the cage's current shape-key mix.

    One pass over key_blocks per redraw instead of one pass per landmark.
    """
    # getattr, not attribute access: a draw handler runs on whatever the
    # scene properties happen to point at, and a cage that is not a mesh (an
    # old file, a script, a slot restored from a broken .blend) raised here on
    # EVERY redraw - an exception in a draw callback is the worst place for it.
    me = getattr(cage, "data", None)
    keys = getattr(me, "shape_keys", None)
    if keys is None:
        return None
    kb = keys.key_blocks
    if not kb:
        return None
    return (cage.as_pointer(), keys.as_pointer(), len(kb),
            tuple((i, round(k.value, 6)) for i, k in enumerate(kb)
                  if i and abs(k.value) > 1e-9))


def _current_offsets(cage):
    """The live offset table for `cage`, emptied when the mix moved.

    The mix signature is read ONCE here, not once per landmark: walking
    key_blocks per point is the very cost this cache exists to remove.
    """
    if cage is None:
        return None, ()
    mix = _wrap_mix_key(cage)
    if mix is None:
        return None, ()
    if _wrap_cache["key"] != mix:
        _wrap_cache["key"] = mix
        _wrap_cache["offsets"] = {}
    return _wrap_cache["offsets"], mix[3]


def _fill_offset(cage, offsets, active, vidx):
    off = offsets.get(vidx)
    if off is not None:
        return off
    kb = cage.data.shape_keys.key_blocks
    if not active or vidx < 0 or vidx >= len(kb[0].data):
        offsets[vidx] = _ZERO
        return _ZERO
    base = kb[0].data[vidx].co
    off = Vector((0.0, 0.0, 0.0))
    for i, value in active:
        k = kb[i]
        if vidx >= len(k.data):
            continue
        off += (k.data[vidx].co - base) * value
    off.freeze()
    offsets[vidx] = off
    return off


def wrap_offsets(cage, landmarks):
    """``{vidx: Vector}`` covering every landmark, in ONE shape-key pass.

    Callers that place many markers at once (the overlay, the tool's hit test)
    take this and index it, instead of asking per point and re-deriving the
    mix each time.
    """
    offsets, active = _current_offsets(cage)
    if offsets is None:
        return {}
    for lm in landmarks:
        if lm.has_src:
            _fill_offset(cage, offsets, active, int(lm.src_vidx))
    return offsets


def _src_wrap_offset(cage, vidx):
    """Delta the nearest cage vertex gets from ALL relative shape keys at
    their current values (Wrapped, MouthOpen, CloseEyes, ...), so source
    markers visually follow the cage in every pose - including the Weight
    Cleanup guide poses."""
    if vidx < 0:
        return _ZERO
    offsets, active = _current_offsets(cage)
    if offsets is None:
        return _ZERO
    return _fill_offset(cage, offsets, active, vidx)


def src_world(cage, lm, offsets=None):
    """Display position: rides along with the cage's current shape-key mix."""
    if offsets is None:
        off = _src_wrap_offset(cage, int(lm.src_vidx))
    else:
        off = offsets.get(int(lm.src_vidx), _ZERO)
    return cage.matrix_world @ (Vector(lm.src_co) + off)


def src_world_basis(cage, lm):
    """Solver position: always on the Basis (undeformed) shape."""
    return cage.matrix_world @ Vector(lm.src_co)


def tgt_world(target, lm):
    return target.matrix_world @ Vector(lm.tgt_co)


def wrap_pairs(mh):
    """[(src_world_on_basis, tgt_world), ...] for the wrap solver."""
    cage, target = mh.cage, mh.target
    out = []
    if not cage or not target:
        return out
    for l in mh.landmarks:
        if l.has_src and l.has_tgt:
            out.append((src_world_basis(cage, l), tgt_world(target, l)))
    return out


def display_entries(context):
    """Per-pair drawing info for the overlay and the landmark tool.

    Runs on every redraw of every 3D viewport, so everything that does not
    change per landmark - the two world matrices, the shape-key offset table -
    is derived once up front.  Reading ``matrix_world`` per point built a fresh
    Matrix each time, which at a few hundred landmarks was most of the cost.
    """
    mh = getattr(context.scene, "mhfrt", None)
    if mh is None:
        return []
    cage, target = mh.cage, mh.target
    landmarks = mh.landmarks
    offsets = wrap_offsets(cage, landmarks) if cage is not None else {}
    cage_mw = cage.matrix_world.copy() if cage is not None else None
    target_mw = target.matrix_world.copy() if target is not None else None
    entries = []
    for i, l in enumerate(landmarks):
        has_src = l.has_src
        has_tgt = l.has_tgt
        src = None
        if has_src and cage_mw is not None:
            src = cage_mw @ (Vector(l.src_co)
                             + offsets.get(int(l.src_vidx), _ZERO))
        tgt = None
        if has_tgt and target_mw is not None:
            tgt = target_mw @ Vector(l.tgt_co)
        entries.append({
            "index": i,
            "label": l.label,
            "src": src,
            "tgt": tgt,
            "complete": has_src and has_tgt,
            "curve": int(l.curve_id),
            "closed": bool(l.curve_closed),
        })
    return entries


# ------------------------------------------------- legacy Empty migration ---

def migrate_legacy(context):
    """Convert old Empty-based markers (pre-0.2) into scene data, then remove
    the objects. Returns how many pairs were migrated."""
    mh = context.scene.mhfrt
    cage, target = mh.cage, mh.target
    src, tgt = {}, {}
    for o in list(bpy.data.objects):
        pair = o.get("mhfrt_pair")
        kind = o.get("mhfrt_kind")
        if pair is None or kind not in ("src", "tgt"):
            continue
        (src if kind == "src" else tgt)[int(pair)] = o
    if not src and not tgt:
        return 0

    made = 0
    for i in sorted(set(src) | set(tgt)):
        lm, _ = add_pair(mh)
        if i in src and cage:
            lm.src_co = cage.matrix_world.inverted() @ src[i].matrix_world.translation
            lm.has_src = True
        if i in tgt and target:
            lm.tgt_co = target.matrix_world.inverted() @ tgt[i].matrix_world.translation
            lm.has_tgt = True
        made += 1

    for o in list(src.values()) + list(tgt.values()):
        bpy.data.objects.remove(o, do_unlink=True)
    coll = bpy.data.collections.get(LEGACY_COLL)
    if coll is not None and not coll.objects:
        bpy.data.collections.remove(coll)
    return made
