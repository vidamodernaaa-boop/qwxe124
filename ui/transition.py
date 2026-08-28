"""Purely visual morph transition after Wrap / Update Rig - ZWrap-style.

Instead of the viewport snapping to the finished result, the previous shape
morphs into it over ~half a second: the real object is hidden for the
duration and a GPU overlay draws the interpolation (mesh surfaces as a
monochrome shaded morph, bones as ink sticks gliding to their new rest).
The overlay's end state IS the addon's computed result - nothing here ever
writes to mesh, armature, key or scene data beyond a temporary hide/unhide.

Contract with the rest of the add-on (why it cannot affect results):

* Host operators only call `mesh_snapshot()` / `bones_snapshot()` (pure
  numpy reads) at the top of execute() and `request()` right before their
  successful return.  Both swallow every exception internally: a transition
  failure can never fail the operator.
* Nothing visual happens inside the host operator itself.  `request()` only
  parks numpy arrays and registers a self-removing one-shot
  `depsgraph_update_post` bridge (plus a 0 s timer fallback).  The bridge
  runs AFTER the operator's undo push but BEFORE the next redraw, so:
  - the hide -> unhide pair lives entirely between two undo pushes and is
    invisible to undo AND redo (no "redo brings it back hidden" bug),
  - the finished result is never shown for a single frame before the
    morph starts (no flash).
* While the morph plays, an INTERNAL modal operator (no 'UNDO' flag -> no
  undo entry) swallows every event, so no clicks, shortcuts, undo, mode or
  step changes can slip in.  Esc skips straight to the end.  The modal's
  cancel(), transient undo/redo/load/save-pre guards and a watchdog
  bpy.app.timer each independently restore everything if the session is
  interrupted (file load, quit, scripted undo, add-on disable...).
* Depsgraph cost: exactly one hide + one unhide tag per transitioned
  object; the morph itself is a static GPU batch holding BOTH end states,
  re-blended per frame by a single `u_t` uniform - per-frame cost does not
  scale with vertex count, so heavy meshes stay responsive.
* In background mode (headless) everything no-ops.  Outside a transition
  this module owns no handler, timer, operator instance or GPU resource.
"""

import time

import numpy as np
import bpy
import gpu

from . import gizmo_draw as gd

DURATION = 1.1          # seconds, whole morph
BONES_DURATION = 1.6    # the rig flying onto the head is the one worth watching
TIMER_STEP = 1.0 / 60.0  # modal tick / redraw pacing
PENDING_GRACE = 2.0     # max seconds allowed between request() and setup
ACTIVE_GRACE = 3.0      # watchdog kills a stuck session this long after DURATION
WATCHDOG_STEP = 0.25
MAX_VERTS = 2_000_000   # above this a morph batch is not worth the upload
MAX_BONES = 5_000
MIN_DELTA = 1e-7        # world units: below this nothing visibly moved
MOVE_FRAC = 2e-4        # ...and below this fraction of the object, not worth it

# Monochrome clay, shaded in the shader. These are LINEAR values: the viewport
# applies its display transform on top (~x^(1/2.2)), so 0.36 linear reads as
# mid-grey on screen -- close to the real object, which keeps the hand-back at
# t=1 from flashing.
MESH_INK = (0.36, 0.37, 0.39, 1.0)

# The one mutable session. phase: IDLE -> PENDING (request parked) ->
# ACTIVE (objects hidden, overlay drawing) -> IDLE.
_S = {
    "phase": 'IDLE',
    "items": [],        # per-object morph payloads (see request())
    "vis": [],          # [(object, hide_get() before we touched it)]
    "win": None,        # window the request came from
    "t0": 0.0,
    "dur": DURATION,
    "deadline": 0.0,
    "draw_handle": None,
    "modal": False,
}

_shader_cache = None


def is_active():
    return _S["phase"] != 'IDLE'


# ------------------------------------------------------------- snapshots ---
# Pure reads. They run inside the host operator BEFORE it mutates anything,
# and must never raise into it.

def _visible_local_coords(obj):
    """What the artist sees, in local space: the relative shape-key mix
    (house rule: every key is relative to Basis - see op_wrap), or the raw
    vertices when there are no keys. Modifiers are intentionally ignored:
    the cage is modifier-free while wrapping, and this is only a preview."""
    me = obj.data
    n = len(me.vertices)
    buf = np.empty(n * 3, dtype=np.float64)
    kbs = me.shape_keys.key_blocks if me.shape_keys else None
    if not kbs:
        me.vertices.foreach_get("co", buf)
        return buf.reshape(n, 3).copy()
    if obj.show_only_shape_key and obj.active_shape_key is not None:
        obj.active_shape_key.data.foreach_get("co", buf)
        return buf.reshape(n, 3).copy()
    kbs[0].data.foreach_get("co", buf)
    basis = buf.reshape(n, 3).copy()
    out = basis.copy()
    for k in kbs[1:]:
        v = float(k.value)
        if k.mute or abs(v) <= 1e-9:
            continue
        k.data.foreach_get("co", buf)
        out += v * (buf.reshape(n, 3) - basis)
    return out


def _world(obj, co):
    mw = np.asarray(obj.matrix_world, dtype=np.float64)
    return co @ mw[:3, :3].T + mw[:3, 3]


def _evaluated_world(obj, depsgraph=None):
    """World-space vertices as the viewport actually draws them, modifiers
    included. A head bound to the facial rig is deformed by its Armature
    modifier, so its visible shape changes the moment the rest pose is refitted
    - that deformation is exactly what has to be morphed, and it is invisible
    to the shape-key path above."""
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    if me is None:
        return None
    try:
        n = len(me.vertices)
        if not (0 < n <= MAX_VERTS):
            return None
        buf = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", buf)
        co = buf.reshape(n, 3).copy()
    finally:
        ev.to_mesh_clear()
    return _world(ev, co)


def _evaluated_geometry(obj, depsgraph=None):
    """(world coords, triangles) of the evaluated mesh, in one pass."""
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    if me is None:
        return None, None
    try:
        n = len(me.vertices)
        if not (0 < n <= MAX_VERTS):
            return None, None
        buf = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", buf)
        co = buf.reshape(n, 3).copy()
        me.calc_loop_triangles()
        m = len(me.loop_triangles)
        if not (0 < m * 3 <= MAX_VERTS * 6):
            return None, None
        tris = np.empty(m * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(m, 3).copy()
    finally:
        ev.to_mesh_clear()
    return _world(ev, co), tris


def mesh_snapshot(obj, evaluated=False):
    """Visible world-space coords of a mesh object, or None (=> no morph).

    `evaluated=True` captures the mesh WITH its modifiers applied - use it for
    geometry whose visible shape is produced by something the operator is about
    to change (a head bound to the rig being refitted). The default reads the
    shape-key mix only, which is what the wrap needs."""
    try:
        if bpy.app.background or obj is None or obj.type != 'MESH':
            return None
        if not (0 < len(obj.data.vertices) <= MAX_VERTS):
            return None
        if evaluated:
            co = _evaluated_world(obj)
            return None if co is None else {"co": co, "evaluated": True}
        return {"co": _world(obj, _visible_local_coords(obj)),
                "evaluated": False}
    except Exception as exc:  # noqa: BLE001 - cosmetics must never break ops
        print(f"[MHFRT] transition: mesh snapshot skipped ({exc!r})")
        return None


def bones_snapshot(obj):
    """Visible world-space bone sticks {name: (head, tail)} of an armature.
    Pass None for a rig that does not exist yet: every final bone then
    grows into place instead of morphing. Returns None on failure."""
    try:
        if bpy.app.background:
            return None
        if obj is None:
            return {"bones": {}}
        if obj.type != 'ARMATURE' or len(obj.pose.bones) > MAX_BONES:
            return None
        mw = obj.matrix_world
        return {"bones": {pb.name: (np.array(mw @ pb.head, dtype=np.float64),
                                    np.array(mw @ pb.tail, dtype=np.float64))
                          for pb in obj.pose.bones}}
    except Exception as exc:  # noqa: BLE001
        print(f"[MHFRT] transition: bones snapshot skipped ({exc!r})")
        return None


# --------------------------------------------------------------- request ---

def request(context, mesh=None, bones=None, meshes=()):
    """Ask for a morph from the snapshotted state to the CURRENT state.

    mesh   = (mesh_object, mesh_snapshot result)
    meshes = iterable of the same, for operators that change several at once
             (refitting the rig re-deforms every mesh bound to it)
    bones  = (armature_object, bones_snapshot result, may_hide_armature)

    Call as the operator's last act before returning FINISHED. Never raises.
    """
    try:
        if bpy.app.background:
            return
        from ..core import render_state
        if render_state.is_rendering():
            # The morph hides and unhides real objects and blocks the window
            # with a modal. Neither belongs on top of a running render.
            return
        _finalize()                      # a stale session must never survive
        items = []
        for entry in ((mesh,) if mesh is not None else ()) + tuple(meshes):
            if entry is None:
                continue
            item = _mesh_item(*entry)
            if item is not None:
                items.append(item)
        if bones is not None:
            item = _bones_item(*bones)
            if item is not None:
                items.append(item)
        # A skipped transition must always say why: it is invisible by nature,
        # so without this the only symptom is "the animation doesn't play".
        if not items:
            print("[MHFRT] transition: nothing to morph (the bones and meshes "
                  "are already where the result puts them)")
            return
        if not _any_overlay_view3d():    # overlays off => our draw never runs
            print("[MHFRT] transition: skipped - no 3D viewport has Overlays "
                  "enabled, so the morph could not be drawn")
            return
        _S.update(items=items, win=context.window, phase='PENDING',
                  t0=0.0, deadline=time.monotonic() + PENDING_GRACE)
        _guards_add()
        # The bridge fires after this operator's undo push (push happens as
        # the operator returns) and before the next viewport draw - the only
        # spot where hiding can neither flash nor leak into undo/redo.
        if _deg_bridge not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_deg_bridge)
        if not bpy.app.timers.is_registered(_start_blocker):
            bpy.app.timers.register(_start_blocker, first_interval=0.0)
        if not bpy.app.timers.is_registered(_watchdog):
            bpy.app.timers.register(_watchdog, first_interval=WATCHDOG_STEP)
    except Exception as exc:  # noqa: BLE001
        print(f"[MHFRT] transition: request skipped ({exc!r})")
        try:
            _finalize()
        except Exception:  # noqa: BLE001
            pass


def _moved_enough(delta, obj):
    """Is this displacement worth hiding the real object and animating?

    The threshold is relative to the object, and deliberately not near zero:
    hiding a mesh to morph it by a few microns is all risk and no benefit."""
    try:
        size = max(float(max(obj.dimensions)), 0.0)
    except Exception:  # noqa: BLE001
        size = 0.0
    floor = max(MIN_DELTA, MOVE_FRAC * size)
    return float(np.abs(delta).max()) > floor


def _mesh_item(obj, snap):
    if snap is None or obj is None:
        return None
    try:
        if obj.type != 'MESH' or not obj.visible_get():
            return None
        if snap.get("evaluated"):
            # The end state depends on data the operator just changed, and the
            # depsgraph has not re-evaluated yet - resolve it in the bridge,
            # exactly like the bone sticks.
            return {"kind": 'MESH', "obj": obj, "hide": True, "pending": True,
                    "A": np.ascontiguousarray(snap["co"], dtype=np.float32),
                    "B": None, "tris": None, "batch": None, "dead": False}
        me = obj.data
        n = len(me.vertices)
        A = snap["co"]
        if len(A) != n:                     # topology changed: nothing to morph
            return None
        B = _world(obj, _visible_local_coords(obj))
        if not _moved_enough(B - A, obj):
            return None                     # result identical: don't block anyone
        me.calc_loop_triangles()
        m = len(me.loop_triangles)
        if not (0 < m * 3 <= MAX_VERTS * 6):
            return None
        tris = np.empty(m * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        return {"kind": 'MESH', "obj": obj, "hide": True, "pending": False,
                "A": np.ascontiguousarray(A, dtype=np.float32),
                "B": np.ascontiguousarray(B, dtype=np.float32),
                "tris": tris.reshape(m, 3), "batch": None, "dead": False}
    except Exception as exc:  # noqa: BLE001
        print(f"[MHFRT] transition: mesh morph skipped ({exc!r})")
        return None


def _bones_item(obj, snap, may_hide):
    if snap is None or obj is None:
        return None
    try:
        if obj.type != 'ARMATURE':
            return None
        # No visibility requirement, unlike a mesh morph: the sticks are OUR
        # overlay, not the armature's own drawing. The rig is hidden most of the
        # time - the display bar's bones toggle calls hide_set(), and a fitted
        # rig lives in the character collection - and the artist still wants to
        # watch it travel onto the head. _begin() only hides what was visible,
        # so a hidden armature stays hidden and nothing is restored wrongly.
        if len(obj.pose.bones) > MAX_BONES:
            return None
        # Final ("to") sticks are read in _setup(), after the depsgraph has
        # evaluated the refit pose - reading here could still see stale data.
        return {"kind": 'BONES', "obj": obj, "hide": bool(may_hide),
                "prev": snap["bones"], "sticks": None,
                "in_front": bool(obj.show_in_front), "dead": False}
    except Exception as exc:  # noqa: BLE001
        print(f"[MHFRT] transition: bone morph skipped ({exc!r})")
        return None


def _any_overlay_view3d():
    wm = bpy.context.window_manager
    if wm is None:
        return False
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type == 'VIEW_3D' and space.overlay.show_overlays:
                    return True
    return False


# ----------------------------------------------------------------- setup ---

def _deg_bridge(_scene, depsgraph=None):
    try:
        bpy.app.handlers.depsgraph_update_post.remove(_deg_bridge)
    except ValueError:
        pass
    try:
        _setup(depsgraph)
    except Exception as exc:  # noqa: BLE001
        print(f"[MHFRT] transition: setup failed ({exc!r})")
        _finalize()


def _setup(depsgraph):
    """Hide the real objects and start drawing. Runs once, between the host
    operator's undo push and the next redraw (or from the timer fallback)."""
    if _S["phase"] != 'PENDING':
        return
    items = []
    for item in _S["items"]:
        try:
            if item["kind"] == 'BONES':
                item = _resolve_bone_sticks(item, depsgraph)
            elif item.get("pending"):
                item = _resolve_mesh_item(item, depsgraph)
            if item is not None:
                items.append(item)
        except Exception as exc:  # noqa: BLE001
            print(f"[MHFRT] transition: item dropped ({exc!r})")
    _S["items"] = items
    if not items:
        _finalize()
        return
    for item in items:
        if not item["hide"]:
            continue
        obj = item["obj"]
        try:
            prev = obj.hide_get()
            if not prev:
                obj.hide_set(True)
            _S["vis"].append((obj, prev))
        except Exception:  # noqa: BLE001 - keep drawing over the visible object
            item["hide"] = False
    if _S["draw_handle"] is None:
        _S["draw_handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
    _S["t0"] = time.monotonic()
    # The rig flying onto the head is the slowest, most watchable part; a mesh
    # morph on its own stays snappy.
    _S["dur"] = (BONES_DURATION
                 if any(i["kind"] == 'BONES' for i in items) else DURATION)
    _S["deadline"] = _S["t0"] + _S["dur"] + ACTIVE_GRACE
    _S["phase"] = 'ACTIVE'
    _tag_redraw_3d()


def _resolve_mesh_item(item, depsgraph):
    """Finish an evaluated-mesh item once the depsgraph carries the operator's
    result: read the deformed end state and its triangulation."""
    obj = item["obj"]
    B, tris = _evaluated_geometry(obj, depsgraph)
    if B is None or tris is None:
        return None
    A = item["A"]
    if len(A) != len(B):            # a modifier changed the topology: skip
        return None
    if not _moved_enough(B - A, obj):
        return None                 # this mesh did not move: nothing to show
    item["B"] = np.ascontiguousarray(B, dtype=np.float32)
    item["tris"] = tris
    item["pending"] = False
    return item


def _resolve_bone_sticks(item, depsgraph):
    """Pair previous sticks with the depsgraph-evaluated final pose:
    moved bones morph, brand-new bones grow out of their head, removed
    bones fade where they were. Bones that did not move are drawn static
    only when the real armature is hidden in their place."""
    obj = item["obj"]
    src = obj
    try:
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        src = obj.evaluated_get(depsgraph)
    except Exception:  # noqa: BLE001 - fall back to last flushed pose
        src = obj
    mw = src.matrix_world
    prev = item["prev"]
    moved, grown, gone, still = [], [], [], []
    for pb in src.pose.bones:
        h = np.array(mw @ pb.head, dtype=np.float64)
        t = np.array(mw @ pb.tail, dtype=np.float64)
        old = prev.get(pb.name)
        if old is None:
            grown.append((h, h, h, t))          # head->head grows to head->tail
        elif (np.abs(h - old[0]).max() > MIN_DELTA
              or np.abs(t - old[1]).max() > MIN_DELTA):
            moved.append((old[0], old[1], h, t))
        else:
            still.append((h, t))
    known = {pb.name for pb in src.pose.bones}
    for name, (h, t) in prev.items():
        if name not in known:
            gone.append((h, t))
    if not moved and not grown and not gone:
        print(f"[MHFRT] transition: '{obj.name}' bones did not move "
              f"({len(still)} unchanged) - nothing to animate")
        return None                              # rig unchanged: skip morph
    if not item["hide"]:
        still = []                               # real bones already show them

    def _pack(rows, ai, bi):
        if not rows:
            return None
        a = np.empty((len(rows) * 2, 3), dtype=np.float64)
        b = np.empty_like(a)
        for k, r in enumerate(rows):
            a[2 * k], a[2 * k + 1] = r[ai[0]], r[ai[1]]
            b[2 * k], b[2 * k + 1] = r[bi[0]], r[bi[1]]
        return a, b

    item["sticks"] = {
        "moved": _pack(moved, (0, 1), (2, 3)),
        "grown": _pack(grown, (0, 1), (2, 3)),
        "gone": _pack([(h, t, h, t) for h, t in gone], (0, 1), (2, 3)),
        "still": _pack([(h, t, h, t) for h, t in still], (0, 1), (2, 3)),
    }
    item.pop("prev", None)
    return item


def _start_blocker():
    """0 s timer: put the blocking modal in charge (and cover the rare case
    where no depsgraph update followed the operator)."""
    try:
        if _S["phase"] == 'PENDING':
            _setup(None)
        if _S["phase"] != 'ACTIVE' or _S["modal"]:
            return None
        win = _S["win"]
        wm = bpy.context.window_manager
        if wm is None:
            _finalize()
            return None
        if win is None or win not in list(wm.windows):
            win = wm.windows[0] if wm.windows else None
        if win is None:
            _finalize()
            return None
        with bpy.context.temp_override(window=win):
            ret = bpy.ops.mhfrt.visual_transition('INVOKE_DEFAULT')
        if 'RUNNING_MODAL' not in ret:
            _play_unblocked()   # no input blocking, but still show the morph
    except Exception as exc:  # noqa: BLE001
        print(f"[MHFRT] transition: blocker not started ({exc!r})")
        _play_unblocked()
    return None


def _play_unblocked():
    """Drive the morph from a plain timer when the blocking modal cannot run.

    Losing the modal used to throw the whole transition away, so the artist saw
    NOTHING - the one outcome worse than an unblocked morph. Input is not
    swallowed for these few frames; the watchdog still bounds the session.
    """
    if _S["phase"] != 'ACTIVE' or bpy.app.timers.is_registered(_play_tick):
        return
    bpy.app.timers.register(_play_tick, first_interval=0.0)


def _play_tick():
    if _S["phase"] != 'ACTIVE' or _S["modal"]:
        return None                      # finished, or the modal took over
    if time.monotonic() - _S["t0"] >= _S.get("dur", DURATION):
        _finalize()
        return None
    _tag_redraw_3d()
    return TIMER_STEP


def _watchdog():
    """Last line of defence: whatever happens to the modal or the handlers,
    a transition can never outlive DURATION + grace."""
    if _S["phase"] == 'IDLE':
        return None
    if time.monotonic() > _S["deadline"]:
        _finalize()
        return None
    return WATCHDOG_STEP


# ------------------------------------------------------- guards / cleanup ---

def _abort(*_args):
    """undo/redo/load/save about to happen mid-morph: restore instantly so
    the file and the undo stream only ever contain the addon's real state."""
    _finalize()


_GUARD_LISTS = ("undo_pre", "redo_pre", "load_pre", "save_pre")


def _guards_add():
    for name in _GUARD_LISTS:
        lst = getattr(bpy.app.handlers, name)
        if _abort not in lst:
            lst.append(_abort)


def _guards_remove():
    for name in _GUARD_LISTS:
        try:
            getattr(bpy.app.handlers, name).remove(_abort)
        except (ValueError, AttributeError):
            pass


def _finalize():
    """Idempotent teardown: restore visibility, drop every handler, free GPU
    payloads, redraw. Every exit path (natural end, Esc, cancel(), guards,
    watchdog, unregister) funnels through here."""
    if _S["phase"] == 'IDLE' and not _S["vis"] and _S["draw_handle"] is None:
        return
    _S["phase"] = 'IDLE'
    try:
        bpy.app.handlers.depsgraph_update_post.remove(_deg_bridge)
    except ValueError:
        pass
    _guards_remove()
    if _S["draw_handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_S["draw_handle"], 'WINDOW')
        except Exception:  # noqa: BLE001
            pass
        _S["draw_handle"] = None
    for obj, prev in reversed(_S["vis"]):
        try:
            obj.hide_set(prev)
        except Exception:  # noqa: BLE001 - object may be gone (new file, delete)
            pass
    _S["vis"] = []
    _S["items"] = []
    _S["win"] = None
    try:
        _tag_redraw_3d()
    except Exception:  # noqa: BLE001
        pass


def force_finish():
    """Public: snap any running transition to its end state immediately."""
    _finalize()


def _tag_redraw_3d():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# -------------------------------------------------------- blocking modal ---

class MHFRT_OT_visual_transition(bpy.types.Operator):
    """Internal: swallows input while the wrap / rig morph preview plays"""
    bl_idname = "mhfrt.visual_transition"
    bl_label = "Morphing"
    bl_options = {'INTERNAL'}   # deliberately no 'UNDO': zero undo entries

    _timer = None

    @classmethod
    def poll(cls, context):
        return _S["phase"] == 'ACTIVE'

    def invoke(self, context, _event):
        if _S["phase"] != 'ACTIVE' or _S["modal"]:
            return {'CANCELLED'}
        wm = context.window_manager
        self._timer = wm.event_timer_add(TIMER_STEP, window=context.window)
        wm.modal_handler_add(self)
        _S["modal"] = True
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if _S["phase"] != 'ACTIVE':          # finished/aborted behind our back
            return self._done(context)
        if event.type == 'TIMER':
            if time.monotonic() - _S["t0"] >= _S.get("dur", DURATION):
                _finalize()
                return self._done(context)
            _tag_redraw_3d()
            return {'RUNNING_MODAL'}
        if event.type == 'ESC' and event.value == 'PRESS':
            _finalize()                      # skip straight to the real result
            return self._done(context)
        return {'RUNNING_MODAL'}             # block everything else

    def cancel(self, context):
        # Blender ends us itself (file load, window closed, quit).
        self._done(context)
        _finalize()

    def _done(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:  # noqa: BLE001
                pass
            self._timer = None
        _S["modal"] = False
        return {'FINISHED'}


# ---------------------------------------------------------------- drawing ---

def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)   # smootherstep


def _shader():
    """Morph surface shader: blends the two baked positions on the GPU and
    shades flat with a headlight - one uniform per frame, no CPU work."""
    global _shader_cache
    if _shader_cache is not None:
        return _shader_cache
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC4', "u_color")
    info.push_constant('VEC3', "u_eye")
    info.push_constant('FLOAT', "u_t")
    info.vertex_in(0, 'VEC3', "pos_a")
    info.vertex_in(1, 'VEC3', "pos_b")
    iface = gpu.types.GPUStageInterfaceInfo("mhfrt_morph_iface")
    iface.smooth('VEC3', "vPos")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source("""
void main()
{
    vec3 p = mix(pos_a, pos_b, u_t);
    vPos = p;                       // shade the true position
    /* The morph ends up exactly ON the target surface, so the last frames
       would z-fight it into random stipple. Nudge the drawn position a hair
       toward the eye: moving toward the camera always reduces depth, whatever
       the backend's depth convention, and 0.15% of the view distance is far
       below anything visible. */
    vec3 to_eye = u_eye - p;
    float d = length(to_eye);
    vec3 biased = (d > 1e-9) ? p + (to_eye / d) * (d * 0.0015) : p;
    gl_Position = u_mvp * vec4(biased, 1.0);
}
""")
    info.fragment_source("""
void main()
{
    vec3 nrm = cross(dFdx(vPos), dFdy(vPos));
    float nl = length(nrm);
    nrm = nl > 0.0 ? nrm / nl : vec3(0.0, 0.0, 1.0);
    vec3 v = normalize(u_eye - vPos);
    float ndv = clamp(abs(dot(nrm, v)), 0.0, 1.0);
    float shade = 0.30 + 0.70 * ndv;                  // headlight lambert
    shade *= 0.72 + 0.28 * smoothstep(0.0, 0.45, ndv); // darker grazing rim
    fragColor = vec4(u_color.rgb * shade, u_color.a);
}
""")
    _shader_cache = gpu.shader.create_from_info(info)
    return _shader_cache


def _mesh_batch(item):
    if item["batch"] is None:
        # NOTE: the format is built by hand to match the vertex_in names in
        # _shader(). Do NOT switch to shader.format_calc() - for shaders made
        # from GPUShaderCreateInfo it does not carry the custom attribute
        # names and attr_fill then raises "Unknown attribute 'pos_a'".
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos_a", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="pos_b", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(format=fmt, len=len(item["A"]))
        vbo.attr_fill("pos_a", item["A"])
        vbo.attr_fill("pos_b", item["B"])
        ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=item["tris"])
        item["batch"] = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo)
    return item["batch"]


def _landmark_session():
    """Is the landmark tool running? Only ITS local views own the stage - an
    artist who simply isolated the head with `/` must still see the morph."""
    try:
        from ..ops import op_pairs
        return bool(op_pairs.is_running())
    except Exception:  # noqa: BLE001
        return False


def _draw():
    if _S["phase"] != 'ACTIVE':
        return
    space = bpy.context.space_data
    if space is None:
        return
    if getattr(space, "local_view", None) and _landmark_session():
        return          # landmark dual-view local views keep their own stage
    t = _ease((time.monotonic() - _S["t0"]) / _S.get("dur", DURATION))
    try:
        mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        eye = gpu.matrix.get_model_view_matrix().inverted().translation
    except Exception:  # noqa: BLE001
        return
    for item in _S["items"]:
        if item.get("dead"):
            continue
        try:
            if item["kind"] == 'MESH':
                _draw_mesh(item, mvp, eye, t)
            else:
                _draw_bones(item, t)
        except Exception as exc:  # noqa: BLE001 - never break viewport drawing
            item["dead"] = True
            print(f"[MHFRT] transition: draw dropped ({exc!r})")
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set('NONE')
    gpu.state.face_culling_set('NONE')
    gd.finish()


def _draw_mesh(item, mvp, eye, t):
    sh = _shader()
    batch = _mesh_batch(item)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')   # scene still occludes the morph
    gpu.state.depth_mask_set(True)
    gpu.state.face_culling_set('NONE')
    sh.bind()
    sh.uniform_float("u_mvp", mvp)
    sh.uniform_float("u_eye", eye)
    sh.uniform_float("u_color", MESH_INK)
    sh.uniform_float("u_t", t)
    batch.draw(sh)


def _draw_bones(item, t):
    """Sticks, drawn like the armature they stand in for (organize_skeleton sets
    display_type='STICK', show_in_front=True). Facial joints are small and sit
    inside the head, so each run gets a dark underlay first: without it the
    motion is easy to lose against the mesh."""
    sticks = item["sticks"]
    if not sticks:
        return
    gpu.state.depth_mask_set(False)
    # Always x-ray. Facial joints sit INSIDE the head, so depth-testing them
    # against it hides the entire preview behind the face - the artist would
    # see the rig change with no transition at all. The real armature is hidden
    # for these few frames anyway, so nothing competes with the sticks.
    gpu.state.depth_test_set('ALWAYS')

    def _blend(pair):
        a, b = pair
        return (a + (b - a) * t).tolist()

    def _run(pair, color, width, outline=True):
        pts = _blend(pair)
        if outline:
            gd.segments(pts, gd.OUTLINE, width + 2.2)
        gd.segments(pts, color, width)
        return pts

    if sticks["still"] is not None:
        _run(sticks["still"], gd.MID, 1.6)
    if sticks["gone"] is not None:
        c = gd.DIM
        _run(sticks["gone"], (c[0], c[1], c[2], c[3] * (1.0 - t)), 1.6,
             outline=False)
    for key in ("moved", "grown"):
        pair = sticks[key]
        if pair is None:
            continue
        # The joints in flight ARE the preview: draw them fat and bright.
        # 838 facial sticks inside a head read as a faint smudge at 2.4 px -
        # the artist reported seeing "zero animation" at that weight.
        _run(pair, gd.WHITE, 3.6)


# ---------------------------------------------------------- registration ---

_classes = (MHFRT_OT_visual_transition,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    # Disabling the add-on mid-morph must leave the scene exactly as the
    # host operator finished it: visible objects, no handlers, no overlay.
    try:
        _finalize()
    except Exception:  # noqa: BLE001
        pass
    global _shader_cache
    _shader_cache = None
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
