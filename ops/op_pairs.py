"""PRO landmark tool - surface curves, ZRemesher style.

One modal tool does everything, ZWrap-style split view + ZBrush-style curves:

* SPLIT VIEW: entering the tool splits the 3D viewport in two - the HEAD
  CAGE alone on the left, YOUR HEAD alone on the right (each in its own
  temporary Local View with an independent user-perspective camera, exactly
  like ZWrap's side-by-side layout). Finishing the tool restores the
  original single viewport.
* hold LMB and DRAW a curve on the cage surface (like drawing a ZRemesher
  guide curve), release -> the tool pauses in a Ctrl+R-style preview:
  MOVING THE MOUSE left/right changes how many evenly spaced landmarks ride
  the curve (scrolling and typing a number do the same), LMB confirms,
  RMB / Esc discards the whole temporary curve.  The mouse dial is the one
  that always works - the wheel is unreliable across keymaps and trackpads
* then draw the matching curve on the head - it automatically uses the SAME
  landmark count (locked, nothing to dial), and point 1 pairs with point 1
* after confirmation points can be DRAGGED to reshape the curve; dragging one
  endpoint onto the other merges them and turns the curve into a closed loop
* X / Delete removes the whole hovered curve, Ctrl+Z undoes any action
* auto-mirroring across X for new curves (symmetry toggle); points within
  the centre tolerance stay single (centre merge)
* when the viewport cannot be split (tiny area, fullscreen), the tool falls
  back to the classic single view with auto-solo (H shows both meshes)

While the tool runs, the cage is shown in its ORIGINAL (Basis) shape -
correspondences are defined on the undeformed cage, so editing after a wrap
stays correct. The previous Wrapped value is restored on exit.

Curves are pure data (scene properties) drawn with a monochrome GPU overlay;
there are no curve or marker objects in the scene, and a curve that was never
confirmed never touches the data at all.
"""

import math
import re

import bpy
import numpy as np
from mathutils import Vector, Quaternion
from bpy_extras import view3d_utils
from bpy_extras.view3d_utils import location_3d_to_region_2d

from ..core import landmarks as lmdata
from ..core import nav
from ..core import organization
from ..ui import gizmo_draw as gd

WRAPPED_KEY = lmdata.WRAPPED_KEY
PICK_PX = 14.0          # hover radius in pixels
DRAG_START_PX = 3.0

# --- surface-curve drawing ---------------------------------------------------
CURVE_MIN = 3           # a curve carries at least 3 landmarks
CURVE_MAX = 32
CURVE_DEFAULT = 5
STROKE_SAMPLE_PX = 4.0  # min screen-space gap between stroke samples
STROKE_MIN_PX = 12.0    # shorter strokes count as a stray click, not a curve
STROKE_MAX_SAMPLES = 4000

# --- lazy mouse (ZBrush-style trailing pen) ----------------------------------
# The stroke is not drawn at the pointer.  A "pen" is dragged along behind it
# on a leash of `landmark_lazy_radius` pixels and eased toward that position by
# `landmark_lazy_smooth`; the surface is sampled at the PEN.  Hand tremor,
# tablet jitter and the little hook everyone makes at the end of a fast stroke
# all happen inside the leash, so they never reach the curve - the same reason
# every sculpting app ships this.
LAZY_FLUSH_STEPS = 40   # retained for the legacy helper; release skips it
LAZY_CATCHUP = 0.34
# Ctrl+R-style point-count dial.  The wheel is only ever a shortcut here: on
# Industry Compatible the wheel is bound through WHEELIN/OUTMOUSE, laptop
# trackpads and Magic Mice send TRACKPAD* instead of a wheel at all, and some
# artists simply have no wheel.  So the dial that everyone gets is MOUSE
# MOVEMENT plus typing a number - the wheel and the keys below are extras that
# land on top of it, and no keymap preset can take the tool away.
_DIAL_UP = {'WHEELUPMOUSE', 'WHEELINMOUSE', 'UP_ARROW', 'RIGHT_ARROW',
            'PAGE_UP', 'NUMPAD_PLUS', 'EQUAL', 'PLUS'}
_DIAL_DOWN = {'WHEELDOWNMOUSE', 'WHEELOUTMOUSE', 'DOWN_ARROW', 'LEFT_ARROW',
              'PAGE_DOWN', 'NUMPAD_MINUS', 'MINUS'}
_WHEEL = {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE'}

# Move-the-mouse dial: pixels of horizontal travel per extra landmark, and the
# slack around the release point so a twitchy hand never re-counts a curve.
DIAL_STEP_PX = 26.0
DIAL_DEAD_PX = 16.0

_DIGITS = {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
           'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9}
_NUMPAD_DIGITS = {f'NUMPAD_{d}': d for d in range(10)}


def _digit(event):
    """The number the artist just typed, or None.

    Number-row keys always count.  Numpad keys only count when *Emulate
    Numpad* is on - with real numpad keys those are view shortcuts, and
    stealing them mid-preview would break navigation.
    """
    d = _DIGITS.get(event.type)
    if d is not None:
        return d
    try:
        emulate = bpy.context.preferences.inputs.use_emulate_numpad
    except AttributeError:
        emulate = False
    return _NUMPAD_DIGITS.get(event.type) if emulate else None

# area pointer -> 'CAGE' | 'TARGET' while the tool's split layout is live.
# The persistent overlay reads this to draw only that side's markers in each
# of the two viewports.
_DUAL_SIDES = {}

# True while the landmark tool is modal.  The rest of Blender stays usable
# during the tool (events outside its viewports pass through), so operators
# that would fight it - re-entry, tab jumps, character switches - check this.
_RUNNING = False


def is_running():
    return _RUNNING


# Front view (numpad 1): look down -Y with +Z up.  Both split viewports open
# on the front of the head instead of inheriting the working viewport's angle.
_FRONT_QUAT = Quaternion((0.7071068, 0.7071068, 0.0, 0.0))


def _face_front(rv3d):
    """Point one viewport at the front of its framed mesh, in perspective."""
    if rv3d is None:
        return
    rv3d.view_rotation = _FRONT_QUAT.copy()
    rv3d.view_perspective = 'PERSP'


# ---- ZWrap-style camera sync between the two split viewports --------------
# A bpy.app timer mirrors orbit + zoom from whichever side the artist moves
# onto the other.  A timer (not the modal) is used on purpose: it keeps
# running while Blender's own view-navigation modal has control of events, so
# the sync stays live *during* an orbit, not only after it ends.
_CAM_SYNC = {"window": None, "areas": (), "last": [None, None]}


def _rv3d_snapshot(rv3d):
    return (tuple(round(c, 6) for c in rv3d.view_rotation),
            round(rv3d.view_distance, 6),
            rv3d.view_perspective,
            tuple(round(c, 6) for c in rv3d.view_location))


def _sync_area_rv3d(window, area_ptr):
    area = next((a for a in window.screen.areas
                 if a.as_pointer() == area_ptr), None)
    if area is None:
        return None, None
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    rv3d = getattr(region, "data", None) if region is not None else None
    return area, rv3d


def _start_cam_sync(window, left, right):
    _CAM_SYNC["window"] = window
    _CAM_SYNC["areas"] = (left.as_pointer(), right.as_pointer())
    _CAM_SYNC["last"] = [None, None]
    if not bpy.app.timers.is_registered(_cam_sync_tick):
        bpy.app.timers.register(_cam_sync_tick, first_interval=0.05)


def _stop_cam_sync():
    _CAM_SYNC["window"] = None
    _CAM_SYNC["areas"] = ()
    _CAM_SYNC["last"] = [None, None]


def _force_cam_align():
    """Snap the target view onto the cage view (used when sync is switched on
    mid-session so the two don't stay diverged until the next move)."""
    window = _CAM_SYNC["window"]
    if window is None or not _CAM_SYNC["areas"]:
        return
    ptr_a, ptr_b = _CAM_SYNC["areas"]
    area_a, rv_a = _sync_area_rv3d(window, ptr_a)
    area_b, rv_b = _sync_area_rv3d(window, ptr_b)
    if rv_a is None or rv_b is None:
        return
    rv_b.view_rotation = rv_a.view_rotation
    rv_b.view_distance = rv_a.view_distance
    rv_b.view_perspective = rv_a.view_perspective
    _CAM_SYNC["last"] = [None, None]
    area_a.tag_redraw()
    area_b.tag_redraw()


def _cam_sync_tick():
    """Mirror orbit/zoom between the two landmark viewports while enabled."""
    if not _RUNNING or _CAM_SYNC["window"] is None:
        return None                     # tool ended -> stop the timer
    try:
        window = _CAM_SYNC["window"]
        wm = bpy.context.window_manager
        if window not in list(wm.windows):
            _stop_cam_sync()
            return None
        mh = getattr(bpy.context.scene, "mhfrt", None)
        if mh is None or not mh.landmark_sync_view:
            _CAM_SYNC["last"] = [None, None]
            return 0.08                 # sync off: idle-poll, don't mirror
        ptr_a, ptr_b = _CAM_SYNC["areas"]
        area_a, rv_a = _sync_area_rv3d(window, ptr_a)
        area_b, rv_b = _sync_area_rv3d(window, ptr_b)
        if rv_a is None or rv_b is None:
            return 0.08
        snap_a, snap_b = _rv3d_snapshot(rv_a), _rv3d_snapshot(rv_b)
        last_a, last_b = _CAM_SYNC["last"]
        if last_a is None or last_b is None:
            _CAM_SYNC["last"] = [snap_a, snap_b]
            return 0.016
        src = dst = redraw = prev = None
        if snap_a != last_a:
            src, dst, redraw, prev = rv_a, rv_b, area_b, last_a
        elif snap_b != last_b:
            src, dst, redraw, prev = rv_b, rv_a, area_a, last_b
        if src is not None:
            dst.view_rotation = src.view_rotation
            dst.view_distance = src.view_distance
            dst.view_perspective = src.view_perspective
            # Pan: apply the source's world-space location DELTA rather than
            # its absolute location, so each view keeps its own centering on
            # its own head instead of jumping onto the other one.  prev[3] is
            # the source's previous view_location (snapshot's 4th field).
            dst.view_location = dst.view_location + (
                src.view_location - Vector(prev[3]))
            _CAM_SYNC["last"] = [_rv3d_snapshot(rv_a), _rv3d_snapshot(rv_b)]
            redraw.tag_redraw()
        else:
            _CAM_SYNC["last"] = [snap_a, snap_b]
        return 0.016
    except (ReferenceError, AttributeError, RuntimeError):
        return 0.08


def _force_default_cursor(window):
    """Pop any live modal cursor and reset the pointer shape to DEFAULT.

    Blender's cursor bookkeeping doesn't always refresh the on-screen shape
    when a modal ends - the fix needs three prongs:
      1. ``cursor_modal_restore`` pops any modal cursor we (or anyone else)
         pushed while the tool ran.
      2. ``cursor_modal_set('DEFAULT')`` + immediate restore forces the
         modal stack to a known empty state even if a stray push wasn't
         balanced by a pop.
      3. ``cursor_set('DEFAULT')`` writes DEFAULT to the window's normal
         cursor so the shape shown when there's no modal is the arrow.
    Cheap redundancy - running these on every exit path is a lot cheaper
    than a stuck crosshair that outlives the tool."""
    for step in (
        lambda: window.cursor_modal_restore(),
        lambda: window.cursor_modal_set('DEFAULT'),
        lambda: window.cursor_modal_restore(),
        lambda: window.cursor_set('DEFAULT'),
    ):
        try:
            step()
        except (RuntimeError, ReferenceError, TypeError):
            pass


def _delayed_default_cursor(window_ptr):
    """Backup cursor reset run one tick after the modal returns.

    A timer runs OUTSIDE Blender's event dispatch, so any post-modal cursor
    change Blender does internally has already happened when this fires. That
    makes it the last word on the pointer shape."""
    try:
        for w in bpy.context.window_manager.windows:
            if w.as_pointer() == window_ptr:
                _force_default_cursor(w)
                break
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    return None


def dual_side_of(area):
    """'CAGE'/'TARGET' when the landmark tool's split layout owns this area."""
    if not _DUAL_SIDES or area is None:
        return None
    return _DUAL_SIDES.get(area.as_pointer())


# ---------------------------------------------------------------- raycast ---

def _region_ray(region, rv3d, mouse):
    if region is None or rv3d is None:
        return None, None
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse).normalized()
    return origin, direction


def _nearest_vidx(obj, local_co, poly_idx):
    me = obj.data
    if 0 <= poly_idx < len(me.polygons):
        verts = me.polygons[poly_idx].vertices
        return int(min(verts,
                       key=lambda v: (me.vertices[v].co - local_co).length_squared))
    return -1


def _cast(context, obj, origin, direction, want_vidx=True):
    mwi = obj.matrix_world.inverted()
    ro = mwi @ Vector(origin)
    rd = mwi.to_3x3() @ Vector(direction)
    if rd.length < 1e-12:
        return None
    rd.normalize()
    try:
        ok, loc, nor, idx = obj.ray_cast(
            ro, rd, depsgraph=context.evaluated_depsgraph_get())
    except RuntimeError:
        return None
    if not ok:
        return None
    mw = obj.matrix_world
    normal = (mw.to_3x3() @ nor).normalized() if nor is not None else None
    return {
        "world": mw @ loc,
        "normal": normal,
        "local": loc.copy(),
        # only the points that become DATA need their nearest vertex; the
        # stroke samples and the hover ring never read it, and looking it up
        # is four RNA vertex reads we would otherwise pay on every mouse move
        "vidx": _nearest_vidx(obj, loc, idx) if want_vidx else -1,
        "view": Vector(direction),
    }


def raycast(context, obj, mouse, region, rv3d, want_vidx=True):
    """Viewport-ray hit on obj in one specific region (tries both ray
    directions, like the old tool, to survive cameras inside the mesh)."""
    origin, direction = _region_ray(region, rv3d, mouse)
    if origin is None:
        return None
    return (_cast(context, obj, origin, direction, want_vidx)
            or _cast(context, obj, origin, -direction, want_vidx))


# ------------------------------------------------------------- lazy mouse ---

def lazy_settings(mh):
    """(radius_px, smoothing) for the trailing pen - (0, 0) when it is off."""
    if not getattr(mh, "landmark_lazy", False):
        return 0.0, 0.0
    return (max(0.0, float(getattr(mh, "landmark_lazy_radius", 0.0))),
            min(max(float(getattr(mh, "landmark_lazy_smooth", 0.0)), 0.0), 0.95))


def lazy_step(pen, mouse, radius, smooth):
    """Advance the trailing pen one event toward the pointer.

    Two effects stack: a leash of `radius` pixels the pen never leaves (small
    movements inside it move nothing at all, which is what kills tremor), and
    an ease of `smooth` toward wherever the leash allows (which is what makes
    a fast stroke curve instead of corner).
    """
    if radius <= 0.0 and smooth <= 0.0:
        return mouse.copy()
    d = mouse - pen
    dist = d.length
    if radius > 0.0:
        goal = pen if dist <= radius else pen + d * ((dist - radius) / dist)
    else:
        goal = mouse
    if smooth > 0.0:
        return pen.lerp(goal, max(1.0 - smooth, 0.05))
    return Vector(goal)


# --------------------------------------------------------------- symmetry ---

# Local-X half-width, memoized per mesh.
#
# This is the ruler behind every symmetry decision - the centre-line clip runs
# it on EVERY stroke sample, the drag sync on every mouse move, the preview on
# every count change - and it used to be a Python list comprehension over the
# whole vertex array. On the 6k cage nobody could tell; on a two-million-vertex
# head it is roughly a second of RNA traffic per mouse move, which is the whole
# reason drawing on a dense mesh felt like the tool had frozen. One numpy pass,
# cached per mesh for the session.
_HALFWIDTH_CACHE = {}


def _clear_halfwidth_cache():
    _HALFWIDTH_CACHE.clear()


def local_halfwidth_x(obj):
    me = getattr(obj, "data", None)
    n = len(me.vertices) if me is not None else 0
    if not n:
        return 1e-6
    key = me.as_pointer()
    cached = _HALFWIDTH_CACHE.get(key)
    if cached is not None and cached[0] == n:
        return cached[1]
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    xs = co[0::3]
    half = max(1e-6, float(xs.max() - xs.min()) * 0.5)
    _HALFWIDTH_CACHE[key] = (n, half)
    return half


def _local_bbox_extent(obj):
    """Largest edge of the object's LOCAL-space bounding box.

    ``obj.dimensions`` reports WORLD-space size, which is wrong to use as a
    ray span in local-space raycasts - a cage scaled 0.3x ends up with a
    span smaller than the local mesh, so the mirror ray can start inside
    the head and miss its front."""
    xs = [c[0] for c in obj.bound_box]
    ys = [c[1] for c in obj.bound_box]
    zs = [c[2] for c in obj.bound_box]
    return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def mirror_on_surface(context, obj, local_co, view_dir_world):
    """Mirror a local point across local X=0 and re-project it onto the surface
    using the mirrored view ray (finds the partner point the same way the
    original was clicked, from the opposite viewing side)."""
    ml = Vector(local_co)
    ml.x = -ml.x
    dg = context.evaluated_depsgraph_get()

    if view_dir_world is not None:
        ld = obj.matrix_world.inverted().to_3x3() @ Vector(view_dir_world)
        if ld.length > 1e-9:
            ld.x = -ld.x
            ld.normalize()
            # Local-space span so a moved/scaled cage still gets a ray start
            # well outside its mesh.
            span = max(_local_bbox_extent(obj), 1e-6) * 4.0
            for start, direction in ((ml - ld * span, ld), (ml + ld * span, -ld)):
                try:
                    ok, loc, _nor, idx = obj.ray_cast(start, direction, depsgraph=dg)
                except RuntimeError:
                    ok = False
                if ok:
                    return loc.copy(), _nearest_vidx(obj, loc, idx), True

    try:
        ok, loc, _nor, idx = obj.closest_point_on_mesh(ml, depsgraph=dg)
    except RuntimeError:
        ok = False
    if ok:
        return loc.copy(), _nearest_vidx(obj, loc, idx), False
    return ml, -1, False


# ------------------------------------------------------------ stroke math ---

def _smooth_pts(pts, passes=2):
    """Light Laplacian smoothing of interior samples (kills raycast jitter
    without dragging the stroke off its endpoints)."""
    out = [Vector(p) for p in pts]
    if len(out) < 3:
        return out
    for _ in range(passes):
        prev = [v.copy() for v in out]
        for i in range(1, len(out) - 1):
            out[i] = prev[i] * 0.5 + (prev[i - 1] + prev[i + 1]) * 0.25
    return out


def _resample_polyline(pts, views, count):
    """`count` (world, view) pairs spread evenly by arc length along the
    sampled stroke - endpoints included, so a curve drawn eye-corner to
    eye-corner lands landmarks exactly on the corners. count == 1 takes the
    stroke's midpoint (a single landmark)."""
    segs, total = [], 0.0
    for a, b in zip(pts, pts[1:]):
        d = (b - a).length
        segs.append(d)
        total += d
    if total <= 1e-9:
        return [(pts[0].copy(), Vector(views[0]))] * count
    if count == 1:
        targets = [total * 0.5]
    else:
        targets = [total * i / (count - 1) for i in range(count)]
    out = []
    si, acc = 0, 0.0
    for t in targets:
        while si < len(segs) - 1 and acc + segs[si] < t:
            acc += segs[si]
            si += 1
        d = segs[si]
        f = 0.0 if d <= 1e-12 else min(max((t - acc) / d, 0.0), 1.0)
        out.append((pts[si].lerp(pts[si + 1], f), Vector(views[si])))
    return out


def _project_stroke_points(context, obj, pairs):
    """Snap resampled stroke points back onto the mesh surface (resampling
    cuts chords between samples; the landmarks must sit ON the skin)."""
    dg = context.evaluated_depsgraph_get()
    mw = obj.matrix_world
    mwi = mw.inverted()
    out = []
    for w, view in pairs:
        local = mwi @ Vector(w)
        try:
            ok, loc, _nor, poly = obj.closest_point_on_mesh(local, depsgraph=dg)
        except RuntimeError:
            ok = False
        if not ok:
            loc, poly = local.copy(), -1
        out.append({
            "local": loc.copy(),
            "world": mw @ loc,
            "vidx": _nearest_vidx(obj, loc, poly),
            "view": Vector(view),
        })
    return out


def _mirror_keep_indices(mh):
    """Which points of the pending cage curve got a mirrored twin - every
    point beyond the centre tolerance. Recomputed from data on demand, so
    in-tool undo can never desync it."""
    pending = lmdata.pending_curve(mh)
    if pending is None or mh.cage is None:
        return []
    half = local_halfwidth_x(mh.cage)
    thr = max(mh.symmetry_center_threshold * half, 1e-9)
    _cid, idxs = pending
    return [k for k, i in enumerate(idxs)
            if abs(mh.landmarks[i].src_co[0]) > thr]


# ------------------------------------------------------------- view modes ---

def apply_view_mode(context):
    """Hide/show cage and target per the scene 'view_mode'. Marker visibility
    is handled by the GPU overlay (markers are data, not objects)."""
    mh = getattr(context.scene, "mhfrt", None)
    if mh is None:
        return
    mode = mh.view_mode
    for obj, hide in ((mh.cage, mode == 'TARGET'), (mh.target, mode == 'CAGE')):
        if obj:
            try:
                obj.hide_set(hide)
            except RuntimeError:
                pass


def _undo_push(msg):
    try:
        bpy.ops.ed.undo_push(message=msg)
    except Exception:
        pass


def _redraw_viewports():
    """Force every 3D viewport to redraw (the GPU overlay only repaints on
    redraw, so data-only changes like Clear All must request one)."""
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _visible(obj):
    """Actual viewport visibility (covers view_mode hiding AND tool solo)."""
    if obj is None:
        return False
    try:
        return obj.visible_get()
    except (RuntimeError, AttributeError):
        return True


def _win_region(area):
    return next((r for r in area.regions if r.type == 'WINDOW'), None)


def _region_rv3d(area, region):
    """The RegionView3D that region ACTUALLY renders with.

    A freshly split area's `space.region_3d` can disagree with its window
    region's live view data - rays built from the stale one silently miss
    everything - so always prefer `region.data`."""
    rv3d = getattr(region, "data", None) if region is not None else None
    if rv3d is not None:
        return rv3d
    space = area.spaces.active if area is not None else None
    return space.region_3d if space is not None else None


# ------------------------------------------------- local view slot hygiene ---
# Blender hands out local views from a pool of only 16 slots shared by the
# whole .blend, and a slot is released solely when its viewport LEAVES local
# view.  Closing a window does NOT release it: the closed window's screen
# survives (Blender only deletes screens flagged temporary), so a viewport
# left isolated keeps its slot for the rest of the session.  The tool takes
# two slots per session, so an artist who opened it eight times used to hit
# "No more than 16 local views" and lose the split layout from then on.
#
# Two defences: leave local view before the tool window closes (below, in
# _close_window), and sweep up slots stranded by earlier builds or by an
# artist closing the floating window with its X (_release_stuck_local_views).

# marks the screens this tool creates, so the sweep never touches a layout
# the artist isolated themselves
_TOOL_SCREEN_KEY = "mhfrt_landmark_window"
# Blender names the screen of a Python-opened window "temp"/"temp.001"/...
# - that catches windows leaked by versions before the marker existed.
_TEMP_SCREEN_NAMES = re.compile(r"^temp(\.\d+)?$")


def _v3d_actives(screen):
    """(area, region, space) for every viewport in a screen that is currently
    showing a 3D view (only the active space of an area can be operated on)."""
    for area in screen.areas:
        space = area.spaces.active
        region = _win_region(area)
        if (space is not None and space.type == 'VIEW_3D'
                and region is not None):
            yield area, region, space


def _leave_local_view(window, screen, area, region):
    """Take one viewport out of local view, releasing its slot.  The screen is
    overridden alongside the window so this also works on the screen of a
    window that is already gone."""
    try:
        with bpy.context.temp_override(window=window, screen=screen,
                                       area=area, region=region):
            bpy.ops.view3d.localview(frame_selected=False)
    except (RuntimeError, ReferenceError, TypeError):
        return False
    return True


def _release_stuck_local_views(context):
    """Give back local view slots stranded by tool windows that are gone.

    Only screens this tool created and that NO open window is showing are
    touched, so an artist's own isolated viewport - including one in a
    workspace they have switched away from - is left alone."""
    wm = getattr(context, "window_manager", None)
    if wm is None or not wm.windows:
        return 0
    host = getattr(context, "window", None) or wm.windows[0]
    live = {w.screen.as_pointer() for w in wm.windows}
    freed = 0
    for screen in bpy.data.screens:
        if screen.as_pointer() in live:
            continue
        if not (screen.get(_TOOL_SCREEN_KEY)
                or _TEMP_SCREEN_NAMES.match(screen.name)):
            continue
        for area, region, space in _v3d_actives(screen):
            if (space.local_view is not None
                    and _leave_local_view(host, screen, area, region)):
                freed += 1
    return freed


# ------------------------------------------------------------ tool drawing ---

def _project_runs(region, rv3d, world_pts, closed=False):
    """Project a world polyline to region px, split where it leaves the view.

    Returns ``[(run, is_loop), ...]``.  ``is_loop`` is True only when the curve
    is closed AND survived the projection whole - a ring with a clipped-away
    arc is no longer a ring, and closing what is left would draw a chord
    straight across the missing part.
    """
    runs, cur = [], []
    for w in world_pts:
        p = location_3d_to_region_2d(region, rv3d, w)
        if p is None:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
        else:
            cur.append((p[0], p[1]))
    if len(cur) >= 2:
        runs.append(cur)
    whole = bool(closed) and len(runs) == 1 and len(runs[0]) == len(world_pts)
    return [(run, whole) for run in runs]


def _curve_is_closed(points):
    """True when this group of landmarks was drawn as one loop."""
    return any(bool(getattr(l, "curve_closed", False)) for l in points)


def _polyline_point_dir(run, frac):
    """(position, direction) at a fraction of a 2D polyline's arc length."""
    lens, total = [], 0.0
    for a, b in zip(run, run[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        lens.append(d)
        total += d
    if total <= 1e-6:
        return None
    t = total * frac
    acc = 0.0
    for (a, b), d in zip(zip(run, run[1:]), lens):
        if acc + d >= t:
            f = 0.0 if d <= 1e-9 else (t - acc) / d
            p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            return p, (b[0] - a[0], b[1] - a[1])
        acc += d
    a, b = run[-2], run[-1]
    return b, (b[0] - a[0], b[1] - a[1])


def _arrow_at_end(runs, size, color, width=1.8):
    if not runs or len(runs[-1][0]) < 2:
        return
    last = runs[-1][0]
    d = (last[-1][0] - last[-2][0], last[-1][1] - last[-2][1])
    gd.arrow_head(last[-1], d, size, color, width)


def _draw_pending_curve(op, mh, side, region, rv3d, pending):
    """The confirmed cage curve awaiting its match glows white - with its
    direction arrow - so the artist sees exactly what to redraw on the head."""
    if pending is None or mh.cage is None or side not in {'CAGE', 'BOTH'}:
        return
    _cid, idxs = pending
    points = [mh.landmarks[i] for i in idxs if 0 <= i < len(mh.landmarks)]
    world = [lmdata.src_world(mh.cage, l) for l in points]
    if not world:
        return
    runs = _project_runs(region, rv3d, world, _curve_is_closed(points))
    for run, loop in runs:
        gd.curve_stroke(gd.smooth_polyline(run, closed=loop), gd.CYAN, 1.8)
    _arrow_at_end(runs, 9.0, gd.CYAN)
    if len(world) == 1:
        p = location_3d_to_region_2d(region, rv3d, world[0])
        if p is not None:
            gd.marker_src(p, 9.0, gd.WHITE, 1.8)


def _draw_hover_curve(op, mh, side, region, rv3d, all_curves):
    """Hover / drag emphasis: the whole curve under the cursor brightens and
    the hovered point gets the big white marker (filled while dragging)."""
    hov = op.drag or op.hover
    if hov is None:
        return
    i, hkind = hov
    if not (0 <= i < len(mh.landmarks)):
        return
    kind_side = 'CAGE' if hkind == 'SRC' else 'TARGET'
    if side not in {kind_side, 'BOTH'}:
        return
    obj = mh.cage if hkind == 'SRC' else mh.target
    if obj is None:
        return
    lm = mh.landmarks[i]
    cid = int(lm.curve_id)
    if cid >= 0:
        # A closed curve is highlighted as a LOOP. Drawn open, this bright
        # stroke both left the seam unjoined and - with Catmull-Rom inventing
        # its end tangents - flew off the ring in a straight tail, next to the
        # correctly closed loop the persistent overlay was drawing underneath.
        world, points = [], []
        for _j, l in all_curves.get(cid, ()):
            if hkind == 'SRC' and l.has_src:
                world.append(lmdata.src_world(obj, l))
            elif hkind == 'TGT' and l.has_tgt:
                world.append(lmdata.tgt_world(obj, l))
            else:
                continue
            points.append(l)
        for run, loop in _project_runs(region, rv3d, world,
                                       _curve_is_closed(points)):
            gd.curve_stroke(gd.smooth_polyline(run, closed=loop), gd.CYAN, 1.7)
    w3 = (lmdata.src_world(obj, lm) if hkind == 'SRC'
          else lmdata.tgt_world(obj, lm))
    p = location_3d_to_region_2d(region, rv3d, w3)
    if p is not None:
        fill = op.drag is not None
        if hkind == 'SRC':
            gd.marker_src(p, 9.5, gd.WHITE, 2.0, fill=fill)
        else:
            gd.marker_tgt(p, 9.5, gd.WHITE, 2.0, fill=fill)
        # armed endpoint merge: cyan capture ring + centre tick on the
        # dragged point - release collapses it onto its mirrored sister
        if (getattr(op, "_merge_ready", None) is not None
                and op.drag is not None and i == op.drag[0]):
            gd.ring(p, 14.0, gd.CYAN, 1.6)
            gd.tick((p[0], p[1] + 18.0), 4.5, gd.CYAN)
        if lm.label:
            gd.text(p[0] + 13, p[1] - 4, lm.label, 11, gd.WHITE)


def _draw_stroke(op, mh, region, rv3d):
    """The live stroke and its Ctrl+R-style landmark preview, drawn only in
    the viewport the curve is being drawn in."""
    st = op.stroke
    # During the drag, show the hand stroke. After release, replace it with
    # the finalized landmark curve so a closed preview is drawn only once.
    preview = op.state == 'PREVIEW' and bool(st["points"])
    world = ([p["world"] for p in st["points"]] if preview
             else [s["world"] for s in st["samples"]])
    closed = preview and bool(st.get("closed")) and len(world) > 2
    runs = _project_runs(region, rv3d, world, closed)
    for run, loop in runs:
        path = gd.smooth_polyline(run, closed=loop) if preview else run
        gd.curve_stroke(path, gd.CYAN, 2.0)
    start = (location_3d_to_region_2d(region, rv3d, world[0])
             if world else None)
    if start is not None:
        gd.dot(start, 4.2, gd.OUTLINE)
        gd.dot(start, 2.6, gd.CYAN)
    _arrow_at_end(runs, 9.0, gd.CYAN)

    if op.state != 'PREVIEW':
        # The lazy-mouse leash: a faint tether from the trailing pen (where
        # the curve is actually being drawn) to the pointer, so the lag reads
        # as deliberate instead of as the tool losing track of the cursor.
        if st["lazy_radius"] > 0.0 or st["lazy_smooth"] > 0.0:
            pen, raw = st["lazy"], st["raw"]
            if (raw - pen).length > 2.0:
                gd.dashed_polyline([(pen.x, pen.y), (raw.x, raw.y)],
                                   gd.CYAN_FAINT, 1.1)
            gd.ring((pen.x, pen.y), 5.0, gd.CYAN_SOFT, 1.3)
            gd.dot((raw.x, raw.y), 2.0, gd.MID)
        return

    # direction chevrons at thirds of the longest visible run
    if runs:
        main = max((run for run, _loop in runs), key=len)
        for frac in (0.35, 0.7):
            pd = _polyline_point_dir(main, frac)
            if pd is not None:
                gd.arrow_head(pd[0], pd[1], 6.5, gd.CYAN_SOFT, 1.5)

    # the landmarks this curve will create
    merged = st["merged"]
    for k, p in enumerate(st["points"]):
        p2 = location_3d_to_region_2d(region, rv3d, p["world"])
        if p2 is None:
            continue
        if st["kind"] == 'SRC':
            gd.marker_src(p2, 6.5, gd.WHITE, 1.7)
        else:
            gd.marker_tgt(p2, 6.5, gd.WHITE, 1.7)
        if k < len(merged) and merged[k]:
            gd.tick((p2[0], p2[1] + 12.0), 4.0, gd.MID)

    # auto-mirrored twin, faint and dashed - created for free on confirm
    if st["mirror"]:
        m2 = [location_3d_to_region_2d(region, rv3d, p["world"])
              for p in st["mirror"]]
        whole = all(p is not None for p in m2)
        m2 = [(p[0], p[1]) for p in m2 if p is not None]
        mloop = bool(st.get("closed")) and whole and len(m2) > 2
        if len(m2) >= 2:
            gd.dashed_polyline(gd.smooth_polyline(m2, closed=mloop),
                               gd.CYAN_FAINT, 1.2)
        for p in m2:
            gd.dot(p, 3.2, gd.OUTLINE)
            gd.dot(p, 2.0, gd.MID)
        # Centre merge: join the stroke to its twin ONLY at an endpoint that
        # actually merged on the centre line, so the pair previews as one
        # continuous curve there. A corner endpoint stays unjoined.
        ends = [(0, 0)] if len(st["points"]) == 1 else [(0, 0), (-1, -1)]
        for pk, mk in ends:
            if not (merged and merged[pk]):
                continue
            a = location_3d_to_region_2d(region, rv3d,
                                         st["points"][pk]["world"])
            b = location_3d_to_region_2d(region, rv3d,
                                         st["mirror"][mk]["world"])
            if a is not None and b is not None:
                gd.curve_stroke([(a[0], a[1]), (b[0], b[1])],
                                gd.CYAN_FAINT, 1.2)

    # the Ctrl+R readout: big point count floating over the curve
    anchor = (location_3d_to_region_2d(region, rv3d, world[len(world) // 2])
              if world else None)
    if anchor is not None:
        ax, ay = anchor[0], anchor[1] + 30.0
    else:
        ax, ay = st["last_mouse"][0], st["last_mouse"][1] + 30.0
    n = st["count"]
    gd.text(ax, ay, str(n), 22, gd.WHITE, align='CENTER')
    label = "point" if n == 1 else "points"
    if st["locked"]:
        label += "  ·  matched"
    gd.text(ax, ay - 14.0, label, 10, gd.MID, align='CENTER')


def _draw_hud(op):
    ctx = bpy.context
    region = ctx.region
    area = ctx.area
    if region is None or region.type != 'WINDOW' or area is None:
        return
    side = op._area_sides.get(area.as_pointer())
    if side is None:
        return
    mh = getattr(ctx.scene, "mhfrt", None)
    if mh is None:
        return
    rv3d = ctx.region_data

    st = op.stroke
    # One grouping pass for the whole HUD: the pending curve, the hover
    # highlight and the curve counter all read it.
    all_curves = lmdata.curves(mh)
    pending = lmdata.pending_curve(mh, all_curves)

    # ---- 3D-anchored visuals (projected to px) ------------------------------
    if rv3d is not None:
        _draw_pending_curve(op, mh, side, region, rv3d, pending)
        _draw_hover_curve(op, mh, side, region, rv3d, all_curves)
        if st is not None and st["region_ptr"] == region.as_pointer():
            _draw_stroke(op, mh, region, rv3d)

    # ---- top-center text block ----------------------------------------------
    title = {"CAGE": "HEAD CAGE", "TARGET": "YOUR HEAD"}.get(
        side, "LANDMARK CURVES")

    stroke_here = st is not None and (st["side"] == side or side == 'BOTH')
    if op.state == 'DRAW':
        if not stroke_here:
            hint = ""
        elif st["locked"]:
            n = st["count"]
            hint = (f"Release to place {n} point{'s' if n != 1 else ''} "
                    "matching the cage curve")
        else:
            hint = "Release to preview the landmarks"
    elif op.state == 'PREVIEW':
        if stroke_here:
            n = st["count"]
            pts = f"{n} point{'s' if n != 1 else ''}"
            hint = (f"{pts} - move the mouse left / right, scroll, or type a "
                    "number   ·   click to confirm   ·   right-click to cancel")
        else:
            hint = ""
    elif side == 'CAGE':
        hint = ("This curve waits on the head  ->" if pending
                else "Hold LMB and draw a curve along a feature")
    elif side == 'TARGET':
        hint = ("Draw the matching curve - same start, same direction"
                if pending else "<-  Start on the cage")
    else:
        hint = ("Draw the matching curve on the head - same direction"
                if pending else "Hold LMB and draw a curve on the cage")

    # Every toggle reads as its own word in its own colour: green while it is
    # doing something, red while it is not. The state is then a glance rather
    # than a sentence to parse, which matters most while the artist is mid
    # stroke and looking at the mesh, not at the HUD.
    ncurves = len(all_curves)

    def toggle(label, enabled, on="ON", off="OFF"):
        return [(f"      {label} ", gd.MID),
                (on if enabled else off, gd.state_color(enabled))]

    info = [(f"{ncurves} curve{'s' if ncurves != 1 else ''}  ·  "
             f"{lmdata.complete_count(mh)} points", gd.MID)]
    info += toggle("symmetry", bool(mh.symmetry))
    info += toggle("lazy", bool(getattr(mh, "landmark_lazy", False)))
    if side == 'BOTH':
        info += toggle("showing", not op.solo, "both meshes", "one mesh")
    else:
        info += toggle("sync", bool(mh.landmark_sync_view))

    keys = "LMB drag = draw curve   ·   drag ends together = loop   ·   " \
           "X delete curve   ·   Ctrl+Z undo   ·   S symmetry   ·   L lazy"
    if side == 'BOTH':
        keys += "   ·   H " + ("show both" if op.solo else "solo")
    else:
        keys += "   ·   C sync"
    keys += "   ·   Esc finish"

    cx = region.width * 0.5
    y = region.height - 54
    gd.text(cx, y, title, 17, gd.WHITE, align='CENTER')
    w = max(gd.text_width(title, 17), 240.0)
    gd.stroke([(cx - w * 0.5, y - 9), (cx + w * 0.5, y - 9)], gd.DIM, 1.0)
    if hint:
        gd.text(cx, y - 26, hint, 12, gd.MID, align='CENTER')
        info_y = y - 44
    else:
        info_y = y - 26
    gd.text_runs(cx, info_y, info, 12, align='CENTER')
    gd.text(cx, info_y - 18, keys, 11, gd.DIM, align='CENTER')
    if op.msg:
        gd.text(cx, info_y - 38, op.msg, 12, gd.WHITE, align='CENTER')
    gd.finish()


def _draw_cursor(op):
    """Surface-following ring at the raycast hit (POST_VIEW), only in the
    viewport currently under the mouse."""
    ctx = bpy.context
    region = ctx.region
    if region is None or region.as_pointer() != op._hot_region_ptr:
        return
    hit = op.cursor_hit
    rv3d = ctx.region_data
    if hit is None or rv3d is None:
        return
    r = rv3d.view_distance * 0.016
    gd.ring_3d(hit["world"], hit["normal"], r, gd.INK, 1.6)
    gd.finish()


# --------------------------------------------------------------- the tool ---

class MHFRT_OT_edit_landmarks(bpy.types.Operator):
    bl_idname = "mhfrt.edit_landmarks"
    bl_label = "Draw Landmark Curves"
    bl_description = ("ZWrap-style split view: the cage alone on the left, your "
                      "head alone on the right, each with its own camera. Hold "
                      "LMB and draw a curve along a feature, then move the "
                      "mouse (or scroll, or type a number) to choose how many "
                      "landmarks ride it (like Ctrl+R), click to "
                      "confirm - then draw the matching curve on your head. "
                      "Drag any point to reshape a curve, X deletes a curve, "
                      "Ctrl+Z undoes - Esc restores the original viewport")
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        global _RUNNING
        from .op_live import stop_running
        stop_running()          # this tool rides the Wrapped key value
        nav.refresh()           # honour the artist's own navigation bindings
        mh = context.scene.mhfrt
        if _RUNNING:
            self.report({'INFO'}, "Landmark mode is already running - press "
                                  "Esc in the split view to finish")
            return {'CANCELLED'}
        if not mh.cage or not mh.target:
            self.report({'ERROR'}, "Set both Head Cage and Head Target first")
            return {'CANCELLED'}
        if mh.cage == mh.target:
            self.report({'ERROR'}, "Head Cage and Head Target must be different objects")
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run this from the 3D Viewport")
            return {'CANCELLED'}

        migrated = lmdata.migrate_legacy(context)
        if migrated:
            self.report({'INFO'}, f"Migrated {migrated} old marker pair(s)")
        # the symmetry ruler is cached per mesh for the session; a fresh tool
        # run re-measures once so a mesh edited in between cannot go stale
        _clear_halfwidth_cache()

        self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self.hover = None          # (pair_index, 'SRC'|'TGT')
        self.drag = None
        self.press = None          # mouse position at LMB press (drag threshold)
        self.cursor_hit = None
        self.solo = True           # single-view fallback: show only one mesh
        self.msg = ""
        self.state = 'IDLE'        # IDLE | DRAW | PREVIEW (Ctrl+R moment)
        self.stroke = None         # live surface curve, never data until confirm
        self.last_count = CURVE_DEFAULT   # remembered point count per session
        self.typed = ""            # digits typed into the preview's count
        # ('CENTER', i_src, i_twin) or ('LOOP', i_dragged, i_stationary).
        self._merge_ready = None
        self.undo_stack = []
        self._hot_region_ptr = 0   # WINDOW region currently under the mouse
        self._cursor_hot = False   # crosshair only over the tool viewports
        self._area_sides = {}      # area_ptr -> 'CAGE'|'TARGET'|'BOTH'
        self._created_ptr = 0
        self._orig_ptr = 0
        self._view_prev = None
        self._sel_prev = []
        self._act_prev = None

        # edit correspondences on the ORIGINAL cage shape
        self._wrap_prev = None
        cage = mh.cage
        if cage.data.shape_keys:
            wk = cage.data.shape_keys.key_blocks.get(WRAPPED_KEY)
            if wk is not None:
                self._wrap_prev = wk.value
                wk.value = 0.0
                cage.data.update()
        # ... and with any Weight Cleanup pose silenced on BOTH meshes:
        # a point placed on an open-mouthed surface would otherwise store
        # the posed position as the neutral one
        self._pose_prev = []
        for obj in (mh.cage, mh.target):
            if obj is None or obj.data.shape_keys is None:
                continue
            kb = obj.data.shape_keys.key_blocks
            for name in ("MouthOpen", "CloseEyes"):
                k = kb.get(name)
                if k is not None and abs(k.value) > 1e-9:
                    self._pose_prev.append((obj, name, k.value))
                    k.value = 0.0
                    obj.data.update()

        # ... and with every armature that deforms either mesh held at REST.
        # A landmark is picked by raycasting the surface Blender is DRAWING and
        # stored as a mesh-local coordinate; on a head still deformed by a posed
        # rig those two disagree, so the point lands somewhere else the moment
        # the pose changes - and the wrap that follows aims at the wrong place
        # with nothing on screen to explain it.  A modal, so this is a snapshot
        # here and a restore in _finish rather than a `with` block; _finish also
        # runs from cancel(), which is what covers the artist closing the tool
        # window instead of pressing Esc.
        self._rest_prev = []
        for armature in organization.deforming_armatures(mh.cage, mh.target):
            data = armature.data
            if data is None or data.pose_position == 'REST':
                continue
            self._rest_prev.append((data, data.pose_position))
            data.pose_position = 'REST'
        if self._rest_prev:
            context.view_layer.update()

        # both layouts below isolate meshes in local view, and Blender only
        # has 16 of those per file - reclaim any that an earlier session left
        # behind before asking for two more
        _release_stuck_local_views(context)

        # ZWrap layout: cage | target, each isolated in its own local view.
        # Primary path opens a THROWAWAY floating window with its own screen,
        # so the artist's working layout is never split or moved; exiting just
        # closes that window.  Falls back to the in-place split, then a single
        # auto-solo view, when a new window can't be opened.
        self._tool_window = None
        self.dual = False
        if self._enter_window(context, mh):
            self._area_sides = dict(_DUAL_SIDES)
        elif self._enter_dual(context, mh):
            self.dual = True
            self._area_sides = dict(_DUAL_SIDES)
        else:
            self._area_sides = {context.area.as_pointer(): 'BOTH'}
        self._sync_visibility(context, mh)

        self._hud = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (self,), 'WINDOW', 'POST_PIXEL')
        self._ring = bpy.types.SpaceView3D.draw_handler_add(
            _draw_cursor, (self,), 'WINDOW', 'POST_VIEW')
        # NO custom cursor - keep the OS default arrow.  Every attempt to
        # push a modal cursor has ended up leaving a stuck crosshair when
        # the modal exits through some path Blender doesn't clean up,
        # and the 3D ring drawn by _draw_cursor already gives the "you're
        # in the tool" visual feedback we need.
        self._cursor_hot = False
        # Blender dispatches modal events per WINDOW: the handler must live on
        # the window whose viewports the artist actually clicks in.  When the
        # tool runs in its own floating window, add the handler THERE (via a
        # context override) so its events reach us; otherwise the modal would
        # never see a click in the new window.
        if self._tool_window is not None:
            with context.temp_override(window=self._tool_window):
                context.window_manager.modal_handler_add(self)
        else:
            context.window_manager.modal_handler_add(self)
        _RUNNING = True
        self._tag_redraw_sides(context)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        """Blender's external-termination hook: fires if the artist closes the
        floating tool window (or a file load / add-on reload ends the modal)
        instead of pressing Esc.  Runs the same cleanup so draw handlers, the
        Wrapped shape key, and the running flag are never left dangling."""
        self._finish(context)

    # ---------------------------------------------------- floating window --

    def _enter_window(self, context, mh):
        """Open a temporary floating window whose 3D view is split into
        cage (left) | target (right), each soloed in its own local view.

        The new window carries its own screen datablock, so the split never
        touches the artist's working layout - exiting simply closes the whole
        window (no split to unwind).  Returns True when the window layout is
        live, False to fall back to the in-place split."""
        window = context.window
        wm = context.window_manager
        view_layer = context.view_layer
        # selection is view-layer wide, so soloing meshes here also changes it
        # in the working window; snapshot it now and restore it on close
        self._sel_prev = [o for o in view_layer.objects if o.select_get()]
        self._act_prev = view_layer.objects.active

        before = set(wm.windows)
        try:
            with context.temp_override(window=window):
                bpy.ops.wm.window_new()
        except RuntimeError:
            return False
        fresh = [w for w in wm.windows if w not in before]
        if not fresh:
            return False
        tool_win = fresh[0]
        self._tool_window = tool_win
        screen = tool_win.screen
        # tag the throwaway screen: closing the window does not delete it, so
        # this is what tells a later sweep which stranded local views are ours
        screen[_TOOL_SCREEN_KEY] = True

        # the biggest 3D viewport in the fresh window becomes the cage side
        area = max((a for a in screen.areas if a.type == 'VIEW_3D'),
                   key=lambda a: a.width * a.height, default=None)
        region = _win_region(area) if area is not None else None
        if area is None or region is None:
            self._close_window(context)
            return False

        before_areas = {a.as_pointer() for a in screen.areas}
        try:
            with context.temp_override(window=tool_win, area=area,
                                       region=region):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
        except RuntimeError:
            self._close_window(context)
            return False
        new_area = next((a for a in screen.areas
                         if a.as_pointer() not in before_areas), None)
        if new_area is None:
            self._close_window(context)
            return False

        left, right = sorted((area, new_area), key=lambda a: a.x)
        try:
            for side_area, obj in ((left, mh.cage), (right, mh.target)):
                side_region = _win_region(side_area)
                side_space = side_area.spaces.active
                if side_region is None or side_space is None:
                    raise RuntimeError("viewport region missing")
                try:
                    obj.hide_set(False)
                except RuntimeError:
                    pass
                for other in view_layer.objects:
                    if other.select_get():
                        other.select_set(False)
                obj.select_set(True)
                view_layer.objects.active = obj
                with context.temp_override(window=tool_win, area=side_area,
                                           region=side_region):
                    if side_space.local_view:
                        bpy.ops.view3d.localview(frame_selected=False)
                    bpy.ops.view3d.localview(frame_selected=True)
                _face_front(_region_rv3d(side_area, side_region))
        except (RuntimeError, ReferenceError):
            self._close_window(context)
            return False

        _DUAL_SIDES.clear()
        _DUAL_SIDES[left.as_pointer()] = 'CAGE'
        _DUAL_SIDES[right.as_pointer()] = 'TARGET'
        _start_cam_sync(tool_win, left, right)
        return True

    def _close_window(self, context):
        """Close the temporary tool window and restore the artist's selection.

        The close is deferred one tick: the finishing Esc may be dispatched
        while the tool window is the context window, and closing a window from
        inside its own event handling is fragile - a 0-delay timer runs after
        the modal has returned, matching how the cursor reset is deferred."""
        win = getattr(self, "_tool_window", None)
        self._tool_window = None
        _DUAL_SIDES.clear()
        _stop_cam_sync()
        if win is not None:
            # Leave local view FIRST: the closed window's screen outlives the
            # window, and a viewport still isolated when it goes keeps one of
            # Blender's 16 local view slots for good.
            win_ptr = 0
            try:
                win_ptr = win.as_pointer()
                screen = win.screen
                for area, region, space in _v3d_actives(screen):
                    if space.local_view is not None:
                        _leave_local_view(win, screen, area, region)
            except (RuntimeError, ReferenceError, AttributeError):
                pass

            def _do_close(ptr=win_ptr):
                try:
                    wm = bpy.context.window_manager
                    target = next((w for w in wm.windows
                                   if w.as_pointer() == ptr), None) if ptr else None
                    if target is not None:
                        with bpy.context.temp_override(window=target):
                            bpy.ops.wm.window_close()
                except (RuntimeError, ReferenceError, AttributeError):
                    pass
                # second pass, now that the window is really gone: catches the
                # case where the artist closed it with its X instead of Esc,
                # which strands the local views before the code above can run
                try:
                    _release_stuck_local_views(bpy.context)
                except (RuntimeError, ReferenceError, AttributeError):
                    pass
                return None

            bpy.app.timers.register(_do_close, first_interval=0.0)

        # bring back the pre-tool selection in the working window
        try:
            view_layer = context.view_layer
            for obj in list(view_layer.objects):
                if obj.select_get():
                    obj.select_set(False)
            for obj in self._sel_prev:
                if obj and obj.name in view_layer.objects:
                    obj.select_set(True)
            if (self._act_prev is not None
                    and self._act_prev.name in view_layer.objects):
                view_layer.objects.active = self._act_prev
        except (AttributeError, ReferenceError, RuntimeError):
            pass

    # ------------------------------------------------------- split layout --

    def _enter_dual(self, context, mh):
        """Split the invoking viewport into cage | target local views.
        Returns True when the ZWrap layout is live; False falls back to the
        classic single view."""
        window, area = context.window, context.area
        screen = window.screen
        if screen.show_fullscreen or area.width < 300:
            return False
        region = _win_region(area)
        if region is None:
            return False

        space = area.spaces.active
        rv3d = _region_rv3d(area, region)
        self._view_prev = (rv3d.view_location.copy(),
                           rv3d.view_rotation.copy(),
                           rv3d.view_distance, rv3d.view_perspective)
        view_layer = context.view_layer
        self._sel_prev = [o for o in view_layer.objects if o.select_get()]
        self._act_prev = view_layer.objects.active

        try:
            # never split while the invoking viewport is in a local view
            if space.local_view:
                with context.temp_override(window=window, area=area,
                                           region=region):
                    bpy.ops.view3d.localview(frame_selected=False)
            before = {a.as_pointer() for a in screen.areas}
            with context.temp_override(window=window, area=area,
                                       region=region):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
        except RuntimeError:
            return False
        new_area = next((a for a in screen.areas
                         if a.as_pointer() not in before), None)
        if new_area is None:
            return False
        self._created_ptr = new_area.as_pointer()
        self._orig_ptr = area.as_pointer()

        left, right = sorted((area, new_area), key=lambda a: a.x)
        try:
            for other in self._sel_prev:
                other.select_set(False)
            prev = None
            for side_area, obj in ((left, mh.cage), (right, mh.target)):
                side_region = _win_region(side_area)
                side_space = side_area.spaces.active
                if side_region is None or side_space is None:
                    raise RuntimeError("viewport region missing")
                if prev is not None:
                    prev.select_set(False)
                try:
                    obj.hide_set(False)
                except RuntimeError:
                    pass
                obj.select_set(True)
                view_layer.objects.active = obj
                with context.temp_override(window=window, area=side_area,
                                           region=side_region):
                    if side_space.local_view:
                        bpy.ops.view3d.localview(frame_selected=False)
                    bpy.ops.view3d.localview(frame_selected=True)
                _face_front(_region_rv3d(side_area, side_region))
                prev = obj
        except (RuntimeError, ReferenceError):
            self._created_ptr = new_area.as_pointer()
            self._exit_dual(context)        # unwind the half-built layout
            return False

        _DUAL_SIDES.clear()
        _DUAL_SIDES[left.as_pointer()] = 'CAGE'
        _DUAL_SIDES[right.as_pointer()] = 'TARGET'
        return True

    def _exit_dual(self, context):
        """Restore the single viewport: leave both local views, close the
        created area, bring back the original camera and selection.

        Must survive degraded contexts: the finishing event can arrive with
        the mouse over the status bar (no selected_objects in context) or
        during window teardown (no window at all)."""
        try:
            window = getattr(context, "window", None)
            screen = window.screen if window else None
            if screen is not None:
                areas = {a.as_pointer(): a for a in screen.areas}

                side_ptrs = set(_DUAL_SIDES) | {self._created_ptr,
                                                self._orig_ptr}
                side_ptrs.discard(0)
                for ptr in side_ptrs:
                    side_area = areas.get(ptr)
                    if side_area is None or side_area.type != 'VIEW_3D':
                        continue
                    space = side_area.spaces.active
                    region = _win_region(side_area)
                    if (space is not None and space.local_view
                            and region is not None):
                        try:
                            with context.temp_override(window=window,
                                                       area=side_area,
                                                       region=region):
                                bpy.ops.view3d.localview(frame_selected=False)
                        except RuntimeError:
                            pass

                created = (areas.get(self._created_ptr)
                           if self._created_ptr else None)
                if created is not None:
                    try:
                        with context.temp_override(window=window, area=created,
                                                   region=_win_region(created)):
                            bpy.ops.screen.area_close()
                    except RuntimeError:
                        self.report({'WARNING'},
                                    "Could not close the split viewport - "
                                    "join it manually (drag its corner)")

                orig = areas.get(self._orig_ptr)
                if (orig is not None and orig.type == 'VIEW_3D'
                        and self._view_prev is not None):
                    rv3d = _region_rv3d(orig, _win_region(orig))
                    if rv3d is not None:
                        (rv3d.view_location, rv3d.view_rotation,
                         rv3d.view_distance,
                         rv3d.view_perspective) = self._view_prev

            try:
                view_layer = context.view_layer
                for obj in list(view_layer.objects):
                    if obj.select_get():
                        obj.select_set(False)
                for obj in self._sel_prev:
                    if obj and obj.name in view_layer.objects:
                        obj.select_set(True)
                if (self._act_prev is not None
                        and self._act_prev.name in view_layer.objects):
                    view_layer.objects.active = self._act_prev
            except (AttributeError, ReferenceError, RuntimeError):
                pass
        finally:
            # never leave stale pointers behind: a later exit pass must not
            # close whatever area a recycled pointer might match
            self._created_ptr = 0
            _DUAL_SIDES.clear()

    # ------------------------------------------------------------- helpers --

    def _expected(self, mh):
        """The side the next curve must be drawn on: ('TGT', target) while a
        cage curve waits for its match, else ('SRC', cage)."""
        if lmdata.pending_curve(mh) is not None:
            return 'TGT', mh.target
        return 'SRC', mh.cage

    def _tag_redraw_sides(self, context):
        # the tool viewports may live in a separate floating window, so scan
        # every window's screen rather than just the one under the mouse
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.as_pointer() in self._area_sides:
                    area.tag_redraw()

    def _hot_area(self, context, event):
        """The tool viewport the pointer is over (area-level test only)."""
        mx, my = event.mouse_x, event.mouse_y
        screen = getattr(context.window, "screen", None)
        if screen is None:
            return None
        for area in screen.areas:
            if (area.as_pointer() in self._area_sides
                    and area.x <= mx < area.x + area.width
                    and area.y <= my < area.y + area.height):
                return area
        return None

    def _tag_redraw_event(self, context, event):
        """Repaint what this event can actually have changed.

        A tag_redraw is a full re-render of everything that viewport shows,
        and the tool owns two of them. Doing both on every event - mouse moves
        included - means a high-poly head is rasterised twice per pointer
        sample, which is most of why drawing on a dense mesh crawled. A move
        only ever changes the viewport it happens in; everything else (a
        confirm, a delete, an undo, a symmetry flip) can change both.
        """
        if event.type == 'MOUSEMOVE' or event.type.startswith('TRACKPAD'):
            area = self._hot_area(context, event)
            if area is not None:
                area.tag_redraw()
                return
        self._tag_redraw_sides(context)

    def _hot_lookup(self, context, event):
        """(side, area, region, rv3d, region_mouse) for the tool viewport
        under the mouse, or None."""
        mx, my = event.mouse_x, event.mouse_y
        screen = context.window.screen
        for area in screen.areas:
            side = self._area_sides.get(area.as_pointer())
            if side is None:
                continue
            if not (area.x <= mx < area.x + area.width
                    and area.y <= my < area.y + area.height):
                continue
            for region in area.regions:
                if (region.type == 'WINDOW'
                        and region.x <= mx < region.x + region.width
                        and region.y <= my < region.y + region.height):
                    rv3d = _region_rv3d(area, region)
                    if rv3d is None:
                        return None
                    mouse = Vector((mx - region.x, my - region.y))
                    return side, area, region, rv3d, mouse
        return None

    def _sync_visibility(self, context, mh):
        """Split view: both meshes stay visible - each viewport's local view
        does the isolating. Single view: auto-solo shows only the mesh the
        next curve lands on (H toggles both-visible for orientation)."""
        cage, target = mh.cage, mh.target
        if not cage or not target:
            return
        if self.dual or getattr(self, "_tool_window", None) is not None:
            # split layout (in-window or floating window): each viewport's
            # own local view isolates a mesh, so both stay globally visible
            hide_cage = hide_tgt = False
        elif self.solo:
            kind, _obj = self._expected(mh)
            hide_cage, hide_tgt = kind != 'SRC', kind != 'TGT'
        else:
            hide_cage = hide_tgt = False
        for obj, hide in ((cage, hide_cage), (target, hide_tgt)):
            try:
                obj.hide_set(hide)
            except RuntimeError:
                pass

    def _pick(self, mh, side, region, rv3d):
        """Marker under the cursor (screen-space), respecting the viewport's
        side: cage markers left, target markers right, both in single view."""
        if region is None or rv3d is None:
            return None
        # Matches what the overlay draws while the tool runs: every curve the
        # viewport is responsible for stays on screen, so every one of those
        # points also stays grabbable.  Keying this off mesh visibility made
        # points the artist could plainly see refuse to be picked.
        want_src = side in {'CAGE', 'BOTH'} and mh.cage is not None
        want_tgt = side in {'TARGET', 'BOTH'} and mh.target is not None
        if not (want_src or want_tgt):
            return None
        cage, target = mh.cage, mh.target
        landmarks = mh.landmarks
        mouse = self.mouse
        best, best_d = None, PICK_PX * PICK_PX
        project = location_3d_to_region_2d
        # This runs on EVERY mouse move, so the two matrices and the cage's
        # shape-key offsets are read once instead of once per landmark.
        offsets = lmdata.wrap_offsets(cage, landmarks) if want_src else {}
        cage_mw = cage.matrix_world.copy() if want_src else None
        target_mw = target.matrix_world.copy() if want_tgt else None
        zero = lmdata.ZERO
        for i, lm in enumerate(landmarks):
            if want_src and lm.has_src:
                w = cage_mw @ (Vector(lm.src_co)
                               + offsets.get(int(lm.src_vidx), zero))
                p = project(region, rv3d, w)
                if p is not None:
                    d = (p - mouse).length_squared
                    if d < best_d:
                        best_d, best = d, (i, 'SRC')
            if want_tgt and lm.has_tgt:
                p = project(region, rv3d, target_mw @ Vector(lm.tgt_co))
                if p is not None:
                    d = (p - mouse).length_squared
                    if d < best_d:
                        best_d, best = d, (i, 'TGT')
        return best

    def _push_undo(self, mh):
        self.undo_stack.append(lmdata.snapshot(mh))
        if len(self.undo_stack) > 64:
            self.undo_stack.pop(0)

    # -------------------------------------------------- surface curves --

    def _probe_obj(self, mh, side):
        """The mesh the surface cursor should follow - only where the next
        curve can actually start."""
        pending = lmdata.pending_curve(mh)
        if side == 'CAGE':
            return mh.cage if pending is None else None
        if side == 'TARGET':
            return mh.target if pending is not None else None
        _kind, obj = self._expected(mh)
        return obj

    def _start_stroke(self, context, mh, side, region, rv3d):
        """LMB pressed on empty surface: begin drawing a curve."""
        pending = lmdata.pending_curve(mh)
        if side == 'CAGE':
            kind = 'SRC'
        elif side == 'TARGET':
            kind = 'TGT'
        else:
            kind = 'TGT' if pending else 'SRC'
        if kind == 'SRC' and pending is not None:
            self.msg = ("Draw the matching curve on the head first  ->"
                        if side == 'CAGE'
                        else "Draw the matching curve on the head first")
            return
        if kind == 'TGT' and pending is None:
            self.msg = ("<-  Start on the cage" if side == 'TARGET'
                        else "Start on the cage")
            return
        obj = mh.cage if kind == 'SRC' else mh.target
        if obj is None:
            self.msg = "Set both Head Cage and Head Target in Setup"
            return
        hit = raycast(context, obj, self.mouse, region, rv3d, want_vidx=False)
        if hit is None:
            which = "cage" if kind == 'SRC' else "head"
            self.msg = f"Start the curve on the {which} surface"
            return
        radius, smooth = lazy_settings(mh)
        self.msg = ""
        self.state = 'DRAW'
        self.stroke = {
            "kind": kind,
            "side": side,
            "region_ptr": region.as_pointer(),
            "samples": [{"world": hit["world"], "local": hit["local"],
                         "view": hit["view"], "vidx": hit["vidx"]}],
            "points": [],       # resampled landmark preview (PREVIEW state)
            "mirror": [],       # auto-mirrored twin preview
            "merged": [],       # per point: centre-merged (stays single)
            "count": len(pending[1]) if kind == 'TGT' else self.last_count,
            "locked": kind == 'TGT',
            "screen_len": 0.0,
            "last_mouse": self.mouse.copy(),
            # lazy mouse: the pen the surface is sampled at trails `raw`
            "lazy": self.mouse.copy(),
            "raw": self.mouse.copy(),
            "lazy_radius": radius,
            "lazy_smooth": smooth,
            "side_sign": 0.0,   # which half of X the stroke lives on
            "clipped": False,   # stroke already cut at the centre line
            "dial_anchor": None,  # PREVIEW: where the move-the-mouse dial sits
            "dial_base": 0,
        }
        self.typed = ""
        self.hover = None
        self.drag = None
        self.press = None
        self.cursor_hit = hit

    def _grow_stroke(self, context, event, mh):
        st = self.stroke
        if st is None:
            self.state = 'IDLE'
            return
        hot = self._hot_lookup(context, event)
        if hot is None:
            return
        _side, _area, region, rv3d, mouse = hot
        if region.as_pointer() != st["region_ptr"]:
            return                  # the stroke lives in one viewport only
        self.mouse = mouse
        self._hot_region_ptr = region.as_pointer()
        # The pointer moves; the PEN follows it on the lazy-mouse leash, and
        # the surface is sampled at the pen - never at the raw pointer.
        st["raw"] = mouse.copy()
        pen = lazy_step(st["lazy"], mouse, st["lazy_radius"], st["lazy_smooth"])
        st["lazy"] = pen
        step = (pen - st["last_mouse"]).length
        if step < STROKE_SAMPLE_PX or len(st["samples"]) >= STROKE_MAX_SAMPLES:
            return
        obj = mh.cage if st["kind"] == 'SRC' else mh.target
        if obj is None:
            return
        hit = raycast(context, obj, pen, region, rv3d, want_vidx=False)
        if hit is None:
            return                  # off the mesh: pause, resume when back on
        if mh.symmetry and self._clip_at_center(context, mh, st, hit):
            return                  # the stroke ends on the centre line
        st["screen_len"] += step
        st["last_mouse"] = pen.copy()
        st["samples"].append({"world": hit["world"], "local": hit["local"],
                              "view": hit["view"], "vidx": hit["vidx"]})
        self.cursor_hit = hit

    def _stroke_region(self, context):
        """(region, rv3d) the live stroke is being drawn in, or (None, None)."""
        st = self.stroke
        if st is None:
            return None, None
        want = st["region_ptr"]
        for area in context.window.screen.areas:
            if area.as_pointer() not in self._area_sides:
                continue
            for region in area.regions:
                if region.type == 'WINDOW' and region.as_pointer() == want:
                    return region, _region_rv3d(area, region)
        return None, None

    def _flush_lazy(self, context, mh):
        """On release, walk the trailing pen the rest of the way to where the
        pointer actually is, sampling as it goes.

        A landmark curve's ENDPOINTS are the whole point - a stroke drawn eye
        corner to eye corner must put a point on each corner - so leaving the
        curve short by the leash radius, which is what a plain lazy mouse
        does, is not acceptable here. The lag smooths the stroke; it does not
        get to move where it ends.
        """
        st = self.stroke
        if st is None or (st["lazy_radius"] <= 0.0 and st["lazy_smooth"] <= 0.0):
            return
        region, rv3d = self._stroke_region(context)
        obj = mh.cage if st["kind"] == 'SRC' else mh.target
        if region is None or rv3d is None or obj is None:
            return
        pen, raw = Vector(st["lazy"]), Vector(st["raw"])
        for _ in range(LAZY_FLUSH_STEPS):
            gap = raw - pen
            if gap.length <= 1.0 or len(st["samples"]) >= STROKE_MAX_SAMPLES:
                break
            pen = pen + gap * LAZY_CATCHUP
            step = (pen - st["last_mouse"]).length
            if step < STROKE_SAMPLE_PX:
                continue
            hit = raycast(context, obj, pen, region, rv3d, want_vidx=False)
            if hit is None:
                break
            if mh.symmetry and self._clip_at_center(context, mh, st, hit):
                break
            st["screen_len"] += step
            st["last_mouse"] = pen.copy()
            st["samples"].append({"world": hit["world"], "local": hit["local"],
                                  "view": hit["view"], "vidx": hit["vidx"]})
        st["lazy"] = pen

    def _clip_at_center(self, context, mh, st, hit):
        """Symmetry: a stroke may not cross into the other half. When it
        does, it is cut at the centre line and ends with one sample placed
        EXACTLY there - the final landmark lands on the centre and merges.
        Returns True when the new hit was consumed by the clip."""
        if st["clipped"]:
            return True             # already ended on the centre line
        obj = mh.cage if st["kind"] == 'SRC' else mh.target
        half = local_halfwidth_x(obj)
        thr = max(mh.symmetry_center_threshold * half, 1e-9)
        x = hit["local"].x
        if st["side_sign"] == 0.0:
            x0 = st["samples"][0]["local"].x
            ref = x0 if abs(x0) > thr else x
            if abs(ref) <= thr:
                return False        # still hugging the centre: no side yet
            st["side_sign"] = 1.0 if ref > 0.0 else -1.0
        if x * st["side_sign"] >= -thr:
            return False            # still on its own half
        # crossing: intersect the last stroke segment with the centre plane
        prev = st["samples"][-1]["local"]
        span = prev.x - x
        f = prev.x / span if abs(span) > 1e-12 else 0.0
        mid = prev.lerp(hit["local"], min(max(f, 0.0), 1.0))
        mid.x = 0.0
        dg = context.evaluated_depsgraph_get()
        try:
            ok, loc, _nor, poly = obj.closest_point_on_mesh(mid, depsgraph=dg)
        except RuntimeError:
            ok = False
        if not ok:
            loc, poly = mid, -1
        st["samples"].append({"world": obj.matrix_world @ loc,
                              "local": loc.copy(),
                              "view": hit["view"],
                              "vidx": _nearest_vidx(obj, loc, poly)})
        st["clipped"] = True
        self.msg = "Clipped at the centre line"
        return True

    def _end_stroke(self, context, mh):
        """LMB released: pause in the Ctrl+R-style landmark preview."""
        # Never sample the raw release position. The endpoint is the last
        # position reached by the lazy pen; catching it up to the hardware
        # cursor here defeats stabilization and creates a final hook.
        st = self.stroke
        if (st is None or len(st["samples"]) < 2
                or st["screen_len"] < STROKE_MIN_PX):
            self._cancel_stroke("Hold LMB and drag to draw a curve"
                                if st is not None else "")
            return
        self.state = 'PREVIEW'
        self.hover = None
        self.cursor_hit = None
        self._dial_rebase()     # the dial starts where the stroke ended
        self._build_preview(context, mh)
        if self.stroke is None:
            return
        if not self.stroke["points"]:
            self._cancel_stroke()
            return
        # The head-side curve has nothing to decide - its point count is
        # locked to the cage curve - so it commits the moment the stroke
        # ends. No confirmation step to sit through; Ctrl+Z still undoes it.
        if self.stroke["locked"]:
            self._confirm_preview(context, mh)

    def _cancel_stroke(self, msg=""):
        """Discard the whole temporary curve - it never became data."""
        self.state = 'IDLE'
        self.stroke = None
        self.typed = ""
        self.msg = msg

    def _build_preview(self, context, mh):
        st = self.stroke
        if st is None:
            return
        obj = mh.cage if st["kind"] == 'SRC' else mh.target
        if obj is None or len(st["samples"]) < 2:
            self._cancel_stroke()
            return
        world = _smooth_pts([Vector(s["world"]) for s in st["samples"]])
        views = [Vector(s["view"]) for s in st["samples"]]
        pairs = _resample_polyline(world, views, st["count"])
        points = _project_stroke_points(context, obj, pairs)
        st["closed"] = self._is_loop(mh, obj, points)
        if st["closed"]:
            # Re-sample one extra and drop the duplicate end: the curve then
            # closes AND still carries exactly the number of points the artist
            # dialled (the target side's count is locked to the cage curve, so
            # it has to land on it exactly).
            pairs = _resample_polyline(world, views, st["count"] + 1)
            points = _project_stroke_points(context, obj, pairs)
            if len(points) > st["count"]:
                points.pop()
        st["points"] = points
        self._build_mirror_preview(context, mh)

    @staticmethod
    def _is_loop(mh, obj, points):
        """Did the stroke come back onto its own start?

        Eye rims, lips and nostrils are closed contours. Left alone, such a
        stroke ends with two landmarks a hair apart - two pins fighting over
        one spot in the wrap - so the ends can merge into one closed curve.

        Loop Merge Distance controls the capture radius as a fraction of the
        mesh half-width, so artists can choose exactly how close the endpoints
        must be before the curve closes.
        """
        if obj is None or len(points) < 4:
            return False            # too short to tell a loop from a stroke
        first = Vector(points[0]["world"])
        limit = max(float(mh.landmark_loop_merge_threshold)
                    * max(local_halfwidth_x(obj), 1e-9), 1e-9)
        return (Vector(points[-1]["world"]) - first).length <= limit

    def _build_mirror_preview(self, context, mh):
        """Preview of the auto-mirrored twin curve (and the centre-merge
        flags), recomputed whenever the point count changes."""
        st = self.stroke
        st["mirror"] = []
        st["merged"] = [False] * len(st["points"])
        if not st["points"]:
            return
        if st["kind"] == 'SRC':
            obj = mh.cage
            if not mh.symmetry or obj is None:
                return
            half = local_halfwidth_x(obj)
            thr = max(mh.symmetry_center_threshold * half, 1e-9)
            st["merged"] = [abs(p["local"].x) <= thr for p in st["points"]]
            mirror = []
            for p, merged in zip(st["points"], st["merged"]):
                if merged:
                    continue
                mloc, mvidx, _ok = mirror_on_surface(
                    context, obj, p["local"], p["view"])
                mirror.append({"local": mloc,
                               "world": obj.matrix_world @ mloc,
                               "vidx": mvidx})
            st["mirror"] = mirror
        else:
            # head side: the twin exists only if the cage curve was mirrored
            obj = mh.target
            if obj is None or lmdata.pending_mirror_curve(mh) is None:
                return
            mirror = []
            for k in _mirror_keep_indices(mh):
                if k >= len(st["points"]):
                    break
                p = st["points"][k]
                mloc, mvidx, _ok = mirror_on_surface(
                    context, obj, p["local"], p["view"])
                mirror.append({"local": mloc,
                               "world": obj.matrix_world @ mloc,
                               "vidx": mvidx})
            st["mirror"] = mirror

    def _dial_count(self, context, mh, delta):
        """Nudge the landmark count by one step (wheel, arrows, + / -)."""
        st = self.stroke
        if st is None:
            return
        self._set_count(context, mh, st["count"] + delta)
        self._dial_rebase()     # keep moving the mouse from where it is now

    def _set_count(self, context, mh, n):
        """Change how many landmarks ride the curve."""
        st = self.stroke
        if st is None:
            return
        if st["locked"]:
            self.msg = f"{st['count']} points - matched to the cage curve"
            return
        n = max(CURVE_MIN, min(CURVE_MAX, int(n)))
        if n == st["count"]:
            return
        st["count"] = n
        self.last_count = n
        self.msg = ""
        self._build_preview(context, mh)

    def _dial_rebase(self):
        """Anchor the move-the-mouse dial at the cursor and the live count."""
        st = self.stroke
        if st is None:
            return
        st["dial_anchor"] = self.mouse.copy()
        st["dial_base"] = st["count"]
        self.typed = ""

    def _dial_mouse(self, context, mh, event):
        """The dial everybody has: slide the mouse right for more landmarks,
        left for fewer.  No wheel, no keymap, no modifier involved."""
        st = self.stroke
        if st is None or st["locked"]:
            return
        hot = self._hot_lookup(context, event)
        if hot is None:
            return
        _side, _area, region, _rv3d, mouse = hot
        if region.as_pointer() != st["region_ptr"]:
            return              # the preview belongs to one viewport only
        self.mouse = mouse
        self._hot_region_ptr = region.as_pointer()
        anchor = st.get("dial_anchor")
        if anchor is None:
            self._dial_rebase()
            return
        dx = mouse.x - anchor.x
        if abs(dx) <= DIAL_DEAD_PX:
            steps = 0
        else:
            steps = int((abs(dx) - DIAL_DEAD_PX) // DIAL_STEP_PX) + 1
            steps = steps if dx > 0.0 else -steps
        self._set_count(context, mh, st["dial_base"] + steps)

    def _dial_typed(self, context, mh, digit):
        """Type an exact count, like Blender's own numeric input."""
        st = self.stroke
        if st is None or st["locked"]:
            self._set_count(context, mh, 0)     # reports the locked message
            return
        buf = (self.typed + str(digit))[-2:]
        if int(buf) > CURVE_MAX:
            buf = str(digit)
        n = int(buf)
        self.typed = buf
        if n < CURVE_MIN:
            # a leading "1" of "12": wait for the second digit
            self.msg = f"{buf}...  ({CURVE_MIN}-{CURVE_MAX} points)"
            return
        self._set_count(context, mh, n)
        st["dial_anchor"] = self.mouse.copy()
        st["dial_base"] = st["count"]
        self.typed = buf

    def _confirm_preview(self, context, mh):
        """LMB in the preview: the temporary curve becomes landmark data."""
        st = self.stroke
        if st is None or not st["points"]:
            self._cancel_stroke()
            return
        obj = mh.cage if st["kind"] == 'SRC' else mh.target
        if obj is None:
            self._cancel_stroke("Mesh is gone - check Setup")
            return
        self.msg = ""
        self._push_undo(mh)

        if st["kind"] == 'SRC':
            cid = lmdata.next_curve_id(mh)
            first = len(mh.landmarks)
            merged = st["merged"]
            closed = bool(st.get("closed"))
            for k, p in enumerate(st["points"]):
                lm, _i = lmdata.add_pair(mh)
                lm.curve_id = cid
                lm.curve_closed = closed
                lm.src_co = p["local"]
                lm.src_vidx = p["vidx"]
                lm.has_src = True
                # a centre-line point gets no twin; record that, so only it
                # may later be joined to the mirrored curve
                lm.center_merged = bool(k < len(merged) and merged[k])
            if st["mirror"]:
                mcid = lmdata.next_curve_id(mh)
                for p in st["mirror"]:
                    lm, _i = lmdata.add_pair(mh)
                    lm.curve_id = mcid
                    lm.curve_closed = closed   # a mirrored loop is a loop too
                    lm.mirror_of = cid
                    lm.src_co = p["local"]
                    lm.src_vidx = p["vidx"]
                    lm.has_src = True
                    lm.mirror_pending = True
            mh.landmark_active = first
            _undo_push("Draw landmark curve")
        else:
            pending = lmdata.pending_curve(mh)
            if pending is None:
                self._cancel_stroke("The cage curve is gone - draw a new one")
                return
            _cid, idxs = pending
            if len(st["points"]) != len(idxs):
                # data changed under the preview (undo, tolerance edit):
                # refit the stroke to the count that is now required
                st["count"] = len(idxs)
                self._build_preview(context, mh)
                if self.stroke is None or not st["points"]:
                    return
            # the keep-list must be read BEFORE the fill consumes the pending
            keep = _mirror_keep_indices(mh)
            mirror = lmdata.pending_mirror_curve(mh)
            pts = st["points"]
            for i, p in zip(idxs, pts):
                lm = mh.landmarks[i]
                lm.tgt_co = p["local"]
                lm.has_tgt = True
            if mirror is not None and mh.target is not None:
                _mcid, midxs = mirror
                if len(keep) != len(midxs):
                    keep = list(range(min(len(midxs), len(pts))))
                fallback = False
                for mi, k in zip(midxs, keep):
                    p = pts[k]
                    mloc, _mv, ok = mirror_on_surface(
                        context, mh.target, p["local"], p["view"])
                    lm = mh.landmarks[mi]
                    lm.tgt_co = mloc
                    lm.has_tgt = True
                    lm.mirror_pending = False
                    fallback = fallback or not ok
                if fallback:
                    self.msg = ("Mirror used nearest-surface fallback - "
                                "check the mirrored curve")
            mh.landmark_active = idxs[0]
            _undo_push("Match landmark curve")

        self.state = 'IDLE'
        self.stroke = None
        lmdata.save_active(mh)
        self._sync_visibility(context, mh)

    def _delete_hovered_curve(self, context, mh):
        """X: remove the whole curve under the cursor (points are never
        removed one by one - a curve's count is fixed at confirmation)."""
        if self.hover is None:
            self.msg = "Hover a curve to delete it"
            return
        i = self.hover[0]
        if not (0 <= i < len(mh.landmarks)):
            self.hover = None
            return
        self._push_undo(mh)
        cid = int(mh.landmarks[i].curve_id)
        if cid >= 0:
            mirror = lmdata.pending_mirror_curve(mh)
            lmdata.remove_curve(mh, cid, save=False)
            # the twin born from the same stroke dies with its partner while
            # it is still pending (it could never be matched on its own)
            if mirror is not None and mirror[0] == cid + 1:
                lmdata.remove_curve(mh, mirror[0], save=False)
            lmdata.save_active(mh)
            self.msg = "Curve removed"
        else:
            lmdata.remove_at(mh, i)     # legacy single point
            self.msg = "Point removed"
        self.hover = None
        self.drag = None
        self.press = None
        self._merge_ready = None
        _undo_push("Delete landmark curve")
        self._sync_visibility(context, mh)

    def _apply_drag(self, context, mh, side, region, rv3d):
        i, kind = self.drag
        if not (0 <= i < len(mh.landmarks)):
            self.drag = None
            return
        # a cage marker only slides in the cage viewport (and vice versa)
        kind_side = 'CAGE' if kind == 'SRC' else 'TARGET'
        if side not in {kind_side, 'BOTH'}:
            return
        obj = mh.cage if kind == 'SRC' else mh.target
        if obj is None:
            return
        hit = raycast(context, obj, self.mouse, region, rv3d)
        if hit is None:
            return  # cursor off the mesh: marker keeps its last valid spot
        lm = mh.landmarks[i]
        if kind == 'SRC':
            lm.src_co = hit["local"]
            lm.src_vidx = hit["vidx"]
        else:
            lm.tgt_co = hit["local"]
        self.cursor_hit = hit
        # symmetry: the mirrored sister follows every move
        if mh.symmetry:
            self._sync_drag_sister(context, mh, i, kind, hit)
        loop = self._loop_close_candidate(mh, i, kind)
        if loop is not None:
            self._merge_ready = ('LOOP', *loop)
            self.msg = "Release to close the curve into a loop"
        elif mh.symmetry:
            centre = self._endpoint_merge_candidate(mh, i, kind)
            self._merge_ready = (('CENTER', *centre)
                                 if centre is not None else None)
            if self._merge_ready is not None:
                self.msg = "Release to merge at the centre"
        else:
            self._merge_ready = None
        if (self._merge_ready is None and self.msg in {
                "Release to merge at the centre",
                "Release to close the curve into a loop"}):
            self.msg = ""

    def _sync_drag_sister(self, context, mh, i, kind, hit):
        """Mirror the dragged point's new position onto its symmetric sister
        (same mesh, opposite side of X)."""
        j = lmdata.mirror_partner(mh, i)
        if j < 0:
            return
        obj = mh.cage if kind == 'SRC' else mh.target
        if obj is None:
            return
        mloc, mvidx, _ok = mirror_on_surface(
            context, obj, hit["local"], hit["view"])
        sister = mh.landmarks[j]
        if kind == 'SRC':
            sister.src_co = mloc
            sister.src_vidx = mvidx
        else:
            sister.tgt_co = mloc

    def _endpoint_merge_candidate(self, mh, i, kind):
        """(index_on_source_curve, index_on_twin) when the dragged point is a
        curve END hovering the centre line and its sister is the matching end
        of the mirrored twin - the only pair that is ever allowed to merge."""
        if not (0 <= i < len(mh.landmarks)):
            return None
        lm = mh.landmarks[i]
        if int(lm.curve_id) < 0:
            return None
        j = lmdata.mirror_partner(mh, i)
        if j < 0:
            return None
        pts = lmdata.curves(mh).get(int(lm.curve_id)) or []
        if len(pts) < 2 or i not in (pts[0][0], pts[-1][0]):
            return None
        sister = mh.landmarks[j]
        jpts = lmdata.curves(mh).get(int(sister.curve_id)) or []
        if not jpts or j not in (jpts[0][0], jpts[-1][0]):
            return None
        obj = mh.cage if kind == 'SRC' else mh.target
        if obj is None:
            return None
        half = local_halfwidth_x(obj)
        snap = max(mh.symmetry_center_threshold * half, 1e-9)
        co = lm.src_co if kind == 'SRC' else lm.tgt_co
        if abs(co[0]) > snap:
            return None
        # the SOURCE curve keeps the merged point; the twin's end dies
        return (j, i) if int(lm.mirror_of) >= 0 else (i, j)

    def _loop_close_candidate(self, mh, i, kind):
        """Return the two ends when an existing open curve is dragged shut.

        The dragged endpoint is removed on release and the stationary endpoint
        survives.  The capture distance is the same Loop Merge Distance used
        while initially drawing a loop.
        """
        if not (0 <= i < len(mh.landmarks)):
            return None
        lm = mh.landmarks[i]
        cid = int(lm.curve_id)
        if cid < 0 or lm.curve_closed:
            return None
        pts = lmdata.curves(mh).get(cid) or []
        if len(pts) < 4:
            return None
        first, last = pts[0][0], pts[-1][0]
        if i == first:
            other = last
        elif i == last:
            other = first
        else:
            return None
        obj = mh.cage if kind == 'SRC' else mh.target
        if obj is None:
            return None
        other_lm = mh.landmarks[other]
        if kind == 'SRC':
            if not (lm.has_src and other_lm.has_src):
                return None
            a, b = Vector(lm.src_co), Vector(other_lm.src_co)
        else:
            if not (lm.has_tgt and other_lm.has_tgt):
                return None
            a, b = Vector(lm.tgt_co), Vector(other_lm.tgt_co)
        limit = max(float(mh.landmark_loop_merge_threshold)
                    * max(local_halfwidth_x(obj), 1e-9), 1e-9)
        return (i, other) if (a - b).length <= limit else None

    def _close_curve_endpoints(self, context, mh, i_drop, i_keep):
        """Merge an existing curve's two ends and mark it cyclic.

        One pair is removed, so source and target keep identical point counts.
        When the curve has an auto-mirrored twin, its corresponding endpoint is
        removed and that curve becomes cyclic in the same operation.
        """
        if not (0 <= i_drop < len(mh.landmarks)
                and 0 <= i_keep < len(mh.landmarks)):
            return
        cid = int(mh.landmarks[i_keep].curve_id)
        if cid < 0 or int(mh.landmarks[i_drop].curve_id) != cid:
            return

        twin_drop = lmdata.mirror_partner(mh, i_drop)
        twin_keep = lmdata.mirror_partner(mh, i_keep)
        close_ids = {cid}
        doomed = {i_drop}
        if (twin_drop >= 0 and twin_keep >= 0
                and twin_drop != twin_keep
                and int(mh.landmarks[twin_drop].curve_id)
                    == int(mh.landmarks[twin_keep].curve_id)):
            close_ids.add(int(mh.landmarks[twin_keep].curve_id))
            doomed.add(twin_drop)

        for _curve_id, pts in lmdata.curves(mh).items():
            if _curve_id in close_ids:
                for _idx, point in pts:
                    point.curve_closed = True

        new_active = i_keep - sum(1 for index in doomed if index < i_keep)
        for index in sorted(doomed, reverse=True):
            lmdata.remove_at(mh, index, save=False)
        mh.landmark_active = min(new_active, len(mh.landmarks) - 1)
        lmdata.save_active(mh)
        _undo_push("Close landmark loop")
        self.msg = "Endpoints merged - curve is now a loop"
        self._sync_visibility(context, mh)

    def _merge_endpoints(self, context, mh, i_src, i_twin):
        """Collapse a curve end onto its mirrored sister: one centre landmark
        survives on the source curve, the twin end is removed, and the two
        curves join into one stroke (via the centre-merge link)."""
        if not (0 <= i_src < len(mh.landmarks)
                and 0 <= i_twin < len(mh.landmarks)):
            return
        dg = context.evaluated_depsgraph_get()
        lm = mh.landmarks[i_src]
        # snap the survivor exactly onto the centre line, on each surface
        for obj, attr, has, is_src in (
                (mh.cage, "src_co", lm.has_src, True),
                (mh.target, "tgt_co", lm.has_tgt, False)):
            if obj is None or not has:
                continue
            co = Vector(getattr(lm, attr))
            co.x = 0.0
            try:
                ok, loc, _nor, poly = obj.closest_point_on_mesh(
                    co, depsgraph=dg)
            except RuntimeError:
                ok = False
            if ok:
                setattr(lm, attr, loc)
                if is_src:
                    lm.src_vidx = _nearest_vidx(obj, loc, poly)
        lm.center_merged = True
        lmdata.remove_at(mh, i_twin, save=False)
        mh.landmark_active = i_src - (1 if i_twin < i_src else 0)
        lmdata.save_active(mh)
        _undo_push("Merge landmarks at centre")
        self.msg = "Merged at the centre - curves joined"
        self._sync_visibility(context, mh)

    # --------------------------------------------------------------- modal --

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as e:
            self.report({'ERROR'}, f"Landmark tool error: {e}")
            self._finish(context)
            return {'CANCELLED'}

    def _modal(self, context, event):
        self._tag_redraw_event(context, event)
        mh = context.scene.mhfrt

        # ---- Ctrl+R moment: move / scroll / type the count, click confirms --
        if self.state == 'PREVIEW':
            # The wheel first, and outside the PRESS gate: some devices and
            # keymaps deliver wheel events with a value the gate would drop.
            if event.type in _WHEEL:
                self._dial_count(context, mh,
                                 +1 if event.type in _DIAL_UP else -1)
                return {'RUNNING_MODAL'}
            # The universal dial - works with any mouse, trackpad or keymap.
            if event.type == 'MOUSEMOVE':
                self._dial_mouse(context, mh, event)
                return {'RUNNING_MODAL'}
            if event.value == 'PRESS':
                if nav.is_navigation(event):
                    # same rule as the middle-mouse orbit below: reaching for
                    # navigation means the artist is done deciding
                    self._confirm_preview(context, mh)
                    return {'PASS_THROUGH'}
                digit = _digit(event)
                if digit is not None:
                    self._dial_typed(context, mh, digit)
                    return {'RUNNING_MODAL'}
                if event.type == 'BACK_SPACE':
                    self.typed = ""
                    self.msg = ""
                    return {'RUNNING_MODAL'}
                if event.type in _DIAL_UP:
                    self._dial_count(context, mh, +1)
                    return {'RUNNING_MODAL'}
                if event.type in _DIAL_DOWN:
                    self._dial_count(context, mh, -1)
                    return {'RUNNING_MODAL'}
                if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER', 'SPACE'}:
                    self._confirm_preview(context, mh)
                    return {'RUNNING_MODAL'}
                if event.type in {'RIGHTMOUSE', 'ESC'} or (
                        event.type == 'Z' and event.ctrl):
                    # cancel before confirmation: the whole temporary curve
                    # disappears - nothing was ever written to the scene
                    self._cancel_stroke()
                    return {'RUNNING_MODAL'}
                if event.type == 'S' and not event.ctrl:
                    mh.symmetry = not mh.symmetry
                    self._build_preview(context, mh)
                    return {'RUNNING_MODAL'}
                if (event.type == 'MIDDLEMOUSE'
                        or event.type.startswith('NUMPAD')):
                    # starting to orbit means the artist is done deciding:
                    # confirm the curve, then let the navigation through
                    self._confirm_preview(context, mh)
                    return {'PASS_THROUGH'}
            if event.type in {'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'}:
                self._confirm_preview(context, mh)
                return {'PASS_THROUGH'}
            if (event.type == 'MIDDLEMOUSE'
                    or event.type.startswith('NUMPAD')
                    or nav.is_navigation(event)):
                return {'PASS_THROUGH'}     # release of the confirming orbit
            return {'RUNNING_MODAL'}

        # ---- stroke being drawn ---------------------------------------------
        if self.state == 'DRAW':
            if event.type == 'MOUSEMOVE':
                self._grow_stroke(context, event, mh)
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self._end_stroke(context, mh)
                return {'RUNNING_MODAL'}
            if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC'}:
                self._cancel_stroke()
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        # ---- idle: navigate, hover, drag points, start curves ---------------
        # free navigation - each split viewport orbits/zooms independently
        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}
                or event.type.startswith('NUMPAD')):
            return {'PASS_THROUGH'}

        # Alt+LMB / Alt+RMB and friends when nothing is being held or dragged
        if (self.drag is None and self.press is None
                and nav.is_navigation(event)):
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            hot = self._hot_lookup(context, event)
            if hot is None:
                self._hot_region_ptr = 0
                self.cursor_hit = None
                if self.drag is not None:
                    return {'RUNNING_MODAL'}    # keep the grab alive
                self.hover = None
                return {'PASS_THROUGH'}         # the rest of the UI stays live
            side, _area, region, rv3d, self.mouse = hot
            self._hot_region_ptr = region.as_pointer()
            if self.drag is not None:
                self._apply_drag(context, mh, side, region, rv3d)
            else:
                self.hover = self._pick(mh, side, region, rv3d)
                if self.hover is None:
                    probe = self._probe_obj(mh, side)
                    self.cursor_hit = (raycast(context, probe, self.mouse,
                                               region, rv3d, want_vidx=False)
                                       if probe is not None else None)
                else:
                    self.cursor_hit = None
                # arm a drag once the press point moved far enough on a marker
                if (self.press is not None and self.hover is not None
                        and (self.mouse - self.press).length > DRAG_START_PX):
                    self._push_undo(mh)
                    self.drag = self.hover
                    mh.landmark_active = self.hover[0]
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                hot = self._hot_lookup(context, event)
                if hot is None:
                    # buttons, panels and other editors keep working
                    return {'PASS_THROUGH'}
                side, _area, region, rv3d, self.mouse = hot
                self._hot_region_ptr = region.as_pointer()
                if self.hover is not None:
                    # wait for movement: click on a point selects it,
                    # dragging it reshapes the curve
                    self.press = self.mouse.copy()
                else:
                    self._start_stroke(context, mh, side, region, rv3d)
                return {'RUNNING_MODAL'}
            # RELEASE
            if self.drag is not None:
                merge = self._merge_ready
                self._merge_ready = None
                self.drag = None
                self.press = None
                if merge is not None:
                    # collapse the curve end onto its mirrored sister -
                    # indices shifted, so the hover is stale too
                    self.hover = None
                    action, *indices = merge
                    if action == 'LOOP':
                        self._close_curve_endpoints(context, mh, *indices)
                    else:
                        self._merge_endpoints(context, mh, *indices)
                else:
                    lmdata.save_active(mh)
                    _undo_push("Move landmark")
                return {'RUNNING_MODAL'}
            if self.press is not None and self.hover is not None:
                # Plain click on a point: make it the active one.  No save -
                # nothing about the landmarks changed, and serializing the
                # whole set costs milliseconds that scale with how many points
                # the artist has drawn.  The index is persisted with the next
                # real edit, and by _finish on the way out.
                mh.landmark_active = self.hover[0]
                self.press = None
                return {'RUNNING_MODAL'}
            self.press = None
            if self._hot_lookup(context, event) is not None:
                return {'RUNNING_MODAL'}    # release of a click we consumed
            return {'PASS_THROUGH'}

        if event.value == 'PRESS':
            if event.type == 'Z' and event.ctrl:
                # window-wide on purpose: Blender's global undo mid-tool
                # would yank the meshes out from under the session
                if self.undo_stack:
                    lmdata.restore(mh, self.undo_stack.pop())
                    self.msg = ""
                    self.drag = None
                    self.hover = None
                    self.press = None
                    self._merge_ready = None
                    self._sync_visibility(context, mh)
                else:
                    self.msg = "Nothing to undo"
                return {'RUNNING_MODAL'}

            if event.type == 'ESC':
                self._finish(context)
                return {'FINISHED'}

            # remaining hotkeys act only over the tool's viewports; anywhere
            # else the key belongs to whatever editor is under the mouse
            hot_key = self._hot_lookup(context, event) is not None
            if not hot_key:
                return {'PASS_THROUGH'}

            if event.type in {'X', 'DEL'}:
                self._delete_hovered_curve(context, mh)
                return {'RUNNING_MODAL'}

            if event.type == 'S' and not event.ctrl:
                mh.symmetry = not mh.symmetry
                return {'RUNNING_MODAL'}

            if event.type == 'L' and not event.ctrl:
                mh.landmark_lazy = not mh.landmark_lazy
                self.msg = ("Lazy mouse ON - the stroke trails the cursor"
                            if mh.landmark_lazy else "Lazy mouse OFF")
                return {'RUNNING_MODAL'}

            if event.type == 'C':
                mh.landmark_sync_view = not mh.landmark_sync_view
                if mh.landmark_sync_view:
                    _force_cam_align()
                self.msg = ("View sync ON" if mh.landmark_sync_view
                            else "View sync OFF")
                return {'RUNNING_MODAL'}

            if event.type == 'H':
                if self.dual:
                    self.msg = "Split view - each side already shows one mesh"
                else:
                    self.solo = not self.solo
                    self._sync_visibility(context, mh)
                return {'RUNNING_MODAL'}

            if event.type == 'RIGHTMOUSE':
                self._finish(context)
                return {'FINISHED'}

            # Only swallow the Blender keys that would fight the tool: G/R
            # would grab the isolated meshes, TAB would enter Edit mode, A/B
            # would run box/select and steal the click.  Everything else
            # (N, T, mode toggles, workspace shortcuts) passes through so the
            # N-panel, properties, and other viewport features remain usable
            # even while the modal is running.
            if event.type in {'G', 'R', 'A', 'B', 'TAB'}:
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def _finish(self, context):
        global _RUNNING
        if getattr(self, "_finished", False):
            return
        self._finished = True
        _RUNNING = False
        # a stroke or preview still live is pure runtime state - dissolve it
        self.state = 'IDLE'
        self.stroke = None
        if getattr(self, "_hud", None):
            bpy.types.SpaceView3D.draw_handler_remove(self._hud, 'WINDOW')
            self._hud = None
        if getattr(self, "_ring", None):
            bpy.types.SpaceView3D.draw_handler_remove(self._ring, 'WINDOW')
            self._ring = None
        if getattr(self, "_tool_window", None) is not None:
            self._close_window(context)
        elif getattr(self, "dual", False):
            self.dual = False
            self._exit_dual(context)
        scene = getattr(context, "scene", None)
        mh = getattr(scene, "mhfrt", None) if scene else None
        if mh is None:
            return
        # restore the wrapped shape the user had before editing - unless a
        # re-wrap (panel stays usable now) already set its own value
        if self._wrap_prev is not None and mh.cage and mh.cage.data.shape_keys:
            wk = mh.cage.data.shape_keys.key_blocks.get(WRAPPED_KEY)
            if wk is not None and abs(wk.value) < 1e-6:
                wk.value = self._wrap_prev
                mh.cage.data.update()
        # ... and the cleanup pose that was showing when the tool opened
        for obj, name, value in getattr(self, "_pose_prev", ()):
            try:
                kb = obj.data.shape_keys.key_blocks if obj.data.shape_keys \
                    else None
                k = kb.get(name) if kb else None
                if k is not None and abs(k.value) < 1e-6:
                    k.value = value
                    obj.data.update()
            except (ReferenceError, AttributeError, RuntimeError):
                pass
        # ... and every armature we parked in REST to draw on the neutral shape
        restored = False
        for data, position in getattr(self, "_rest_prev", ()):
            try:
                data.pose_position = position
                restored = True
            except (ReferenceError, AttributeError):
                pass
        self._rest_prev = []
        if restored:
            try:
                context.view_layer.update()
            except (AttributeError, RuntimeError):
                pass
        window = getattr(context, "window", None)
        if window is not None:
            # Force the crosshair out of the way immediately, then queue two
            # backup resets at successively later ticks - after area_close
            # inside _exit_dual, and after Blender's own post-modal handling
            # has had a chance to touch the cursor again.  Belt, suspenders,
            # and parachute.
            self._cursor_hot = False
            _force_default_cursor(window)
            win_ptr = window.as_pointer()
            for delay in (0.05, 0.25):
                bpy.app.timers.register(
                    lambda p=win_ptr: _delayed_default_cursor(p),
                    first_interval=delay)
        apply_view_mode(context)
        # a cage curve never matched on the head is useless to the wrap -
        # it does not outlive the tool (a curve exists whole or not at all)
        removed = lmdata.remove_incomplete(mh, save=False)
        lmdata.save_active(mh)
        _undo_push("Edit landmarks")
        _redraw_viewports()
        n = lmdata.complete_count(mh)
        ncurves = lmdata.curve_count(mh)
        msg = f"{ncurves} curve(s) · {n} landmark point(s)"
        if removed:
            msg += "  ·  unfinished curve removed"
        self.report({'INFO'}, msg)
        # NO auto-advance to the next step: Esc just closes the tool (the user
        # often enters only to toggle symmetry or inspect markers)


# --------------------------------------------------------- misc operators ---

class MHFRT_OT_clear_pairs(bpy.types.Operator):
    bl_idname = "mhfrt.clear_pairs"
    bl_label = "Clear All Landmarks"
    bl_description = "Remove every landmark curve and point"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        mh = context.scene.mhfrt
        lmdata.migrate_legacy(context)
        lmdata.clear(mh)
        _redraw_viewports()
        self.report({'INFO'}, "Cleared all landmarks")
        return {'FINISHED'}


_classes = (
    MHFRT_OT_edit_landmarks,
    MHFRT_OT_clear_pairs,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    global _RUNNING
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
    _DUAL_SIDES.clear()
    _RUNNING = False
    _stop_cam_sync()
    if bpy.app.timers.is_registered(_cam_sync_tick):
        try:
            bpy.app.timers.unregister(_cam_sync_tick)
        except (ValueError, RuntimeError):
            pass
