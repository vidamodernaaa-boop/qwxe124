"""Persistent viewport overlay for landmark curves - cyan ink style.

Each confirmed curve is drawn as a smooth light-cyan stroke through its
landmark points with the curve number at its start point, so matching cage
curve N to head curve N is a glance. Points are small dots; the active point
is a white marker. Incomplete (still pending) curves are dimmed. Where a
symmetry centre merge happened, a curve and its mirrored twin are joined at
the merged endpoint so they read as one stroke across the face. Legacy single
points from the click-click era keep their ring / diamond markers and
numbers. A faint dashed line links the two sides of the ACTIVE pair only,
when both meshes share the viewport. Independent of viewport shading; toggle
from the panel.
"""

import bpy
from bpy_extras.view3d_utils import location_3d_to_region_2d

from ..core import landmarks as lmdata
from . import gizmo_draw as gd

_handle = None

_DIM_LINE = (0.42, 0.62, 0.70, 0.45)   # pending curve bodies (dimmed cyan)


def _runs(pts2d, closed=False):
    """Split a projected polyline where points leave the view.

    Returns ``[(run, is_loop), ...]``.  A closed curve stays ONE loop only
    while every one of its points is on screen: once part of the ring has been
    clipped away what is left is an arc, and closing that would draw a chord
    straight across the gap.
    """
    runs, cur = [], []
    for p in pts2d:
        if p is None:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
        else:
            cur.append((p[0], p[1]))
    if len(cur) >= 2:
        runs.append(cur)
    whole = bool(closed) and len(runs) == 1 and len(runs[0]) == len(pts2d)
    return [(run, whole) for run in runs]


def _draw():
    ctx = bpy.context
    mh = getattr(ctx.scene, "mhfrt", None)
    if mh is None or not mh.show_pairs_overlay or not len(mh.landmarks):
        return
    # landmarks play no role in the live session's physics - hide their
    # markers entirely while it runs so the viewport shows only the sim
    from ..ops import op_live
    if op_live.is_running():
        return
    region = ctx.region
    rv3d = ctx.region_data
    if region is None or rv3d is None or region.type != 'WINDOW':
        return

    # follow ACTUAL mesh visibility (covers the view-mode switch), so markers
    # never float over a hidden mesh
    def _vis(obj):
        if obj is None:
            return False
        try:
            return obj.visible_get()
        except (RuntimeError, AttributeError):
            return True

    from ..ops import op_pairs
    drawing = op_pairs.is_running()
    if drawing:
        # INSIDE the landmark tool nothing hides a curve behind the artist's
        # back.  The tool solos meshes as the workflow moves from the cage to
        # the head, and deriving marker visibility from the mesh meant the
        # curve you had just drawn disappeared the moment the tool switched
        # sides.  The only filter left is the split layout's own per-viewport
        # side, applied below - everything else stays on screen until the
        # artist leaves the tool.
        show = {"src": mh.cage is not None, "tgt": mh.target is not None}
    else:
        show = {"src": _vis(mh.cage), "tgt": _vis(mh.target)}

    # landmark tool split layout: the cage viewport draws only cage markers,
    # the target viewport only target markers (local view hides the meshes,
    # but markers are data and need their own per-viewport filter)
    side = op_pairs.dual_side_of(ctx.area)
    if side == 'CAGE':
        show["tgt"] = False
    elif side == 'TARGET':
        show["src"] = False

    curves, legacy, by_index = {}, [], {}
    for e in lmdata.display_entries(ctx):
        by_index[e["index"]] = e
        if e["curve"] >= 0:
            curves.setdefault(e["curve"], []).append(e)
        else:
            legacy.append(e)

    active = mh.landmark_active
    active_entry = None

    # ---- curves: smooth stroke + dots + number at the start ----------------
    #
    # The per-POINT work is collected and drawn in one GPU batch each at the
    # end: every batch_for_shader call builds a vertex buffer, and two per
    # marker is what made a face carrying hundreds of landmarks crawl.  Curve
    # bodies stay per-curve - there are only ever a handful of them, and they
    # carry alpha, which does not survive being merged into one buffer.
    outline_dots = []
    ink_dots = []
    dim_dots = []
    labels = []                     # (x, y, text, size, color)
    active_markers = []             # (point, key) - drawn last, on top

    for num, entries in enumerate(curves.values(), start=1):
        complete = all(e["complete"] for e in entries)
        is_active = any(e["index"] == active for e in entries)
        for key in ("src", "tgt"):
            if not show[key]:
                continue
            pts2 = [None if e[key] is None else
                    location_3d_to_region_2d(region, rv3d, e[key])
                    for e in entries]
            # A closed curve links its last point back to its first: it was
            # drawn as one loop (eye rim, lips) and has to read as one - one
            # PERIODIC curve, so the seam is no different from any other joint.
            closed = bool(entries[0].get("closed")) and len(pts2) > 2
            line = gd.CYAN_SOFT if complete else _DIM_LINE
            if is_active and complete:
                line = gd.CYAN
            for run, loop in _runs(pts2, closed):
                gd.curve_stroke(gd.smooth_polyline(run, closed=loop), line,
                                1.7 if is_active else 1.3)
            for e, p in zip(entries, pts2):
                if p is None:
                    continue
                if e["index"] == active:
                    active_entry = e
                    active_markers.append((p, key))
                else:
                    outline_dots.append(p)
                    (ink_dots if e["complete"] else dim_dots).append(p)
            first = next((p for p in pts2 if p is not None), None)
            if first is not None:
                color = gd.WHITE if is_active else (gd.INK if complete
                                                    else gd.DIM)
                labels.append((first[0] + 8, first[1] + 8, str(num),
                               12 if is_active else 11, color))

    # ---- symmetry merge links: twin curves join at the centre line ----------
    # drawn in the same ink as the curve bodies, so a curve and its mirrored
    # twin read as ONE continuous stroke across the face
    for ia, ib in lmdata.merge_links(mh):
        ea, eb = by_index.get(ia), by_index.get(ib)
        if ea is None or eb is None:
            continue
        for key in ("src", "tgt"):
            if not show[key] or ea[key] is None or eb[key] is None:
                continue
            a = location_3d_to_region_2d(region, rv3d, ea[key])
            b = location_3d_to_region_2d(region, rv3d, eb[key])
            if a is not None and b is not None:
                gd.curve_stroke([(a[0], a[1]), (b[0], b[1])],
                                gd.CYAN_SOFT, 1.3)

    # the dots ride on top of every curve body, then the numbers on top again
    gd.dots(outline_dots, 3.4, gd.OUTLINE)
    gd.dots(ink_dots, 2.1, gd.INK)
    gd.dots(dim_dots, 2.1, gd.DIM)
    for p, key in active_markers:
        marker = gd.marker_src if key == "src" else gd.marker_tgt
        marker(p, 7.0, gd.WHITE, 1.7)
    for x, y, text, size, color in labels:
        gd.text(x, y, text, size, color)

    # ---- legacy single points (pre-curve files) -----------------------------
    for e in legacy:
        s2 = (location_3d_to_region_2d(region, rv3d, e["src"])
              if (e["src"] is not None and show["src"]) else None)
        t2 = (location_3d_to_region_2d(region, rv3d, e["tgt"])
              if (e["tgt"] is not None and show["tgt"]) else None)
        is_active = e["index"] == active
        if is_active:
            active_entry = e
        color = gd.WHITE if is_active else (gd.INK if e["complete"] else gd.DIM)
        radius = 8.0 if is_active else 6.5
        for p, marker in ((s2, gd.marker_src), (t2, gd.marker_tgt)):
            if p is None:
                continue
            marker(p, radius, color, 1.6)
            gd.text(p[0] + 9, p[1] + 7, str(e["index"] + 1),
                    12 if is_active else 11, color)
            if e["label"] and not e["complete"]:
                gd.text(p[0] + 9, p[1] - 8, e["label"], 10, gd.DIM)

    # ---- one dashed link, on the active pair only ---------------------------
    if (active_entry is not None and show["src"] and show["tgt"]
            and active_entry["src"] is not None
            and active_entry["tgt"] is not None):
        s2 = location_3d_to_region_2d(region, rv3d, active_entry["src"])
        t2 = location_3d_to_region_2d(region, rv3d, active_entry["tgt"])
        if s2 is not None and t2 is not None:
            gd.dashed(s2, t2, gd.FAINT, 1.1)

    gd.finish()


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
