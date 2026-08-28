"""Live soft session - Softwrap-style interactive softbody refinement.

Press once to start, again (or Esc) to stop. A timer runs a spring simulation
on the wrapped cage at ~30 fps: edge + quad-diagonal + bending springs (with
Softwrap-style plastic rest lengths) hold the wrapped shape, a gentle
Laplacian keeps the quad flow even, and every vertex is continuously pulled
onto its cached closest point on the target head (the point - not just the
plane - so the registration is anchored tangentially and never slides).
Vertices found INSIDE the target are never abandoned: they are always pulled
back out to the surface even when their normal disagrees. Two surgical
recovery passes run per tick: UNTANGLE reflects vertices that folded through
their neighbours (relative to the session-start shape) back to the correct
side and leaves healthy mesh completely untouched, and KEEP OUTSIDE projects
any vertex that sank under the target skin back onto it. While it runs:

* drag anywhere on the cage to GRAB an elastic region (view-plane drag, the
  surrounding mesh follows and re-snaps live),
* Shift+click to drop a PIN (a persistent little anchor you can drag around;
  the mesh keeps chasing it), X deletes a hovered pin,
* tweak every slider in the panel mid-simulation - events pass through, the
  panel stays fully interactive,
* Space pauses, Ctrl+Z undoes interactions, Esc/Enter finishes.

Everything happens inside the 'Wrapped' shape key - Basis is never touched.
Region_Mask verts (if the wrap setting is on) and pinned borders stay locked,
and mh.symmetry mirrors grabs & pins across X, so the session agrees with
every other step. Any operator that rewrites the cage calls stop_running()
first; the session also stops itself if the mesh changes under it.

The physics lives in the bpy-free _Sim class so it can be tested headless.
"""

import time
from math import atan2

import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.kdtree import KDTree
from bpy_extras import view3d_utils
import bpy.app.handlers as handlers

from ..core import bvh as bvhmod
from ..core import nav
from ..core import wrap as wrapmod
from ..ui import gizmo_draw as gd
from .op_wrap import (WRAPPED_KEY, region_mask_frozen, get_basis_local,
                      pose_offset_local)

TICK = 1.0 / 30.0
SUBSTEPS = 2
DAMPING = 0.25
SNAP_RATE = 0.45       # max pull toward the cached surface point per substep
INSIDE_W_MIN = 0.25    # inside verts are ALWAYS pulled out, even bad normals
SMOOTH_RATE = 0.40     # max tangential relax per tick at slider = 1
UNTANGLE_RATE = 0.50   # max fold-recovery blend per tick at slider = 1
GRAB_RATE = 0.90       # how hard the mouse constraint pulls per substep
PIN_RATE = 0.60        # how hard pins pull per substep
PLASTIC_UPDATE = 0.04  # spring rest-length creep toward the current length
PLASTIC_RESTORE = 0.02  # and its slow recovery back to the original length
PLASTIC_MIN = 0.3      # rest-length scale clamp (Softwrap's min/max scaling)
PLASTIC_MAX = 3.0
BEND_DISTANCE = 2      # bending springs reach this many edges in a line
KEEP_OUT_MAX = 0.25    # keep-outside push per tick, as a fraction of v_max
UNDO_MAX = 30
PIN_PICK_PX = 14.0

_running = None        # the single active session (like Softwrap's running_op)


def is_running():
    return _running is not None


def running_scope():
    return getattr(_running, "scope", None)


def stop_running():
    """Immediately and safely end the active session (called by any operator
    that is about to rewrite the cage, and by the load_pre handler)."""
    if _running is not None:
        _running.stop(bpy.context)


# ------------------------------------------------------------------ physics ---

def _quad_diagonals(mesh):
    """Shear springs: both diagonals of every quad."""
    pairs = []
    for p in mesh.polygons:
        if p.loop_total == 4:
            v = p.vertices
            pairs.append((v[0], v[2]))
            pairs.append((v[1], v[3]))
    return (np.array(pairs, dtype=np.int64) if pairs
            else np.empty((0, 2), np.int64))


def _sorted_link_edges(vert):
    """A vertex's edges sorted by angle around its normal (Softwrap)."""
    u = vert.normal.orthogonal().normalized()
    w = vert.normal.cross(u)

    def angle(edge):
        vec = vert.co - edge.other_vert(vert).co
        return atan2(vec.dot(u), vec.dot(w))

    return sorted(vert.link_edges, key=angle)


def _loop_pairs(elems):
    """Pairs of roughly opposite elements of an angularly sorted ring."""
    n = len(elems)
    half = n // 2
    for i in range(half + n % 2):
        yield elems[i], elems[(i + half) % n]


def _ternary_links(bm, keep=None):
    """(center, a, b) triples of opposite-neighbour pairs around each vertex -
    Softwrap's topologic-smooth links, the fold/overlap detector."""
    links = []
    for vert in bm.verts:
        if keep is not None and not keep[vert.index]:
            continue
        for ea, eb in _loop_pairs(_sorted_link_edges(vert)):
            a = ea.other_vert(vert).index
            b = eb.other_vert(vert).index
            if len({vert.index, a, b}) == 3:
                links.append((vert.index, a, b))
    return (np.array(links, dtype=np.int64) if links
            else np.empty((0, 3), np.int64))


def _bending_springs(bm, distance=BEND_DISTANCE, keep=None):
    """Springs that continue straight across the quad grid for `distance`
    edges (Softwrap's bending springs) - they resist folding and collapse."""
    springs = set()
    for vert in bm.verts:
        if keep is not None and not keep[vert.index]:
            continue
        for loop in vert.link_loops:
            walk = loop
            for _ in range(max(distance, 1)):
                try:
                    walk = walk.link_loop_next
                    for _ in range(len(walk.vert.link_edges) // 2 - 1):
                        walk = walk.link_loop_radial_next.link_loop_next
                except (AttributeError, IndexError):
                    break
                other = walk.vert.index
                if other == vert.index:
                    continue
                springs.add((min(vert.index, other), max(vert.index, other)))
    return (np.array(sorted(springs), dtype=np.int64) if springs
            else np.empty((0, 2), np.int64))


class _Sim:
    """Vectorized position-based spring engine (bpy-free, testable headless).

    P/P_prev are world-space (n,3). Edge + shear + bending springs rest at
    session-start lengths (with Softwrap-style plastic rest scaling) so the
    sim defends the wrapped shape; snapping pulls toward cached closest
    POINTS on the target BVH (tangential anchor - the registration cannot
    slide), refreshed in round-robin chunks. Ternary links (side measured on
    the session-start shape, like Softwrap) surgically recover folds that
    appear during the session, and a keep-outside pass projects sunken
    vertices back onto the surface.
    """

    def __init__(self, P0, topo, shear, bvh, locked, max_dist, weights=None,
                 bend=None, tern=None, frozen=None):
        n = len(P0)
        self.n = n
        self.P = P0.copy()
        self.P_prev = P0.copy()
        self.P_lock = P0.copy()
        self.topo = topo
        self.bvh = bvh
        self.locked = locked          # bool (n,) - never moved by the sim
        self.max_dist = max_dist
        self.weights = None
        if weights is not None:
            self.weights = np.clip(np.asarray(weights, dtype=float).reshape(n, 1),
                                   0.0, 1.0)

        if bend is None:
            bend = np.empty((0, 2), np.int64)
        e = np.vstack([topo["edges"], shear, bend])
        # shear springs count at half weight, like Softwrap's separate groups
        k_edge = np.ones(len(e))
        ne = len(topo["edges"])
        k_edge[ne:ne + len(shear)] = 0.5
        # springs between two permanently frozen verts can never do anything -
        # drop them (in masked cleanup mode this removes almost all of them)
        if frozen is not None and frozen.any():
            live_spring = ~(frozen[e[:, 0]] & frozen[e[:, 1]])
            e = e[live_spring]
            k_edge = k_edge[live_spring]
        self.e0, self.e1 = e[:, 0], e[:, 1]
        self.k_edge = k_edge
        d = P0[self.e1] - P0[self.e0]
        self.rest = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        # plastic rest-length scale (Softwrap): springs slowly adapt to a held
        # deformation instead of fighting it forever, then creep back
        self.rest_scale = np.ones(len(e))
        cnt = (np.bincount(self.e0, minlength=n)
               + np.bincount(self.e1, minlength=n)).astype(float)
        self.counts = np.maximum(cnt, 1.0)[:, None]

        # ternary (opposite-neighbour) links: fold detector. Side + distance
        # ratio are measured on the SESSION-START shape (like Softwrap's
        # displacements_update): the current wrap is the ground truth, so
        # healthy geometry is never touched - only later fold-overs are.
        if tern is None or not len(tern):
            tern = np.empty((0, 3), np.int64)
        self.tc, self.ta, self.tb = tern[:, 0], tern[:, 1], tern[:, 2]
        Nref = wrapmod.vertex_normals(P0, topo)
        avg = (P0[self.ta] + P0[self.tb]) * 0.5
        dref = P0[self.tc] - avg
        self.t_side = (dref * Nref[self.tc]).sum(axis=1) > 0.0
        self.t_ratio = (np.linalg.norm(dref, axis=1)
                        / (np.linalg.norm(P0[self.ta] - P0[self.tb], axis=1)
                           + 1e-12))

        # cached correspondences (closest point + its surface normal),
        # refreshed a chunk per tick - only for verts that can ever move
        self.T = P0.copy()
        self.Tn = np.zeros((n, 3))
        self.T_ok = np.zeros(n, dtype=bool)
        self._active = (np.flatnonzero(~frozen) if frozen is not None
                        else np.arange(n))
        self._chunk = max(256, len(self._active) // 10)
        self._cursor = 0
        size = max_dist if max_dist > 0 else 1.0
        self.v_max = 0.05 * size * 8.0   # runaway guard

        # external constraints, rebuilt by the operator every tick
        self.grabs = []                  # (indices, weights (m,1), target (m,3))
        self.pin_pulls = []              # (indices, weights (m,1), delta (3,))

    def refresh_chunk(self):
        """Update closest-point targets for the next round-robin chunk.
        Every hit within range is kept - even with a disagreeing normal -
        because the snap step decides per-tick what to do with bad ones
        (push the vertex out) instead of abandoning them inside."""
        na = len(self._active)
        if na == 0:
            return
        sel = (self._cursor + np.arange(min(self._chunk, na))) % na
        self._cursor = int((self._cursor + len(sel)) % na)
        find = self.bvh.find_nearest
        T, Tn, ok, P = self.T, self.Tn, self.T_ok, self.P
        for i in self._active[sel]:
            loc, nor, _fi, _d = find(Vector(P[i]), self.max_dist)
            if loc is None or nor is None:
                ok[i] = False
                continue
            T[i] = (loc.x, loc.y, loc.z)
            Tn[i] = (nor.x, nor.y, nor.z)
            ok[i] = True

    def _springs(self, k, plastic=False):
        P = self.P
        d = P[self.e1] - P[self.e0]
        L = np.sqrt(np.einsum('ij,ij->i', d, d))
        L[L < 1e-12] = 1e-12
        rest = self.rest * self.rest_scale
        c = ((L - rest) / L * 0.5 * k * self.k_edge)[:, None] * d
        acc = np.empty_like(P)
        for ax in range(3):
            acc[:, ax] = (np.bincount(self.e0, weights=c[:, ax], minlength=self.n)
                          - np.bincount(self.e1, weights=c[:, ax], minlength=self.n))
        P += acc / self.counts
        if plastic:
            ratio = np.clip(L / self.rest, PLASTIC_MIN, PLASTIC_MAX)
            self.rest_scale += (ratio - self.rest_scale) * PLASTIC_UPDATE
            self.rest_scale += (1.0 - self.rest_scale) * PLASTIC_RESTORE

    def _snap(self, N, snap_f):
        """Pull vertices toward their cached closest POINT on the target.
        The full 3D pull (not just the plane) is what anchors the wrap
        tangentially so it cannot slide. Outside vertices snap only where
        cage/target normals agree (lips never grab the chin); a vertex found
        INSIDE the target is always pulled back out to the surface - the
        closest point is on the skin by definition - even when its normal
        disagrees (this is the fix for 'it just accepts being inside')."""
        P = self.P
        pull = self.T_ok & ~self.locked
        if not pull.any():
            return
        Tn = self.Tn[pull]
        v = self.T[pull] - P[pull]           # toward the cached surface point
        inside = (v * Tn).sum(axis=1) > 0.0  # behind the surface
        align = (N[pull] * Tn).sum(axis=1)
        w = np.where(inside,
                     np.maximum(align * align, INSIDE_W_MIN),
                     np.clip(align, 0.0, None) ** 2)[:, None]
        if self.weights is not None:
            w = w * self.weights[pull]
        P[pull] += v * (snap_f * w)

    def _untangle(self, factor):
        """Surgical fold recovery (Softwrap's topologic smooth, applied ONLY
        to violated links): a vertex that moved to the wrong side of its
        opposite-neighbour midpoints - compared against the session-start
        shape - is reflected back across its normal and blended toward the
        correct-side offset. Healthy geometry is left completely untouched,
        so this force can never make the mesh slide, shrink, or crumple."""
        if not len(self.tc) or factor <= 0.0:
            return
        P = self.P
        N = wrapmod.vertex_normals(P, self.topo)
        avg = (P[self.ta] + P[self.tb]) * 0.5
        d = P[self.tc] - avg
        Nc = N[self.tc]
        dot = (d * Nc).sum(axis=1)
        flipped = (dot > 0.0) != self.t_side
        if not flipped.any():
            return
        c = self.tc[flipped]
        d = d[flipped] - 2.0 * dot[flipped, None] * Nc[flipped]  # reflect back
        avg = avg[flipped]
        ln = np.sqrt(np.einsum('ij,ij->i', d, d))[:, None]
        ln[ln < 1e-12] = 1.0
        ab = P[self.ta[flipped]] - P[self.tb[flipped]]
        span = np.sqrt(np.einsum('ij,ij->i', ab, ab))[:, None]
        tgt = avg + d / ln * (self.t_ratio[flipped, None] * span)
        acc = np.zeros_like(P)
        for ax in range(3):
            acc[:, ax] = np.bincount(c, weights=tgt[:, ax], minlength=self.n)
        cnt = np.bincount(c, minlength=self.n).astype(float)
        move = (cnt > 0) & ~self.locked
        acc[move] /= cnt[move, None]
        P[move] += (acc[move] - P[move]) * factor

    def _keep_outside(self):
        """Hard constraint (Softwrap's OUTSIDE snap mode): any vertex behind
        its cached target plane is pushed back out onto it, so the cage never
        settles under the skin. The push per tick is capped: cached planes
        can be a few ticks stale, and an uncapped teleport onto a stale plane
        pops the mesh around - capped, a sunken vertex still surfaces within
        a few ticks."""
        P = self.P
        m = self.T_ok & ~self.locked
        if not m.any():
            return
        Tn = self.Tn[m]
        dist = ((self.T[m] - P[m]) * Tn).sum(axis=1)   # > 0 = inside
        push = np.clip(dist, 0.0, KEEP_OUT_MAX * self.v_max)[:, None]
        if self.weights is not None:
            push = push * self.weights[m]
        P[m] += Tn * push

    def _clamp(self):
        """Re-assert locked verts and the painted-mask blend."""
        P, locked = self.P, self.locked
        if self.weights is not None:
            free = ~locked
            P[free] = (self.P_lock[free]
                       + (P[free] - self.P_lock[free]) * self.weights[free])
        P[locked] = self.P_lock[locked]

    def step(self, stiffness, smooth, snap, pin_force,
             untangle=0.0, keep_outside=False):
        P = self.P

        # verlet inertia with damping (the softbody feel), clamped for safety
        v = (P - self.P_prev) * (1.0 - DAMPING)
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        np.clip(vn, None, self.v_max, out=vn)
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        v = v / nrm * vn
        self.P_prev = P.copy()
        P += v

        # one set of normals per tick, like Softwrap's update_mesh_normals
        N = wrapmod.vertex_normals(P, self.topo)
        snap_f = (snap ** 2) * SNAP_RATE
        for sub in range(SUBSTEPS):
            if stiffness > 0.0:
                self._springs(min(stiffness, 1.0), plastic=(sub == 0))
            if snap_f > 0.0:
                self._snap(N, snap_f)
            for idx, w, tgt in self.grabs:
                P[idx] += (tgt - P[idx]) * (w * GRAB_RATE)
            for idx, w, delta in self.pin_pulls:
                P[idx] += delta[None, :] * (w * PIN_RATE * pin_force)
            self._clamp()

        if smooth > 0.0:
            free = ~self.locked
            d = wrapmod.neighbor_average(P, self.topo) - P
            d -= (d * N).sum(axis=1, keepdims=True) * N   # tangential: no shrink
            P[free] += d[free] * (smooth * SMOOTH_RATE)
        if untangle > 0.0:
            self._untangle(untangle * UNTANGLE_RATE)
        # last word: the frame never ends with vertices under the skin
        if keep_outside:
            self._keep_outside()
        self._clamp()

    def snapshot(self):
        return (self.P.copy(), self.P_prev.copy(), self.rest_scale.copy())

    def restore(self, snap):
        self.P, self.P_prev = snap[0].copy(), snap[1].copy()
        self.rest_scale = snap[2].copy()
        self.T_ok[:] = False   # stale correspondences


# ------------------------------------------------------------------ drawing ---

def _draw_hud(op):
    ctx = bpy.context
    region = ctx.region
    if region is None or region.type != 'WINDOW' or region != op._region:
        return
    mh = getattr(ctx.scene, "mhfrt", None)
    if mh is None:
        return
    cx = region.width * 0.5
    y = region.height - 54
    title = getattr(op, "scope_label", "LIVE SOFT SESSION")
    if mh.live_pause:
        title += "  ·  PAUSED"
    gd.text(cx, y, title, 17, gd.WHITE, align='CENTER')
    w = max(gd.text_width(title, 17), 340.0)
    gd.stroke([(cx - w * 0.5, y - 9), (cx + w * 0.5, y - 9)], gd.DIM, 1.0)
    mask = getattr(op, "mask_label", "")
    detail = f"      {mask}" if mask else ""
    gd.text(cx, y - 26,
            f"drag = grab      Shift+click = pin      pins: {len(op.pins)}"
            f"      symmetry {'ON' if mh.symmetry else 'off'}{detail}",
            12, gd.MID, align='CENTER')
    gd.text(cx, y - 44,
            "X delete pin   ·   [ ] radius   ·   S symmetry   ·   B borders"
            "   ·   Space pause   ·   Ctrl+Z undo   ·   Esc done",
            11, gd.DIM, align='CENTER')
    if op.msg:
        gd.text(cx, y - 64, op.msg, 12, gd.WHITE, align='CENTER')

    # pins + their leashes, in pixel space (same look as the landmark overlay)
    r3d = ctx.space_data.region_3d if ctx.space_data else None
    if r3d is not None:
        for k, pin in enumerate(op.pins):
            vp = view3d_utils.location_3d_to_region_2d(
                region, r3d, Vector(op.sim.P[pin["vidx"]]))
            pp = view3d_utils.location_3d_to_region_2d(region, r3d,
                                                       Vector(pin["pos"]))
            if pp is None:
                continue
            hovered = (k == op.hover_pin)
            color = gd.WHITE if hovered else gd.INK
            if vp is not None:
                gd.dashed(vp, pp, gd.FAINT, 1.2)
                gd.dot(vp, 2.5, gd.MID)
            gd.marker_tgt(pp, r=7.0 if hovered else 6.0, color=color,
                          fill=(k == op.drag_pin))
    gd.finish()


def _draw_ring(op):
    ctx = bpy.context
    if ctx.region != op._region:
        return
    if op.grab is not None:
        center = Vector(op.grab["anchor"] + op.grab["delta"])
        gd.ring_3d(center, op.grab["normal"], op.radius, gd.WHITE, 1.8)
    elif op.hover_hit is not None and op.hover_pin < 0:
        pos, normal = op.hover_hit
        gd.ring_3d(pos, normal, op.radius, gd.INK, 1.4)
    gd.finish()


# ----------------------------------------------------------------- operator ---

class MHFRT_OT_live_session(bpy.types.Operator):
    bl_idname = "mhfrt.live_session"
    bl_label = "Live Session"
    bl_description = ("Interactive softbody refinement (Softwrap-style): a live "
                      "simulation keeps the cage snapped to the head while you "
                      "grab and drag the mesh, drop pins with Shift+click, and "
                      "tune every slider in real time. Press again or Esc to stop")
    bl_options = {'REGISTER'}

    cleanup_feature: bpy.props.EnumProperty(items=[
        ('NONE', "Neutral Wrap", "Refine the neutral cage wrap"),
        ('MOUTH', "Mouth Cleanup", "Live-wrap the target mouth cleanup key"),
        ('EYES', "Eyes Cleanup", "Live-wrap the target eyes cleanup key"),
    ], default='NONE', options={'SKIP_SAVE'})

    def invoke(self, context, event):
        global _running
        if _running is not None:            # toggle: second press stops
            _running.stop(context)
            return {'CANCELLED'}

        nav.refresh()           # honour the artist's own navigation bindings
        mh = context.scene.mhfrt
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run this from the 3D Viewport")
            return {'CANCELLED'}

        setup = self._session_setup(context, self.cleanup_feature)
        if setup is None:
            return {'CANCELLED'}
        edit_obj, snap_obj, write_key, mask_weights, scope, label = setup

        me = edit_obj.data
        wk = me.shape_keys.key_blocks.get(WRAPPED_KEY) if me.shape_keys else None
        if write_key != WRAPPED_KEY:
            wk = me.shape_keys.key_blocks.get(write_key) if me.shape_keys else None
        if wk is None:
            self.report({'ERROR'}, f"Missing '{write_key}' shape key")
            return {'CANCELLED'}

        wk.value = 1.0
        me.update()

        n = len(me.vertices)
        co = np.empty(n * 3)
        wk.data.foreach_get("co", co)
        self.edit_obj = edit_obj
        self.snap_obj = snap_obj
        self.write_key = write_key
        self.scope = scope
        self.scope_label = label
        self.mask_weights = mask_weights
        self.mask_label = ""
        if mask_weights is not None:
            self.mask_label = "painted mask only"
        self._dynamic_snap = mask_weights is not None
        self.mw = np.array(edit_obj.matrix_world)
        self.mwi = np.array(edit_obj.matrix_world.inverted())
        # sim the POSED surface (what the artist sees): other shape keys at
        # their current values - e.g. the Weight Cleanup mouth/eyes pose -
        # are added here and subtracted again on every write-back, so
        # refining while posed still edits the session's key correctly
        pose_off = pose_offset_local(edit_obj, exclude={write_key})
        P0 = self._affine(self.mw, co.reshape(-1, 3) + pose_off)

        topo = wrapmod.topology(me)
        self.boundary = wrapmod.boundary_verts(me)
        if mask_weights is None:
            frozen = (region_mask_frozen(edit_obj) if mh.wrap_use_region_mask
                      else np.zeros(n, dtype=bool))
        else:
            frozen = mask_weights <= 1e-4
        self.frozen = frozen
        locked = frozen.copy()
        if mh.brush_pin_boundary:
            locked |= self.boundary

        size = max(max(snap_obj.dimensions), 1e-6)
        self.size = size
        bvh = bvhmod.build_world_bvh(snap_obj, context.evaluated_depsgraph_get())
        # bending + ternary (anti-fold) links, walked on the clean basis mesh;
        # in masked cleanup mode only the painted region is linked
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        bm.verts.ensure_lookup_table()
        keep = None if mask_weights is None else (mask_weights > 1e-4)
        bend = _bending_springs(bm, BEND_DISTANCE, keep)
        tern = _ternary_links(bm, keep)
        bm.free()
        self.sim = _Sim(P0, topo, _quad_diagonals(me), bvh, locked,
                        max_dist=mh.wrap_maxdist_frac * size,
                        weights=mask_weights,
                        bend=bend, tern=tern, frozen=frozen)
        self.sim_symmap = self._symmetry_map(edit_obj, n)
        self._snap_sig = self._snap_signature()

        self.radius = max(mh.brush_radius * size, 1e-9)
        self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self.kd = None
        self.grab = None          # {"aff","w","P0","anchor","press","delta","normal","twin"}
        self.pins = []            # {"vidx","pos","aff","w","twin"}
        self.hover_pin = -1
        self.drag_pin = -1
        self._pin_drag = None
        self.hover_hit = None
        self.undo = []
        self.msg = ""
        self.ticks = 0
        self._region = context.region
        self._area = context.area
        self._n = n

        self._hud = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (self,), 'WINDOW', 'POST_PIXEL')
        self._ring = bpy.types.SpaceView3D.draw_handler_add(
            _draw_ring, (self,), 'WINDOW', 'POST_VIEW')
        self._timer = context.window_manager.event_timer_add(
            TICK, window=context.window)
        context.window_manager.modal_handler_add(self)
        _running = self
        mh.live_pause = False
        # Softwrap-style session display defaults: both meshes visible, the
        # wireframe cage drawn through the head (like its wire + in-front)
        if mh.view_mode != 'BOTH':
            mh.view_mode = 'BOTH'
        if not mh.cage_in_front:
            mh.cage_in_front = True
        self._area.tag_redraw()
        return {'RUNNING_MODAL'}

    # -------------------------------------------------------------- helpers --

    @staticmethod
    def _affine(M, pts):
        h = np.hstack([pts, np.ones((len(pts), 1))])
        return (h @ np.asarray(M).T)[:, :3]

    @staticmethod
    def _group_weights(obj, group_name):
        vg = obj.vertex_groups.get(group_name)
        if vg is None:
            return None
        weights = np.zeros(len(obj.data.vertices), dtype=float)
        gi = vg.index
        for i, v in enumerate(obj.data.vertices):
            for g in v.groups:
                if g.group == gi:
                    weights[i] = max(0.0, min(1.0, g.weight))
                    break
        return weights

    @staticmethod
    def _mode_object(context):
        if context.object and context.object.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass

    def _session_setup(self, context, cleanup_feature):
        mh = context.scene.mhfrt
        cage, target = mh.cage, mh.target
        if not cage or not target or cage == target:
            self.report({'ERROR'}, "Set Head Cage and Head Target first")
            return None

        if cleanup_feature == 'NONE':
            self._mode_object(context)
            me = cage.data
            wk = me.shape_keys.key_blocks.get(WRAPPED_KEY) if me.shape_keys else None
            if wk is None:
                self.report({'ERROR'}, "Wrap the cage first")
                return None
            return (cage, target, WRAPPED_KEY, None,
                    'WRAP', "LIVE SOFT SESSION")

        from .op_mouth import ensure_cleanup_pose
        spec = ensure_cleanup_pose(context, self, cleanup_feature,
                                   reset_amount=False)
        if spec is None:
            return None
        weights = self._group_weights(target, spec["group"])
        if weights is None or not np.any(weights > 1e-4):
            self.report(
                {'ERROR'},
                f"Paint some vertices into '{spec['group']}' before starting live cleanup",
            )
            return None
        return (target, cage, spec["key"], weights,
                f"CLEANUP_{cleanup_feature}",
                f"LIVE {spec['label'].upper()} CLEANUP")

    def _snap_signature(self):
        obj = getattr(self, "snap_obj", None)
        if obj is None:
            return None
        values = ()
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else ()
        if keys:
            values = tuple(round(float(k.value), 6) for k in keys)
        matrix = tuple(round(float(v), 6) for row in obj.matrix_world for v in row)
        return (obj.name, values, matrix)

    def _refresh_snap_bvh(self, context):
        if not getattr(self, "_dynamic_snap", False):
            return
        sig = self._snap_signature()
        if sig == self._snap_sig:
            return
        self.sim.bvh = bvhmod.build_world_bvh(
            self.snap_obj, context.evaluated_depsgraph_get())
        self.sim.T_ok[:] = False
        self._snap_sig = sig

    def _symmetry_map(self, cage, n):
        """vidx -> mirrored vidx across local X on the Basis shape (-1 = none,
        self = centreline). Used to mirror grabs and pins, never positions."""
        basis = get_basis_local(cage)
        kd = KDTree(n)
        for i, p in enumerate(basis):
            kd.insert(Vector(p), i)
        kd.balance()
        eps = 1e-3 * max(np.ptp(basis[:, 0]), 1e-9)
        m = np.full(n, -1, dtype=np.int64)
        for i, p in enumerate(basis):
            co, j, d = kd.find(Vector((-p[0], p[1], p[2])))
            if co is not None and d < eps:
                m[i] = j
        return m

    def _mirror_vec(self, v):
        """Mirror a world-space vector across the cage's local X."""
        lv = self.mwi[:3, :3] @ np.asarray(v, dtype=float)
        lv[0] = -lv[0]
        return self.mw[:3, :3] @ lv

    def _ensure_kd(self):
        if self.kd is None:
            P = self.sim.P
            kd = KDTree(len(P))
            for i, p in enumerate(P):
                kd.insert(Vector(p), i)
            kd.balance()
            self.kd = kd

    def _falloff_set(self, center, mh):
        """(indices, weights) of movable verts around a world point."""
        self._ensure_kd()
        found = self.kd.find_range(Vector(center), self.radius)
        idx, dist = [], []
        for _co, i, d in found:
            if self.frozen[i]:
                continue
            if mh.brush_pin_boundary and self.boundary[i]:
                continue
            idx.append(i)
            dist.append(d)
        if not idx:
            return None, None
        idx = np.array(idx, dtype=np.int64)
        d = np.array(dist) / self.radius
        w = ((1.0 - np.clip(d, 0.0, 1.0) ** 2) ** 2)[:, None]
        if self.mask_weights is not None:
            w *= self.mask_weights[idx, None]
        return idx, w

    def _hit(self, context):
        """Raycast the displayed editable mesh under the mouse."""
        region, rv3d = self._region, context.region_data
        obj = getattr(self, "edit_obj", None)
        if region is None or rv3d is None or not obj:
            return None
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse)
        mwi = obj.matrix_world.inverted()
        ro = mwi @ origin
        rd = mwi.to_3x3() @ direction
        if rd.length < 1e-12:
            return None
        rd.normalize()
        try:
            ok, loc, nor, _ = obj.ray_cast(
                ro, rd, depsgraph=context.evaluated_depsgraph_get())
        except RuntimeError:
            return None
        if not ok:
            return None
        mw = obj.matrix_world
        return (mw @ loc, (mw.to_3x3() @ nor).normalized() if nor else None)

    def _mouse_in_workspace(self, context, event):
        """True when the mouse is over our 3D view's WINDOW region and not on
        any overlapping panel (Softwrap's areas_under_mouse check)."""
        mx, my = event.mouse_x, event.mouse_y
        area = self._area
        if not (area.x <= mx < area.x + area.width
                and area.y <= my < area.y + area.height):
            return False
        for r in area.regions:
            if r.type in {'UI', 'TOOLS', 'HEADER', 'TOOL_HEADER'}:
                if r.x <= mx < r.x + r.width and r.y <= my < r.y + r.height:
                    return False
        return True

    def _push_undo(self):
        if len(self.undo) >= UNDO_MAX:
            self.undo.pop(0)
        pins = [dict(p, pos=p["pos"].copy()) for p in self.pins]
        self.undo.append((self.sim.snapshot(), pins))

    def _pop_undo(self, context):
        if not self.undo:
            self.msg = "Nothing to undo"
            return
        snap, pins = self.undo.pop()
        self.sim.restore(snap)
        self.pins = pins
        self.grab = None
        self.drag_pin = -1
        self._pin_drag = None
        self.kd = None
        self.msg = ""
        self._write(context)

    def _write(self, context):
        obj = getattr(self, "edit_obj", None)
        if not obj or not obj.data.shape_keys:
            return
        wk = obj.data.shape_keys.key_blocks.get(self.write_key)
        if wk is None:
            return
        # the sim runs on the POSED surface; store key data without the pose
        # (recomputed each write, so a pose change never leaves a stale shift)
        local = (self._affine(self.mwi, self.sim.P)
                 - pose_offset_local(obj, exclude={self.write_key}))
        wk.data.foreach_set("co", np.ascontiguousarray(local.reshape(-1)))
        obj.data.update()

    def _scale_radius(self, mh, fac):
        mh.brush_radius = min(0.5, max(0.005, mh.brush_radius * fac))
        self.radius = mh.brush_radius * self.size

    # ----------------------------------------------------------------- grab --

    def _grab_begin(self, context, mh):
        hit = self._hit(context)
        if hit is None:
            return False
        pos, normal = hit
        idx, w = self._falloff_set(pos, mh)
        if idx is None:
            return False
        self._push_undo()
        region, rv3d = self._region, context.region_data
        press = view3d_utils.region_2d_to_location_3d(region, rv3d,
                                                      self.mouse, pos)
        twin = None
        if mh.symmetry:
            m = self.sim_symmap[idx]
            keep = (m >= 0) & (m != idx)
            if keep.any():
                twin = {"aff": m[keep], "w": w[keep], "P0": self.sim.P[m[keep]].copy()}
        self.grab = {"aff": idx, "w": w, "P0": self.sim.P[idx].copy(),
                     "anchor": np.array(pos), "press": np.array(press),
                     "delta": np.zeros(3), "normal": normal, "twin": twin}
        return True

    def _grab_update(self, context):
        g = self.grab
        region, rv3d = self._region, context.region_data
        cur = view3d_utils.region_2d_to_location_3d(
            region, rv3d, self.mouse, Vector(g["anchor"]))
        g["delta"] = np.array(cur) - g["press"]

    def _grab_constraints(self):
        """Feed the current grab into the sim (called every tick)."""
        self.sim.grabs = []
        g = self.grab
        if g is None:
            return
        self.sim.grabs.append((g["aff"], g["w"], g["P0"] + g["delta"][None, :] * g["w"]))
        if g["twin"] is not None:
            t = g["twin"]
            md = self._mirror_vec(g["delta"])
            self.sim.grabs.append((t["aff"], t["w"], t["P0"] + md[None, :] * t["w"]))

    # ----------------------------------------------------------------- pins --

    def _pin_add(self, context, mh):
        hit = self._hit(context)
        if hit is None:
            self.msg = "Click on the cage to pin it"
            return False
        pos, _normal = hit
        self._ensure_kd()
        co, vidx, _d = self.kd.find(pos)
        if co is None:
            return False
        idx, w = self._falloff_set(self.sim.P[vidx], mh)
        if idx is None:
            self.msg = "Those verts are locked"
            return False
        self._push_undo()
        pin = {"vidx": int(vidx), "pos": self.sim.P[vidx].copy(),
               "aff": idx, "w": w, "twin": -1}
        self.pins.append(pin)
        if mh.symmetry:
            mv = int(self.sim_symmap[vidx])
            if mv >= 0 and mv != vidx:
                midx, mw_ = self._falloff_set(self.sim.P[mv], mh)
                if midx is not None:
                    tp = {"vidx": mv, "pos": self.sim.P[mv].copy(),
                          "aff": midx, "w": mw_, "twin": len(self.pins) - 1}
                    self.pins.append(tp)
                    pin["twin"] = len(self.pins) - 1
        self.msg = ""
        return True

    def _pin_hover(self, context):
        region, rv3d = self._region, context.region_data
        if rv3d is None:
            self.hover_pin = -1
            return
        best, best_d = -1, PIN_PICK_PX
        for k, pin in enumerate(self.pins):
            p2 = view3d_utils.location_3d_to_region_2d(region, rv3d,
                                                       Vector(pin["pos"]))
            if p2 is None:
                continue
            d = (p2 - self.mouse).length
            if d < best_d:
                best, best_d = k, d
        self.hover_pin = best

    def _pin_drag_begin(self, context):
        k = self.hover_pin
        self._push_undo()
        pin = self.pins[k]
        region, rv3d = self._region, context.region_data
        press = view3d_utils.region_2d_to_location_3d(
            region, rv3d, self.mouse, Vector(pin["pos"]))
        starts = {k: pin["pos"].copy()}
        if pin["twin"] >= 0:
            starts[pin["twin"]] = self.pins[pin["twin"]]["pos"].copy()
        self.drag_pin = k
        self._pin_drag = {"anchor": pin["pos"].copy(),
                          "press": np.array(press), "starts": starts}

    def _pin_drag_update(self, context):
        pd = self._pin_drag
        region, rv3d = self._region, context.region_data
        cur = view3d_utils.region_2d_to_location_3d(
            region, rv3d, self.mouse, Vector(pd["anchor"]))
        delta = np.array(cur) - pd["press"]
        md = self._mirror_vec(delta)
        for k, start in pd["starts"].items():
            self.pins[k]["pos"] = start + (delta if k == self.drag_pin else md)

    def _pin_delete_hovered(self):
        k = self.hover_pin
        if k < 0:
            self.msg = "Hover a pin to delete it"
            return
        self._push_undo()
        dead = {k}
        if self.pins[k]["twin"] >= 0:
            dead.add(self.pins[k]["twin"])
        remap = {}
        kept = []
        for i, p in enumerate(self.pins):
            if i not in dead:
                remap[i] = len(kept)
                kept.append(p)
        for p in kept:
            p["twin"] = remap.get(p["twin"], -1)
        self.pins = kept
        self.hover_pin = -1
        self.msg = ""

    def _pin_constraints(self):
        self.sim.pin_pulls = []
        for pin in self.pins:
            delta = pin["pos"] - self.sim.P[pin["vidx"]]
            self.sim.pin_pulls.append((pin["aff"], pin["w"], delta))

    # ----------------------------------------------------------------- tick --

    def _valid(self, context):
        mh = getattr(context.scene, "mhfrt", None)
        if mh is None:
            return False
        edit_obj = getattr(self, "edit_obj", None)
        snap_obj = getattr(self, "snap_obj", None)
        write_key = getattr(self, "write_key", WRAPPED_KEY)
        try:
            if (not edit_obj or not snap_obj
                    or len(edit_obj.data.vertices) != self._n
                    or not edit_obj.data.shape_keys
                    or write_key not in edit_obj.data.shape_keys.key_blocks
                    or edit_obj.mode != 'OBJECT'):
                return False
        except ReferenceError:
            return False
        return True

    def _tick(self, context):
        mh = context.scene.mhfrt
        self.ticks += 1
        # keep matrices fresh: the sim owns WORLD positions, so object moves
        # during a session do not corrupt the written shape key.
        self.mw = np.array(self.edit_obj.matrix_world)
        self.mwi = np.array(self.edit_obj.matrix_world.inverted())
        if not mh.live_pause:
            self._refresh_snap_bvh(context)
            self.sim.refresh_chunk()
            self._grab_constraints()
            self._pin_constraints()
            self.sim.step(mh.live_stiffness, mh.live_smooth,
                          mh.live_snap, mh.live_pin_force,
                          untangle=mh.live_untangle,
                          keep_outside=mh.live_keep_outside)
            self.kd = None                     # positions moved
            self._write(context)
        self._area.tag_redraw()

    # ---------------------------------------------------------------- modal --

    def modal(self, context, event):
        if _running is not self:               # stopped externally
            return {'FINISHED'}
        try:
            return self._modal(context, event)
        except Exception as e:
            self.report({'ERROR'}, f"Live session error: {e}")
            self.stop(context)
            return {'CANCELLED'}

    def _modal(self, context, event):
        mh = context.scene.mhfrt

        if not self._valid(context):
            self.stop(context)
            return {'FINISHED'}

        if event.type == 'TIMER':
            self._tick(context)
            return {'RUNNING_MODAL'} if (self.grab is not None
                                         or self.drag_pin >= 0) else {'PASS_THROUGH'}

        # active drags own the mouse
        if self.grab is not None or self.drag_pin >= 0:
            if event.type == 'MOUSEMOVE':
                self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                if self.grab is not None:
                    self._grab_update(context)
                else:
                    self._pin_drag_update(context)
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self.grab = None
                self.sim.grabs = []
                self.drag_pin = -1
                self._pin_drag = None
                return {'RUNNING_MODAL'}
            if event.type == 'ESC':            # abort the drag, not the session
                self._pop_undo(context)
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        # Alt+LMB / Alt+RMB and friends belong to the viewport, not the session
        if nav.is_navigation(event):
            return {'PASS_THROUGH'}

        inside = self._mouse_in_workspace(context, event)

        if event.type == 'MOUSEMOVE':
            self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            if inside:
                self._pin_hover(context)
                self.hover_hit = self._hit(context) if self.hover_pin < 0 else None
            else:
                self.hover_pin = -1
                self.hover_hit = None
            return {'PASS_THROUGH'}

        if not inside:
            return {'PASS_THROUGH'}

        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            if event.ctrl:
                self._scale_radius(mh, 1.15 if event.type == 'WHEELUPMOUSE'
                                   else 1 / 1.15)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        if event.value == 'PRESS':
            if event.type == 'LEFT_BRACKET':
                self._scale_radius(mh, 1 / 1.15)
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHT_BRACKET':
                self._scale_radius(mh, 1.15)
                return {'RUNNING_MODAL'}
            if event.type == 'S':
                mh.symmetry = not mh.symmetry
                return {'RUNNING_MODAL'}
            if event.type == 'B':
                mh.brush_pin_boundary = not mh.brush_pin_boundary
                self.sim.locked = self.frozen | (self.boundary
                                                 if mh.brush_pin_boundary
                                                 else False)
                # newly locked verts hold where they are NOW, no jumping back
                self.sim.P_lock = self.sim.P.copy()
                return {'RUNNING_MODAL'}
            if event.type == 'SPACE':
                mh.live_pause = not mh.live_pause
                return {'RUNNING_MODAL'}
            if event.type == 'Z' and event.ctrl:
                self._pop_undo(context)
                return {'RUNNING_MODAL'}
            if event.type in {'X', 'DEL'}:
                self._pin_delete_hovered()
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE':
                if self.hover_pin >= 0:
                    self._pin_drag_begin(context)
                    return {'RUNNING_MODAL'}
                if event.shift:
                    self._pin_add(context, mh)
                    return {'RUNNING_MODAL'}
                if self._grab_begin(context, mh):
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            if event.type in {'ESC', 'RET'}:
                self.stop(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    # ----------------------------------------------------------------- stop --

    def stop(self, context):
        """Tear the session down NOW (also called from other operators)."""
        global _running
        if _running is not self:
            return
        _running = None
        if getattr(self, "_hud", None):
            bpy.types.SpaceView3D.draw_handler_remove(self._hud, 'WINDOW')
            self._hud = None
        if getattr(self, "_ring", None):
            bpy.types.SpaceView3D.draw_handler_remove(self._ring, 'WINDOW')
            self._ring = None
        if getattr(self, "_timer", None):
            bpy.context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._valid(context):
            self._write(context)
        try:
            bpy.ops.ed.undo_push(message=self.scope_label.title())
        except Exception:
            pass
        try:
            self._area.tag_redraw()
        except ReferenceError:
            pass
        self.report({'INFO'},
                    f"{self.scope_label}: {self.ticks} frames, {len(self.pins)} pin(s)")


@handlers.persistent
def _load_pre(_scene):
    stop_running()


_classes = (MHFRT_OT_live_session,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    handlers.load_pre.append(_load_pre)


def unregister():
    stop_running()
    if _load_pre in handlers.load_pre:
        handlers.load_pre.remove(_load_pre)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
