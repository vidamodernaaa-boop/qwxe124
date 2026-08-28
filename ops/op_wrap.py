"""Operator: wrap the cage onto the target head using the landmark pairs.

Pipeline:  A) Procrustes similarity -> B) Thin-Plate-Spline -> C) non-rigid
surface registration (ZWrap-style coarse-to-fine, see core.wrap).

Output is stored as a 'Wrapped' SHAPE KEY. The 'Basis' shape key always holds
the FIRST (original) cage shape and is never overwritten, so re-wrapping
recomputes from the first shape and you can always slide back to it. Skeleton
fitting reads both Basis (rest) and Wrapped (final) coordinates from here.
"""

import time

import numpy as np
import bpy

from ..core import align, tps, bvh as bvhmod, wrap as wrapmod
from ..core import landmarks as lmdata
from ..core import organization

REST_ATTR = "mhfrt_rest"           # legacy (pre-shape-key) storage, migrated below
SOLVER_ATTR = "mhfrt_solver_wrap"  # last RAW solver output (local coords) -
                                   # the baseline hand refinements are measured
                                   # against, so re-wrapping can keep them
REGION_MASK_GROUP = "Region_Mask"
WRAPPED_KEY = "Wrapped"
MOUTH_OPEN_KEY = "MouthOpen"        # authored on Ada (cage) / sculpted on target
CLOSE_EYES_KEY = "CloseEyes"        # authored on Ada (cage) / sculpted on target
MOUTH_CLEAN_GROUP = "MHFRT_Mouth_Cleanup"
EYES_CLEAN_GROUP = "MHFRT_Eyes_Cleanup"

# ZWrap-style presets: (outer correspondence stages, inner relax iterations,
# stiffness anneal range, pull step, normal agreement threshold)
PRESETS = {
    'DRAFT':    dict(stages=4,  inner=6,  stiff_hi=0.85, stiff_lo=0.30, step=0.70, normal_limit=0.00),
    'BALANCED': dict(stages=8,  inner=10, stiff_hi=0.92, stiff_lo=0.18, step=0.60, normal_limit=0.10),
    'HIGH':     dict(stages=14, inner=12, stiff_hi=0.95, stiff_lo=0.10, step=0.50, normal_limit=0.15),
}


# --------------------------------------------------------------- helpers ---

def region_mask_frozen(obj):
    """Boolean (n,) array of cage verts in the 'Region_Mask' vertex group."""
    n = len(obj.data.vertices)
    frozen = np.zeros(n, dtype=bool)
    vg = obj.vertex_groups.get(REGION_MASK_GROUP)
    if vg is None:
        return frozen
    gi = vg.index
    for i, v in enumerate(obj.data.vertices):
        for g in v.groups:
            if g.group == gi and g.weight > 0.0:
                frozen[i] = True
                break
    return frozen


def _apply_affine(M, pts):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ np.asarray(M).T)[:, :3]


def _read_key(kb, n):
    a = np.empty(n * 3)
    kb.data.foreach_get("co", a)
    return a.reshape(n, 3)


def _write_key(kb, coords):
    kb.data.foreach_set("co", np.ascontiguousarray(coords.reshape(-1)))


def ensure_basis(cage):
    """Guarantee a Basis shape key holding the FIRST shape; migrate legacy rest."""
    me = cage.data
    n = len(me.vertices)
    if me.shape_keys is None:
        attr = me.attributes.get(REST_ATTR)
        if attr is not None and len(attr.data) == n:
            co = np.empty(n * 3)
            attr.data.foreach_get("vector", co)
            me.vertices.foreach_set("co", co)   # restore original before snapshotting Basis
            me.update()
        if attr is not None:
            me.attributes.remove(attr)
        cage.shape_key_add(name="Basis", from_mix=False)
    return me.shape_keys.key_blocks[0]


def get_basis_local(cage):
    me = cage.data
    n = len(me.vertices)
    if me.shape_keys:
        return _read_key(me.shape_keys.key_blocks[0], n)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def get_wrapped_local(cage):
    me = cage.data
    n = len(me.vertices)
    if me.shape_keys and WRAPPED_KEY in me.shape_keys.key_blocks:
        return _read_key(me.shape_keys.key_blocks[WRAPPED_KEY], n)
    return None


def pose_offset_local(obj, exclude=frozenset({WRAPPED_KEY})):
    """Per-vertex local offset added by every non-Basis shape key at its
    CURRENT value, except `exclude`. The live tools sim/brush the POSED
    surface (what the artist actually sees - e.g. the Weight Cleanup
    mouth-open pose) and subtract this again before writing back into their
    session key, so refining while posed edits the right data."""
    me = obj.data
    n = len(me.vertices)
    out = np.zeros((n, 3))
    if not me.shape_keys:
        return out
    kb = me.shape_keys.key_blocks
    basis = _read_key(kb[0], n)
    buf = np.empty(n * 3)
    for k in kb[1:]:
        if k.name in exclude or abs(k.value) <= 1e-9:
            continue
        k.data.foreach_get("co", buf)
        out += k.value * (buf.reshape(n, 3) - basis)
    return out


def _mute_cleanup_pose(cage, target):
    """Zero the Weight Cleanup guide keys on both meshes. The wrap solver
    must see the NEUTRAL pair - registering the neutral cage onto an
    open-mouthed head would corrupt the neutral wrap. Returns True when a
    pose was actually showing."""
    posed = False
    for obj in (cage, target):
        if obj is None or obj.type != 'MESH' or obj.data.shape_keys is None:
            continue
        keys = obj.data.shape_keys.key_blocks
        touched = False
        for name in (MOUTH_OPEN_KEY, CLOSE_EYES_KEY):
            kb = keys.get(name)
            if kb is not None and abs(kb.value) > 1e-6:
                kb.value = 0.0
                touched = True
        if touched:
            obj.data.update()
            posed = True
    return posed


def _read_solver_result(me):
    """The previous RAW solver output, or None (legacy file / first wrap)."""
    attr = me.attributes.get(SOLVER_ATTR)
    n = len(me.vertices)
    if attr is None or attr.domain != 'POINT' or len(attr.data) != n:
        return None
    out = np.empty(n * 3)
    attr.data.foreach_get("vector", out)
    return out.reshape(n, 3)


def _store_solver_result(me, coords):
    attr = me.attributes.get(SOLVER_ATTR)
    if attr is not None and (attr.domain != 'POINT'
                             or attr.data_type != 'FLOAT_VECTOR'):
        me.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.attributes.new(SOLVER_ATTR, 'FLOAT_VECTOR', 'POINT')
    attr.data.foreach_set("vector", np.ascontiguousarray(coords.reshape(-1)))


# High-resolution hardening, applied to every quality (see core.wrap).
# These are what make a real scan or a subdivided sculpt wrap like the clean
# test heads do: the search narrows as the fit settles, rejected vertices get
# a second chance along their own normal, whatever is still unmatched follows
# its neighbours instead of staying behind as a spike, and the final exact
# snap is no longer allowed to grab a back-facing surface (the inside of the
# mouth, the underside of an eyelid) just because it happens to be closest.
_HIRES = dict(
    dist_taper=0.45,
    snap_normal_limit=-0.35,
    ray_guided=True,
    fill_passes=4,
    robust=True,
)


def _solver_params(mh):
    if mh.wrap_quality != 'CUSTOM':
        params = dict(PRESETS[mh.wrap_quality])
    else:
        params = dict(
            stages=max(2, min(24, mh.wrap_iterations // 10)),
            inner=10,
            stiff_hi=0.92,
            stiff_lo=min(0.60, 0.06 + 0.5 * mh.wrap_smooth),
            step=mh.wrap_step,
            normal_limit=0.10,
        )
    params.update(_HIRES)
    return params


# ------------------------------------------------------------- operators ---

class MHFRT_OT_wrap(bpy.types.Operator):
    bl_idname = "mhfrt.wrap"
    bl_label = "Wrap Head Cage to Head"
    bl_description = ("Align + warp the cage onto the target head (coarse-to-fine "
                      "surface registration). Result is a 'Wrapped' shape key; "
                      "'Basis' keeps the original shape. Re-wrapping keeps your "
                      "hand refinements (brushes, live session) on top of the "
                      "new solve")
    bl_options = {'REGISTER', 'UNDO'}

    fresh: bpy.props.BoolProperty(
        name="Fresh Wrap",
        description="Discard hand refinements (brushes, live session) and "
                    "recompute the wrap purely from the landmarks",
        default=False,
        options={'SKIP_SAVE'},
    )

    def execute(self, context):
        from .op_live import stop_running
        stop_running()          # the solver owns the Wrapped key now
        mh = context.scene.mhfrt
        cage, target = mh.cage, mh.target
        if not cage or not target or cage == target:
            self.report({'ERROR'}, "Set Head Cage and Head Target first")
            return {'CANCELLED'}

        lmdata.migrate_legacy(context)
        pairs = lmdata.wrap_pairs(mh)
        if len(pairs) < 3:
            self.report({'ERROR'},
                        f"Need at least 3 complete landmark pairs (have {len(pairs)})")
            return {'CANCELLED'}

        # Visual-only: remember the shape on screen right now, so the finished
        # wrap can morph from it instead of snapping (see ui/transition.py).
        # Pure numpy read - the wrap math below is untouched by it.
        from ..ui import transition as vtrans
        cage_prev = vtrans.mesh_snapshot(cage)

        # a Weight Cleanup pose may be showing - solve on the NEUTRAL pair,
        # then re-sync the pose from the sliders after the wrap (the head's
        # pose key regenerates against the new wrap automatically)
        posed = _mute_cleanup_pose(cage, target)

        t0 = time.perf_counter()
        me = cage.data
        n = len(me.vertices)

        basis = ensure_basis(cage)            # first shape, never overwritten
        rest_local = _read_key(basis, n)
        mw = np.array(cage.matrix_world)
        mwi = np.array(cage.matrix_world.inverted())
        rest_world = _apply_affine(mw, rest_local)

        src = np.array([list(s) for s, _t in pairs], dtype=float)
        dst = np.array([list(t) for _s, t in pairs], dtype=float)

        # A) Procrustes similarity
        s, R, t = align.similarity_transform(src, dst)
        A = align.apply_similarity(rest_world, s, R, t)
        src_a = align.apply_similarity(src, s, R, t)

        # B) Thin-Plate-Spline (needs >= 4 points)
        B = A
        if len(pairs) >= 4:
            try:
                model = tps.tps_fit(src_a, dst, reg=1e-6)
                B = tps.tps_apply(model, A)
            except np.linalg.LinAlgError:
                self.report({'WARNING'}, "TPS solve failed; used affine align only")
                B = A

        # C) non-rigid surface registration
        used_reg = False
        n_frozen = 0
        n_faces = -1
        if mh.wrap_use_icp:
            # The solver registers the cage against the head's EVALUATED
            # surface, so a head still deformed by a posed rig - the artist's
            # body armature parked on an animation frame, or our own board mid
            # expression - would be wrapped onto that pose and the pose would be
            # baked into the cage as if it were the neutral shape. Held at rest
            # for the solve only; the pose itself is never touched and comes
            # straight back (core.organization.meshes_at_rest).
            with organization.meshes_at_rest(
                    cage, target, view_layer=context.view_layer) as rested:
                if rested:
                    self.report({'INFO'},
                                "Solved against the rest pose of "
                                + ", ".join(rested))
                depsgraph = context.evaluated_depsgraph_get()
                # No mesh copy: queries run against the BVH Blender already
                # keeps on the evaluated head, so a multi-million-face target
                # costs nothing to set up (it used to be duplicated into a
                # BMesh first, which is what crashed Blender on press).
                tree = bvhmod.build_world_bvh(target, depsgraph)
                n_faces = bvhmod.evaluated_face_count(target, depsgraph)
                if getattr(tree, "approximate", False):
                    # only reachable on a head that is BOTH non-uniformly
                    # scaled and too dense for the exact world-space tree
                    self.report({'WARNING'},
                                "Head has a non-uniform scale and is too dense "
                                "for an exact fit - apply its scale "
                                "(Ctrl+A > Scale) and wrap again")
                topo = wrapmod.topology(me)

                pins = {}
                if mh.wrap_use_region_mask:
                    frozen = region_mask_frozen(cage)
                    for i in np.nonzero(frozen)[0]:
                        pins[int(i)] = B[int(i)]
                    n_frozen = int(frozen.sum())
                if mh.wrap_pin_landmarks:
                    for q in dst:
                        i = int(((B - q) ** 2).sum(axis=1).argmin())
                        pins[i] = q

                # Scale reference: the WARPED CAGE, never the target object.
                # `target.dimensions` measures the whole mesh the artist
                # picked - on a full-body character that is metres, and 15%
                # of metres is a search radius wide enough for a jaw vertex
                # to find a shoulder. After the TPS warp the cage is already
                # sitting on the head, so its own extent is the only ruler
                # that means "head-sized" no matter what the target contains.
                span = float(np.ptp(B, axis=0).max()) if len(B) else 0.0
                if span <= 0.0:
                    span = float(max(target.dimensions)) or 1.0
                max_dist = mh.wrap_maxdist_frac * span
                params = _solver_params(mh)

                wm = context.window_manager
                total = params["stages"] + 3
                wm.progress_begin(0, total)
                try:
                    B = wrapmod.register_surface(
                        B, topo, tree, pins,
                        max_dist=max_dist,
                        progress=wm.progress_update,
                        **params,
                    )
                finally:
                    wm.progress_end()
            used_reg = True

        # store the result in the 'Wrapped' shape key (Basis stays = first shape)
        solver_local = _apply_affine(mwi, B)
        local = solver_local

        # ---- refinement layer ------------------------------------------------
        # Hand refinements (brushes, live session) live only in the Wrapped
        # key, which is about to be overwritten. Lift the artist's tweaks off
        # the PREVIOUS solver result and re-apply them on top of the new one,
        # so a re-wrap updates the fit without throwing hand work away. With
        # no edits in between the delta is zero and nothing changes.
        wk = me.shape_keys.key_blocks.get(WRAPPED_KEY)
        tweaked = 0
        if not self.fresh and wk is not None:
            prev_solver = _read_solver_result(me)
            if prev_solver is not None:
                delta = _read_key(wk, n) - prev_solver
                tweaked = int((np.abs(delta).max(axis=1) > 1e-9).sum())
                if tweaked:
                    local = solver_local + delta
        _store_solver_result(me, solver_local)

        if wk is None:
            wk = cage.shape_key_add(name=WRAPPED_KEY, from_mix=False)
        _write_key(wk, local)
        wk.slider_min, wk.slider_max = 0.0, 1.0
        wk.value = 1.0
        cage.active_shape_key_index = me.shape_keys.key_blocks.find(WRAPPED_KEY)
        me.update()

        dt = time.perf_counter() - t0
        quality = mh.wrap_quality.title() if used_reg else "warp only"
        extra = f", {n_frozen} Region_Mask frozen" if n_frozen else ""
        if n_faces >= 100_000:
            extra += f", {n_faces / 1_000_000.0:.1f}M-face head"
        if tweaked:
            extra += f", kept your refinements on {tweaked} verts"
        elif self.fresh:
            extra += ", refinements discarded"
        self.report({'INFO'},
                    f"Wrapped in {dt:.1f}s - {len(pairs)} pairs, {quality}{extra}")
        # bring back the cleanup pose the artist was looking at - regenerated
        # against the wrap that was just computed
        if posed:
            from . import op_mouth
            for key_name, amount in ((MOUTH_OPEN_KEY, mh.mouth_open_amount),
                                     (CLOSE_EYES_KEY, mh.eyes_close_amount)):
                if amount > 1e-6:
                    op_mouth.apply_cleaning_amount(context, key_name, amount)
        # stay on the Wrap tab (refinement happens here) but unfold the brushes
        mh.ui_sec_refine = True
        # last act before returning (transition contract): morph the viewport
        # from the pre-wrap shape to the result that was just computed
        vtrans.request(context, mesh=(cage, cage_prev))
        return {'FINISHED'}


class MHFRT_OT_wrap_reset(bpy.types.Operator):
    bl_idname = "mhfrt.wrap_reset"
    bl_label = "Show First Shape"
    bl_description = "Slide the cage back to its original (Basis) shape, non-destructively"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .op_live import stop_running
        stop_running()
        mh = context.scene.mhfrt
        cage = mh.cage
        if not cage:
            self.report({'ERROR'}, "No cage set")
            return {'CANCELLED'}
        me = cage.data
        if me.shape_keys and WRAPPED_KEY in me.shape_keys.key_blocks:
            from ..ui import transition as vtrans
            cage_prev = vtrans.mesh_snapshot(cage)
            me.shape_keys.key_blocks[WRAPPED_KEY].value = 0.0
            me.update()
            self.report({'INFO'}, "Showing first shape (Basis)")
            vtrans.request(context, mesh=(cage, cage_prev))
            return {'FINISHED'}
        self.report({'WARNING'}, "Cage not wrapped yet")
        return {'CANCELLED'}


_classes = (MHFRT_OT_wrap, MHFRT_OT_wrap_reset)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
