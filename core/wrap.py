"""ZWrap-style non-rigid surface registration (numpy, vectorized).

Coarse-to-fine scheme: an outer loop refreshes closest-point correspondences
(filtered by distance and by cage/target normal agreement so lips don't grab
the chin and eyelids don't grab the brow), while an inner loop pulls the cage
toward those correspondences under a stiffness-annealed Laplacian that keeps
the MetaHuman quad flow even. Early stages are stiff (the global shape moves
as one), late stages are loose (fine detail); the smoothing turns tangential
once the shape has settled so the surface can relax without shrinking.
A final pass snaps every matched vertex exactly onto the target and polishes
edge flow.
"""

import numpy as np
from mathutils import Vector


def topology(mesh):
    """Edge/triangle index arrays used by the solver (built once per wrap)."""
    n = len(mesh.vertices)
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    deg = np.zeros(n)
    np.add.at(deg, ev[:, 0], 1.0)
    np.add.at(deg, ev[:, 1], 1.0)
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return {
        "n": n,
        "edges": ev,
        "degree": np.maximum(deg, 1.0)[:, None],
        "tris": tris.reshape(-1, 3),
    }


def boundary_verts(mesh):
    """Boolean (n,) mask of vertices on the mesh's open borders (neck seam,
    eye/mouth openings). An edge is a border when it doesn't have exactly two
    faces; wire and non-manifold edges are treated as borders too, which is
    the safe choice for pinning."""
    n = len(mesh.vertices)
    mask = np.zeros(n, dtype=bool)
    if not len(mesh.edges) or not len(mesh.loops):
        return mask
    le = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", le)
    counts = np.bincount(le, minlength=len(mesh.edges))
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    mask[ev[counts != 2].ravel()] = True
    return mask


def vertex_normals(P, topo):
    """Area-weighted vertex normals of the deforming cage."""
    n = topo["n"]
    tris = topo["tris"]
    fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    N = np.zeros((n, 3))
    for k in range(3):  # bincount: much faster than np.add.at
        col = tris[:, k]
        for ax in range(3):
            N[:, ax] += np.bincount(col, weights=fn[:, ax], minlength=n)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return N / ln


def neighbor_average(P, topo):
    n = topo["n"]
    e = topo["edges"]
    acc = np.empty_like(P)
    for ax in range(3):
        acc[:, ax] = (np.bincount(e[:, 0], weights=P[e[:, 1], ax], minlength=n)
                      + np.bincount(e[:, 1], weights=P[e[:, 0], ax], minlength=n))
    return acc / topo["degree"]


def _nearest_loop(P, bvh, max_dist, active):
    """Per-point fallback for plain ``mathutils`` BVH trees (no array API)."""
    n = len(P)
    T = P.copy()
    HN = np.zeros((n, 3))
    ok = np.zeros(n, dtype=bool)
    find = bvh.find_nearest
    indices = range(n) if active is None else np.flatnonzero(active).tolist()
    for i in indices:
        loc, nor, _idx, _d = find(Vector(P[i]), max_dist)
        if loc is None:
            continue
        T[i] = (loc.x, loc.y, loc.z)
        if nor is not None:
            HN[i] = (nor.x, nor.y, nor.z)
        ok[i] = True
    return T, ok, HN


def _reject_outliers(P, T, ok, max_dist):
    """Drop correspondences that are wildly farther than the rest of the mesh
    agreed on - the ear a jaw vertex grabbed across a gap, the shoulder a neck
    vertex found on a full-body target.

    Robust statistics (median + MAD), never a hard threshold, so a wrap that
    legitimately starts far away is not gutted on its first stage. The
    ``max_dist`` floor guarantees nothing close is ever thrown out.
    """
    idx = np.flatnonzero(ok)
    if len(idx) < 32:
        return ok
    d = np.linalg.norm(T[idx] - P[idx], axis=1)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    limit = max(med + 6.0 * (1.4826 * mad), 3.0 * med, 0.05 * max_dist)
    bad = idx[d > limit]
    if len(bad):
        ok = ok.copy()
        ok[bad] = False
    return ok


def correspondences(P, N, bvh, max_dist, normal_limit, active=None,
                    ray_guided=False, robust=False):
    """Closest surface point per vertex. Rejects hits farther than max_dist or
    whose surface normal disagrees with the cage normal (normal_limit <= -1
    disables the normal test - used for the final exact snap).

    `active` is an optional boolean mask of vertices worth querying. Both call
    sites already discard the result for pinned vertices (`ok & free`), so
    passing `free` here changes nothing about the outcome and simply skips the
    BVH work - which is the whole cost of this function. It matters most when a
    solve deliberately freezes most of the mesh, as the Weight Cleanup pose
    does.

    `ray_guided` gives every vertex the closest point could not serve - no hit
    in range, or a hit whose normal disagrees - a second chance along its own
    normal. On a dense head that is the difference between a fit and a mess:
    closest-point cheerfully pulls a lip across a sealed mouth or an eyelid
    through the brow above it, and a ray can only ever find the surface the
    vertex is actually facing. It costs ray casts for the leftovers only, so
    on a clean pair (where almost nothing is rejected) it costs nothing.
    """
    P = np.asarray(P, dtype=float)
    batch = getattr(bvh, "nearest_arrays", None)
    if batch is not None:
        T, ok, HN = batch(P, max_dist, active)
    else:
        T, ok, HN = _nearest_loop(P, bvh, max_dist, active)

    if normal_limit > -1.0:
        # a hit with no normal at all is accepted, as it always was
        has_n = (HN * HN).sum(axis=1) > 1e-18
        ok &= ((HN * N).sum(axis=1) >= normal_limit) | ~has_n

    if ray_guided:
        rays = getattr(bvh, "ray_arrays", None)
        if rays is not None:
            retry = ~ok if active is None else (active & ~ok)
            if retry.any():
                RT, rok, RN = rays(P, N, max_dist, retry)
                if normal_limit > -1.0:
                    rok &= (RN * N).sum(axis=1) >= normal_limit
                if rok.any():
                    T[rok] = RT[rok]
                    ok |= rok

    if robust:
        ok = _reject_outliers(P, T, ok, max_dist)
    return T, ok


def fill_correspondences(P, T, ok, topo, passes=4):
    """Give a vertex with no correspondence the displacement its matched
    neighbours agreed on, spreading a few edge rings at a time.

    Without this, a rejected vertex simply stays where the warp left it while
    everything around it snaps onto the head - which is exactly the pinched,
    spiky wrap a detailed target produces, because detail is what makes the
    normal test reject in the first place. The spread is bounded, so a region
    with no target anywhere near it (the back of the cage, a neck past the end
    of the head) still holds still instead of being dragged in.

    Returns ``(T, ok)`` with the newly filled vertices marked usable.
    """
    if not passes or ok.all() or not ok.any():
        return T, ok
    n = topo["n"]
    e = topo["edges"]
    D = np.zeros_like(P)
    D[ok] = T[ok] - P[ok]
    filled = ok.copy()
    for _ in range(int(passes)):
        w = filled.astype(float)
        cnt = (np.bincount(e[:, 0], weights=w[e[:, 1]], minlength=n)
               + np.bincount(e[:, 1], weights=w[e[:, 0]], minlength=n))
        new = (~filled) & (cnt > 0.0)
        if not new.any():
            break
        acc = np.empty((n, 3))
        for ax in range(3):
            wd = w * D[:, ax]
            acc[:, ax] = (np.bincount(e[:, 0], weights=wd[e[:, 1]], minlength=n)
                          + np.bincount(e[:, 1], weights=wd[e[:, 0]], minlength=n))
        D[new] = acc[new] / cnt[new][:, None]
        filled |= new
        if filled.all():
            break
    T = P + D
    return T, filled


def register_surface(P0, topo, bvh, pins, *, stages=8, inner=10,
                     stiff_hi=0.92, stiff_lo=0.18, step=0.6,
                     max_dist=1.0, normal_limit=0.10,
                     dist_taper=1.0, snap_normal_limit=-2.0,
                     ray_guided=False, fill_passes=0, robust=False,
                     progress=None):
    """Non-rigid registration of the point cloud P0 onto the BVH surface.

    pins: {vertex_index: position} hard constraints (landmark anchors and
    Region_Mask verts that must keep their warped shape).
    progress: optional callable(step_index); total steps = stages + 3.

    The four keywords below are the high-resolution hardening; they all
    default OFF so every existing caller solves exactly as it did before, and
    the head wrap turns them on:

    dist_taper        end the anneal searching only ``taper * max_dist``, so
                      late stages cannot reach across to a wrong surface
    snap_normal_limit normal agreement still required by the final exact snap
                      (-2 = none, the old behaviour: on a dense head that let
                      a vertex snap to the *inside* of the mouth or eyelid)
    ray_guided        retry rejected vertices along their own normal
    fill_passes       diffuse the matched displacement into the vertices that
                      still have no correspondence
    robust            drop statistical outlier correspondences
    """
    P = np.asarray(P0, dtype=float).copy()
    n = topo["n"]
    pin_idx = (np.fromiter(pins.keys(), dtype=np.int64)
               if pins else np.empty(0, np.int64))
    pin_pos = (np.array([pins[int(i)] for i in pin_idx], dtype=float)
               if len(pin_idx) else np.empty((0, 3)))
    free = np.ones(n, dtype=bool)
    free[pin_idx] = False

    stages = max(1, int(stages))
    taper = min(max(float(dist_taper), 0.05), 1.0)
    for s in range(stages):
        t = s / max(stages - 1, 1)
        stiff = stiff_hi * (stiff_lo / stiff_hi) ** t  # geometric anneal
        dist = max_dist * (1.0 - (1.0 - taper) * t)
        N = vertex_normals(P, topo)
        T, ok = correspondences(P, N, bvh, dist, normal_limit, active=free,
                                # the first stage is still a long reach from
                                # the warp; rays only start paying once the
                                # cage is roughly on the head
                                ray_guided=ray_guided and s >= 1,
                                robust=robust)
        if fill_passes:
            T, ok = fill_correspondences(P, T, ok, topo, fill_passes)
        pull = ok & free
        # the first stages may shrink slightly to untangle folds; after that,
        # smoothing is restricted to the tangent plane so the surface relaxes
        # without losing volume
        tangential = s >= 2
        for _ in range(max(1, int(inner))):
            P[pull] += step * (T[pull] - P[pull])
            d = neighbor_average(P, topo) - P
            if tangential:
                d -= (d * N).sum(axis=1, keepdims=True) * N
            P[free] += stiff * d[free]
            if len(pin_idx):
                P[pin_idx] = pin_pos
        if progress:
            progress(s + 1)

    # exact snap + tangential polish passes to even out the quads
    snap_dist = max_dist * taper
    for k in range(3):
        N = vertex_normals(P, topo)
        T, ok = correspondences(P, N, bvh, snap_dist, snap_normal_limit,
                                active=free, ray_guided=ray_guided,
                                robust=robust)
        if fill_passes:
            T, ok = fill_correspondences(P, T, ok, topo, fill_passes)
        snap = ok & free
        P[snap] = T[snap]
        if len(pin_idx):
            P[pin_idx] = pin_pos
        if k < 2:
            d = neighbor_average(P, topo) - P
            d -= (d * N).sum(axis=1, keepdims=True) * N
            P[free] += 0.5 * d[free]
        if progress:
            progress(stages + k + 1)
    return P
