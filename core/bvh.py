"""Nearest-surface queries against the target head, at ANY resolution.

The old implementation copied the target's evaluated mesh into a BMesh and
handed that to ``BVHTree.FromBMesh``.  That is fine on a 6k cage and fatal on
a real production head: a scan or a subdivided sculpt is a few million
triangles, a BMesh costs hundreds of bytes per element, and when the
allocation could not be served Blender did not raise - it walked off the end
of the buffer inside ``BM_mesh_bm_from_me`` and took the whole session down
(EXCEPTION_ACCESS_VIOLATION the moment Wrap was pressed).

No BMesh is built here, by either backend:

``SurfaceQuery``  queries the BVH Blender already caches on the evaluated
                  mesh, through ``Object.closest_point_on_mesh`` / ``ray_cast``.
                  Nothing is copied and nothing is built, so a five-million-
                  triangle head costs exactly as much to set up as a
                  five-thousand one: nothing.  Those queries minimise distance
                  in the object's LOCAL space, which equals world space for
                  any uniformly scaled object - i.e. virtually every head.

``TreeQuery``     a real world-space ``BVHTree`` assembled straight from numpy
                  vertex/triangle arrays (``FromPolygons``, ~0.5 s per million
                  triangles and no BMesh anywhere).  Used when the target's
                  transform is non-uniform or sheared, where local-space
                  distance is genuinely the wrong measure.

Both expose the same world-space contract as ``mathutils``' BVHTree -
``find_nearest`` / ``ray_cast`` - plus the array methods the wrap solver runs
on, which lift the world<->local transforms out of the per-vertex Python loop
and into one numpy pass.
"""

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# mathutils' own "no limit" sentinel is ~1.8e19; anything at or above this is
# treated as "search the whole mesh".
FAR = 1.0e18

_MISS = (None, None, None, None)

# Above this the exact world-space tree is not worth building: the transient
# Python lists ``FromPolygons`` needs would be hundreds of megabytes. Such a
# target falls back to the zero-copy object queries, which are exact anyway
# unless its transform is also non-uniform.
TRI_BUDGET = 1_500_000

# Surface-walk steps used to pull a LOCAL-space closest point toward the
# WORLD-space one when neither exact path is available.
_REFINE_PASSES = 3


def _affine(M, pts):
    """Apply a 4x4 (numpy) to an (n,3) array of points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return (np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(M).T)[:, :3]


class _Query:
    """Array queries expressed on top of a single-point backend."""

    #: True when nearest-point answers are approximate (see SurfaceQuery)
    approximate = False

    def nearest_arrays(self, P, max_dist=FAR, active=None):
        """``(T, ok, Nrm)`` - hit position (misses keep their input), a mask,
        and the world hit normal, for many world points at once."""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        n = len(P)
        T, Nrm = P.copy(), np.zeros((n, 3))
        ok = np.zeros(n, dtype=bool)
        find = self.find_nearest
        idxs = range(n) if active is None else np.flatnonzero(active).tolist()
        for i in idxs:
            loc, nor, _idx, _d = find(Vector(P[i]), max_dist)
            if loc is None:
                continue
            T[i] = (loc.x, loc.y, loc.z)
            if nor is not None:
                Nrm[i] = (nor.x, nor.y, nor.z)
            ok[i] = True
        return T, ok, Nrm

    def ray_arrays(self, P, D, max_dist=FAR, active=None, both_ways=True):
        """Ray hits along per-point world directions ``D``; same returns.

        Shooting along the cage's own normal is what keeps a dense head
        honest: a closest-point query on a detailed surface happily grabs the
        eyelid from under the brow or the far lip across a sealed mouth, while
        a ray along the normal can only find the surface the vertex is
        actually looking at.
        """
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        D = np.asarray(D, dtype=float).reshape(-1, 3)
        n = len(P)
        T, Nrm = P.copy(), np.zeros((n, 3))
        ok = np.zeros(n, dtype=bool)
        if n == 0:
            return T, ok, Nrm
        # start behind the point so a vertex sitting exactly on the surface,
        # or just inside it, still registers a hit
        back = max_dist * 0.5 if max_dist < FAR else 0.0
        origins = P - D * back
        cast = self.ray_cast
        span = max_dist * 1.5 if max_dist < FAR else FAR
        idxs = range(n) if active is None else np.flatnonzero(active).tolist()
        for i in idxs:
            d = Vector(D[i])
            if d.length < 1e-12:
                continue
            d.normalize()
            o = Vector(origins[i])
            loc, nor, _idx, _dist = cast(o, d, span)
            if loc is None and both_ways:
                loc, nor, _idx, _dist = cast(o, -d, span)
            if loc is None:
                continue
            if max_dist < FAR and (loc - Vector(P[i])).length > max_dist:
                continue
            T[i] = (loc.x, loc.y, loc.z)
            if nor is not None:
                Nrm[i] = (nor.x, nor.y, nor.z)
            ok[i] = True
        return T, ok, Nrm


class TreeQuery(_Query):
    """Exact world-space queries against a real ``mathutils`` BVHTree."""

    def __init__(self, tree):
        self.tree = tree

    def find_nearest(self, co, distance=FAR):
        return self.tree.find_nearest(Vector(co), distance)

    def ray_cast(self, origin, direction, distance=FAR):
        return self.tree.ray_cast(Vector(origin), Vector(direction), distance)

    def find_nearest_range(self, co, distance=FAR):
        return self.tree.find_nearest_range(Vector(co), distance)


class SurfaceQuery(_Query):
    """Zero-copy queries against an object's evaluated mesh."""

    def __init__(self, obj, depsgraph=None):
        self.obj = obj
        self._dg = depsgraph
        self.refresh_matrix()

    # -- transforms ---------------------------------------------------------

    def refresh_matrix(self):
        """Re-read the object transform (call after the object moved)."""
        mw = self.obj.matrix_world.copy()
        self._mw = mw
        self._mwi = mw.inverted_safe()
        self._mw_np = np.array(mw)
        self._mwi_np = np.array(self._mwi)
        self._rot = mw.to_3x3()
        self._rot_inv = self._mwi.to_3x3()
        # normals transform by the inverse transpose (non-uniform scale)
        self._nmat = self._rot_inv.transposed()
        self._nmat_np = np.array(self._nmat)
        # Conservative world->local distance factor: the SMALLEST axis scale,
        # so a world radius always maps to a local radius that is big enough.
        # Hits are re-checked against the real world distance afterwards, so
        # searching a little too far is free; searching too little would
        # silently drop correspondences on a scaled rig.
        cols = [self._rot.col[i] for i in range(3)]
        lens = [c.length for c in cols]
        self._scale_min = max(min(lens), 1e-12)
        # `closest_point_on_mesh` minimises distance in LOCAL space, which is
        # the same point as in world space only under a similarity transform.
        # Anything else (non-uniform scale, shear) gets the refinement walk
        # below and is reported as approximate.
        unit = [c / max(l, 1e-12) for c, l in zip(cols, lens)]
        ortho = max(abs(unit[0].dot(unit[1])), abs(unit[0].dot(unit[2])),
                    abs(unit[1].dot(unit[2])))
        self.approximate = (max(lens) > min(lens) * (1.0 + 1e-4)
                            or ortho > 1e-4)

    def _local_radius(self, distance):
        if distance is None or distance >= FAR:
            return FAR
        return float(distance) / self._scale_min

    def _world_normal(self, nor):
        if nor is None:
            return None
        n = self._nmat @ nor
        ln = n.length
        return n / ln if ln > 1e-12 else n

    def _world_normals(self, nors):
        wn = np.asarray(nors, dtype=float) @ self._nmat_np.T
        ln = np.linalg.norm(wn, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        return wn / ln

    # -- depsgraph lifetime -------------------------------------------------
    # Unlike a copied tree, these queries need the depsgraph to still be alive
    # at CALL time, and a modal tool can outlive the one it was handed (a file
    # load, an undo that rebuilds the graph). Every query re-acquires from the
    # context once before giving up, so a stale handle costs one wasted call
    # rather than the tool going quietly blind.

    def refresh(self, depsgraph=None):
        """Point the queries at a current depsgraph and re-read the matrix."""
        self._dg = depsgraph if depsgraph is not None else self._live_dg()
        self.refresh_matrix()

    @staticmethod
    def _live_dg():
        try:
            return bpy.context.evaluated_depsgraph_get()
        except (AttributeError, RuntimeError):
            return None

    def _recover(self):
        dg = self._live_dg()
        if dg is None or dg == self._dg:
            return False
        self._dg = dg
        return True

    # -- single-point API (mathutils BVHTree compatible) --------------------

    def find_nearest(self, co, distance=FAR):
        """(world location, world normal, poly index, distance) or 4x None."""
        origin = self._mwi @ Vector(co)
        radius = self._local_radius(distance)
        try:
            hit, loc, nor, idx = self.obj.closest_point_on_mesh(
                origin, distance=radius, depsgraph=self._dg)
        except (RuntimeError, ReferenceError, TypeError, SystemError):
            if not self._recover():
                return _MISS
            try:
                hit, loc, nor, idx = self.obj.closest_point_on_mesh(
                    origin, distance=radius, depsgraph=self._dg)
            except (RuntimeError, ReferenceError, TypeError, SystemError):
                return _MISS
        if not hit:
            return _MISS
        world = self._mw @ loc
        normal = self._world_normal(nor)
        if self.approximate:
            world, normal = self._refine_one(Vector(co), world, normal)
        d = (world - Vector(co)).length
        if distance < FAR and d > distance:
            return _MISS
        return world, normal, idx, d

    def ray_cast(self, origin, direction, distance=FAR):
        """(world location, world normal, poly index, distance) or 4x None."""
        ro = self._mwi @ Vector(origin)
        rd = self._rot_inv @ Vector(direction)
        if rd.length < 1e-12:
            return _MISS
        rd.normalize()
        try:
            hit, loc, nor, idx = self.obj.ray_cast(
                ro, rd, distance=self._local_radius(distance),
                depsgraph=self._dg)
        except (RuntimeError, ReferenceError, TypeError, SystemError):
            return _MISS
        if not hit:
            return _MISS
        world = self._mw @ loc
        d = (world - Vector(origin)).length
        if distance < FAR and d > distance:
            return _MISS
        return world, self._world_normal(nor), idx, d

    # -- world-space refinement (non-similarity transforms only) ------------

    def _refine_one(self, p, world, normal):
        best, bestn = Vector(world), normal
        bestd = (p - best).length
        cpm = self.obj.closest_point_on_mesh
        for _ in range(_REFINE_PASSES):
            if bestn is None:
                break
            r = p - best
            rt = r - r.dot(bestn) * bestn
            if rt.length <= 1e-9:
                break
            try:
                hit, loc, nor, _i = cpm(self._mwi @ (best + rt),
                                        depsgraph=self._dg)
            except (RuntimeError, ReferenceError, TypeError, SystemError):
                break
            if not hit:
                break
            cand = self._mw @ loc
            d = (p - cand).length
            if d >= bestd - 1e-12:
                break
            best, bestn, bestd = cand, self._world_normal(nor), d
        return best, bestn

    def _refine_world(self, P, world, wn):
        """Batch version: slide each hit along the surface by the tangential
        part of the world residual and re-project, keeping a candidate only
        when it is genuinely closer."""
        cpm = self.obj.closest_point_on_mesh
        dg = self._dg
        best, bestn = world.copy(), wn.copy()
        bestd = np.linalg.norm(P - best, axis=1)
        for _ in range(_REFINE_PASSES):
            r = P - best
            rt = r - (r * bestn).sum(axis=1, keepdims=True) * bestn
            live = np.flatnonzero(np.linalg.norm(rt, axis=1) > 1e-9)
            if not len(live):
                break
            probe = _affine(self._mwi_np, best[live] + rt[live]).tolist()
            hits, locs, nors = [], [], []
            try:
                for k, q in zip(live.tolist(), probe):
                    hit, loc, nor, _i = cpm(q, depsgraph=dg)
                    if not hit:
                        continue
                    hits.append(k)
                    locs.append((loc.x, loc.y, loc.z))
                    nors.append((nor.x, nor.y, nor.z))
            except (RuntimeError, ReferenceError, TypeError, SystemError):
                break
            if not hits:
                break
            hits = np.array(hits, dtype=np.int64)
            cand = _affine(self._mw_np, np.array(locs, dtype=float))
            candn = self._world_normals(np.array(nors, dtype=float))
            cd = np.linalg.norm(P[hits] - cand, axis=1)
            better = cd < bestd[hits] - 1e-12
            if not better.any():
                break
            sel = hits[better]
            best[sel] = cand[better]
            bestn[sel] = candn[better]
            bestd[sel] = cd[better]
        return best, bestn

    # -- array API (what the wrap solver runs on) ---------------------------

    def nearest_arrays(self, P, max_dist=FAR, active=None):
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        n = len(P)
        T, Nrm = P.copy(), np.zeros((n, 3))
        ok = np.zeros(n, dtype=bool)
        if n == 0:
            return T, ok, Nrm
        local = _affine(self._mwi_np, P).tolist()
        radius = self._local_radius(max_dist)
        cpm = self.obj.closest_point_on_mesh
        dg = self._dg
        idxs = range(n) if active is None else np.flatnonzero(active).tolist()
        rows, locs, nors = [], [], []
        try:
            for i in idxs:
                hit, loc, nor, _f = cpm(local[i], distance=radius, depsgraph=dg)
                if not hit:
                    continue
                rows.append(i)
                locs.append((loc.x, loc.y, loc.z))
                nors.append((nor.x, nor.y, nor.z))
        except (RuntimeError, ReferenceError, TypeError, SystemError):
            if not rows and self._recover():
                return self.nearest_arrays(P, max_dist, active)
        if not rows:
            return T, ok, Nrm
        rows = np.array(rows, dtype=np.int64)
        world = _affine(self._mw_np, np.array(locs, dtype=float))
        wn = self._world_normals(np.array(nors, dtype=float))
        if self.approximate:
            world, wn = self._refine_world(P[rows], world, wn)
        if max_dist < FAR:
            keep = np.linalg.norm(world - P[rows], axis=1) <= max_dist
            rows, world, wn = rows[keep], world[keep], wn[keep]
            if not len(rows):
                return T, ok, Nrm
        T[rows] = world
        Nrm[rows] = wn
        ok[rows] = True
        return T, ok, Nrm


# ------------------------------------------------------------- factories ---

def world_tree(obj, depsgraph, budget=TRI_BUDGET):
    """Exact world-space BVHTree of ``obj``'s evaluated mesh, or None.

    Built from numpy arrays through ``FromPolygons`` - no BMesh, so duplicate
    and degenerate faces are harmless and the memory is just the tree. Returns
    None when the mesh is over ``budget`` (or cannot be read), which is the
    caller's cue to fall back to the zero-copy object queries.
    """
    try:
        ob_eval = obj.evaluated_get(depsgraph)
        me = ob_eval.to_mesh()
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        if not nv or not nt or nt > budget:
            return None
        co = np.empty(nv * 3)
        me.vertices.foreach_get("co", co)
        tris = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
    except (RuntimeError, ValueError, MemoryError):
        return None
    finally:
        try:
            ob_eval.to_mesh_clear()
        except (AttributeError, ReferenceError, RuntimeError):
            pass
    world = _affine(np.array(obj.matrix_world), co.reshape(nv, 3))
    try:
        tree = BVHTree.FromPolygons(world.tolist(),
                                    tris.reshape(nt, 3).tolist(),
                                    all_triangles=True)
    except (ValueError, MemoryError):
        return None
    return TreeQuery(tree)


def build_world_bvh(obj, depsgraph):
    """Nearest-surface accessor for ``obj``'s evaluated mesh, in WORLD space.

    Kept under its historical name (and its historical ``find_nearest``
    contract) so the live session, the slide brush and the wrap solver all
    keep working unchanged - but it no longer duplicates the mesh into a
    BMesh, which is what crashed Blender on a high-poly head.

    A uniformly transformed object needs nothing built at all. Only a target
    whose transform distorts distance (non-uniform scale, shear) gets the
    exact world-space tree, and only while it fits the triangle budget.
    """
    query = SurfaceQuery(obj, depsgraph)
    if query.approximate:
        tree = world_tree(obj, depsgraph)
        if tree is not None:
            return tree
    return query


def evaluated_face_count(obj, depsgraph):
    """Face count of the evaluated mesh, or -1 when it cannot be read.

    Reporting only: it lets the wrap tell the artist it really is running
    against a multi-million-face head rather than silently taking minutes.
    """
    try:
        me = obj.evaluated_get(depsgraph).data
        if not isinstance(me, bpy.types.Mesh):
            return -1
        return len(me.polygons)
    except (AttributeError, ReferenceError, RuntimeError):
        return -1
