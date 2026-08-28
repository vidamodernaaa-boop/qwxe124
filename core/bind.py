"""Surface binding: attach arbitrary points (bone joints) to a cage so they
follow its deformation. Each point binds to the nearest cage triangle as
barycentric weights + a signed normal offset. Reconstructing with the deformed
triangle moves the point exactly with the cage (linear in the shape-key value).
"""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def triangles_of(mesh):
    """(T,3) array of vertex indices for the mesh's (constant) triangulation."""
    mesh.calc_loop_triangles()
    return np.array([list(lt.vertices) for lt in mesh.loop_triangles], dtype=np.int64)


def build_bvh(coords, tris):
    """BVH whose polygon index == triangle index in `tris`.

    Built straight from the triangle list instead of through a BMesh. Real
    production characters do ship duplicate faces - double-sided cloth, a
    mirrored half welded without a cleanup, a hair card stacked on itself -
    and bmesh refused the second copy of a triangle with "face already
    exists", which crashed the whole transfer on a mesh Blender itself is
    perfectly happy with. FromPolygons stores the triangles as plain geometry,
    so duplicates and slivers are harmless, the polygon index still matches
    `tris` row for row, and no caller reads the hit normal (they all recompute
    it from the triangle).
    """
    verts = np.asarray(coords, dtype=np.float64).reshape(-1, 3).tolist()
    polys = np.asarray(tris, dtype=np.int64).reshape(-1, 3).tolist()
    return BVHTree.FromPolygons(verts, polys, all_triangles=True)


def _barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1.0 - v - w, v, w])


def _tri_normal(a, b, c):
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def _frame(a, b, c):
    """Orthonormal 3x3 frame [T, B, N] (columns) for a triangle."""
    t = b - a
    tn = np.linalg.norm(t)
    T = t / tn if tn > 1e-12 else np.array([1.0, 0.0, 0.0])
    N = _tri_normal(a, b, c)
    B = np.cross(N, T)
    bn = np.linalg.norm(B)
    B = B / bn if bn > 1e-12 else np.array([0.0, 1.0, 0.0])
    T = np.cross(B, N)            # re-orthogonalize
    return np.column_stack([T, B, N])


def bind_oriented(head, rot, rest_coords, tris, bvh):
    """Bind a joint (world head position + world 3x3 rotation) to the rest cage.

    Returns ``(tri_index, bary[3], offset[3], R_rel[3x3])`` or None. Both the
    offset and ``R_rel`` are expressed in the triangle's own frame, so the joint
    travels and turns with the surface on reconstruction (Surface-Deform style).

    The offset keeps all THREE components. The nearest point on a cage is only
    directly "below" the joint when it lands inside a triangle; for a joint deep
    under a coarse cage - the jaw pivot, the skull - it usually lands on an edge
    instead, and the vector out to the joint then has a large tangential part.
    Storing just the normal component silently discarded it, so the joint did
    not return to its own rest even when the cage had not moved at all: on Ada's
    LOD5 cage that put FACIAL_C_Jaw 11.8 mm out, and every chin and lower-lip
    bone inherited the error through the hierarchy.
    """
    loc, nor, idx, dist = bvh.find_nearest(Vector(head))
    if loc is None:
        return None
    a, b, c = (np.asarray(rest_coords[i], float) for i in tris[idx])
    loc = np.asarray(loc, float)
    bary = _barycentric(loc, a, b, c)
    F0 = _frame(a, b, c)
    # Measure the offset from the point the barycentric weights REBUILD, not
    # from the point the BVH returned. They are the same wherever the solve is
    # well conditioned, but eyelash and eyelid geometry has slivers thin enough
    # to make it degenerate, and there _barycentric falls back to the first
    # corner. Measuring against the fallback keeps the round trip exact anyway,
    # instead of leaving the joint up to 1.9 mm from where it started.
    surface = bary[0] * a + bary[1] * b + bary[2] * c
    offset = F0.T @ (np.asarray(head, float) - surface)
    R_rel = F0.T @ np.asarray(rot, float)
    return (int(idx), bary, offset, R_rel)


def reconstruct_oriented(bind, deformed_coords, tris):
    """Returns (pos[3], R[3x3]) on the deformed cage, or (None, None)."""
    if bind is None:
        return None, None
    idx, bary, offset, R_rel = bind
    a, b, c = (np.asarray(deformed_coords[v], float) for v in tris[idx])
    F1 = _frame(a, b, c)
    surf = bary[0] * a + bary[1] * b + bary[2] * c
    pos = surf + F1 @ offset
    R = F1 @ R_rel
    return pos, R


def bind_points(points, rest_coords, tris, bvh):
    """Bind each world point to the rest cage. Returns list of
    (tri_index, bary[3], normal_offset)."""
    binds = []
    for p in points:
        loc, nor, idx, dist = bvh.find_nearest(Vector(p))
        if loc is None:
            binds.append(None)
            continue
        a, b, c = (rest_coords[i] for i in tris[idx])
        bary = _barycentric(np.array(loc), a, b, c)
        offset = float(np.dot(np.array(p) - np.array(loc), _tri_normal(a, b, c)))
        binds.append((int(idx), bary, offset))
    return binds


def reconstruct(binds, deformed_coords, tris):
    """Reconstruct bound points on the deformed cage. Returns (m,3); rows for
    unbound points are NaN."""
    out = np.full((len(binds), 3), np.nan)
    for i, b in enumerate(binds):
        if b is None:
            continue
        tri_idx, bary, offset = b
        a, p1, p2 = (deformed_coords[v] for v in tris[tri_idx])
        surf = bary[0] * a + bary[1] * p1 + bary[2] * p2
        out[i] = surf + offset * _tri_normal(a, p1, p2)
    return out
