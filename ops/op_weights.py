"""Transfer skin weights from the wrapped cage to the user's head, then bind the
head to the fitted skeleton with an Armature modifier.

This is a real surface transfer: for every target vertex we find the closest
POINT on the wrapped cage surface (not just the nearest vertex), take its triangle
+ barycentric coordinates, and blend the three corner vertices' weights. That gives
smooth, high-quality weights even though the cage is low-poly (564 verts). Weights
are clamped and normalized against the complete source skin, while only FACIAL_*
groups are written so an existing body rig's weights remain untouched.

When the facial rig follows an existing character armature, the opposite
surface transfer is also performed for body weights: Head Target -> Head Cage.
That gives the cage the character's real head/neck blend instead of letting
its neck inherit the head rigidly.
"""

import json

import numpy as np
import bpy
from mathutils import Vector

from ..props import RIG_TARGET_PROP, RIG_MERGED_PROP, RIG_ID_PROP
from .op_wrap import (WRAPPED_KEY, MOUTH_OPEN_KEY, CLOSE_EYES_KEY,
                      get_wrapped_local)
from ..core import bind as bindmod
from ..core import riglogic

ARM_MOD_NAME = "Facial Rig"
CAGE_BODY_MOD_NAME = "Character Body Preview"
CAGE_BODY_GROUPS_PROP = "mhfrt_cage_body_groups"
CAGE_BODY_ARMATURE_PROP = "mhfrt_cage_body_armature"
CAGE_BODY_BACKUPS_PROP = "mhfrt_cage_body_group_backups"
CAGE_BACKUP_PREFIX = "__MHFRT_ORIGINAL__"

# Weight-transfer tuning
MAX_INFLUENCES = 8        # cap bones per target vertex; drops thin-feature bleed
NORMAL_DOT_MIN = 0.0      # cage surface must face the same way as the target vertex
GAP_FRAC = 0.06           # search this fraction of head size past the nearest hit


def restore_cage_facial_weights(context, cage):
    """Put the cage's FACIAL_* weights back to the bundled asset's values.

    Those weights are SOURCE data - every head weight is interpolated from them
    - and nothing in the workflow asks the artist to paint them, so the asset is
    always the truth. A merge used to be able to damage them: the face's share
    was taken out of one head bone, and on a character whose head is skinned to
    its own face rig that bone held only a fraction, so the cage's facial
    weights were scaled down (or dropped) and the cage stopped following the
    controls. The rule is fixed, but files bound by the old one need repairing,
    and re-checking here costs a compare on a mesh of a few thousand verts.

    Only runs on a cage a merge has touched, and only writes the vertices that
    actually differ. Returns how many were repaired.
    """
    if cage is None or cage.type != 'MESH':
        return 0
    from . import op_merge
    if not cage.get(op_merge.MERGED_WEIGHTS_PROP):
        return 0
    from .op_load_cage import SOURCE_TEMPLATE
    from .op_skeleton import LOD_BY_VERTS
    from ..prefs import get_cage_blend_path

    n = len(cage.data.vertices)
    lod = LOD_BY_VERTS.get(n)
    blend = get_cage_blend_path(context)
    if lod is None or not blend:
        return 0                    # not a bundled cage: nothing to compare to
    source_name = SOURCE_TEMPLATE.format(lod=lod)
    before = set(bpy.data.objects)
    try:
        with bpy.data.libraries.load(blend, link=False) as (have, want):
            if source_name not in have.objects:
                return 0
            want.objects = [source_name]
    except (OSError, RuntimeError):
        return 0
    source = next((o for o in bpy.data.objects if o.name not in
                   {b.name for b in before} and o.type == 'MESH'), None)
    if source is None or len(source.data.vertices) != n:
        for obj in set(bpy.data.objects) - before:
            bpy.data.objects.remove(obj, do_unlink=True)
        return 0

    try:
        src_name_of = {g.index: g.name for g in source.vertex_groups
                       if g.name.startswith("FACIAL_")}
        truth = {}                  # vertex -> {group name: weight}
        for vert in source.data.vertices:
            row = {src_name_of[e.group]: e.weight for e in vert.groups
                   if e.group in src_name_of and e.weight > 0.0}
            if row:
                truth[vert.index] = row

        cage_name_of = {g.index: g.name for g in cage.vertex_groups
                        if g.name.startswith("FACIAL_")}
        repaired = []
        for vert in cage.data.vertices:
            row = {cage_name_of[e.group]: e.weight for e in vert.groups
                   if e.group in cage_name_of and e.weight > 0.0}
            want_row = truth.get(vert.index, {})
            if set(row) != set(want_row) or any(
                    abs(row[name] - want_row[name]) > 1e-5 for name in row):
                repaired.append(vert.index)
        if not repaired:
            return 0

        groups = {g.name: g for g in cage.vertex_groups}
        for name in src_name_of.values():
            if name not in groups:
                groups[name] = cage.vertex_groups.new(name=name)
        clear = {}
        for index in repaired:
            want_row = truth.get(index, {})
            for name, weight in want_row.items():
                groups[name].add([index], weight, 'REPLACE')
            for name in cage_name_of.values():
                if name not in want_row:
                    clear.setdefault(name, []).append(index)
        for name, indices in clear.items():
            group = groups.get(name)
            if group is not None:
                group.remove(indices)
        return len(repaired)
    finally:
        # the bundled cage pulls its armature in with it - purge every
        # hitchhiker, or each re-bind leaves another orphan in the file
        leftovers = []
        for obj in set(bpy.data.objects) - before:
            leftovers.append(obj.data)
            bpy.data.objects.remove(obj, do_unlink=True)
        for data in leftovers:
            try:
                if data is None or data.users:
                    continue
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Armature):
                    bpy.data.armatures.remove(data)
            except (ReferenceError, RuntimeError):
                pass


def _source_body_weight_groups():
    try:
        names = riglogic.meta().get("joint_names", ())
    except Exception:
        return set()
    return {name for name in names if not name.startswith("FACIAL_")}


def _stored_cage_body_groups(cage):
    raw = str(cage.get(CAGE_BODY_GROUPS_PROP, "") or "") if cage else ""
    if not raw:
        return set()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    return {str(name) for name in value if isinstance(name, str)}


def _stored_cage_group_backups(cage):
    raw = str(cage.get(CAGE_BODY_BACKUPS_PROP, "") or "") if cage else ""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(original): str(backup)
        for original, backup in value.items()
        if isinstance(original, str) and isinstance(backup, str)
    }


def clear_cage_body_weights(cage):
    """Remove only body-preview data previously copied by this add-on."""
    if cage is None or cage.type != 'MESH':
        return 0
    removed = 0
    for name in _stored_cage_body_groups(cage):
        group = cage.vertex_groups.get(name)
        if group is not None:
            cage.vertex_groups.remove(group)
            removed += 1
    for original, backup in _stored_cage_group_backups(cage).items():
        group = cage.vertex_groups.get(backup)
        if group is None:
            continue
        collision = cage.vertex_groups.get(original)
        if collision is not None:
            cage.vertex_groups.remove(collision)
        group.name = original
    for mod in list(cage.modifiers):
        if mod.type == 'ARMATURE' and mod.name == CAGE_BODY_MOD_NAME:
            cage.modifiers.remove(mod)
    for prop in (CAGE_BODY_GROUPS_PROP, CAGE_BODY_ARMATURE_PROP,
                 CAGE_BODY_BACKUPS_PROP):
        if prop in cage:
            del cage[prop]
    return removed


def _basis_local(obj):
    """Unposed local coordinates, including the Basis shape when present."""
    count = len(obj.data.vertices)
    coords = np.empty(count * 3)
    keys = obj.data.shape_keys
    if keys is not None and keys.reference_key is not None:
        keys.reference_key.data.foreach_get("co", coords)
    else:
        obj.data.vertices.foreach_get("co", coords)
    return coords.reshape(count, 3)


def transfer_body_weights_to_cage(cage, target, body, facial=None):
    """Reverse-transfer the target's character skin onto its wrapped cage.

    Only groups belonging to deform bones on ``body`` are sampled. Facial and
    utility cage groups remain untouched. The copied body skin is limited and
    normalized independently, then evaluated before the facial modifier; after
    a one-armature merge the normal merge reconciliation combines both skins.
    """
    if cage is None or cage.type != 'MESH' \
            or target is None or target.type != 'MESH' \
            or body is None or body.type != 'ARMATURE':
        return 0, 0
    if not (cage.data.shape_keys
            and WRAPPED_KEY in cage.data.shape_keys.key_blocks):
        return 0, 0

    deform_names = {
        bone.name for bone in body.data.bones
        if bone.use_deform and not bone.name.startswith("FACIAL_")
    }
    target_index_to_name = {
        group.index: group.name for group in target.vertex_groups
        if group.name in deform_names
    }
    if not target_index_to_name:
        return 0, 0

    target_local = _basis_local(target)
    target_mw = np.asarray(target.matrix_world)
    target_world = (
        np.hstack([target_local, np.ones((len(target_local), 1))])
        @ target_mw.T
    )[:, :3]
    target_tris = bindmod.triangles_of(target.data)
    if not len(target_tris):
        return 0, 0
    target_bvh = bindmod.build_bvh(target_world, target_tris)
    # One geometric normal per target triangle.
    a = target_world[target_tris[:, 0]]
    b = target_world[target_tris[:, 1]]
    c = target_world[target_tris[:, 2]]
    tri_normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    tri_normals = tri_normals / np.where(lengths > 1e-12, lengths, 1.0)

    target_weights = []
    for vert in target.data.vertices:
        target_weights.append({
            target_index_to_name[element.group]: element.weight
            for element in vert.groups
            if element.group in target_index_to_name and element.weight > 0.0
        })

    cage_local = get_wrapped_local(cage)
    cage_mw = np.asarray(cage.matrix_world)
    cage_world = (
        np.hstack([cage_local, np.ones((len(cage_local), 1))])
        @ cage_mw.T
    )[:, :3]
    cage_tris = bindmod.triangles_of(cage.data)
    if not len(cage_tris):
        return 0, 0
    cage_normals = _vertex_normals(cage_world, cage_tris)
    size = max(float(max(target.dimensions)), 1e-6)
    gap = GAP_FRAC * size

    assignments = {}
    assigned = 0
    for cage_index, point in enumerate(cage_world):
        location, tri_index = _normal_aware_hit(
            target_bvh,
            Vector(point),
            cage_normals[cage_index],
            tri_normals,
            gap,
            NORMAL_DOT_MIN,
        )
        if location is None:
            continue
        tri = target_tris[tri_index]
        bary = _barycentric(
            np.asarray(location, float),
            target_world[tri[0]],
            target_world[tri[1]],
            target_world[tri[2]],
        )
        weights = {}
        for corner, factor in zip(tri, bary):
            if factor <= 0.0:
                continue
            for name, value in target_weights[corner].items():
                weights[name] = weights.get(name, 0.0) + factor * value
        if len(weights) > MAX_INFLUENCES:
            weights = dict(sorted(
                weights.items(), key=lambda item: item[1], reverse=True
            )[:MAX_INFLUENCES])
        total = sum(weights.values())
        if total <= 1e-12:
            continue
        for name, value in weights.items():
            if value > 0.0:
                assignments.setdefault(name, []).append(
                    (cage_index, value / total))
        assigned += 1

    if not assigned or not assignments:
        return 0, 0

    previous = _stored_cage_body_groups(cage)
    backups = _stored_cage_group_backups(cage)
    for name in previous:
        group = cage.vertex_groups.get(name)
        if group is not None:
            cage.vertex_groups.remove(group)

    # A character bone may have the same name as a bundled source-body group
    # (for example a plain "head") or a utility group. Preserve that original
    # group under an internal non-bone name, then restore it on Standalone.
    for name in assignments:
        group = cage.vertex_groups.get(name)
        if group is None or name in backups:
            continue
        backup = f"{CAGE_BACKUP_PREFIX}{name}"
        suffix = 1
        while cage.vertex_groups.get(backup) is not None:
            backup = f"{CAGE_BACKUP_PREFIX}{name}.{suffix:03d}"
            suffix += 1
        group.name = backup
        backups[name] = backup

    # Keep the bundled source-body groups: facial transfer uses them only as
    # the denominator that defines the original facial falloff. They do not
    # match bones on the fitted facial rig and therefore do not deform.
    for name in assignments:
        group = cage.vertex_groups.get(name)
        if group is not None:
            cage.vertex_groups.remove(group)

    used = set()
    for name, values in assignments.items():
        group = cage.vertex_groups.new(name=name)
        for vertex_index, value in values:
            group.add([vertex_index], value, 'REPLACE')
        used.add(name)

    cage[CAGE_BODY_GROUPS_PROP] = json.dumps(sorted(used))
    cage[CAGE_BODY_BACKUPS_PROP] = json.dumps(backups, sort_keys=True)
    cage[CAGE_BODY_ARMATURE_PROP] = body
    # A fresh target-derived body baseline supersedes any previous merged
    # facial take-over on the cage; the merge pass will reconcile it again.
    for prop in ("mhfrt_merged_weights", "mhfrt_merged_weight_carrier"):
        if prop in cage:
            del cage[prop]
    modifier = cage.modifiers.get(CAGE_BODY_MOD_NAME)
    if modifier is not None and modifier.type != 'ARMATURE':
        modifier = None
    if modifier is None:
        modifier = cage.modifiers.new(CAGE_BODY_MOD_NAME, 'ARMATURE')
    modifier.object = body
    modifier.use_vertex_groups = True
    modifier.show_in_editmode = True
    modifier.show_on_cage = True

    if facial is not None:
        body_index = cage.modifiers.find(modifier.name)
        facial_index = next(
            (index for index, candidate in enumerate(cage.modifiers)
             if candidate.type == 'ARMATURE'
             and candidate.object == facial
             and candidate != modifier),
            -1,
        )
        if facial_index >= 0 and body_index > facial_index:
            cage.modifiers.move(body_index, facial_index)
    return assigned, len(used)


def _current_local(obj):
    """Current shape-key-mixed local coords (relative keys at their slider values),
    or the base vertices if there are no shape keys. This lets the weight transfer
    use whatever pose the user is showing right now (e.g. mouth open) with no extra
    step -- open the mouth, hit Transfer again, done."""
    me = obj.data
    n = len(me.vertices)
    if not me.shape_keys:
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        return co.reshape(n, 3)
    kb = me.shape_keys.key_blocks
    basis = np.empty(n * 3)
    kb[0].data.foreach_get("co", basis)
    basis = basis.reshape(n, 3)
    mixed = basis.copy()
    buf = np.empty(n * 3)
    for k in kb[1:]:                          # relative keys add their delta * value
        if k.value == 0.0:
            continue
        k.data.foreach_get("co", buf)
        mixed += k.value * (buf.reshape(n, 3) - basis)
    return mixed


def _vertex_normals(coords, tris):
    """Area-weighted per-vertex normals from positions + triangulation. Computed
    from the actual sampled positions so it stays correct in the open-mouth pose."""
    a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    fn = np.cross(b - a, c - a)               # area-weighted face normals
    vn = np.zeros_like(coords)
    np.add.at(vn, tris[:, 0], fn)
    np.add.at(vn, tris[:, 1], fn)
    np.add.at(vn, tris[:, 2], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(ln > 1e-12, ln, 1.0)


def _normal_aware_hit(bvh, p, vn, tri_normals, gap, dotmin):
    """Closest cage surface point whose face normal agrees with the target vertex
    normal `vn`. If the plain nearest faces the wrong way (e.g. the opposite lip
    across the thin mouth gap, or the other eyelid), search nearby for a same-facing
    surface instead; fall back to the plain nearest when nothing qualifies.

    Triangle normals are taken from `tri_normals` (computed from geometry), not from
    the BVH hit, so the test is reliable. Returns (location, tri_index) or (None, None).
    """
    loc, nor, idx, dist = bvh.find_nearest(p)
    if loc is None:
        return None, None
    if float(np.dot(tri_normals[idx], vn)) >= dotmin:
        return loc, idx
    best = None
    best_d = None
    for cloc, cnor, cidx, cdist in bvh.find_nearest_range(p, dist + gap):
        if float(np.dot(tri_normals[cidx], vn)) >= dotmin and (best_d is None or cdist < best_d):
            best = (cloc, cidx)
            best_d = cdist
    return best if best is not None else (loc, idx)


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
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
    bary = np.array([1.0 - v - w, v, w])
    bary = np.clip(bary, 0.0, None)          # closest point may sit on an edge
    s = bary.sum()
    return bary / s if s > 1e-12 else np.array([1.0, 0.0, 0.0])


class MHFRT_OT_transfer_weights(bpy.types.Operator):
    bl_idname = "mhfrt.transfer_weights"
    bl_label = "Transfer Weights & Bind"
    bl_description = ("Surface-interpolated weight transfer from the wrapped cage to the "
                      "head, then add an Armature modifier to the fitted skeleton")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Bind, with the cleanup pose applied but never shown.

        The pose is a step inside the bind, not a change to the artist's scene:
        whatever is on screen when they press the button is what is on screen
        when it returns - including after a failure, which used to strand both
        meshes mid-pose. Pose Meshes remains the only way to actually look at it.

        The one thing that does NOT survive the bind is a cleanup pose left on
        screen: the mouth-open / eyes-closed keys go straight back to zero, so
        the artist looks at the neutral head they just bound instead of a face
        still hanging open. The SLIDERS keep their values - they are the
        setting the bind was made with, so Re-Bind repeats it without
        re-dialling - only the shape keys are cleared.
        """
        from . import op_mouth
        mh = context.scene.mhfrt
        shown = op_mouth._visible_pose_state(mh.cage, mh.target)
        try:
            return self._bind(context)
        finally:
            op_mouth._restore_pose_state(shown)
            self._clear_cleanup_pose(context)

    @staticmethod
    def _clear_cleanup_pose(context):
        """Put the mouth-open / eyes-closed keys back to zero on both meshes."""
        from . import op_mouth
        for spec in op_mouth._CLEANUP_SPECS.values():
            try:
                op_mouth.apply_cleaning_amount(context, spec["key"], 0.0)
            except (ReferenceError, RuntimeError, AttributeError):
                pass          # a clear must never mask the bind's own result

    def _bind(self, context):
        mh = context.scene.mhfrt
        skel = mh.skeleton
        if skel is not None:
            from ..core import rest_tuning
            if skel.get(rest_tuning.TONGUE_SESSION_PROP):
                self.report({'ERROR'}, "Finish or Cancel Tongue Edit before binding")
                return {'CANCELLED'}
        from .op_live import stop_running
        stop_running()          # weights read the cage's current shape
        if context.object and context.object.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass
        cage, target, skel = mh.cage, mh.target, mh.skeleton
        if not cage or not target or cage == target:
            self.report({'ERROR'}, "Set Head Cage and Head Target first")
            return {'CANCELLED'}
        if not skel:
            self.report({'ERROR'}, "Fit the skeleton first")
            return {'CANCELLED'}
        me = cage.data
        if not (me.shape_keys and WRAPPED_KEY in me.shape_keys.key_blocks):
            self.report({'ERROR'}, "Wrap the cage first (need a 'Wrapped' shape key)")
            return {'CANCELLED'}
        # On a MERGED rig this bind rebalances the artist's own head-region
        # weights (the face takes its share out of them), so their weights are
        # copied into the character file first - the last moment they are
        # theirs.  Taken once: a re-bind snapshots nothing, because the first
        # snapshot is the pristine one.  An unmerged bind only ever rewrites
        # our own FACIAL_ groups and needs none of this.
        if skel.get(RIG_MERGED_PROP):
            from ..core import registry as _registry, provenance as _provenance
            _record = (_registry.for_object(context.scene, target)
                       or _registry.active(context.scene))
            _provenance.snapshot_object(_record, target,
                                        role=_provenance.ROLE_HEAD,
                                        weights=True, adopt=True)
        # Apply the Weight Cleanup sliders behind the scenes. They deliberately
        # do nothing while dragged (one pose is a switched-wrap solve), so this
        # is where they take effect: binding gives the same result whether or
        # not the artist pressed Pose Meshes to look at it first. An already
        # applied - or hand-sculpted - pose is kept as it is.
        #
        # `execute` restores the visible state afterwards, so this only ever
        # reaches the sampling below, never the viewport. What it returns is
        # what was actually posed - a slider that is up but could not be
        # generated (no landmark pairs in this file, say) is worth saying out
        # loud rather than binding a neutral mouth and calling it cleaned.
        from . import op_mouth
        posed_features = op_mouth.sync_cleanup_pose(context)
        wanted_features = [
            spec["label"] for spec in op_mouth._CLEANUP_SPECS.values()
            if float(getattr(mh, spec["amount_prop"], 0.0)) > 1e-6
        ]
        unposed = [name for name in wanted_features
                   if name not in posed_features]
        # Every head weight is interpolated from the cage's own facial weights,
        # so they must be the asset's before anything is sampled (a merge bound
        # by the pre-1.15.2 rule could have scaled them down - see the function).
        cage_repaired = restore_cage_facial_weights(context, cage)
        # Transfer in whatever pose is now shown. With both mouths open (cage
        # 'MouthOpen' + the head's generated/sculpted 'MouthOpen') the lips are
        # separated so correspondence is clean. Weights are pose-independent, so
        # the neutral final rig is still correct.
        ck = me.shape_keys.key_blocks
        mouth_open = (MOUTH_OPEN_KEY in ck and ck[MOUTH_OPEN_KEY].value > 1e-4)
        eyes_closed = (CLOSE_EYES_KEY in ck and ck[CLOSE_EYES_KEY].value > 1e-4)
        cage_src_local = _current_local(cage)

        mw = np.array(cage.matrix_world)
        cage_world = (np.hstack([cage_src_local, np.ones((len(cage_src_local), 1))]) @ mw.T)[:, :3]
        tris = bindmod.triangles_of(me)
        bvh = bindmod.build_bvh(cage_world, tris)
        # per-triangle outward normals from geometry (reliable for normal-aware match)
        A = cage_world[tris[:, 0]]
        B = cage_world[tris[:, 1]]
        C = cage_world[tris[:, 2]]
        fn = np.cross(B - A, C - A)
        fln = np.linalg.norm(fn, axis=1, keepdims=True)
        tri_normals = fn / np.where(fln > 1e-12, fln, 1.0)

        # Sample the complete MetaHuman skin so each facial weight keeps its
        # original share of the total, but WRITE facial groups only.  Body groups
        # on a pre-rigged target belong to the existing character armature and
        # existing armature and must never be deleted or replaced here.
        bone_names = {
            b.name for b in skel.data.bones if b.name.startswith("FACIAL_")
        }
        gi_to_name = {
            group.index: group.name for group in cage.vertex_groups
            if group.name in bone_names
        }
        source_backups = _stored_cage_group_backups(cage)
        for source_name in _source_body_weight_groups():
            actual_name = source_backups.get(source_name, source_name)
            group = cage.vertex_groups.get(actual_name)
            if group is not None:
                # The name is irrelevant to output (body entries only
                # contribute to the denominator), but keep source semantics.
                gi_to_name[group.index] = source_name
        facial_names = set(gi_to_name.values()) & bone_names
        # per cage vertex: dict bone_name -> weight
        cage_vw = []
        for v in me.vertices:
            d = {}
            for g in v.groups:
                name = gi_to_name.get(g.group)
                if name is not None and g.weight > 0.0:
                    d[name] = g.weight
            cage_vw.append(d)

        # On a merged rig the facial groups hold weight taken from the head
        # bone; hand it back before the transfer rewrites them, so the artist's
        # painted weights are the baseline again (see op_merge).
        merged_carrier = None
        merged_body_names = set()
        if skel.get(RIG_MERGED_PROP):
            from . import op_merge
            merged_body_names = {b.name for b in skel.data.bones
                                 if not b.name.startswith("FACIAL_")
                                 and b.use_deform}
            merged_carrier = op_merge.weight_carrier(
                target,
                op_merge.merged_deform_bone(skel),
                merged_body_names,
                op_merge.facial_bone_names(skel),
                skel,
            )
            op_merge.release_weights_from_merged_rig(
                target, op_merge.facial_bone_names(skel), merged_carrier)

        # The cage needs the same character-body field as the target. Without
        # it, its facial groups inherit the head while its neck has no matching
        # neck deformation and visibly pulls away when the head turns.
        body_for_cage = None
        if skel.get(RIG_MERGED_PROP):
            body_for_cage = skel
        elif (skel.parent is not None
              and skel.parent.type == 'ARMATURE'
              and skel.parent_type == 'BONE'
              and skel.parent_bone):
            body_for_cage = skel.parent
        cage_body_refreshed = False
        if body_for_cage is not None:
            cage_vertices, _cage_groups = transfer_body_weights_to_cage(
                cage, target, body_for_cage, skel)
            cage_body_refreshed = bool(cage_vertices)

        # Refresh this add-on's facial groups only.  Every existing body group is
        # intentionally preserved byte-for-byte for its original armature.
        for name in list(target.vertex_groups.keys()):
            if name in facial_names:
                target.vertex_groups.remove(target.vertex_groups[name])
        tgroups = {
            name: target.vertex_groups.new(name=name) for name in facial_names
        }

        # target verts in world space, in its CURRENT pose (sculpted mouth included)
        tmw = np.array(target.matrix_world)
        tlocal = _current_local(target)
        tworld = (np.hstack([tlocal, np.ones((len(tlocal), 1))]) @ tmw.T)[:, :3]

        # target vertex normals from the POSED positions (consistent with tworld)
        ttris = bindmod.triangles_of(target.data)
        tnorm_world = _vertex_normals(tworld, ttris)
        size = max(target.dimensions)
        gap = GAP_FRAC * (size if size > 0 else 1.0)

        n_assigned = 0
        for ti, p in enumerate(tworld):
            loc, idx = _normal_aware_hit(
                bvh, Vector(p), tnorm_world[ti], tri_normals, gap, NORMAL_DOT_MIN)
            if loc is None:
                continue
            tri = tris[idx]
            a, b, c = cage_world[tri[0]], cage_world[tri[1]], cage_world[tri[2]]
            bary = _barycentric(np.asarray(loc, float), a, b, c)
            acc = {}
            for k in range(3):
                bw = bary[k]
                if bw <= 0.0:
                    continue
                for name, w in cage_vw[tri[k]].items():
                    acc[name] = acc.get(name, 0.0) + bw * w
            if len(acc) > MAX_INFLUENCES:    # keep only the strongest (cleanup pass)
                acc = dict(sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:MAX_INFLUENCES])
            total = sum(acc.values())
            if total <= 1e-9:
                continue
            assigned = False
            for name, w in acc.items():
                group = tgroups.get(name)
                if group is None or w <= 0.0:
                    continue
                group.add([ti], w / total, 'REPLACE')
                assigned = True
            if assigned:
                n_assigned += 1

        # Identify our modifier by the armature it actually uses, not by name.
        # A user's body modifier is allowed to have any name (even "Facial Rig")
        # and must never be repointed to this skeleton.
        mod = next(
            (candidate for candidate in target.modifiers
             if candidate.type == 'ARMATURE' and candidate.object == skel),
            None,
        )
        if mod is None:
            mod = target.modifiers.new(ARM_MOD_NAME, 'ARMATURE')
        elif mod.name != ARM_MOD_NAME and skel.get(RIG_MERGED_PROP):
            # Merged rig: the character's single Armature modifier is also the
            # facial one - a second modifier for the same armature would
            # deform the head twice.  Claim the existing one by name so the
            # workflow still sees the head as bound.
            if not any(m.name == ARM_MOD_NAME for m in target.modifiers):
                mod.name = ARM_MOD_NAME
        mod.object = skel
        mod.use_vertex_groups = True
        mod.show_in_editmode = True     # posed result visible while editing
        mod.show_on_cage = True         # and editable directly on the deformed mesh
        # The facial deformation must consume the already body-deformed mesh.
        # A reused modifier may have been moved by the user, so put it after
        # every other Armature modifier while leaving later surface/detail
        # modifiers in their existing relative order.
        facial_index = target.modifiers.find(mod.name)
        last_other_armature = max(
            (index for index, candidate in enumerate(target.modifiers)
             if candidate != mod and candidate.type == 'ARMATURE'),
            default=-1,
        )
        if facial_index < last_other_armature:
            target.modifiers.move(facial_index, last_other_armature)
        # link the head into the character chain (append pulls it along)
        skel[RIG_TARGET_PROP] = target

        # On a merged rig the facial and body groups share one modifier, so the
        # face takes its share out of the whole head region - the neck, spine
        # and shoulder weights the artist painted stay exactly as they are, and
        # a face rig the character already had gives the face up (see op_merge).
        rebalanced = 0
        missed = 0
        if skel.get(RIG_MERGED_PROP):
            from . import op_merge
            rebalanced, missed = op_merge.bind_weights_to_merged_rig(
                target,
                op_merge.facial_bone_names(skel),
                merged_body_names,
                op_merge.merged_deform_bone(skel),
                str(skel.get(RIG_ID_PROP) or ""),
                skel,
                op_merge.merged_parent_bone(skel),
            )
            if cage_body_refreshed:
                op_merge.dedupe_armature_modifiers(cage, skel)
                op_merge.bind_weights_to_merged_rig(
                    cage,
                    op_merge.facial_bone_names(skel),
                    merged_body_names,
                    op_merge.merged_deform_bone(skel),
                    str(skel.get(RIG_ID_PROP) or ""),
                    skel,
                    op_merge.merged_parent_bone(skel),
                )

        # No _set_neutral_display here any more: `execute`'s finally puts the
        # meshes back exactly as the artist had them, which is neutral for
        # everyone who did not press Pose Meshes, and their own pose for
        # everyone who did. The cleanup sliders keep their values either way -
        # they are the setting for the bind, not a picture of what is showing,
        # so a second Re-Bind repeats the same pose without re-dialling it.

        if unposed:
            self.report({'WARNING'},
                        "Weight Cleanup could not pose "
                        + " + ".join(unposed)
                        + " - bound without it")

        active_pose = []
        if mouth_open:
            active_pose.append("mouth-open")
        if eyes_closed:
            active_pose.append("eyes-closed")
        pose = " + ".join(active_pose) + " pose" if active_pose else "neutral pose"
        self.report(
            {'INFO'},
            f"Surface-transferred {len(tgroups)} groups to {n_assigned}/"
            f"{len(target.data.vertices)} verts ({pose}); bound to '{skel.name}'"
            + (f"; {rebalanced} merged weights limited and normalized"
               if rebalanced else "")
            + (f"; repaired the cage's facial weights on {cage_repaired} verts"
               if cage_repaired else "")
            + (f"; {missed} verts the face drives have no head-region weight"
               if missed else ""),
        )
        return {'FINISHED'}


_classes = (MHFRT_OT_transfer_weights,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
