"""Build / update a character's facial rig: import its skeleton (once per
character) and bind/deform the joints onto that character's wrapped cage.

Each FACIAL_* joint binds to the cage's FIRST shape (Basis) and is reconstructed
on the LAST shape (Wrapped) with full orientation (Surface-Deform style). The
joint's original rest (head/orientation/length) is stored as props so UPDATING
re-fits the SAME skeleton object (no re-import) - change the character, re-wrap
or slide, then press Update and the rig follows. After the rest pose is fitted,
the DNA delta-to-bone conversion is rebuilt so RigLogic drives the new bone
axes, and the always-on rig runtime (op_rig) gets its control board.

Every character keeps its OWN skeleton: they are matched through an ID-pointer
property linking skeleton -> cage, never by object name.
"""

import re
from contextlib import contextmanager

import numpy as np
import bpy
from mathutils import Vector, Matrix

from ..prefs import get_cage_blend_path
from ..props import (RIG_ID_PROP, RIG_CAGE_PROP, RIG_MERGED_PROP,
                     RIG_SOURCE_SCALE_PROP, RIG_SOURCE_BASIS_PROP,
                     RIG_REST_HEAD_PROP)
from ..core import bind as bindmod
from ..core import align
from ..core import organization
from ..core import rest_tuning
from . import op_rig
from .op_wrap import get_basis_local, get_wrapped_local, WRAPPED_KEY

SKEL_SOURCE = "Ada_Head_head_rig"
SKEL_NAME = "Facial_Skeleton"   # legacy single-character name (pre-0.3 scenes)
FACIAL_PREFIX = "FACIAL_"
MIN_BONE_LEN = 1e-4

REST_HEAD = RIG_REST_HEAD_PROP
REST_ROT = "mhfrt_rest_rot"
REST_LEN = "mhfrt_rest_len"
REST_SOURCE_TO_RIG_BASIS = RIG_SOURCE_BASIS_PROP
REST_SOURCE_TO_RIG_SCALE = RIG_SOURCE_SCALE_PROP
MERGED_BODY_PROP = RIG_MERGED_PROP

# cage vertex count -> Ada LOD index (used to find the matching source rest)
LOD_BY_VERTS = {564: 5, 1291: 4, 2548: 3, 5999: 2, 11977: 1, 24049: 0}


def _apply_affine(M, pts):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ np.asarray(M).T)[:, :3]


def _mode_object(context):
    if context.object and context.object.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass


def _strip_body_bones(context, skel):
    """Remove Ada body helper bones from a freshly imported facial skeleton."""
    if skel is None or skel.type != 'ARMATURE':
        return 0
    _mode_object(context)
    prev_active = context.view_layer.objects.active
    prev_selected = list(context.selected_objects)
    for obj in prev_selected:
        obj.select_set(False)
    removed = 0
    with organization.shown_for_edit(
            skel, view_layer=context.view_layer):
        context.view_layer.update()
        skel.select_set(True)
        context.view_layer.objects.active = skel
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in list(skel.data.edit_bones):
            if not bone.name.startswith(FACIAL_PREFIX):
                skel.data.edit_bones.remove(bone)
                removed += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        skel.select_set(False)
    for obj in prev_selected:
        if context.view_layer.objects.get(obj.name):
            obj.select_set(True)
    if prev_active and context.view_layer.objects.get(prev_active.name):
        context.view_layer.objects.active = prev_active
    return removed


def _deformed_meshes(context, skel):
    """Visible meshes this skeleton deforms through an Armature modifier.

    Used only by the visual transition: refitting the rest pose changes what
    these meshes look like, so they have to be morphed alongside the bones."""
    out = []
    if skel is None or skel.type != 'ARMATURE':
        return out
    view_layer = getattr(context, "view_layer", None)
    for obj in getattr(view_layer, "objects", ()) or ():
        if obj.type != 'MESH':
            continue
        if any(m.type == 'ARMATURE' and m.object == skel and m.show_viewport
               for m in obj.modifiers):
            out.append(obj)
    return out


def _changed_bone_count(skel, snapshot, epsilon=1e-7):
    """How many visible bone sticks differ from a transition snapshot.

    The fit loop counts successfully reconstructed joints, not joints whose
    position actually changed. Keeping those concepts separate prevents a
    no-op Update Rig from claiming a visual update while the transition
    correctly has nothing to animate.
    """
    if skel is None or not snapshot:
        return None
    previous = snapshot.get("bones")
    if previous is None:
        return None
    mw = skel.matrix_world
    changed = 0
    current_names = set()
    for pose_bone in skel.pose.bones:
        current_names.add(pose_bone.name)
        old = previous.get(pose_bone.name)
        if old is None:
            changed += 1
            continue
        head = np.asarray(mw @ pose_bone.head, dtype=float)
        tail = np.asarray(mw @ pose_bone.tail, dtype=float)
        if (np.abs(head - old[0]).max() > epsilon
                or np.abs(tail - old[1]).max() > epsilon):
            changed += 1
    changed += sum(name not in current_names for name in previous)
    return changed


def _simplify_bone_name(name):
    simple = name.lower()
    for prefix in ("def-", "def_", "org-", "org_", "mch-", "mch_",
                   "ctrl-", "ctrl_", "mixamorig:"):
        if simple.startswith(prefix):
            simple = simple[len(prefix):]
            break
    simple = re.sub(r"[^a-z0-9]+", "", simple)
    return re.sub(r"\d+", lambda m: str(int(m.group(0))), simple)


def _resolve_bone_name(arm_obj, requested="", candidates=()):
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        return ""
    bones = arm_obj.data.bones
    if requested:
        if requested in bones:
            return requested
        want = _simplify_bone_name(requested)
        for bone in bones:
            if _simplify_bone_name(bone.name) == want:
                return bone.name
        return ""

    # Prefer a literal candidate before fuzzy matching. A control rig can
    # contain several generated variants of the same semantic bone; an exact
    # artist-visible choice is safer than armature order.
    for name in candidates:
        if name and name in bones:
            return name
    wants = [_simplify_bone_name(name) for name in candidates if name]
    for want in wants:
        for bone in bones:
            if _simplify_bone_name(bone.name) == want:
                return bone.name
    suffix_matches = []
    for want in wants:
        suffix_matches.extend(
            bone.name for bone in bones
            if _simplify_bone_name(bone.name).endswith(want))
    if suffix_matches:
        return sorted(set(suffix_matches), key=len)[0]
    return ""


def _target_body_armatures(target, facial):
    """Armatures already driving/parenting the target, excluding this face rig."""
    if target is None or target.type != 'MESH':
        return []
    armatures = []
    parent = target.parent
    if parent is not None and parent.type == 'ARMATURE' and parent != facial:
        armatures.append(parent)
    for mod in target.modifiers:
        arm = mod.object if mod.type == 'ARMATURE' else None
        if arm is not None and arm.type == 'ARMATURE' and arm != facial:
            if arm not in armatures:
                armatures.append(arm)
    return armatures


@contextmanager
def _parent_armature_rest_position(context, skel):
    """Evaluate an attached facial rig against its parent's neutral rest.

    Update Rig rebuilds facial rest bones from unposed cage coordinates.  If the
    body happens to be posed, using that evaluated parent-bone matrix would bake
    the current animation frame into the child's parent-space offset.  Rest mode
    is temporary and always restored before the operator returns.
    """
    parent = skel.parent
    attached = bool(
        parent is not None
        and parent.type == 'ARMATURE'
        and skel.parent_type == 'BONE'
        and skel.parent_bone
    )
    armature = parent.data if attached else None
    previous = armature.pose_position if armature is not None else None
    try:
        if armature is not None and previous != 'REST':
            armature.pose_position = 'REST'
            context.view_layer.update()
        yield
    finally:
        if armature is not None and armature.pose_position != previous:
            armature.pose_position = previous
            context.view_layer.update()


def _source_rest_local(context, vcount):
    """Original Ada cage local verts for the LOD matching `vcount`, read fresh from
    the bundled blend. Lets us recover any transform the user baked into the cage
    (move + Apply) so the skeleton can be aligned to the cage's current rest."""
    lod = LOD_BY_VERTS.get(vcount)
    if lod is None:
        return None
    from .op_load_cage import SOURCE_TEMPLATE
    name = SOURCE_TEMPLATE.format(lod=lod)
    blend = get_cage_blend_path(context)
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(blend, link=False) as (df, dt):
        if name not in df.objects:
            return None
        dt.objects = [name]
    obj = next((o for o in dt.objects if o is not None), None)
    if obj is None:
        return None
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    ref = co.reshape(-1, 3)
    # the load dependency-pulls the asset skeleton via the mesh's armature
    # modifier: purge EVERYTHING this read brought in, not just the mesh
    loaded = set(bpy.data.objects) - before
    datas = {o.data for o in loaded if o.data is not None}
    for o in loaded:
        bpy.data.objects.remove(o, do_unlink=True)
    for d in datas:
        if d.users == 0:
            if isinstance(d, bpy.types.Mesh):
                bpy.data.meshes.remove(d)
            elif isinstance(d, bpy.types.Armature):
                bpy.data.armatures.remove(d)
    return ref


def _global_stretch(rest_w, wrap_w):
    """Uniform scale ratio of the Wrapped cage cloud vs the Basis (Ada) cage cloud.

    RigLogic joint *translations* are baked at Ada's absolute size; multiplying them
    by this ratio keeps expression motion proportional on a larger/smaller head.
    Uses RMS distance from the centroid over all cage verts, so it's robust to the
    low-poly cage (no dependence on any single triangle)."""
    def cloud_scale(P):
        c = P.mean(axis=0)
        return float(np.sqrt(((P - c) ** 2).sum(axis=1).mean()))
    s0 = cloud_scale(rest_w)
    s1 = cloud_scale(wrap_w)
    return s1 / s0 if s0 > 1e-9 else 1.0


def _similarity_matrix(s, R, t):
    M = Matrix.Identity(4)
    for i in range(3):
        M[i][0], M[i][1], M[i][2] = s * R[i][0], s * R[i][1], s * R[i][2]
        M[i][3] = t[i]
    return M


def _orthonormal_basis(matrix):
    """Return the rotation part of a 3x3/4x4 matrix without scale or shear."""
    A = np.asarray(matrix, dtype=float)[:3, :3]
    try:
        U, _sig, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
        return R
    except np.linalg.LinAlgError:
        return np.eye(3)


def _flat_matrix3(matrix):
    M = np.asarray(matrix, dtype=float).reshape(3, 3)
    return [float(M[r, c]) for r in range(3) for c in range(3)]


def _matrix3_prop(values):
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(vals) != 9:
        return None
    return np.array(vals, dtype=float).reshape(3, 3)


def _skeleton_alignment(context, cage, n):
    """World matrix to place the skeleton so its (Ada-space) bones line up with the
    cage's current rest, even if the cage was moved + Apply-ed."""
    ref = _source_rest_local(context, n)
    basis_local = get_basis_local(cage)
    if ref is not None and len(ref) == n:
        s, R, t = align.similarity_transform(ref, basis_local)
        return cage.matrix_world @ _similarity_matrix(s, R, t)
    return cage.matrix_world.copy()


def _uniform_scale(matrix):
    """Cube root of a transform's volume ratio (its uniform scale)."""
    det = abs(np.linalg.det(np.asarray(matrix.to_3x3(), dtype=float)))
    return float(det ** (1.0 / 3.0)) if det > 1e-24 else 1.0


def source_to_rig_scale(skel):
    """Armature-space unit ratio DNA source -> this skeleton (1.0 normally).

    A rig merged into an armature with another object scale evaluates its pose
    in that armature's units, so every joint TRANSLATION has to be rescaled.
    """
    try:
        return max(1e-6, float(skel.get(REST_SOURCE_TO_RIG_SCALE, 1.0)))
    except (TypeError, ValueError):
        return 1.0


def ensure_rest_source_to_rig_basis(context, skel, cage=None):
    """Ensure DNA's source armature basis is mapped into this rig's data space.

    New fits store this explicitly.  The branch below recovers it for rigs
    merged into the character's own armature (by this add-on or by hand)
    before a Fine-Tune refresh touches their cached conversion matrices.
    """
    existing = _matrix3_prop(skel.get(REST_SOURCE_TO_RIG_BASIS))
    if existing is not None:
        return True
    if not skel.get(MERGED_BODY_PROP):
        skel[REST_SOURCE_TO_RIG_BASIS] = _flat_matrix3(np.eye(3))
        return True
    if cage is None:
        cage = skel.get(RIG_CAGE_PROP)
    if not isinstance(cage, bpy.types.Object) or cage.type != 'MESH':
        return False
    source_matrix_world = _skeleton_alignment(
        context, cage, len(cage.data.vertices))
    source_mw3 = np.asarray(source_matrix_world.to_3x3())
    dest_mw3i = np.linalg.inv(np.asarray(skel.matrix_world.to_3x3()))
    skel[REST_SOURCE_TO_RIG_BASIS] = _flat_matrix3(
        _orthonormal_basis(dest_mw3i @ source_mw3))
    skel[REST_SOURCE_TO_RIG_SCALE] = (
        _uniform_scale(source_matrix_world) / _uniform_scale(skel.matrix_world))
    return True


def _store_original_rest(skel):
    """Once: stash each FACIAL bone's rest head/orientation/length.

    These belong to the Ada/source skeleton and stay immutable - every Update
    Rig rebuilds the fit from them.  The RigLogic change of basis is NOT stored
    alongside: ``core/dna_apply`` derives it live from the baked DNA neutral
    frames and the bone's current rest, so it is always in step with the fit.
    """
    for b in skel.data.bones:
        if not b.name.startswith(FACIAL_PREFIX):
            continue
        if REST_HEAD not in b:
            b[REST_HEAD] = [b.head_local.x, b.head_local.y, b.head_local.z]
            b[REST_ROT] = [c for row in b.matrix_local.to_3x3() for c in row]
            b[REST_LEN] = max(b.length, MIN_BONE_LEN)


def _riglogic_bone_count(skel):
    """How many facial bones the RigLogic pose path will drive on this rig."""
    return sum(1 for b in skel.data.bones
               if b.name.startswith(FACIAL_PREFIX) and REST_HEAD in b)


def _find_character_skeleton(context, cage, relink=True):
    """This cage's own skeleton, if it exists. Matched by the ID-pointer link
    (a legacy untagged skeleton picked in the panel is adopted). Returns None
    when the character has none yet.

    ``relink=False`` accepts only a skeleton that is actually in the scene.
    An unlinked one may be the corpse of a hand-made armature join, in which
    case the merged armature (see op_merge) is the better answer; relinking is
    the last resort, tried after adoption fails."""
    from ..core import board
    mh = context.scene.mhfrt
    candidates = []
    sk = mh.skeleton
    # ``owner is None`` accepts a skeleton that was never linked to a cage, so
    # a legacy rig picked in the panel is adopted.  The control board also has
    # no cage pointer and carries the rig id, so without this guard it would be
    # accepted as ANY cage's skeleton - and the fit below would then look for
    # FACIAL_ bones on a board that has none.
    if sk is not None and sk.type == 'ARMATURE' \
            and not board.is_board_armature(sk):
        owner = sk.get(RIG_CAGE_PROP)
        if owner == cage or owner is None:
            candidates.append(sk)
    for o in bpy.data.objects:
        if (o.type == 'ARMATURE' and o.get(RIG_CAGE_PROP) == cage
                and not board.is_board_armature(o)
                and o not in candidates):
            candidates.append(o)
    for skel in candidates:
        if skel.name not in context.view_layer.objects:
            if not relink:
                continue
            try:
                context.scene.collection.objects.link(skel)
            except RuntimeError:
                continue
        return skel
    return None


class MHFRT_OT_fit_skeleton(bpy.types.Operator):
    bl_idname = "mhfrt.fit_skeleton"
    bl_label = "Update Rig"
    bl_description = ("Build this character's facial rig (skeleton + live control "
                      "board), or UPDATE it after you changed the character: re-wrap, "
                      "slide, then press again - same skeleton, no re-import. Each "
                      "character keeps its own rig")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .op_live import stop_running
        stop_running()          # fitting reads the Wrapped key; freeze it
        mh = context.scene.mhfrt
        if (mh.skeleton is not None
                and mh.skeleton.get(rest_tuning.TONGUE_SESSION_PROP)):
            self.report({'ERROR'}, "Finish or Cancel Tongue Edit before Update Rig")
            return {'CANCELLED'}
        cage = mh.cage
        if not cage:
            self.report({'ERROR'}, "Set the Head Cage first")
            return {'CANCELLED'}
        me = cage.data
        if not me.shape_keys or WRAPPED_KEY not in me.shape_keys.key_blocks:
            self.report({'ERROR'}, "Wrap the cage first (need a 'Wrapped' shape key)")
            return {'CANCELLED'}

        mw = np.array(cage.matrix_world)
        rest_w = _apply_affine(mw, get_basis_local(cage))
        wrap_w = _apply_affine(mw, get_wrapped_local(cage))
        tris = bindmod.triangles_of(me)
        bvh = bindmod.build_bvh(rest_w, tris)

        skel = _find_character_skeleton(context, cage, relink=False)
        adopted = False
        if skel is None:
            # The artist may have joined the two armatures by hand: Blender
            # keeps the bones but the facial OBJECT leaves the scene, and with
            # it the rig id and every link.  Recognise that armature and take
            # it back over instead of importing a second skeleton next to it.
            from . import op_merge
            skel = op_merge.adopt_merged_armature(context, cage, mh.target)
            adopted = skel is not None
        if skel is None:
            skel = _find_character_skeleton(context, cage)
        if skel is None:
            skel = self._import_skeleton(context, cage, mh.target)
            if skel is None:
                return {'CANCELLED'}
        # Visual-only: remember where the bones are on screen RIGHT NOW, before
        # the fit rewrites them. On a first build that is Ada's generic rig as
        # imported, so the artist watches it travel onto this character; on an
        # update it is the previous fit. Pure read - the fit is untouched by it.
        from ..ui import transition as vtrans
        bones_prev = vtrans.bones_snapshot(skel)
        # Refitting the REST pose re-deforms everything already bound to this
        # skeleton, so those meshes have to travel with the bones - otherwise
        # the character snaps to its final shape while only the joints glide.
        # Their shape comes from the Armature modifier, hence evaluated=True.
        bound_prev = [(obj, vtrans.mesh_snapshot(obj, evaluated=True))
                      for obj in _deformed_meshes(context, skel)]
        # Fit against the parent armature's REST evaluation.  This makes Update
        # Rig safe even when the character is currently on an animated frame.
        with _parent_armature_rest_position(context, skel):
            # align the skeleton to the cage's CURRENT rest (handles move + Apply)
            n = len(me.vertices)
            source_matrix_world = _skeleton_alignment(context, cage, n)
            if not skel.get(MERGED_BODY_PROP):
                skel.matrix_world = source_matrix_world
            _store_original_rest(skel)
            # head-size ratio for RigLogic translation deltas (see _global_stretch)
            skel["mhfrt_riglogic_scale"] = _global_stretch(rest_w, wrap_w)

            source_mw = np.array(source_matrix_world)
            source_mw3 = np.array(source_matrix_world.to_3x3())
            dest_mw3 = np.array(skel.matrix_world.to_3x3())
            dest_mw3i = np.linalg.inv(dest_mw3)
            dest_mwi = skel.matrix_world.inverted()
            skel[REST_SOURCE_TO_RIG_BASIS] = _flat_matrix3(
                _orthonormal_basis(dest_mw3i @ source_mw3))
            # Bone lengths and RigLogic translations are stored in the DNA
            # source's units.  They match the skeleton's own units unless the
            # rig lives inside a merged armature at another object scale.
            length_scale = (_uniform_scale(source_matrix_world)
                            / _uniform_scale(skel.matrix_world))
            skel[REST_SOURCE_TO_RIG_SCALE] = length_scale

            infos = []  # (name, length, bind)
            for b in skel.data.bones:
                if not b.name.startswith(FACIAL_PREFIX) or REST_HEAD not in b:
                    continue
                rest_head_local = np.array(b[REST_HEAD], dtype=float)
                rest_rot_local = np.array(b[REST_ROT], dtype=float).reshape(3, 3)
                head_world = _apply_affine(source_mw, rest_head_local[None, :])[0]
                rot_world = source_mw3 @ rest_rot_local
                bind = bindmod.bind_oriented(head_world, rot_world, rest_w, tris, bvh)
                infos.append((b.name, float(b[REST_LEN]), bind))
            if not infos:
                self.report({'ERROR'}, "No FACIAL_* bones found")
                return {'CANCELLED'}

            # Update Rig runs on hidden armatures all the time - the facial rig
            # is hidden by the workflow itself, and a MERGED one is the artist's
            # character rig, usually hidden too. Edit Mode needs it shown.
            with organization.shown_for_edit(
                    skel, view_layer=context.view_layer):
                context.view_layer.update()
                context.view_layer.objects.active = skel
                bpy.ops.object.mode_set(mode='EDIT')
                eb = skel.data.edit_bones
                moved = 0
                for name, length, bind in infos:
                    pos_w, R_w = bindmod.reconstruct_oriented(bind, wrap_w, tris)
                    if pos_w is None or np.isnan(pos_w).any():
                        continue
                    bone = eb.get(name)
                    if bone is None:
                        continue
                    head = dest_mwi @ Vector(pos_w)
                    y_axis = dest_mw3i @ R_w[:, 1]
                    z_axis = dest_mw3i @ R_w[:, 2]
                    yn = np.linalg.norm(y_axis)
                    y_axis = (y_axis / yn if yn > 1e-9
                              else np.array([0.0, 1.0, 0.0]))
                    bone.use_connect = False
                    bone.head = head
                    bone.tail = head + Vector(y_axis) * length * length_scale
                    bone.align_roll(Vector(z_axis))
                    moved += 1
                tuned = rest_tuning.apply_manual_deltas_edit_mode(skel)
                bpy.ops.object.mode_set(mode='OBJECT')
            converted = _riglogic_bone_count(skel)

        mh.skeleton = skel
        # the rig is not optional: every fitted skeleton gets its control
        # board and goes live immediately (multi-character safe)
        err = op_rig.setup_rig(context, skel, cage=cage, target=mh.target)
        if err:
            self.report({'WARNING'},
                        f"Fitted {moved} joints, but the rig is not live: {err}")
            vtrans.request(context, meshes=bound_prev,
                           bones=(skel, bones_prev, True))
            return {'FINISHED'}

        changed = _changed_bone_count(skel, bones_prev)
        from ..ui import flow
        # Building the bones is only half of Step 4. Stay here until the
        # artist explicitly chooses Standalone or merges into the character.
        connection_done = flow.rig_done(mh)
        if connection_done:
            flow.advance(context, 'RIG')
        if changed == 0 and not adopted:
            message = (
                f"Rig refreshed: {moved} joints checked; none changed "
                "position, so no transition was needed"
            )
        else:
            message = (
                (f"Adopted merged armature '{skel.name}': " if adopted else
                 "Rig updated: ")
                + f"{moved} joints fitted"
                + (f", {changed} changed position" if changed is not None else "")
                + f", {tuned} fine-tuned, {converted} RigLogic bases, "
                "control board live"
            )
        if not connection_done:
            message += " - now choose Standalone or Merge Into Character"
        self.report({'INFO'}, message)
        # last act before returning (transition contract): glide the bones
        # from where the artist saw them to the rest pose fitted above
        # Hide the armature even when it is a MERGED one carrying the body
        # rig: the overlay redraws every bone, the unchanged ones included, so
        # nothing disappears - and leaving it visible would show the final
        # joints immediately, which reads as the rig snapping.
        vtrans.request(context, meshes=bound_prev,
                       bones=(skel, bones_prev, True))
        return {'FINISHED'}

    def _import_skeleton(self, context, cage, target):
        from ..core import registry, provenance
        blend = get_cage_blend_path(context)
        record = (registry.for_object(context.scene, cage)
                  or registry.for_object(context.scene, target)
                  or registry.active(context.scene))
        with provenance.track(record):
            return self._append_skeleton(context, blend, cage, target)

    def _append_skeleton(self, context, blend, cage, target):
        with bpy.data.libraries.load(blend, link=False) as (data_from, data_to):
            if SKEL_SOURCE not in data_from.objects:
                self.report({'ERROR'}, f"skeleton not found inside {blend}")
                return None
            data_to.objects = [SKEL_SOURCE]
        appended = [o for o in data_to.objects if o is not None]
        if not appended:
            self.report({'ERROR'}, "Skeleton append failed")
            return None
        skel = appended[0]
        context.scene.collection.objects.link(skel)
        base = (target.name if target is not None else cage.name)
        skel.name = f"{base}_FacialRig"
        skel.data.name = skel.name
        _strip_body_bones(context, skel)
        # force: the joints were just (re)fitted and bones may have been
        # stripped, so the previous grouping stamp cannot be trusted.
        organization.organize_skeleton(skel, force=True)
        return skel


_classes = (MHFRT_OT_fit_skeleton,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
