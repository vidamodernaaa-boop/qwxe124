"""Apply Rig Logic joint deltas to Blender pose bones.

The math is the MetaHuman DNA importer's, mirrored function for function
(``_joint_pose_basis_constants`` / ``_joint_behavior_targets`` /
``_apply_joint_outputs_np``), so a transferred rig moves bone for bone like the
reference RigLogic runtime.

RigLogic emits 9 numbers per joint: (tx, ty, tz, rx, ry, rz, sx, sy, sz). The
translation and scale are deltas added to the DNA's neutral local values, but
the rotation is *not* an euler-component addition: the DNA's neutral rotation is
the joint's orient, and RigLogic's rotation is applied about the already-oriented
joint's own axes, i.e. composed after it. The joint's local matrix is rebuilt in
its parent's space as

    local' = T(t0 + dt) . E(r0) . E(dr) . S(1 + ds)       (Maya space, XYZ euler)

Adding the euler components instead (E(r0 + dr)) only agrees when r0 is small.
370 of Ada's 838 facial joints carry an orient above 20 degrees (median 16.9,
max 90), which is why the previous implementation - which never read r0 at all -
drifted by a median of 1.7 mm and up to 65 mm at expression extremes.

Leaf joints are the exception: nothing is parented to them, so Unreal's own
integration leaves them on the added form and the two only differ in the joint's
own axes. That distinction is kept here (``compose``) so the result matches the
reference runtime bone for bone. On Ada, 77 of 838 facial joints have children.

The pure change relative to rest is therefore

    D = (T(t0) . E(r0))^-1 . local' = T(R0^T . dt) . (R0^T . rot . S)

Blender's bones do not have to share the DNA joint's axes - and after a transfer
they certainly do not, because every bone was refitted onto the wrapped cage - so
D is carried into the bone's own rest basis with the constant per-bone change of
basis

    K = B^-1 . W . S3      (B = bone.matrix_local, W = DNA rest world matrix,
                            S3 = Maya Y-up -> Blender Z-up rotation)
    matrix_basis = T(K . dt_local . scale) . (K . D3 . K^-1)

Deriving D in DNA space first is what makes this exact: adding euler components
does not survive a change of basis, so the addition has to happen on the DNA's
own numbers before anything is converted.

Two transfer-specific layers sit on top of the importer's constants, both folded
into K (or into the translation) so the per-frame math stays identical:

* ``source_to_rig`` - the DNA source armature's basis mapped into this rig's
  data space, non-identity only for a rig merged into an armature with another
  orientation/scale (see ops/op_skeleton).
* ``rest_tuning.manual_motion_rotation`` - a hand-tuned rest rotation must not
  change how RigLogic drives the joint, so it is pre-multiplied onto the DNA
  basis and cancels out of K, leaving the automatic fit's motion axes intact.
"""

import numpy as np
from mathutils import Matrix

from . import riglogic
from . import rest_tuning

# Written by versions before the importer-exact pose path; no longer read.
REST_CONV_PROP = "dna_rest_basis_conv"

FACIAL_PREFIX = "FACIAL_"

# Maya(Y-up) -> Blender(Z-up) basis swap (from the importer).
M_SWAP = Matrix((
    (1, 0, 0, 0),
    (0, 0, -1, 0),
    (0, 1, 0, 0),
    (0, 0, 0, 1),
))
M_SWAP_T = M_SWAP.transposed()

# Below this the pose bone is left alone; matches the importer's write gate.
BASIS_EPS = 1e-7


def evaluated_pose_bones(arm_obj, depsgraph):
    """The armature's pose bones as the RENDER depsgraph has them, or None.

    None for a viewport graph, where the original armature is already the
    authority: Blender copies each evaluated pose back onto it, so reading the
    original costs nothing and says the same thing.

    A render graph copies nothing back - the same asymmetry
    ``op_rig._evaluated_controls`` exists for, and it bites harder here.
    Posing this rig is a read-modify-write of the WHOLE armature: the bones the
    DNA does not drive are read off the pose and written straight back so they
    land exactly as they were.  During a render "as they were" came off the
    ORIGINAL, which by then holds whatever pose the viewport last left on it -
    and writing it re-syncs the render's copy from the original without
    re-running the animation, so every bone the artist ANIMATED reverted to
    that stale pose and stayed there for the whole job.  Only bones with an
    fcurve, a driver or an NLA strip are affected; everything the rig writes
    itself is correct either way, which is what makes it look like the body
    animation was never read rather than like the rig doing it.

    Reported by Souhail (2026-08-18): a rigify head keyed over 347 frames,
    turning in the viewport and dead still through a 1090-frame EEVEE render,
    with the face animating correctly on the frozen head.  Measured: the
    original armature sat on frame 1 for the entire job while the render's copy
    moved, and the rendered head matched the original.

    Reading the untouched bones off the render's own copy instead keeps them on
    the frame being rendered.  Verified against the same frames rendered as
    stills - pixel-identical, where before they differed across the whole head.
    """
    if depsgraph is None or getattr(depsgraph, "mode", 'VIEWPORT') != 'RENDER':
        return None
    try:
        bones = arm_obj.evaluated_get(depsgraph).pose.bones
        # A pose the graph has not built to the same length is not a safe
        # source for a positional bulk write; the original still is.
        if len(bones) != len(arm_obj.pose.bones):
            return None
    except (ReferenceError, RuntimeError, AttributeError):
        return None
    return bones


def _orthonormal(matrix):
    """Rotation part of a 3x3/4x4 matrix, without scale or shear."""
    A = np.asarray(matrix, dtype=float)[:3, :3]
    try:
        U, _sigma, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
        return R
    except np.linalg.LinAlgError:
        return np.eye(3)


def matrix3_from_flat(values):
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(vals) != 9:
        return None
    return np.asarray(vals, dtype=float).reshape(3, 3)


def source_to_rig_basis(arm_obj, prop_name):
    """The rig's stored DNA-source -> rig basis, identity when unset."""
    basis = matrix3_from_flat(arm_obj.get(prop_name)) if arm_obj else None
    return np.eye(3) if basis is None else _orthonormal(basis)


def build_targets(arm_obj, joint_names, source_to_rig=None):
    """Resolve the bones this DNA drives, with their pose constants precomputed.

    Mirrors the importer's ``_joint_behavior_targets``.  Every constant is
    derived live from the baked DNA plus the bone's CURRENT rest, so a rig
    saved by an older version starts moving correctly on the next evaluation -
    no re-fit needed.
    """
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        return []
    neutral = riglogic.neutral_joints()
    world_rotations = riglogic.dna_neutral_world_rotations()
    if source_to_rig is None:
        source_to_rig = np.eye(3)

    # Joints something is parented to take the composed rotation (see above).
    parents = neutral["parents"]
    has_children = np.zeros(len(parents), dtype=bool)
    valid = parents >= 0
    has_children[parents[valid]] = True

    swap3 = np.asarray(M_SWAP.to_3x3())
    targets = []
    pose = arm_obj.pose
    for joint_index, joint_name in enumerate(joint_names):
        if not joint_name.startswith(FACIAL_PREFIX):
            continue
        pose_bone = pose.bones.get(joint_name)
        # A pose channel with no ``.bone`` means Blender left this armature's
        # pose desynced from its own bones.  A heavy Scene > New > Full Copy
        # can hand back a copy where EVERY channel is like that (measured:
        # 1263 of 1263 on a 6-character file), and it does NOT heal on a
        # view-layer or depsgraph update.  Reading one raises, and this runs
        # under _rescan inside the depsgraph handler - so the whole rig runtime
        # would go down with it.  Skip: such an armature has no drivable bones
        # anyway, and the next rescan picks it up if Blender rebuilds the pose.
        if pose_bone is None or pose_bone.bone is None:
            continue

        dna_basis = source_to_rig @ world_rotations[joint_index]
        manual_rotation = rest_tuning.manual_motion_rotation(pose_bone.bone)
        if manual_rotation is not None:
            dna_basis = _orthonormal(np.asarray(manual_rotation)) @ dna_basis
        bone_basis = _orthonormal(pose_bone.bone.matrix_local)
        try:
            k_matrix = np.linalg.inv(bone_basis) @ dna_basis @ swap3
            k_inverse = np.linalg.inv(k_matrix)
        except np.linalg.LinAlgError:
            k_matrix = np.eye(3)
            k_inverse = np.eye(3)

        r0 = np.radians(neutral["r"][joint_index])
        rest_rot = riglogic.euler_xyz_matrices(r0)[0]
        # Visual keying and the FBX exporters read euler channels; forcing the
        # mode once here keeps matrix_basis decomposing into rotation_euler.
        if pose_bone.rotation_mode != "XYZ":
            pose_bone.rotation_mode = "XYZ"
        targets.append({
            "bone": pose_bone,
            "name": joint_name,
            "index": joint_index,
            "r0": r0,
            "rest_rot": rest_rot,
            "rest_rot_t": rest_rot.T,
            "k": k_matrix,
            "k_inv": k_inverse,
            "compose": bool(has_children[joint_index]),
            "translation_scale": rest_tuning.manual_translation_scale(
                pose_bone.bone),
        })
    return targets


def build_pose_constants(targets):
    """Stack the per-target constants into the arrays the pose path reads.

    Mirrors the importer's ``_joint_pose_np``, plus ``translation_scale`` for
    this add-on's per-bone Fine-Tune length layer.
    """
    if not targets:
        return None
    return {
        "count": len(targets),
        "index": np.asarray([t["index"] for t in targets], dtype=np.intp),
        "r0": np.asarray([t["r0"] for t in targets], dtype=float),
        "rest_rot": np.asarray([t["rest_rot"] for t in targets], dtype=float),
        "rest_rot_t": np.asarray([t["rest_rot_t"] for t in targets], dtype=float),
        "compose": np.asarray([t["compose"] for t in targets], dtype=bool),
        "k": np.asarray([t["k"] for t in targets], dtype=float),
        "k_inv": np.asarray([t["k_inv"] for t in targets], dtype=float),
        "translation_scale": np.asarray(
            [t["translation_scale"] for t in targets], dtype=float),
        "previous": None,
    }


def pose_bases(constants, rows):
    """The pose basis each target takes for its RigLogic output row.

    ``rows`` is (count, 9) already gathered in target order - the rotation and
    scale two thirds of it, since a basis carries no translation.

    Split out of :func:`apply_joint_outputs` so a caller that only wants to know
    *how a joint would turn* can ask without posing anything.  op_rig's look-at
    measures the eye channels' angular gain that way, and it has to be this
    formula: reading the gain off a different one would leave the solve
    inverting a map the rig does not follow.
    """
    delta_rotation = np.radians(rows[:, 3:6])
    delta_scale = rows[:, 6:9]

    # Leaf joints: E(r0 + dr).  Joints with children: E(r0) . E(dr).
    rotated = riglogic.euler_xyz_matrices(constants["r0"] + delta_rotation)
    compose = constants["compose"]
    if compose.any():
        rotated[compose] = (constants["rest_rot"][compose]
                            @ riglogic.euler_xyz_matrices(delta_rotation[compose]))

    linear = constants["rest_rot_t"] @ rotated
    if np.any(delta_scale):
        # local' = T . E . S(1 + ds): the diagonal scale multiplies the columns.
        linear = linear * (1.0 + delta_scale)[:, None, :]

    return constants["k"] @ linear @ constants["k_inv"]


def joint_pose_arrays(constants, deltas, scale):
    """(basis (n, 3, 3), translation (n, 3)) for one set of joint deltas.

    The arithmetic half of :func:`apply_joint_outputs`, split out so a caller
    that wants the numbers WITHOUT posing anything can ask for them: the
    standalone bake reads one of these per RigLogic input and writes it into an
    Action instead of onto the bones.

    Note the asymmetry the delivered rig relies on - the translation is a
    *linear* map of the delta (so per-input translations may simply be added
    back up), while the rotation is not.
    """
    rows = np.asarray(deltas, dtype=float)[constants["index"]]
    basis = pose_bases(constants, rows)
    local_translation = np.einsum(
        "nij,nj->ni", constants["rest_rot_t"], rows[:, 0:3])
    translation = (np.einsum("nij,nj->ni", constants["k"], local_translation)
                   * (float(scale) * constants["translation_scale"])[:, None])
    return basis, translation


def _bone_rows(arm_obj, targets, constants):
    """Target row -> index into ``arm_obj.pose.bones``, resolved once.

    Re-resolved whenever the armature's bone count changes, which is the only
    thing that can move the positions this depends on.  None means a driven
    bone is not in the armature any more; the caller writes bone by bone then,
    where a missing one costs a ReferenceError the runtime already handles.
    """
    pose = getattr(arm_obj, "pose", None)
    if pose is None:
        return None
    count = len(pose.bones)
    if constants.get("bone_rows") is not None \
            and constants.get("bone_rows_count") == count:
        return constants["bone_rows"]
    index = {bone.name: i for i, bone in enumerate(pose.bones)}
    try:
        rows = np.array([index[t["bone"].name] for t in targets], dtype=int)
    except (KeyError, ReferenceError):
        return None
    constants["bone_rows"] = rows
    constants["bone_rows_count"] = count
    return rows


def _write_bases_bulk(arm_obj, targets, constants, basis, translation,
                      changed_rows, source_bones=None):
    """Write the changed bones in ONE call. True when it happened.

    Assigning ``matrix_basis`` bone by bone runs an RNA update per write, and
    a posed-bone update queues a window-manager notifier.  During a render
    those writes happen on the render job's thread while Blender's main thread
    is walking that same notifier queue - an unsynchronised list, walked and
    appended to at once, which is where Blender died reading a half-linked
    entry.  ``foreach_set`` writes the whole armature without running a single
    update callback, so the render thread stops touching the queue at all; the
    ``update_tag`` that follows is what tells the depsgraph, exactly as the API
    documents for foreach_set.  It is also ~50x faster: 838 bones in 0.3 ms
    against 17 ms one at a time.

    Blender hands out these matrices in its own column-major order, so what
    goes into the buffer is the transpose of the mathutils matrix the per-bone
    path builds - verified against it bone for bone.
    """
    rows = _bone_rows(arm_obj, targets, constants)
    if rows is None:
        return False
    pose_bones = arm_obj.pose.bones
    count = len(pose_bones)
    buffer = constants.get("matrix_buffer")
    if buffer is None or buffer.size != count * 16:
        buffer = np.empty(count * 16, dtype=np.float32)
        constants["matrix_buffer"] = buffer
    # Read first: everything this rig does not drive - the artist's body bones
    # - has to go back exactly as it was.  During a render "as it was" is the
    # frame being RENDERED, which only the render's own copy still knows; see
    # :func:`evaluated_pose_bones`.
    source = pose_bones if source_bones is None else source_bones
    source.foreach_get("matrix_basis", buffer)
    view = buffer.reshape(count, 4, 4)
    picked = rows[changed_rows]
    view[picked, :3, :3] = np.transpose(basis[changed_rows], (0, 2, 1))
    view[picked, 3, :3] = translation[changed_rows]
    view[picked, :3, 3] = 0.0
    view[picked, 3, 3] = 1.0
    pose_bones.foreach_set("matrix_basis", buffer)
    return True


def _seed_from_render(arm_obj, source_bones, constants):
    """Put the frame being rendered back onto the original pose.

    The bulk path gets this for free - its buffer is read from `source_bones`
    to begin with.  The per-bone fallback writes only the DNA's own bones, so
    without this the artist's animated ones would be left on whatever the
    original still held and the render would revert to it; see
    :func:`evaluated_pose_bones`.
    """
    pose_bones = arm_obj.pose.bones
    count = len(pose_bones)
    buffer = constants.get("matrix_buffer")
    if buffer is None or buffer.size != count * 16:
        buffer = np.empty(count * 16, dtype=np.float32)
        constants["matrix_buffer"] = buffer
    try:
        source_bones.foreach_get("matrix_basis", buffer)
        pose_bones.foreach_set("matrix_basis", buffer)
    except (ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def apply_joint_outputs(arm_obj, targets, constants, deltas, scale,
                        bulk=True, depsgraph=None):
    """deltas: (joint_count, 9) [Tx Ty Tz, Rx Ry Rz (deg), Sx Sy Sz].

    Mirrors the importer's ``_apply_joint_outputs_np``: the whole rig is solved
    in one vectorised pass and only the bones whose basis actually moved are
    written back.  ``bulk=False`` forces the bone-by-bone path, which exists
    for the equivalence test and as the fallback when the bone order cannot be
    resolved (see :func:`_write_bases_bulk`).

    Pass `depsgraph` whenever there is one to hand - a handler always has one.
    It is what keeps a render reading the bones it does not drive off the frame
    being rendered instead of off the original (see
    :func:`evaluated_pose_bones`); without it a render is posed correctly and
    animated bones freeze.
    """
    if not targets or constants is None:
        return False
    source_bones = evaluated_pose_bones(arm_obj, depsgraph)
    basis, translation = joint_pose_arrays(constants, deltas, scale)

    flat = np.concatenate(
        (basis.reshape(len(targets), 9), translation), axis=1)
    previous = constants.get("previous")
    if previous is not None and previous.shape == flat.shape:
        changed_rows = np.flatnonzero(
            np.any(np.abs(previous - flat) > BASIS_EPS, axis=1))
    else:
        changed_rows = np.arange(flat.shape[0])
    constants["previous"] = flat

    if changed_rows.size == 0:
        return False

    if not (bulk and _write_bases_bulk(arm_obj, targets, constants, basis,
                                       translation, changed_rows,
                                       source_bones)):
        if source_bones is not None:
            _seed_from_render(arm_obj, source_bones, constants)
        for row in changed_rows.tolist():
            m = basis[row]
            t = translation[row]
            targets[row]["bone"].matrix_basis = Matrix((
                (m[0][0], m[0][1], m[0][2], t[0]),
                (m[1][0], m[1][1], m[1][2], t[1]),
                (m[2][0], m[2][1], m[2][2], t[2]),
                (0.0, 0.0, 0.0, 1.0),
            ))
    arm_obj.update_tag()
    return True


def clear_pose(targets, constants=None):
    """Put every driven bone back on its rest pose (identity basis)."""
    if not targets:
        return False
    identity = Matrix.Identity(4)
    for target in targets:
        target["bone"].matrix_basis = identity
    if constants is not None:
        constants["previous"] = None
    return True
