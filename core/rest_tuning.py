"""Persistent, non-destructive rest-pose tuning for fitted facial bones.

The original ``mhfrt_rest_*`` properties belong to the Ada/source skeleton and
must remain immutable: Update Rig uses them to rebuild the automatic cage fit.
Artist edits therefore live in a separate post-fit matrix/length layer.  The
layer is applied exactly once after every automatic fit, avoiding cumulative
drift while keeping bone names, parents, and RigLogic source data untouched.
"""

from mathutils import Matrix


MANUAL_DELTA_PROP = "mhfrt_manual_rest_delta"
MANUAL_LENGTH_SCALE_PROP = "mhfrt_manual_length_scale"
MANUAL_ROTATION_PROP = "mhfrt_manual_motion_rotation"
MANUAL_HEAD_RADIUS_SCALE_PROP = "mhfrt_manual_head_radius_scale"
MANUAL_TAIL_RADIUS_SCALE_PROP = "mhfrt_manual_tail_radius_scale"

BACKUP_MATRIX_PROP = "mhfrt_tune_backup_matrix"
BACKUP_LENGTH_PROP = "mhfrt_tune_backup_length"
BACKUP_HEAD_RADIUS_PROP = "mhfrt_tune_backup_head_radius"
BACKUP_TAIL_RADIUS_PROP = "mhfrt_tune_backup_tail_radius"

TUNE_DONE_PROP = "mhfrt_tune_done"
EYE_LEFT_DONE_PROP = "mhfrt_tune_eye_left"
EYE_RIGHT_DONE_PROP = "mhfrt_tune_eye_right"
TONGUE_DONE_PROP = "mhfrt_tune_tongue"
TONGUE_SESSION_PROP = "mhfrt_tongue_edit_active"
TONGUE_SESSION_CAGE_PROP = "mhfrt_tongue_edit_cage"
TONGUE_SESSION_TARGET_PROP = "mhfrt_tongue_edit_target"

EYE_BONES = {
    'LEFT': (
        "FACIAL_L_Eye",
        "FACIAL_L_EyeParallel",
        "FACIAL_L_Pupil",
    ),
    'RIGHT': (
        "FACIAL_R_Eye",
        "FACIAL_R_EyeParallel",
        "FACIAL_R_Pupil",
    ),
}

# Every tongue joint the DNA defines, in hierarchy order from the base.
#
# All sixteen, deliberately.  Until 4.16.0 this listed eleven, and the five it
# left out - TongueUpper1, the two TongueSide1 and the two TongueSide4 - are
# tongue joints like any other: the region collection is built by name
# ("tongue" in it, see organization._bone_role), so isolating the tongue put all
# sixteen purple bones on screen while the tool selected, snapshotted, committed
# and restored only eleven.  What the artist saw was a partial selection they
# had to fix with a select-all every single time; what they did not see is that
# the five extras then moved WITHOUT being recorded as a manual correction, so
# the next automatic fit put them back where the cage said and tore them off the
# tongue the other eleven had been shaped into.
TONGUE_BONES = (
    "FACIAL_C_Tongue1",
    "FACIAL_C_TongueUpper1",
    "FACIAL_L_TongueSide1",
    "FACIAL_R_TongueSide1",
    "FACIAL_C_Tongue2",
    "FACIAL_C_TongueUpper2",
    "FACIAL_L_TongueSide2",
    "FACIAL_R_TongueSide2",
    "FACIAL_C_Tongue3",
    "FACIAL_C_TongueUpper3",
    "FACIAL_C_TongueLower3",
    "FACIAL_L_TongueSide3",
    "FACIAL_R_TongueSide3",
    "FACIAL_C_Tongue4",
    "FACIAL_L_TongueSide4",
    "FACIAL_R_TongueSide4",
)


def _flat_matrix(matrix):
    return [float(matrix[row][col])
            for row in range(len(matrix)) for col in range(len(matrix))]


def _matrix_prop(value, size):
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) != size * size:
        return None
    return Matrix(tuple(
        tuple(values[row * size + col] for col in range(size))
        for row in range(size)
    ))


def _basis(matrix):
    """Return an orthonormal rotation matrix, discarding scale/shear."""
    return matrix.to_quaternion().to_matrix()


def _delete_prop(owner, name):
    if name in owner:
        del owner[name]


def manual_delta(bone):
    matrix = _matrix_prop(bone.get(MANUAL_DELTA_PROP), 4)
    return matrix if matrix is not None else Matrix.Identity(4)


def manual_motion_rotation(bone):
    delta = _matrix_prop(bone.get(MANUAL_DELTA_PROP), 4)
    if delta is not None:
        final = bone.matrix_local.copy()
        automatic = final @ delta.inverted_safe()
        return _basis(final) @ _basis(automatic).inverted_safe()
    return _matrix_prop(bone.get(MANUAL_ROTATION_PROP), 3)


def manual_translation_scale(bone):
    try:
        return max(0.001, float(bone.get(MANUAL_LENGTH_SCALE_PROP, 1.0)))
    except (TypeError, ValueError):
        return 1.0


def has_manual_tuning(skel, bone_names):
    if skel is None or skel.type != 'ARMATURE':
        return False
    return any(
        (bone := skel.data.bones.get(name)) is not None
        and MANUAL_DELTA_PROP in bone
        for name in bone_names
    )


def snapshot_bones(skel, bone_names):
    """Save the current fitted/tuned rest so an edit can commit or cancel."""
    saved = 0
    for name in bone_names:
        bone = skel.data.bones.get(name)
        if bone is None:
            continue
        bone[BACKUP_MATRIX_PROP] = _flat_matrix(bone.matrix_local)
        bone[BACKUP_LENGTH_PROP] = max(float(bone.length), 1e-6)
        bone[BACKUP_HEAD_RADIUS_PROP] = max(float(bone.head_radius), 1e-9)
        bone[BACKUP_TAIL_RADIUS_PROP] = max(float(bone.tail_radius), 1e-9)
        saved += 1
    return saved


def clear_snapshots(skel, bone_names):
    for name in bone_names:
        bone = skel.data.bones.get(name)
        if bone is None:
            continue
        _delete_prop(bone, BACKUP_MATRIX_PROP)
        _delete_prop(bone, BACKUP_LENGTH_PROP)
        _delete_prop(bone, BACKUP_HEAD_RADIUS_PROP)
        _delete_prop(bone, BACKUP_TAIL_RADIUS_PROP)


def restore_snapshots_edit_mode(skel, bone_names):
    """Restore snapshot matrices. The armature must currently be in Edit Mode."""
    restored = 0
    for name in bone_names:
        bone = skel.data.bones.get(name)
        edit_bone = skel.data.edit_bones.get(name)
        if bone is None or edit_bone is None:
            continue
        matrix = _matrix_prop(bone.get(BACKUP_MATRIX_PROP), 4)
        if matrix is None:
            continue
        edit_bone.matrix = matrix
        edit_bone.length = max(
            1e-6, float(bone.get(BACKUP_LENGTH_PROP, edit_bone.length)))
        edit_bone.head_radius = max(
            1e-9, float(bone.get(BACKUP_HEAD_RADIUS_PROP,
                                 edit_bone.head_radius)))
        edit_bone.tail_radius = max(
            1e-9, float(bone.get(BACKUP_TAIL_RADIUS_PROP,
                                 edit_bone.tail_radius)))
        restored += 1
    return restored


def commit_snapshots(skel, bone_names):
    """Compose the latest Edit Mode change into each bone's post-fit layer.

    This runs after returning to Object Mode, so ``bone.matrix_local`` is the
    newly edited rest matrix.  Repeated tuning sessions compose with the old
    layer rather than baking the layer twice.
    """
    committed = 0
    for name in bone_names:
        bone = skel.data.bones.get(name)
        if bone is None:
            continue
        before = _matrix_prop(bone.get(BACKUP_MATRIX_PROP), 4)
        if before is None:
            continue
        before_length = max(
            1e-6, float(bone.get(BACKUP_LENGTH_PROP, bone.length)))
        after = bone.matrix_local.copy()
        after_length = max(float(bone.length), 1e-6)
        before_head_radius = max(
            1e-9, float(bone.get(BACKUP_HEAD_RADIUS_PROP, bone.head_radius)))
        before_tail_radius = max(
            1e-9, float(bone.get(BACKUP_TAIL_RADIUS_PROP, bone.tail_radius)))
        after_head_radius = max(float(bone.head_radius), 1e-9)
        after_tail_radius = max(float(bone.tail_radius), 1e-9)

        old_delta = manual_delta(bone)
        automatic = before @ old_delta.inverted_safe()
        incremental = before.inverted_safe() @ after
        new_delta = old_delta @ incremental
        old_scale = manual_translation_scale(bone)
        new_scale = old_scale * (after_length / before_length)
        old_head_scale = max(
            0.001, float(bone.get(MANUAL_HEAD_RADIUS_SCALE_PROP, 1.0)))
        old_tail_scale = max(
            0.001, float(bone.get(MANUAL_TAIL_RADIUS_SCALE_PROP, 1.0)))
        new_head_scale = old_head_scale * (
            after_head_radius / before_head_radius)
        new_tail_scale = old_tail_scale * (
            after_tail_radius / before_tail_radius)

        bone[MANUAL_DELTA_PROP] = _flat_matrix(new_delta)
        bone[MANUAL_LENGTH_SCALE_PROP] = max(0.001, float(new_scale))
        bone[MANUAL_HEAD_RADIUS_SCALE_PROP] = max(
            0.001, float(new_head_scale))
        bone[MANUAL_TAIL_RADIUS_SCALE_PROP] = max(
            0.001, float(new_tail_scale))
        motion_rotation = _basis(after) @ _basis(automatic).inverted_safe()
        bone[MANUAL_ROTATION_PROP] = _flat_matrix(_basis(motion_rotation))
        committed += 1
    clear_snapshots(skel, bone_names)
    return committed


def apply_manual_deltas_edit_mode(skel):
    """Apply all saved post-fit layers to freshly auto-fitted EditBones."""
    applied = 0
    for bone in skel.data.bones:
        if MANUAL_DELTA_PROP not in bone:
            continue
        edit_bone = skel.data.edit_bones.get(bone.name)
        if edit_bone is None:
            continue
        delta = _matrix_prop(bone.get(MANUAL_DELTA_PROP), 4)
        if delta is None:
            continue

        automatic = edit_bone.matrix.copy()
        automatic_length = max(float(edit_bone.length), 1e-6)
        edit_bone.matrix = automatic @ delta
        edit_bone.length = automatic_length * manual_translation_scale(bone)

        final_basis = _basis(edit_bone.matrix)
        automatic_basis = _basis(automatic)
        motion_rotation = final_basis @ automatic_basis.inverted_safe()
        bone[MANUAL_ROTATION_PROP] = _flat_matrix(_basis(motion_rotation))
        applied += 1
    return applied


def clear_manual_tuning(skel, bone_names):
    cleared = 0
    for name in bone_names:
        bone = skel.data.bones.get(name)
        if bone is None:
            continue
        had_tuning = MANUAL_DELTA_PROP in bone
        head_scale = max(
            0.001, float(bone.get(MANUAL_HEAD_RADIUS_SCALE_PROP, 1.0)))
        tail_scale = max(
            0.001, float(bone.get(MANUAL_TAIL_RADIUS_SCALE_PROP, 1.0)))
        bone.head_radius = max(1e-9, float(bone.head_radius) / head_scale)
        bone.tail_radius = max(1e-9, float(bone.tail_radius) / tail_scale)
        for prop in (
                MANUAL_DELTA_PROP,
                MANUAL_LENGTH_SCALE_PROP,
                MANUAL_ROTATION_PROP,
                MANUAL_HEAD_RADIUS_SCALE_PROP,
                MANUAL_TAIL_RADIUS_SCALE_PROP,
                BACKUP_MATRIX_PROP,
                BACKUP_LENGTH_PROP,
                BACKUP_HEAD_RADIUS_PROP,
                BACKUP_TAIL_RADIUS_PROP):
            _delete_prop(bone, prop)
        cleared += int(had_tuning)
    return cleared


def reset_manual_tuning_edit_mode(skel, bone_names):
    """Remove tuning and restore the current automatic-fit rest immediately.

    Because the current rest is ``automatic @ delta``, multiplying by the
    inverse delta returns the latest automatic cage fit without running the
    full fitter or risking loss when its external prerequisites are missing.
    The armature must currently be in Edit Mode.
    """
    reset = 0
    for name in bone_names:
        bone = skel.data.bones.get(name)
        edit_bone = skel.data.edit_bones.get(name)
        if bone is None or edit_bone is None:
            continue
        delta = _matrix_prop(bone.get(MANUAL_DELTA_PROP), 4)
        if delta is None:
            continue
        length_scale = manual_translation_scale(bone)
        head_scale = max(
            0.001, float(bone.get(MANUAL_HEAD_RADIUS_SCALE_PROP, 1.0)))
        tail_scale = max(
            0.001, float(bone.get(MANUAL_TAIL_RADIUS_SCALE_PROP, 1.0)))

        edit_bone.matrix = edit_bone.matrix @ delta.inverted_safe()
        edit_bone.length = max(1e-6, edit_bone.length / length_scale)
        edit_bone.head_radius = max(1e-9, edit_bone.head_radius / head_scale)
        edit_bone.tail_radius = max(1e-9, edit_bone.tail_radius / tail_scale)
        for prop in (
                MANUAL_DELTA_PROP,
                MANUAL_LENGTH_SCALE_PROP,
                MANUAL_ROTATION_PROP,
                MANUAL_HEAD_RADIUS_SCALE_PROP,
                MANUAL_TAIL_RADIUS_SCALE_PROP,
                BACKUP_MATRIX_PROP,
                BACKUP_LENGTH_PROP,
                BACKUP_HEAD_RADIUS_PROP,
                BACKUP_TAIL_RADIUS_PROP):
            _delete_prop(bone, prop)
            _delete_prop(edit_bone, prop)
        reset += 1
    return reset
