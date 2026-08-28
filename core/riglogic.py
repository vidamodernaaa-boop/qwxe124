"""MetaHuman Rig Logic evaluation in pure numpy.

Coefficients are baked from Ada.dna into data/ada_riglogic.npz (+ .json names).
Pipeline:  GUI controls -> raw controls (piecewise linear) -> +PSD products
           -> per joint-group linear maps -> joint delta DOFs (9 per joint:
           Tx,Ty,Tz, Rx,Ry,Rz (deg), Sx,Sy,Sz).

The joint DOFs are deltas against the DNA's NEUTRAL joint transforms, which are
baked alongside them (``j_nt*`` / ``j_nr*`` / ``j_parent``) and served here by
:func:`neutral_joints` and :func:`dna_neutral_world_rotations` - see
``core/dna_apply`` for why the pose path cannot ignore them.

No binary dependency, so it runs on any Blender Python (5.1 = py3.13).
"""

import os
import json
import numpy as np

from .. import paths

_NPZ = None
_META = None
_MORPHS = None
_RAW_MORPHS = None
_PSD = None           # precomputed PSD product structure
_JG = None            # precomputed joint-group matrices
_PER_INPUT = None     # per-input joint-delta columns
_NEUTRAL = None       # DNA neutral joint transforms
_NEUTRAL_WORLD = None  # DNA neutral joint world rotations, Blender space

# Maya (Y-up) -> Blender (Z-up) basis swap: the rotation part of the DNA
# importer's ``M_SWAP``.  Verifies as (x, y, z)_maya -> (x, -z, y)_blender.
S3 = np.array(((1.0, 0.0, 0.0),
               (0.0, 0.0, -1.0),
               (0.0, 1.0, 0.0)))

# Blender ID names are limited to 63 bytes.  These two DNA names differ only
# after that boundary, so letting Blender truncate them produces one ambiguous
# name plus a ``.001`` duplicate.  Keep readable, stable, side-specific aliases
# for the actual KeyBlocks while the picker continues to display the full DNA
# channel names.
_MORPH_KEY_NAME_OVERRIDES = {
    "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtreme_L":
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenX_L",
    "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtreme_R":
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenX_R",
}
_MORPH_SOURCE_BY_KEY_NAME = {
    key_name: source_name
    for source_name, key_name in _MORPH_KEY_NAME_OVERRIDES.items()
}

# Names produced by Blender's old automatic truncation during the removed
# full-stack import.  They are retained only for migration/removal.
_MORPH_LEGACY_KEY_NAMES = {
    "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtreme_L":
        ("McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtre",),
    "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtreme_R":
        ("McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenE.001",),
}

# Boundary tolerance for the GUI->raw piecewise segments, matching the DNA
# importer (_dna_driver_segment): a GUI value within EPS of a segment boundary
# snaps onto it instead of dropping the segment, so float overshoot at segment
# joins doesn't leave a control momentarily undriven.
_SEG_EPS = 1e-5


def _load():
    global _NPZ, _META
    if _NPZ is None:
        npz_path = os.path.join(paths.DATA_DIR, "ada_riglogic.npz")
        json_path = os.path.join(paths.DATA_DIR, "ada_riglogic.json")
        _NPZ = dict(np.load(npz_path))
        with open(json_path, "r", encoding="utf-8") as f:
            _META = json.load(f)
    return _NPZ, _META


def _psd_struct():
    """PSD rows pre-grouped once (they never change after load).

    Returns (rows, starts, cols, vals, nested): `cols`/`vals` are sorted by
    row so each product is the reduceat-range starting at `starts[i]` for
    output input-index `rows[i]`. `nested` is True when any PSD reads another
    PSD's output (needs sequential evaluation in row order).
    """
    global _PSD
    if _PSD is None:
        d, m = _load()
        row, col, val = d["psd_row"], d["psd_col"], d["psd_val"]
        order = np.argsort(row, kind="stable")
        row_s, col_s, val_s = row[order], col[order], val[order]
        rows, starts = np.unique(row_s, return_index=True)
        nested = bool((col_s >= m["raw_count"]).any()) if len(col_s) else False
        _PSD = (rows, starts, col_s, val_s, nested)
    return _PSD


def _joint_groups():
    """Joint-group matrices reshaped once at load (kills the per-eval dict
    lookups + reshape)."""
    global _JG
    if _JG is None:
        d, m = _load()
        groups = []
        for g in range(m["group_count"]):
            inI = d[f"jg{g}_in"]
            outI = d[f"jg{g}_out"]
            vals = d[f"jg{g}_val"]
            if len(inI) == 0 or len(outI) == 0:
                continue
            groups.append((inI, outI, vals.reshape(len(outI), len(inI))))
        _JG = groups
    return _JG


def _load_morphs():
    global _MORPHS
    if _MORPHS is None:
        path = os.path.join(paths.DATA_DIR, "ada_morphs.json")
        if not os.path.exists(path):
            _MORPHS = {"entries": []}
        else:
            with open(path, "r", encoding="utf-8") as f:
                _MORPHS = json.load(f)
    return _MORPHS


# Every facial control the DNA exposes is named CTRL_expressions.<label>.
# Anything else in the raw control list is machinery, not an expression.
_EXPRESSION_PREFIX = "CTRL_expressions."


def _raw_control_label(raw_name):
    return (raw_name[len(_EXPRESSION_PREFIX):]
            if raw_name.startswith(_EXPRESSION_PREFIX) else raw_name)


def _raw_morph_entries():
    """Raw RigLogic controls with no blend-shape channel that DO move bones.

    They are exposed as empty, driven shape keys so an artist can sculpt a
    corrective onto a control that otherwise only moves joints.

    Every expression control gets a row, on equal footing.  A control that
    moves nothing ON ITS OWN is still a real control - ``mouthLipsTogether*``
    only does anything alongside jaw open, and ``jawOpenExtreme`` only past a
    certain jaw angle - so it belongs in the list like any other, not singled
    out.

    What IS left out is anything not named ``CTRL_expressions.*``: a DNA with
    an RBF solver lists its driver inputs among the raw controls (``head.qx``,
    ``neck_01.qw`` and so on, the quaternion components of the bones the solver
    reads). Those are machinery rather than expressions, and a sculptable morph
    row for one is meaningless.
    """
    global _RAW_MORPHS
    if _RAW_MORPHS is not None:
        return _RAW_MORPHS

    _d, meta_data = _load()
    raw_count = int(meta_data.get("raw_count", 0))
    used_raw_inputs = {
        int(entry["input"])
        for entry in _load_morphs().get("entries", ())
        if int(entry.get("input", -1)) < raw_count
    }
    raw_names = meta_data.get("raw_names", ())
    entries = []
    for input_index in range(raw_count):
        if input_index in used_raw_inputs:
            continue
        raw_name = raw_names[input_index]
        if not raw_name.startswith(_EXPRESSION_PREFIX):
            continue
        entries.append({
            "name": _raw_control_label(raw_name),
            "channel": -1,
            "input": input_index,
            "meshes": [],
            "head": True,
            "raw_name": raw_name,
        })
    _RAW_MORPHS = entries
    return _RAW_MORPHS


def available():
    return os.path.exists(os.path.join(paths.DATA_DIR, "ada_riglogic.npz"))


def meta():
    return _load()[1]


def euler_xyz_matrices(angles):
    """Batched ``mathutils.Euler(a, "XYZ").to_matrix()`` for an (n, 3) array.

    Copied from the DNA importer's ``_euler_xyz_matrices_np`` so the pose path
    composes rotations exactly the way the reference runtime does.
    """
    angles = np.asarray(angles, dtype=float).reshape(-1, 3)
    cx, cy, cz = np.cos(angles[:, 0]), np.cos(angles[:, 1]), np.cos(angles[:, 2])
    sx, sy, sz = np.sin(angles[:, 0]), np.sin(angles[:, 1]), np.sin(angles[:, 2])
    m = np.empty((angles.shape[0], 3, 3), dtype=float)
    m[:, 0, 0] = cy * cz
    m[:, 0, 1] = sx * sy * cz - cx * sz
    m[:, 0, 2] = cx * sy * cz + sx * sz
    m[:, 1, 0] = cy * sz
    m[:, 1, 1] = sx * sy * sz + cx * cz
    m[:, 1, 2] = cx * sy * sz - sx * cz
    m[:, 2, 0] = -sy
    m[:, 2, 1] = sx * cy
    m[:, 2, 2] = cx * cy
    return m


def euler_xyz_from_matrices(matrices):
    """Inverse of :func:`euler_xyz_matrices` for an (n, 3, 3) array.

    Batched ``mathutils.Matrix.to_euler("XYZ")``, i.e. the convention a pose
    bone's ``rotation_euler`` uses once its ``rotation_mode`` is XYZ.  That is
    what lets a joint delta be written into an Action as ordinary euler
    channels instead of a matrix (see ops/op_bake_drivers).

    The Y branch is taken from ``-m[2][0] = sin(y)``, which loses its X/Z split
    when |sin(y)| reaches 1; the degenerate branch pins Z at zero and solves X
    from the remaining column, exactly as mathutils does.
    """
    m = np.asarray(matrices, dtype=float).reshape(-1, 3, 3)
    sin_y = -np.clip(m[:, 2, 0], -1.0, 1.0)
    cos_y = np.sqrt(np.maximum(0.0, 1.0 - sin_y * sin_y))
    steady = cos_y > 1e-7
    x = np.where(steady,
                 np.arctan2(m[:, 2, 1], m[:, 2, 2]),
                 np.arctan2(-m[:, 1, 2], m[:, 1, 1]))
    z = np.where(steady, np.arctan2(m[:, 1, 0], m[:, 0, 0]), 0.0)
    return np.stack((x, np.arcsin(sin_y), z), axis=1)


def per_input_joint_deltas():
    """``{input index: (flat DOF indices, values)}`` - one unit input's pose.

    ``inputs_to_joint_deltas`` is linear in the input vector, so column *i* of
    the joint-group matrices already IS input *i*'s entire joint contribution.
    Reading the columns out costs one pass over the non-zeros, where asking the
    evaluator input by input would cost a full dense solve each time - which
    matters when every input needs a baked pose of its own.

    DOF indices are flat into the (joint_count, 9) delta table, so
    ``divmod(dof, 9)`` gives (joint, [Tx Ty Tz Rx Ry Rz Sx Sy Sz]).
    """
    global _PER_INPUT
    if _PER_INPUT is None:
        chunks = {}
        for in_indices, out_indices, matrix in _joint_groups():
            for position, column in enumerate(in_indices.tolist()):
                values = matrix[:, position]
                nonzero = values != 0.0
                if nonzero.any():
                    chunks.setdefault(column, []).append(
                        (out_indices[nonzero], values[nonzero]))
        _PER_INPUT = {
            column: (np.concatenate([part[0] for part in parts]),
                     np.concatenate([part[1] for part in parts]))
            for column, parts in chunks.items()
        }
    return _PER_INPUT


def joint_delta_rows(input_index):
    """(joint_count, 9) deltas for one RigLogic input held at 1.0."""
    _d, m = _load()
    flat = np.zeros(int(m["joint_count"]) * 9)
    found = per_input_joint_deltas().get(int(input_index))
    if found is not None:
        flat[found[0]] = found[1]
    return flat.reshape(-1, 9)


def neutral_joints():
    """Per-joint neutral local transform, straight out of the baked DNA.

    Mirrors the DNA importer's ``_read_neutral_joints``: ``t`` in the DNA's
    translation unit (cm) and ``r`` in degrees, plus the parent table with -1
    for roots.  RigLogic's joint deltas are added to THESE values, so the pose
    path needs them verbatim.
    """
    global _NEUTRAL
    if _NEUTRAL is None:
        d, m = _load()
        count = int(m["joint_count"])
        parents = d["j_parent"].astype(np.intp)
        # The bake keeps the DNA reader's convention of self-parenting a root.
        parents = np.where(parents == np.arange(count), -1, parents)
        _NEUTRAL = {
            "names": list(m["joint_names"]),
            "parents": parents,
            "t": np.stack([d["j_ntx"], d["j_nty"], d["j_ntz"]], axis=1),
            "r": np.stack([d["j_nrx"], d["j_nry"], d["j_nrz"]], axis=1),
        }
    return _NEUTRAL


def dna_neutral_world_rotations():
    """(joint_count, 3, 3) DNA neutral joint world rotations in Blender space.

    Mirrors ``_dna_neutral_world_matrices`` + ``to_blender_matrix``: the neutral
    local matrices are chained down the DNA hierarchy in Maya space, then the
    result is conjugated by the Y-up -> Z-up swap.  Only the rotation is kept;
    the translation is a pure function of the armature scale and the pose path
    never reads it (the joint's rest position is already in the bone).
    """
    global _NEUTRAL_WORLD
    if _NEUTRAL_WORLD is None:
        neutral = neutral_joints()
        parents = neutral["parents"]
        local = euler_xyz_matrices(np.radians(neutral["r"]))
        count = len(local)
        world = np.empty((count, 3, 3))
        for index in range(count):
            parent = int(parents[index])
            # Parents always precede their children in a DNA joint table, so a
            # single forward pass has the parent's world matrix ready.
            if parent < 0 or parent >= index:
                world[index] = local[index]
            else:
                world[index] = world[parent] @ local[index]
        _NEUTRAL_WORLD = S3 @ world @ S3.T
    return _NEUTRAL_WORLD


def morph_entries(head_only=True):
    """Every sculptable morph channel: the DNA's blend shapes, then the raw
    controls that move bones but have no blend shape of their own."""
    entries = list(_load_morphs().get("entries", ()))
    if head_only:
        entries = [entry for entry in entries if entry.get("head", True)]
    entries.extend(_raw_morph_entries())
    return entries


def morph_input_by_name(head_only=True):
    return {entry["name"]: int(entry["input"])
            for entry in morph_entries(head_only=head_only)}


def morph_key_name(source_name):
    """Blender-safe KeyBlock name for one full DNA morph channel name."""
    return _MORPH_KEY_NAME_OVERRIDES.get(source_name, source_name)


def morph_source_name(key_name):
    """Full DNA channel name for a current Blender-safe KeyBlock name."""
    return _MORPH_SOURCE_BY_KEY_NAME.get(key_name, key_name)


def morph_input_by_key_name(head_only=True):
    """RigLogic inputs keyed by the names that can actually exist in Blender."""
    return {
        morph_key_name(entry["name"]): int(entry["input"])
        for entry in morph_entries(head_only=head_only)
    }


def legacy_morph_key_names(source_name=None):
    """Removed full-import aliases, optionally for one source channel."""
    if source_name is not None:
        return _MORPH_LEGACY_KEY_NAMES.get(source_name, ())
    return {
        key_name
        for names in _MORPH_LEGACY_KEY_NAMES.values()
        for key_name in names
    }



def gui_to_raw(gui):
    """GUI control vector (gui_count,) -> raw control vector (raw_count,).

    Piecewise-linear segment evaluation, matching the DNA importer exactly: a
    segment fires only when its GUI value lies within [min(from,to), max(from,to)]
    (using min/max keeps reversed-range segments working, not just from<=to ones);
    values within _SEG_EPS of a boundary snap onto it; anything further out
    contributes zero.
    """
    d, m = _load()
    gi, go = d["gtr_in"], d["gtr_out"]
    gf, gt, gs, gc = d["gtr_from"], d["gtr_to"], d["gtr_slope"], d["gtr_cut"]
    lo = np.minimum(gf, gt)
    hi = np.maximum(gf, gt)
    x = gui[gi].astype(float)
    active = (x >= lo - _SEG_EPS) & (x <= hi + _SEG_EPS)
    xs = np.clip(x, lo, hi)                    # snap tiny boundary overshoot onto the segment
    contrib = np.where(active, xs * gs + gc, 0.0)
    raw = np.zeros(m["raw_count"])
    np.add.at(raw, go, contrib)
    return raw


def raw_to_inputs(raw):
    """raw controls -> full input vector [raw ; PSD] of length raw_count+psd_count.

    PSD row/col indices are in the COMBINED input space (PSD outputs start at
    raw_count). Each PSD is a product of its referenced inputs * weights; rows are
    evaluated in increasing index order so any nested dependency is ready first.
    """
    _d, m = _load()
    rc, pc = m["raw_count"], m["psd_count"]
    inputs = np.zeros(rc + pc)
    inputs[:rc] = raw
    rows, starts, cols, vals, nested = _psd_struct()
    if len(rows) == 0:
        return inputs
    if not nested:
        # all PSD factors are raw inputs -> one vectorized pass
        prods = np.multiply.reduceat(inputs[cols] * vals, starts)
        # PSD activation clamps to [0,1] (importer parity)
        inputs[rows] = np.clip(prods, 0.0, 1.0)
        return inputs
    ends = np.append(starts[1:], len(cols))
    for i, s, e in zip(rows.tolist(), starts.tolist(), ends.tolist()):
        p = 1.0
        for c, v in zip(cols[s:e].tolist(), vals[s:e].tolist()):
            p *= inputs[c] * v
        inputs[i] = min(1.0, max(0.0, p))     # increasing row order -> deps ready
    return inputs


def inputs_to_joint_deltas(inputs, last_inputs=None, last_flat=None):
    """full input vector -> (joint_count, 9) delta DOFs.

    Uses ALL rows of every joint group (matches the importer's evaluation; LOD0 =
    full detail and our skeleton is the full 865-joint rig).

    When the caller passes the previous evaluation (`last_inputs` + the flat
    delta vector it produced), groups whose inputs are bit-identical to last
    time are skipped and their rows reused - the pipeline is deterministic, so
    exact comparison is safe and a single control drag only recomputes the few
    groups it feeds.
    """
    _d, m = _load()
    jc = m["joint_count"]
    groups = _joint_groups()
    if (last_inputs is not None and last_flat is not None
            and len(last_inputs) == len(inputs)):
        changed = inputs != last_inputs
        if not changed.any():
            return last_flat.copy().reshape(jc, 9)
        dirty_groups = []
        # Local control edits touch few groups and benefit greatly from reuse;
        # animation commonly dirties almost every group, where the tests plus
        # copy cost more than a clean dense evaluation.  Bail out once the
        # sparse side crosses half of the immutable group table.
        dense_limit = len(groups) // 2
        for group in groups:
            if changed[group[0]].any():
                dirty_groups.append(group)
                if len(dirty_groups) > dense_limit:
                    out = np.zeros(jc * 9)
                    for inI, outI, M in groups:
                        out[outI] = M @ inputs[inI]
                    return out.reshape(jc, 9)
        out = last_flat.copy()
        for inI, outI, M in dirty_groups:
            out[outI] = M @ inputs[inI]
        return out.reshape(jc, 9)
    out = np.zeros(jc * 9)
    for inI, outI, M in groups:
        out[outI] = M @ inputs[inI]
    return out.reshape(jc, 9)


_GUI_SEGS = None      # per-GUI-channel segment table for inversion


def gui_segments():
    """gui channel index -> (raw_out, lo, hi, slope, cut) arrays; cached.

    The bundled DNA feeds every raw control from exactly ONE GUI channel
    (verified over ada_riglogic.npz), which is what makes per-channel
    inversion exact: no channel ever has to negotiate a shared raw output
    with another channel.
    """
    global _GUI_SEGS
    if _GUI_SEGS is None:
        d, m = _load()
        gi, go = d["gtr_in"], d["gtr_out"]
        gf, gt, gs, gc = d["gtr_from"], d["gtr_to"], d["gtr_slope"], d["gtr_cut"]
        lo = np.minimum(gf, gt)
        hi = np.maximum(gf, gt)
        table = {}
        for k in range(len(gi)):
            table.setdefault(int(gi[k]), []).append(
                (int(go[k]), float(lo[k]), float(hi[k]),
                 float(gs[k]), float(gc[k])))
        _GUI_SEGS = table
    return _GUI_SEGS


def gui_channel_range(g):
    """Total [lo, hi] the DNA defines for GUI channel index `g`."""
    segs = gui_segments().get(g)
    if not segs:
        return (0.0, 0.0)
    return (min(s[1] for s in segs), max(s[2] for s in segs))


def invert_raw_frames(raw):
    """(frames, raw_count) raw control values -> (frames, gui_count) GUI values.

    Per-channel piecewise-linear least squares.  For each GUI channel the
    candidate intervals are the spans between its segments' boundaries; on
    each interval the total squared error against the target raw values is
    quadratic in x, so the minimiser is closed-form; the best interval wins
    per frame.  Exact wherever an exact pre-image exists (i.e. for values
    that actually came out of gui_to_raw).
    """
    raw = np.atleast_2d(np.asarray(raw, float))
    _d, m = _load()
    n = raw.shape[0]
    gui = np.zeros((n, m["gui_count"]))
    for g, segs in gui_segments().items():
        outs = np.array([s[0] for s in segs])
        lo = np.array([s[1] for s in segs])
        hi = np.array([s[2] for s in segs])
        sl = np.array([s[3] for s in segs])
        cu = np.array([s[4] for s in segs])
        targets = raw[:, outs]                      # (n, n_seg)
        if not np.abs(targets).max() > 0.0:
            continue                                # channel never leaves rest
        bounds = np.unique(np.concatenate([lo, hi]))
        best_x = np.zeros(n)
        best_err = np.full(n, np.inf)
        for a, b in zip(bounds[:-1], bounds[1:]):
            mid = 0.5 * (a + b)
            act = (lo <= mid + _SEG_EPS) & (mid - _SEG_EPS <= hi)
            if act.any():
                s_a, c_a = sl[act], cu[act]
                t_a = targets[:, act]
                denom = float((s_a * s_a).sum())
                x = ((t_a - c_a) * s_a).sum(axis=1) / denom if denom > 0 \
                    else np.zeros(n)
                x = np.clip(x, a, b)
                err = (((s_a * x[:, None] + c_a) - t_a) ** 2).sum(axis=1)
            else:
                x = np.full(n, np.clip(0.0, a, b))
                err = np.zeros(n)
            if act.size:
                inact = ~act
                if inact.any():
                    err = err + (targets[:, inact] ** 2).sum(axis=1)
            better = err < best_err - 1e-12
            best_x[better] = x[better]
            best_err[better] = err[better]
        best_x[np.abs(best_x) < 1e-6] = 0.0
        gui[:, g] = best_x
    return gui


def evaluate_gui(gui):
    """GUI controls -> (joint_count, 9) joint deltas."""
    return inputs_to_joint_deltas(raw_to_inputs(gui_to_raw(np.asarray(gui, float))))


def evaluate_gui_with_inputs(gui, last_inputs=None, last_flat=None):
    """GUI controls -> (full input vector, joint deltas).

    `last_inputs`/`last_flat` enable the incremental joint-group skip (see
    inputs_to_joint_deltas)."""
    inputs = raw_to_inputs(gui_to_raw(np.asarray(gui, float)))
    return inputs, inputs_to_joint_deltas(inputs, last_inputs, last_flat)


def evaluate_raw(raw):
    """raw controls -> (joint_count, 9) joint deltas."""
    return inputs_to_joint_deltas(raw_to_inputs(np.asarray(raw, float)))


def psd_factors():
    """``{input index: ((factor input index, weight), ...)}`` for every PSD.

    The baked product structure, in the COMBINED input space (PSD outputs start
    at ``raw_count``), regrouped per output row.  Rows come back in increasing
    index order, so a nested PSD's factors always resolve before it does -
    which is what lets a caller rewrite the whole PSD stage as an acyclic
    expression graph (see ops/op_bake_drivers).
    """
    rows, starts, cols, vals, _nested = _psd_struct()
    ends = np.append(starts[1:], len(cols))
    return {
        int(row): tuple((int(column), float(weight))
                        for column, weight in zip(cols[start:end].tolist(),
                                                  vals[start:end].tolist()))
        for row, start, end in zip(rows.tolist(), starts.tolist(),
                                   ends.tolist())
    }


def raw_inputs_feeding(input_index):
    """Set of raw-control indices that feed one input in the full input vector.

    Raw inputs (index < raw_count) trivially feed themselves.  PSD outputs
    (index >= raw_count) are decomposed once through the PSD rows because a
    PSD is a product of its referenced inputs - recursing keeps working when
    a PSD is itself referenced by another PSD.
    """
    _d, meta_data = _load()
    raw_count = int(meta_data["raw_count"])
    rows, starts, cols, _vals, _nested = _psd_struct()
    ends = np.append(starts[1:], len(cols))
    psd_index = {int(row): (int(s), int(e)) for row, s, e
                 in zip(rows.tolist(), starts.tolist(), ends.tolist())}

    result = set()
    stack = [int(input_index)]
    seen = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current < raw_count:
            result.add(current)
            continue
        interval = psd_index.get(current)
        if interval is None:
            continue
        for column in cols[interval[0]:interval[1]].tolist():
            stack.append(int(column))
    return result


def gui_channels_feeding_raws(raw_indices):
    """GUI channel index -> those in raw_indices they feed.

    Each raw control is fed by exactly one GUI channel in the bundled DNA;
    the mapping is derived from gtr_in / gtr_out.  Used by the morph picker
    to zero the correct board controls when isolating a morph.
    """
    if not raw_indices:
        return {}
    d, _m = _load()
    gtr_in = d["gtr_in"]
    gtr_out = d["gtr_out"]
    wanted = set(int(x) for x in raw_indices)
    out = {}
    for index in range(len(gtr_in)):
        raw_idx = int(gtr_out[index])
        if raw_idx not in wanted:
            continue
        gui_idx = int(gtr_in[index])
        out.setdefault(gui_idx, set()).add(raw_idx)
    return out
