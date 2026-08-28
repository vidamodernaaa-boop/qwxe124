"""Workflow state - the one place that knows the step order.

The panel draws from it; operators call ``advance()`` when they complete a
step so the active tab follows the user through the pipeline (never
backwards: jumping around manually is always allowed and respected).
"""

from types import SimpleNamespace

from ..core import landmarks as lmdata
from ..core import rest_tuning
from ..props import (RIG_ID_PROP, RIG_CAGE_PROP, RIG_TARGET_PROP,
                     RIG_DESTINATION_PROP, RIG_DEFORM_BONE_PROP,
                     RIG_MERGED_PROP)

MIN_PAIRS = 3
RECOMMENDED_PAIRS = 12

STEP_IDS = ('SETUP', 'LANDMARKS', 'WRAP', 'RIG', 'TUNE', 'BIND', 'PARTS',
            'MORPHS', 'ANIM', 'EXPORT')
# Optional steps: the panel's Next button lights up even when they're empty,
# so the artist can skip past parts/morphs/anim on characters that don't need
# them (a stylised head with no extra teeth, a face without custom morphs).
OPTIONAL_STEPS = frozenset({'PARTS', 'MORPHS', 'ANIM', 'EXPORT'})
STEP_LABELS = {
    'SETUP': "Setup",
    'LANDMARKS': "Landmarks",
    'WRAP': "Wrap",
    'RIG': "Rig",
    'TUNE': "Fine-Tune",
    'BIND': "Bind",
    'PARTS': "Parts",
    'MORPHS': "Morphs",
    'ANIM': "Animation",
    'EXPORT': "Export",
}


# ------------------------------------------------------------ step checks ---

def setup_done(mh):
    return bool(mh.cage and mh.target and mh.cage != mh.target)


def landmarks_done(mh):
    return lmdata.complete_count(mh) >= MIN_PAIRS


def wrap_done(mh):
    from ..ops.op_wrap import WRAPPED_KEY
    cage = mh.cage
    keys = getattr(getattr(cage, "data", None), "shape_keys", None)
    return bool(cage and keys and WRAPPED_KEY in keys.key_blocks)


def rig_built(mh):
    skel = mh.skeleton
    if not (skel and skel.get(RIG_ID_PROP)):
        return False
    cage = skel.get(RIG_CAGE_PROP)
    target = skel.get(RIG_TARGET_PROP)
    return ((cage is None or cage == mh.cage)
            and (target is None or target == mh.target))


def rig_done(mh):
    """The rig exists and the artist made the required connection choice."""
    if not rig_built(mh):
        return False
    skel = mh.skeleton
    choice = str(skel.get(RIG_DESTINATION_PROP, ""))
    if mh.rig_destination in {'STANDALONE', 'EXISTING'} \
            and mh.rig_destination != choice:
        return False
    if choice == 'STANDALONE':
        return not skel.get(RIG_MERGED_PROP) and not (
            skel.parent is not None
            and skel.parent.type == 'ARMATURE'
            and skel.parent_type == 'BONE'
            and skel.parent_bone
        )
    if choice != 'EXISTING':
        return False
    deform = str(skel.get(RIG_DEFORM_BONE_PROP, ""))
    if not deform:
        return False
    if skel.get(RIG_MERGED_PROP):
        bone = skel.data.bones.get(deform)
        return bool(bone is not None and bone.use_deform)
    body = skel.parent
    return bool(
        body is not None
        and body.type == 'ARMATURE'
        and skel.parent_type == 'BONE'
        and skel.parent_bone in body.data.bones
        and deform in body.data.bones
        and body.data.bones[deform].use_deform
    )


def bind_done(mh):
    from ..ops.op_weights import ARM_MOD_NAME
    return bool(mh.target and any(
        m.type == 'ARMATURE' and m.name == ARM_MOD_NAME
        for m in mh.target.modifiers))


def tune_done(mh):
    """Per-character confirmation; old already-bound files stay complete."""
    skel = mh.skeleton
    return bool(rig_done(mh) and skel
                and not skel.get(rest_tuning.TONGUE_SESSION_PROP)
                and (skel.get(rest_tuning.TUNE_DONE_PROP) or bind_done(mh)))


def parts_done(mh):
    """Optional step: checked once at least one part mesh is attached."""
    from ..ops.op_attach import parts_done as _done
    return _done(mh)


def morphs_done(mh):
    from ..ops.op_morphs import morphs_done as _done
    return _done(mh)


def anim_done(mh):
    from ..ops.op_anim import face_animation_action
    return bool(mh.skeleton and face_animation_action(mh.skeleton))


def export_ready(mh):
    if not rig_done(mh):
        return False
    skel = mh.skeleton
    destination = str(skel.get(RIG_DESTINATION_PROP, ""))
    return destination == 'STANDALONE' or bool(skel.get(RIG_MERGED_PROP))


_CHECKS = {
    'SETUP': setup_done,
    'LANDMARKS': landmarks_done,
    'WRAP': wrap_done,
    'RIG': rig_done,
    'TUNE': tune_done,
    'BIND': bind_done,
    'PARTS': parts_done,
    'MORPHS': morphs_done,
    'ANIM': anim_done,
    'EXPORT': export_ready,
}


def steps(context):
    """[SimpleNamespace(id, label, done)] evaluated fresh each redraw."""
    mh = context.scene.mhfrt
    return [SimpleNamespace(id=s, label=STEP_LABELS[s], done=_CHECKS[s](mh))
            for s in STEP_IDS]


def requirement(context, step_id):
    """None if the step is workable, else (message, tab_to_fix_it)."""
    mh = context.scene.mhfrt
    if step_id in ('LANDMARKS', 'WRAP', 'RIG', 'TUNE', 'BIND') and not setup_done(mh):
        return "Needs a Head Cage and a Head Target", 'SETUP'
    if step_id == 'WRAP' and not landmarks_done(mh):
        n = lmdata.complete_count(mh)
        return f"Needs {MIN_PAIRS}+ landmark pairs (have {n})", 'LANDMARKS'
    if step_id == 'RIG' and not wrap_done(mh):
        return "Needs a wrapped cage", 'WRAP'
    if step_id == 'TUNE' and not rig_done(mh):
        if rig_built(mh):
            return "Choose Standalone or merge into the character", 'RIG'
        return "Needs the rig built", 'RIG'
    if step_id == 'BIND' and not tune_done(mh):
        return "Review eye and tongue positions first", 'TUNE'
    if step_id == 'PARTS' and not rig_done(mh):
        return "Finish the Rig connection choice", 'RIG'
    if step_id == 'MORPHS' and not bind_done(mh):
        return "Needs the head bound", 'BIND'
    if step_id == 'ANIM' and not rig_done(mh):
        return "Finish the Rig connection choice", 'RIG'
    if step_id == 'EXPORT' and not rig_done(mh):
        return "Finish the Rig connection choice", 'RIG'
    return None


# ------------------------------------------------------------- navigation ---

def _redraw(context):
    try:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    except AttributeError:
        pass  # background mode


def advance(context, completed_step):
    """An operator finished `completed_step`: pull the user to the next tab,
    but only forward - never yank them back from a later tab."""
    mh = context.scene.mhfrt
    i = STEP_IDS.index(completed_step)
    if i + 1 < len(STEP_IDS) and STEP_IDS.index(mh.ui_tab) <= i:
        mh.ui_tab = STEP_IDS[i + 1]
    _redraw(context)
