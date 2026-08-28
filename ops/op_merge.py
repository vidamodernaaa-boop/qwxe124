"""Merge the facial rig into the character's existing armature - ONE skeleton.

Game engines want a single skeleton per character.  This module joins the
fitted facial skeleton into the body armature the artist already has, so the
exported FBX carries one hierarchy: body bones, with the FACIAL_* bones
parented under the head.  The add-on keeps working exactly as before, because
the merged armature simply BECOMES this character's skeleton - same rig id,
same ID-pointer links to cage / head / control board, same RigLogic runtime.

The merged rig is mathematically identical to the two-armature setup:

* the facial bones are parented to the body's head bone, so they inherit the
  body animation the separate armature used to inherit through object bone
  parenting,
* pose-bone locations live in ARMATURE space, so RigLogic joint translations
  are rescaled by the two armatures' object-scale ratio (``merge_scale``),
* deform weights move from the HEAD DEFORM BONE to the facial bones and from
  nowhere
  else: the facial bones hang under the head, so weight handed to them deforms
  identically at rest, while every other bone the artist painted keeps its
  share; affected vertices are limited and normalized after the hand-off (see
  bind_weights_to_merged_rig),
* every Armature modifier collapses to ONE pointing at the merged rig; two
  modifiers on the same armature would deform the mesh twice.

Verified numerically against the two-armature stack: the face deforms within
0.0004 mm of the separate rigs and the head bone still moves exactly the
vertices it was painted to (0.0002 mm). Already-normalized skin totals remain
unchanged.

The bone REST bases change when Blender joins armatures (it re-derives every
bone roll), so a merge always finishes with a full Update Rig: the fit rebuilds
each bone from the cage bind in the merged armature's space and recomputes the
DNA conversion matrices against the bones' real bases.

This IS the non-standalone choice: a character is either a standalone facial
rig or one merged armature. There is no bone-parented two-armature step any
more - it only ever existed as a stop on the way here, and it shipped
characters with two skeletons by accident. Files saved with the old attachment
still open: the merge unparents and joins them exactly the same way, and it
does the cage body-weight transfer that attaching used to do.
"""

import bpy
from mathutils import Matrix

from ..props import (RIG_ID_PROP, RIG_CAGE_PROP, RIG_TARGET_PROP,
                     RIG_GUI_COLL_PROP, RIG_MERGED_PROP, RIG_MERGE_BONE_PROP,
                     RIG_DESTINATION_PROP, RIG_DEFORM_BONE_PROP,
                     RIG_BODY_ARMATURE_PROP)
from ..core import board as boardmod
from ..core import organization, rest_tuning
from .op_weights import (ARM_MOD_NAME, MAX_INFLUENCES,
                         transfer_body_weights_to_cage,
                         clear_cage_body_weights)

FACIAL_PREFIX = "FACIAL_"

# Marks a mesh whose weights the merge has taken over, so the take-over is
# handed back before it is applied again (re-merge, re-transfer, un-merge).
MERGED_WEIGHTS_PROP = "mhfrt_merged_weights"
MERGED_CARRIER_PROP = "mhfrt_merged_weight_carrier"

# Blender's Armature modifier ignores a vertex whose matching bone weights sum
# to 0.0001 or less - it is left completely undeformed (measured exactly:
# 0.000101 deforms, 0.0001 does not).  The weight transfer normalises facial
# weights against the FULL MetaHuman skin, so neck and scalp vertices end up
# with facial shares far below that.  The merge has to use the same threshold
# or those vertices would suddenly be driven by the face.
DEFORM_EPSILON = 1e-4

# Generic words the panel may SUGGEST as the hierarchy parent. The actual
# Parent Bone and Head Deform Bone remain separate, visible artist choices.
HEAD_BONE_CANDIDATES = ("head", "Head")

# Rigs whose head DEFORM bone cannot be reached from the head CONTROL bone.
# Rigify is the one everybody hits: its ``head`` is an unweighted control, its
# deform chain is a separate DEF- hierarchy, and the skin is on DEF-spine.006 -
# so every weight-based rule below finds an unweighted parent, no single painted
# child, and a body full of equally painted DEF bones, and gives up. The artist
# then has to know Rigify's spine numbering to fill the field in by hand.
#
# A NAME is still never trusted on its own: a match here is only offered after
# the same painted() check every other route has to pass, so a rig that merely
# borrows the spelling cannot smuggle in an unweighted bone.
HEAD_DEFORM_CANDIDATES = ("DEF-spine.006", "DEF-spine.06", "DEF-head")


def _name_key(name):
    """Letters and digits only, lower case.

    Bone names survive round trips through FBX, Unity and Unreal with their
    punctuation rewritten - ``DEF-spine.006`` comes back as ``DEF_spine_006``
    or ``DEF_spine006`` - and all of those are the same bone.
    """
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


_HEAD_DEFORM_KEYS = frozenset(_name_key(n) for n in HEAD_DEFORM_CANDIDATES)


# ------------------------------------------------------------------ state ---

def is_merged(skel):
    """True when this skeleton is the character's one merged armature."""
    return bool(skel is not None and skel.get(RIG_MERGED_PROP))


def merged_parent_bone(skel):
    if skel is None:
        return ""
    name = str(skel.get(RIG_MERGE_BONE_PROP, ""))
    return name if name and name in skel.data.bones else ""


def merged_deform_bone(skel):
    if skel is None:
        return ""
    name = str(skel.get(RIG_DEFORM_BONE_PROP, ""))
    bone = skel.data.bones.get(name) if name else None
    return name if bone is not None and bone.use_deform else ""


def facial_bone_names(skel):
    """Bones this add-on owns on `skel` (present on merged rigs too)."""
    if skel is None or skel.type != 'ARMATURE':
        return set()
    return {b.name for b in skel.data.bones if b.name.startswith(FACIAL_PREFIX)}


def _body_bone_names(skel):
    if skel is None or skel.type != 'ARMATURE':
        return set()
    return {b.name for b in skel.data.bones
            if not b.name.startswith(FACIAL_PREFIX) and b.use_deform}


def matrix_scale(matrix):
    """Uniform scale of a transform (cube root of the volume ratio)."""
    try:
        det = abs(matrix.to_3x3().determinant())
    except (AttributeError, ValueError):
        return 1.0
    return det ** (1.0 / 3.0) if det > 1e-24 else 1.0


def merge_scale(source_matrix_world, dest_matrix_world):
    """Armature-space unit ratio source -> destination.

    Pose-bone locations are expressed in the armature object's own space, so a
    facial rig aligned at one scale and merged into an armature at another has
    to have its translations (RigLogic deltas, manual tuning offsets, bone
    lengths) multiplied by this ratio to keep the same world-space motion.
    """
    dest = matrix_scale(dest_matrix_world)
    if dest <= 1e-12:
        return 1.0
    return matrix_scale(source_matrix_world) / dest


# ------------------------------------------------------------- weight work ---

def _group_sets(obj, facial_names, body_names):
    groups = obj.vertex_groups
    facial = {g.index for g in groups if g.name in facial_names}
    body = {g.index for g in groups if g.name in body_names}
    return groups, facial, body


def weight_carrier(obj, deform_bone, body_names, facial_names=(), skel=None):
    """The one body group the face is allowed to take weight from.

    ``deform_bone`` is explicitly chosen by the artist and is independent from
    the facial hierarchy's Parent Bone. This is rig-agnostic: control rigs,
    direct-deform rigs and custom naming all follow the same path. Face-region
    overlap remains only as a migration fallback for old files that predate
    the explicit field.
    """
    if obj is None or obj.type != 'MESH':
        return None
    groups = obj.vertex_groups
    index_of = {g.index: g.name for g in groups if g.name in body_names}
    if not index_of:
        return None

    stored = str(obj.get(MERGED_CARRIER_PROP, "") or "")
    if stored and stored in index_of.values():
        return stored

    # Migration from 1.2.0: that release did not persist its guessed carrier.
    # Reproduce its whole-mesh rule once so the misplaced facial share goes
    # back to the same group it came from. After release, the marker is gone
    # and the new explicit/generic resolver below takes over.
    totals = {}
    for vert in obj.data.vertices:
        for element in vert.groups:
            name = index_of.get(element.group)
            if name is not None and element.weight > 0.0:
                totals[name] = totals.get(name, 0.0) + element.weight
    if obj.get(MERGED_WEIGHTS_PROP) and totals:
        return max(totals.items(), key=lambda item: item[1])[0]

    if deform_bone:
        bone = skel.data.bones.get(deform_bone) if skel is not None else None
        if bone is not None and not bone.use_deform:
            return None
        return deform_bone if deform_bone in index_of.values() else None

    facial_names = set(facial_names)
    facial_idx = {g.index for g in groups if g.name in facial_names}
    overlap = {}
    for vert in obj.data.vertices:
        facial_sum = sum(
            element.weight for element in vert.groups
            if element.group in facial_idx and element.weight > 0.0
        )
        for element in vert.groups:
            name = index_of.get(element.group)
            if name is None or element.weight <= 0.0:
                continue
            if facial_sum > DEFORM_EPSILON:
                overlap[name] = overlap.get(name, 0.0) + (
                    element.weight * min(1.0, facial_sum)
                )
    if overlap:
        return max(overlap.items(), key=lambda item: item[1])[0]
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def head_region_bones(skel, parent_bone="", deform_bone="", obj=None):
    """Every bone the FACE may take its weight from.

    The transferred rig is supposed to OWN the face - that is the whole point of
    transferring weights. So the share comes out of the head region as a whole:
    the Parent Bone and everything hanging under it. On a plain body rig that is
    just the head bone, exactly as before. On a character that already has its
    own face rig (jaw, lips, nose, brows - all parented under the head, which is
    how every such rig is built) it is that whole set, so those groups are
    emptied over the face instead of fighting the new rig for it.

    Bones OUTSIDE the head - neck, spine, clavicles - are never touched, which
    is what keeps the artist's neck blend intact.
    """
    names = set()
    if skel is None or skel.type != 'ARMATURE':
        return names
    bones = skel.data.bones

    def add(bone):
        if bone is None or bone.name.startswith(FACIAL_PREFIX):
            return                       # our own bones are not a source
        if bone.use_deform:
            names.add(bone.name)
        for child in bone.children:
            add(child)

    add(bones.get(parent_bone) if parent_bone else None)
    if deform_bone:                      # may live outside the subtree
        bone = bones.get(deform_bone)
        if bone is not None and bone.use_deform:
            names.add(deform_bone)
    if obj is not None:                  # only groups this mesh actually has
        present = {g.name for g in obj.vertex_groups}
        names &= present
    return names


def _limit_and_normalize_deform_weights(
        obj, facial_names, body_names, vertex_indices,
        max_influences=MAX_INFLUENCES):
    """Limit and normalize the combined skin on facially affected vertices.

    Body and facial weights share one Armature modifier after a merge. Keep at
    least one influence from each side where both are present, preserve each
    side's total share while pruning, then normalize their combined total to
    exactly one. Non-bone groups such as masks and selections are untouched.
    """
    if max_influences < 1:
        return 0, 0
    groups = obj.vertex_groups
    facial_idx = {g.index for g in groups if g.name in facial_names}
    body_idx = {g.index for g in groups if g.name in body_names}
    by_index = {g.index: g for g in groups}
    drop = {}
    cleaned = 0
    pruned = 0

    def keep_strongest(elements, count):
        return sorted(
            elements, key=lambda element: element.weight, reverse=True
        )[:count]

    for vertex_index in vertex_indices:
        vert = obj.data.vertices[vertex_index]
        facial = [
            element for element in vert.groups
            if element.group in facial_idx and element.weight > 1e-8
        ]
        body = [
            element for element in vert.groups
            if element.group in body_idx and element.weight > 1e-8
        ]
        if not facial:
            continue

        if body:
            # Split the slots by how much each side actually OWNS the vertex,
            # never by how many groups it happens to have. A character with its
            # own face rig can paint 8+ bones onto one vertex; giving the face
            # a single slot there collapses a smooth blend of facial bones into
            # one rigid influence - which is exactly what "the cage lost its
            # smooth weight paint" looks like. Each side keeps at least one.
            facial_total = sum(element.weight for element in facial)
            body_total = sum(element.weight for element in body)
            both = facial_total + body_total
            share = (facial_total / both) if both > 1e-12 else 0.5
            facial_slots = int(round(max_influences * share))
            facial_slots = max(1, min(facial_slots, max_influences - 1))
            facial_slots = min(facial_slots, len(facial))
            body_slots = max(1, max_influences - facial_slots)
        else:
            facial_slots = min(len(facial), max_influences)
            body_slots = 0

        kept_facial = keep_strongest(facial, facial_slots)
        kept_body = keep_strongest(body, body_slots)
        kept_indices = {
            element.group for element in kept_facial + kept_body
        }
        for element in facial + body:
            if element.group not in kept_indices:
                drop.setdefault(element.group, []).append(vertex_index)
                pruned += 1

        facial_total = sum(element.weight for element in facial)
        body_total = sum(element.weight for element in body)
        kept_facial_total = sum(element.weight for element in kept_facial)
        kept_body_total = sum(element.weight for element in kept_body)
        if kept_facial_total > 1e-12:
            factor = facial_total / kept_facial_total
            for element in kept_facial:
                element.weight *= factor
        if kept_body_total > 1e-12:
            factor = body_total / kept_body_total
            for element in kept_body:
                element.weight *= factor

        total = facial_total + body_total
        if total > 1e-12:
            factor = 1.0 / total
            for element in kept_facial + kept_body:
                element.weight *= factor
        cleaned += 1

    for index, indices in drop.items():
        group = by_index.get(index)
        if group is not None:
            group.remove(indices)
    return cleaned, pruned


def bind_weights_to_merged_rig(obj, facial_names, body_names, deform_bone,
                               rig_id="", skel=None, parent_bone=""):
    """Give the merged armature ONE coherent set of weights per vertex.

    The facial bones hang under the head bone, so they move exactly like it.
    The face takes its share out of the HEAD REGION - the Parent Bone and every
    deform bone under it (see head_region_bones) - and never out of the neck,
    spine or shoulder weights the artist painted:

        facial share = min(what the transfer asked for, what the head region
                           owns here)
        head region  = the rest of what it owned, split as it was before

    * Over the face the head region owns the vertex outright, so the face takes
      ALL of it and the expression plays at full strength. On a character that
      already has its own face rig, that means its jaw/lip/nose/brow groups go
      to zero right there - transferring the weights is precisely the request
      that the new rig, not the old one, drives the face.
    * Across the neck the artist's painted blend is preserved to the same ratio:
      those bones are outside the head region, keep their share, and the
      expression fades out with it.
    * A vertex the head region never owned keeps its bones and loses the facial
      dust.

    Affected vertices are then limited to eight deform groups and normalized.
    Re-runnable: the previous take-over is handed back first, so re-merging or
    re-transferring weights lands on the same numbers.
    """
    if obj is None or obj.type != 'MESH':
        return 0, 0
    carriers = head_region_bones(skel, parent_bone, deform_bone, obj)
    if obj.get(MERGED_WEIGHTS_PROP):
        release_weights_from_merged_rig(
            obj, facial_names,
            weight_carrier(obj, deform_bone, body_names, facial_names, skel),
            carriers)
    if not carriers:
        # No head region resolved (old file, no Parent Bone): fall back to the
        # single explicitly chosen bone, which is what 1.2.0 did.
        carrier = weight_carrier(
            obj, deform_bone, body_names, facial_names, skel)
        if carrier is None:
            return 0, 0
        carriers = {carrier}
    primary = deform_bone if deform_bone in carriers else sorted(carriers)[0]

    groups, facial_idx, _body_idx = _group_sets(obj, facial_names, body_names)
    carrier_idx = {g.index for g in groups if g.name in carriers}
    if not facial_idx or not carrier_idx:
        return 0, 0

    by_index = {g.index: g for g in groups}
    drop = {}
    unpainted = []      # face-owned vertices the head region does not hold
    touched = []
    changed = 0
    for vert in obj.data.vertices:
        facial_sum = 0.0
        facial_elements = []
        carrier_elements = []
        available = 0.0
        for element in vert.groups:
            if element.group in facial_idx:
                if element.weight > 0.0:
                    facial_sum += element.weight
                    facial_elements.append(element)
            elif element.group in carrier_idx and element.weight > 0.0:
                available += element.weight
                carrier_elements.append(element)
        if facial_sum <= 0.0:
            continue
        share = min(facial_sum, max(0.0, available))
        if share <= DEFORM_EPSILON:
            # The head region owns nothing here, so the face has nothing it may
            # take: this vertex belongs to the bones the artist painted it to.
            for element in facial_elements:
                drop.setdefault(element.group, []).append(vert.index)
            if facial_sum > 0.5:
                unpainted.append(vert.index)
            changed += 1
            continue
        if abs(share - facial_sum) > 1e-9:
            factor = share / facial_sum
            for element in facial_elements:
                element.weight = element.weight * factor
        # take `share` out of the head region proportionally, so the bones that
        # keep something keep it in the ratio the artist painted
        keep = max(0.0, available - share) / available
        for element in carrier_elements:
            element.weight *= keep
            if element.weight <= 1e-9:
                drop.setdefault(element.group, []).append(vert.index)
        touched.append(vert.index)
        changed += 1

    for index, indices in drop.items():
        group = by_index.get(index)
        if group is not None:
            group.remove(indices)
    _limit_and_normalize_deform_weights(
        obj, facial_names, body_names, touched)
    obj[MERGED_CARRIER_PROP] = primary
    if rig_id:
        obj[MERGED_WEIGHTS_PROP] = str(rig_id)
    return changed, len(unpainted)


def release_weights_from_merged_rig(obj, facial_names, carrier, carriers=()):
    """Hand the face's share straight back to the bones it came from.

    The inverse of the take-over above, so un-merging (or re-running the weight
    transfer) restores the artist's painted weights - and the mesh keeps
    deforming with the body rig once the facial bones are gone. Where several
    head-region bones shared the vertex, the share goes back in the ratio they
    still hold; where the face took everything, it goes to `carrier`, the one
    bone that is guaranteed to exist.
    """
    if obj is None or obj.type != 'MESH' or not (carrier or carriers):
        return 0
    groups = obj.vertex_groups
    facial_idx = {g.index for g in groups if g.name in facial_names}
    if not facial_idx:
        return 0
    carrier = carrier or sorted(carriers)[0]
    carrier_group = groups.get(carrier) or groups.new(name=carrier)
    carrier_index = carrier_group.index
    region_idx = {g.index for g in groups if g.name in set(carriers)}
    region_idx.add(carrier_index)

    restored = 0
    give_back = []
    for vert in obj.data.vertices:
        facial_sum = 0.0
        region_elements = []
        region_total = 0.0
        for element in vert.groups:
            if element.group in facial_idx:
                facial_sum += element.weight
            elif element.group in region_idx and element.weight > 0.0:
                region_elements.append(element)
                region_total += element.weight
        if facial_sum <= 0.0:
            continue
        if region_total > 1e-9:
            for element in region_elements:
                element.weight = min(
                    1.0,
                    element.weight
                    + facial_sum * (element.weight / region_total))
        else:
            give_back.append((vert.index, min(1.0, facial_sum)))
        restored += 1
    for index, weight in give_back:
        carrier_group.add([index], weight, 'REPLACE')
    if MERGED_WEIGHTS_PROP in obj:
        del obj[MERGED_WEIGHTS_PROP]
    if MERGED_CARRIER_PROP in obj:
        del obj[MERGED_CARRIER_PROP]
    return restored


# ---------------------------------------------------------------- modifiers ---

def retarget_armature_modifiers(obj, old_armatures=(), skel=None,
                                adopt_empty=False):
    """Point this mesh's facial Armature modifier(s) at `skel`.

    Always called BEFORE the join: while both armatures are alive there is no
    ambiguity about which modifier belongs to the facial rig.  (Blender keeps
    the joined-away armature as a zero-user orphan whenever anything still
    references it, so identifying it afterwards is not reliable.)

    `adopt_empty` also claims Armature modifiers whose object went missing -
    what a hand-made Ctrl+J leaves behind on the meshes.
    """
    old = set(old_armatures)
    had_facial = False
    for mod in obj.modifiers:
        if mod.type != 'ARMATURE':
            continue
        if mod.object is not None and mod.object in old:
            had_facial = True
            mod.object = skel
        elif mod.object is None and adopt_empty:
            had_facial = True
            mod.object = skel
    return had_facial


def dedupe_armature_modifiers(obj, skel, claim_name=False):
    """Leave `obj` with exactly ONE Armature modifier for `skel`.

    Two Armature modifiers driven by the same armature deform the mesh twice,
    so after a merge the facial modifier and the body modifier have to become
    a single one.  The earliest modifier in the stack is kept - that is where
    the body rig already sat, before any surface/detail modifiers - and it
    takes over the add-on's name so the workflow still sees the head as bound.
    """
    keep = None
    for mod in list(obj.modifiers):
        if mod.type != 'ARMATURE' or mod.object != skel:
            continue
        if keep is None:
            keep = mod
        else:
            obj.modifiers.remove(mod)
    if keep is None:
        return None
    keep.use_vertex_groups = True
    keep.show_in_editmode = True
    keep.show_on_cage = True
    if claim_name and keep.name != ARM_MOD_NAME:
        if not any(m.name == ARM_MOD_NAME for m in obj.modifiers if m != keep):
            keep.name = ARM_MOD_NAME
    return keep


def rig_meshes(rid, armatures=()):
    """Meshes belonging to this rig or deformed by one of `armatures`."""
    armatures = set(armatures)
    out = []
    rid = str(rid or "")
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if rid and str(obj.get(RIG_ID_PROP) or "") == rid:
            out.append(obj)
            continue
        if any(m.type == 'ARMATURE' and m.object in armatures
               for m in obj.modifiers):
            out.append(obj)
    return out


def rebind_meshes(context, skel, meshes, claim_for=None):
    """Finish the rebind of every mesh that was facial-bound.

    Runs after the join: one Armature modifier each, and weights that add up
    to one armature's worth (see bind_weights_to_merged_rig).  Only the head
    target takes the add-on's modifier name - that is what the workflow reads
    as "the head is bound".
    """
    facial_names = facial_bone_names(skel)
    body_names = _body_bone_names(skel)
    deform_bone = merged_deform_bone(skel)
    parent_bone = merged_parent_bone(skel)
    rig_id = str(skel.get(RIG_ID_PROP) or "")
    rebound = 0
    verts = 0
    unpainted = 0
    for obj in meshes:
        try:
            if obj.type != 'MESH':
                continue
        except ReferenceError:
            continue
        dedupe_armature_modifiers(obj, skel, claim_name=(obj == claim_for))
        rebound += 1
        moved, missed = bind_weights_to_merged_rig(
            obj, facial_names, body_names, deform_bone, rig_id, skel,
            parent_bone)
        verts += moved
        unpainted += missed
    return rebound, verts, unpainted


# ------------------------------------------------------------------ parenting ---

def _reparent_keep_world(obj, parent, bone=""):
    world = obj.matrix_world.copy()
    obj.parent = parent
    if bone:
        obj.parent_type = 'BONE'
        obj.parent_bone = bone
    else:
        obj.parent_type = 'OBJECT'
        obj.parent_bone = ""
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_world = world


def _mode_object(context):
    if context.object and context.object.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass


def suggest_parent_bone(body):
    """A likely head bone, to PRE-FILL the panel's Parent Bone field.

    This convenience recognizes only the generic word ``head``. Arbitrary rigs
    remain fully supported because the artist sees and confirms this field; no
    deform carrier is inferred from a rig-specific naming convention.
    """
    from .op_skeleton import _resolve_bone_name

    return _resolve_bone_name(body, "", HEAD_BONE_CANDIDATES)


def _resolve_parent_bone(body, requested):
    """The artist's Parent Bone on `body`, or "" - never a guess."""
    from .op_skeleton import _resolve_bone_name

    if not requested:
        return ""
    return _resolve_bone_name(body, requested)


def parent_facial_roots(context, skel, bone_name):
    """Parent every parentless facial bone under `bone_name` (Edit Mode).

    This is what makes the merged rig follow the body: the separate armature
    used to be bone-parented to the head, and inside one armature the same
    inheritance comes from the bone hierarchy.
    """
    if not bone_name or skel.data.bones.get(bone_name) is None:
        return 0
    _mode_object(context)
    previous_active = context.view_layer.objects.active
    parented = 0
    # The merged armature is the artist's own character rig and is very often
    # hidden - and the merge itself puts its hide flags back the moment the
    # join is done, so by the time we get here it is hidden again.
    with organization.shown_for_edit(
        skel, view_layer=context.view_layer):
        context.view_layer.update()      # let the base come back enabled
        context.view_layer.objects.active = skel
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            edit_bones = skel.data.edit_bones
            parent = edit_bones.get(bone_name)
            if parent is not None:
                for bone in edit_bones:
                    if not bone.name.startswith(FACIAL_PREFIX):
                        continue
                    if bone.parent is not None:
                        continue
                    bone.use_connect = False  # never snap the head to the tail
                    bone.parent = parent
                    parented += 1
        finally:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass
            if previous_active is not None and \
                    context.view_layer.objects.get(previous_active.name):
                context.view_layer.objects.active = previous_active
    return parented


def _rescale_manual_tuning(skel, factor):
    """Keep Fine-Tune offsets the same size in the merged armature's units.

    The stored delta is relative to the bone's own basis, and Update Rig
    re-derives that basis identically after the merge, so only the translation
    part carries armature-space units.
    """
    if abs(factor - 1.0) <= 1e-9:
        return 0
    rescaled = 0
    for bone in skel.data.bones:
        raw = bone.get(rest_tuning.MANUAL_DELTA_PROP)
        delta = rest_tuning._matrix_prop(raw, 4)
        if delta is None:
            continue
        delta.translation = delta.translation * factor
        bone[rest_tuning.MANUAL_DELTA_PROP] = rest_tuning._flat_matrix(delta)
        rescaled += 1
    return rescaled


# -------------------------------------------------------------- properties ---

def _prop_value(value):
    """Plain-Python copy of an ID property so it survives its owner."""
    for method in ("to_dict", "to_list"):
        converter = getattr(value, method, None)
        if converter is not None:
            try:
                return converter()
            except (TypeError, ValueError):
                pass
    return value


def _addon_props(datablock):
    """The rig properties that follow the facial armature into the merge.

    Provenance is deliberately NOT among them.  Copying "made by this add-on"
    from the facial rig onto the artist's own armature would tell removal it
    was free to delete their body rig, and copying the origin key would point
    two objects at one ledger entry.  Those two describe where a datablock came
    from, and the merge does not change where the body armature came from.
    """
    from ..core.provenance import MADE_PROP, ORIGIN_KEY_PROP
    skip = {MADE_PROP, ORIGIN_KEY_PROP}
    return {key: _prop_value(value) for key, value in datablock.items()
            if (key.startswith("mhfrt_") or key.startswith("dna_"))
            and key not in skip}


def _apply_props(datablock, props):
    for key, value in props.items():
        try:
            datablock[key] = value
        except (TypeError, ValueError):
            pass


# ---------------------------------------------------------- deform bone ---

def _group_weight_max(obj, name):
    """Highest weight painted into `name`, or None when there is no such group.

    Distinguishing "no group" from "group present but empty" matters: they are
    different mistakes with different fixes, and the artist can see the group
    in the list either way."""
    group = obj.vertex_groups.get(name) if obj is not None else None
    if group is None:
        return None
    index = group.index
    best = 0.0
    for vert in obj.data.vertices:
        for element in vert.groups:
            if element.group == index and element.weight > best:
                best = element.weight
    return best


def _group_has_weight(obj, name, facial_names=None):
    """Does `name` actually deform obj?

    On a mesh already bound to a merged rig the facial bones hold weight taken
    OUT of this very group (see bind_weights_to_merged_rig), so over the face
    the group legitimately reads zero. Count that share back, or a re-run would
    report the artist's own painted head weight as missing."""
    group = obj.vertex_groups.get(name) if obj is not None else None
    if group is None:
        return False
    index = group.index
    facial_idx = set()
    stored_carrier = str(obj.get(MERGED_CARRIER_PROP, "") or "")
    if (facial_names and obj.get(MERGED_WEIGHTS_PROP)
            and (not stored_carrier or stored_carrier == name)):
        facial_idx = {g.index for g in obj.vertex_groups
                      if g.name in facial_names}
    for vert in obj.data.vertices:
        total = 0.0
        for element in vert.groups:
            if element.group == index or element.group in facial_idx:
                total += element.weight
        if total > DEFORM_EPSILON:
            return True
    return False


def head_region_has_weight(body, obj, name, facial_names=None):
    """Does `name` deform obj - itself, or through the bones under it?

    A bone named "head" is very often an empty hierarchy node whose skin lives
    entirely on its children (jaw, lips, nose, brows on a character that has its
    own face rig). It still IS the head bone, and the merge takes the face's
    share from that whole region, so it must count as carrying weight.
    """
    if body is None or obj is None:
        return False
    for bone_name in head_region_bones(body, name, "", obj) or {name}:
        if _group_has_weight(obj, bone_name, facial_names):
            return True
    return False


def suggest_deform_bone(body, obj, parent_name="", requested="",
                        facial_names=None):
    """Return a safe painted deform carrier, or ``""`` when ambiguous.

    Production control rigs commonly expose an unweighted ``head`` parent and
    put the actual skin below it - on one child such as ``facial_master``, or
    spread over a whole face rig of jaw/lip/nose/brow bones.  Prefer the
    artist's current choice when it really carries weight, then the hierarchy
    parent - INCLUDING the case where the parent is empty but the bones under it
    are painted, because that is still the bone the head weight lives under and
    the merge takes the share from the region as a whole.  Only when the parent
    has nothing under it either do we look for one *unambiguous* painted child,
    and only after THAT for a known head deform bone by name
    (:data:`HEAD_DEFORM_CANDIDATES`) - which is what stops a Rigify rig, whose
    deform chain is unreachable from its head control, from leaving the field
    empty with no hint as to what belongs in it.
    """
    if body is None or body.type != 'ARMATURE' \
            or obj is None or obj.type != 'MESH':
        return ""

    def painted(name):
        bone = body.data.bones.get(str(name or ""))
        return bool(
            bone is not None
            and bone.use_deform
            and _group_has_weight(obj, bone.name, facial_names)
        )

    if painted(requested):
        return str(requested)
    if painted(parent_name):
        return str(parent_name)
    # the head bone that really has the weight - through its children
    if parent_name and head_region_has_weight(
            body, obj, str(parent_name), facial_names):
        return str(parent_name)

    parent = body.data.bones.get(str(parent_name or ""))
    if parent is not None:
        children = sorted(
            bone.name for bone in parent.children if painted(bone.name)
        )
        if len(children) == 1:
            return children[0]

    # Last resort before giving up: a rig whose head deform bone is a known one
    # (see HEAD_DEFORM_CANDIDATES) and really carries weight here. Only ever
    # reached when the weight-based rules above found nothing, so it breaks a
    # tie that would otherwise have been an empty field.
    known = sorted(bone.name for bone in body.data.bones
                   if _name_key(bone.name) in _HEAD_DEFORM_KEYS
                   and painted(bone.name))
    if len(known) == 1:
        return known[0]

    candidates = sorted(
        bone.name for bone in body.data.bones if painted(bone.name)
    )
    return candidates[0] if len(candidates) == 1 else ""


def _deform_group_candidates(obj, body, limit=6):
    """Deform bones of `body` that really carry weight on obj - i.e. what the
    Head Deform Bone was probably meant to be."""
    if obj is None or obj.type != 'MESH' or body is None:
        return []
    names = {b.name for b in body.data.bones if b.use_deform}
    idx_to_name = {g.index: g.name for g in obj.vertex_groups
                   if g.name in names}
    if not idx_to_name:
        return []
    best = {}
    for vert in obj.data.vertices:
        for element in vert.groups:
            name = idx_to_name.get(element.group)
            if name is not None and element.weight > DEFORM_EPSILON:
                if element.weight > best.get(name, 0.0):
                    best[name] = element.weight
    return [n for n, _w in sorted(best.items(), key=lambda kv: -kv[1])][:limit]


def _missing_deform_weight_message(obj, body, deform_name):
    """Say what is actually wrong, and which bone to pick instead."""
    best = _group_weight_max(obj, deform_name)
    if best is None:
        detail = f"has no '{deform_name}' vertex group"
    elif best <= 0.0:
        detail = f"has a '{deform_name}' vertex group but it is empty"
    else:
        detail = (f"has no usable '{deform_name}' weight "
                  f"(highest weight {best:.6f}; needs more than "
                  f"{DEFORM_EPSILON:.4f})")
    candidates = _deform_group_candidates(obj, body)
    if candidates:
        return (f"Head Target {detail}. Deform bones that DO carry weight on "
                f"it: {', '.join(candidates)} - set Head Deform Bone to the "
                f"one that moves the head")
    return (f"Head Target {detail}, and no deform bone of '{body.name}' "
            f"carries any weight on it - is Head Target the mesh that "
            f"'{body.name}' actually skins?")


def _store_connection(skel, cage, choice, body=None, deform_bone=""):
    for holder in (skel, cage):
        if holder is None:
            continue
        holder[RIG_DESTINATION_PROP] = choice
        if deform_bone:
            holder[RIG_DEFORM_BONE_PROP] = deform_bone
        elif RIG_DEFORM_BONE_PROP in holder:
            del holder[RIG_DEFORM_BONE_PROP]
        if body is not None:
            holder[RIG_BODY_ARMATURE_PROP] = body
        elif RIG_BODY_ARMATURE_PROP in holder:
            del holder[RIG_BODY_ARMATURE_PROP]


class MHFRT_OT_confirm_standalone(bpy.types.Operator):
    bl_idname = "mhfrt.confirm_standalone_rig"
    bl_label = "Use Standalone Facial Rig"
    bl_description = (
        "Confirm that this facial rig stays independent and continue the "
        "workflow without a character armature")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        return bool(mh and mh.skeleton and mh.skeleton.get(RIG_ID_PROP))

    def execute(self, context):
        from ..ui import flow

        mh = context.scene.mhfrt
        skel = mh.skeleton
        if is_merged(skel):
            self.report(
                {'ERROR'},
                "This rig is already merged into the character armature",
            )
            return {'CANCELLED'}
        if skel.parent is not None and skel.parent.type == 'ARMATURE':
            _reparent_keep_world(skel, None)
        clear_cage_body_weights(mh.cage)
        _store_connection(skel, mh.cage, 'STANDALONE')
        mh.rig_destination = 'STANDALONE'
        flow.advance(context, 'RIG')
        self.report({'INFO'}, "Standalone facial rig confirmed")
        return {'FINISHED'}


# ------------------------------------------------------------------ adopt ---

def _has_facial_rest_bones(obj):
    from .op_skeleton import REST_HEAD
    if obj is None or obj.type != 'ARMATURE':
        return False
    for bone in obj.data.bones:
        if bone.name.startswith(FACIAL_PREFIX) and REST_HEAD in bone:
            return True
    return False


def _in_a_scene(obj):
    """Real, reachable object - not a leftover kept alive by a pointer.

    Joining armatures by hand unlinks the source object but cannot free it
    while anything still references it (the panel's own skeleton pointer
    does), so the file keeps a ghost armature that is in no scene at all.
    """
    try:
        return bool(obj.users_scene)
    except (AttributeError, ReferenceError):
        return False


def ghost_skeletons(rid):
    """Rig-tagged armatures of `rid` that no scene contains any more."""
    rid = str(rid or "")
    if not rid:
        return []
    return [obj for obj in bpy.data.objects
            if obj.type == 'ARMATURE'
            and str(obj.get(RIG_ID_PROP) or "") == rid
            and not _in_a_scene(obj)]


def find_adoptable_armature(cage, target=None):
    """An armature that already carries this character's facial bones.

    Covers the artist who joined the two armatures by hand: Blender keeps the
    bones (with their add-on properties) but the facial OBJECT - and with it
    the rig id and every ID-pointer link - is gone from the scene.  The
    cage/head/board are still tagged, so the character can be recognised and
    the merged armature re-adopted.
    """
    if cage is None:
        return None
    rid = str(cage.get(RIG_ID_PROP) or "")
    if not rid and target is not None:
        rid = str(target.get(RIG_ID_PROP) or "")
    if not rid:
        return None                    # this character never had a rig
    for obj in bpy.data.objects:
        if (obj.type == 'ARMATURE' and _in_a_scene(obj)
                and str(obj.get(RIG_ID_PROP) or "") == rid
                and obj.get(RIG_CAGE_PROP) == cage):
            return None                # its own skeleton is still there
    candidates = []
    for obj in bpy.data.objects:
        if not _in_a_scene(obj) or not _has_facial_rest_bones(obj):
            continue
        own = str(obj.get(RIG_ID_PROP) or "")
        if own and own != rid:
            continue                   # belongs to another character
        candidates.append(obj)
    if not candidates:
        return None
    tagged = [obj for obj in candidates
              if str(obj.get(RIG_ID_PROP) or "") == rid]
    if tagged:
        return tagged[0]
    return candidates[0] if len(candidates) == 1 else None


def _board_collection(rid):
    """The control board collection of a rig whose skeleton link was lost."""
    for control in boardmod.controls_for_rig(RIG_ID_PROP, rid).values():
        owner = boardmod.owner_object(control)
        if owner is None:
            continue
        for coll in owner.users_collection:
            if coll.get(organization.COLL_ROLE_PROP) == organization.ROLE_PANEL:
                return coll
        if owner.users_collection:
            return owner.users_collection[0]
    return None


def adopt_merged_armature(context, cage, target=None):
    """Re-attach the add-on to a hand-merged armature.  Returns it, or None.

    Called by Update Rig when the character has no skeleton object left: it
    restores the rig id and the ID-pointer chain, parents the facial bones
    under the head, and rebinds the meshes.  The caller then runs the normal
    fit, which rebuilds the rest bases in the merged armature's space.
    """
    skel = find_adoptable_armature(cage, target)
    if skel is None:
        return None
    rid = str(cage.get(RIG_ID_PROP) or "")
    if not rid and target is not None:
        rid = str(target.get(RIG_ID_PROP) or "")

    # Drop the leftover facial armature the hand-made join could not free: it
    # still carries the rig tag, so the runtime would drive two skeletons from
    # one control board and the Characters list would show the rig twice.
    for ghost in ghost_skeletons(rid):
        if ghost == skel:
            continue
        try:
            bpy.data.objects.remove(ghost, do_unlink=True)
        except (ReferenceError, RuntimeError):
            pass

    skel[RIG_ID_PROP] = rid
    skel[RIG_MERGED_PROP] = True
    skel[RIG_CAGE_PROP] = cage
    if target is not None:
        skel[RIG_TARGET_PROP] = target
    if not isinstance(skel.get(RIG_GUI_COLL_PROP), bpy.types.Collection):
        gui_coll = _board_collection(rid)
        if gui_coll is not None:
            skel[RIG_GUI_COLL_PROP] = gui_coll

    bone_name = merged_parent_bone(skel)
    if not bone_name:
        mh = getattr(context.scene, "mhfrt", None)
        requested = mh.merge_head_bone if mh is not None else ""
        # Repairing a join the artist already made by hand: the facial bones
        # need a parent to follow, so a suggestion beats leaving them loose.
        bone_name = (_resolve_parent_bone(skel, requested)
                     or suggest_parent_bone(skel))
    if bone_name:
        skel[RIG_MERGE_BONE_PROP] = bone_name
        parent_facial_roots(context, skel, bone_name)

    # What hangs off the armature is the bone board's armature object (or, on a
    # pre-3.0 board, the control frames).  A hand-made join re-parents those to
    # the survivor but keeps the old parent inverse, so they can end up
    # anywhere; put them back beside the head.  Control positions are local to
    # their bone or frame, so the rig is untouched either way.
    #
    # Parented to the skeleton OBJECT, not to the head BONE.  Bone-parenting was
    # a permanent follow-head that the board's own follow-head switch could
    # never turn off, and it made a merged character behave differently from a
    # standalone one for no reason the artist could see.  The switch owns that
    # behaviour now (board.install_follow_head), so an existing merged character
    # is un-bone-parented and its switch is pushed up to 1 - same panel, same
    # motion, and now it can be turned off.
    gui_coll = skel.get(RIG_GUI_COLL_PROP)
    if isinstance(gui_coll, bpy.types.Collection):
        roots = [obj for obj in gui_coll.all_objects if obj.parent == skel]
        was_bone_parented = {
            obj.name for obj in roots
            if obj.parent_type == 'BONE' and obj.parent_bone
        }
        for obj in roots:
            if obj.parent_type == 'BONE':
                _reparent_keep_world(obj, skel, "")
        anchor = target if target is not None else cage
        if roots:
            from . import op_rig
            for obj in roots:
                if not boardmod.is_board_armature(obj):
                    continue
                if anchor is not None:
                    op_rig._place_bone_board(skel, obj, anchor)
                op_rig.install_follow_head(skel, obj)
                if obj.name in was_bone_parented:
                    boardmod.set_follow_head_value(
                        obj, "CTRL_faceGUIfollowHead", 1.0)

    # The hand-made join left the facial modifiers with no object at all.
    bound = []
    for obj in rig_meshes(rid, (skel,)):
        claimed = retarget_armature_modifiers(obj, (), skel, adopt_empty=True)
        drives = any(m.type == 'ARMATURE' and m.object == skel
                     for m in obj.modifiers)
        if claimed or drives:
            bound.append(obj)
    rebind_meshes(context, skel, bound, claim_for=target)
    organization.ensure_character_collections(
        context,
        cage=cage,
        target=target,
        skel=skel,
        gui_coll=(board if isinstance(board, bpy.types.Collection) else None),
    )
    return skel


# ------------------------------------------------------------------ merge ---

class MHFRT_OT_merge_body_rig(bpy.types.Operator):
    bl_idname = "mhfrt.merge_body_rig"
    bl_label = "Merge Into One Armature"
    bl_description = (
        "Join the facial bones INTO the existing character armature so the "
        "character exports to a game engine as ONE skeleton. The facial bones "
        "are parented under the head bone, every mesh keeps a single Armature "
        "modifier with reconciled weights, and the whole facial panel - "
        "control board, morphs, animation - keeps working")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        skel = mh.skeleton if mh else None
        return bool(skel is not None and skel.type == 'ARMATURE'
                    and skel.get(RIG_ID_PROP)
                    and not skel.get(rest_tuning.TONGUE_SESSION_PROP))

    # -- resolution -------------------------------------------------------

    def _resolve_body(self, context, mh, facial):
        from .op_skeleton import _target_body_armatures

        body = mh.merge_body_armature
        if body is facial:
            body = None
        if body is None and facial.parent is not None \
                and facial.parent.type == 'ARMATURE':
            body = facial.parent
        if body is None:
            candidates = _target_body_armatures(mh.target, facial)
            if len(candidates) == 1:
                body = candidates[0]
            elif len(candidates) > 1:
                self.report(
                    {'ERROR'},
                    "More than one existing armature drives the Head Target; "
                    "choose Existing Armature explicitly")
                return None
        if body is None or body.type != 'ARMATURE':
            self.report(
                {'ERROR'},
                "Choose Existing Armature (or add its Armature modifier to "
                "the Head Target)")
            return None
        if body.library is not None or body.data.library is not None:
            self.report({'ERROR'},
                        f"'{body.name}' is linked from another file and "
                        "cannot be merged into")
            return None
        if context.view_layer.objects.get(body.name) is None:
            self.report({'ERROR'},
                        f"'{body.name}' is not in the current view layer "
                        "(its collection is excluded or hidden)")
            return None
        return body

    def _check_collisions(self, facial, body):
        body_names = {b.name for b in body.data.bones}
        clashing = sorted(name for name in facial_bone_names(facial)
                          if name in body_names)
        if not clashing:
            return None
        shown = ", ".join(clashing[:3])
        more = f" (+{len(clashing) - 3} more)" if len(clashing) > 3 else ""
        return (f"'{body.name}' already has bones named {shown}{more}. "
                "Merging would rename them and break the rig - remove those "
                "bones first")

    # -- the merge itself -------------------------------------------------

    def execute(self, context):
        from .op_live import stop_running
        from . import op_rig
        from ..core import registry

        mh = context.scene.mhfrt
        facial = mh.skeleton
        if facial is None or facial.type != 'ARMATURE' \
                or not facial.get(RIG_ID_PROP):
            self.report({'ERROR'}, "Build the facial rig first")
            return {'CANCELLED'}
        if facial.get(rest_tuning.TONGUE_SESSION_PROP):
            self.report({'ERROR'},
                        "Finish or Cancel Tongue Edit before merging")
            return {'CANCELLED'}

        if is_merged(facial):
            return self._refresh_merged(context, mh, facial)
        # No attach step to require: merging IS the non-standalone choice, so
        # this runs straight off the fitted rig. Everything the old attach did
        # for the cage's body weights happens below, before the join.
        body = self._resolve_body(context, mh, facial)
        if body is None:
            return {'CANCELLED'}
        clash = self._check_collisions(facial, body)
        if clash:
            self.report({'ERROR'}, clash)
            return {'CANCELLED'}
        # Never guessed: every facial bone will follow this one, so the artist
        # has to have seen and accepted it (the panel pre-fills a suggestion).
        if not mh.merge_head_bone:
            self.report({'ERROR'},
                        "Choose the Parent Bone first - the head bone of "
                        f"'{body.name}' that the whole face will follow")
            return {'CANCELLED'}
        bone_name = _resolve_parent_bone(body, mh.merge_head_bone)
        if not bone_name:
            self.report({'ERROR'},
                        f"Parent Bone '{mh.merge_head_bone}' was not found on "
                        f"'{body.name}'")
            return {'CANCELLED'}
        requested_deform = str(
            mh.merge_deform_bone
            or facial.get(RIG_DEFORM_BONE_PROP, "")
        )
        facial_names = facial_bone_names(facial)
        suggested_deform = suggest_deform_bone(
            body, mh.target, bone_name, requested_deform, facial_names)
        deform_name = suggested_deform or requested_deform
        if suggested_deform and suggested_deform != requested_deform:
            mh.merge_deform_bone = suggested_deform
        deform = body.data.bones.get(deform_name)
        if deform is None or not deform.use_deform:
            self.report(
                {'ERROR'},
                "Choose a valid Head Deform Bone before merging",
            )
            return {'CANCELLED'}
        # an empty "head" whose children carry the skin is a valid choice: the
        # share comes from the region, not from that one group
        if not (_group_has_weight(mh.target, deform_name, facial_names)
                or head_region_has_weight(
                    body, mh.target, bone_name, facial_names)):
            self.report(
                {'ERROR'},
                _missing_deform_weight_message(
                    mh.target, body, deform_name),
            )
            return {'CANCELLED'}

        if context.view_layer.objects.get(facial.name) is None:
            self.report({'ERROR'},
                        f"'{facial.name}' is not in the current view layer "
                        "(its collection is excluded or hidden)")
            return {'CANCELLED'}

        stop_running()
        _mode_object(context)

        # The scene registry owns a PointerProperty to the facial armature.
        # If that pointer is still live when Blender joins the objects, the
        # joined-away armature cannot be deleted: it survives as a tagged
        # orphan beside the merged body.  The Rig-list duplicate adopter then
        # quite reasonably sees two skeletons claiming one character and
        # creates a second row.  Hand registry ownership over as part of the
        # same transaction as the panel pointer below.
        record = (registry.for_object(context.scene, facial)
                  or registry.active(context.scene))

        # ---- save the artist's armature BEFORE we join anything into it ----
        #
        # This is the last moment their rig exists as they built it.  Three
        # things are written down here, all before the first change:
        #
        #   * every bone's rest position, into the ledger in the .blend - the
        #     join re-derives bone rolls, so "put the bones back" needs the
        #     numbers, not just the names;
        #   * the armature ITSELF, written whole into the character's .mhfrt
        #     file, so it can be recovered even if this object is later lost;
        #   * the vertex weights of every mesh the merge is about to rebalance.
        #     The face takes its share out of the head-region groups, and
        #     handing that back by formula is lossy on any character whose head
        #     is carried by more than one bone.
        #
        # Removing the character replays all three.  Taken here, and only here,
        # because a snapshot taken after the merge would record the merge.
        from ..core import provenance
        rid_before = str(facial.get(RIG_ID_PROP) or "")
        # Exactly the meshes the rebind touches - the head, the parts and
        # anything already facial-bound.  Clothing that only ever followed the
        # body armature is not rebalanced, so its weights are not copied.
        weighted = [obj for obj in rig_meshes(rid_before, (facial,))
                    if obj.type == 'MESH']
        if mh.target is not None and mh.target not in weighted:
            weighted.append(mh.target)
        provenance.snapshot_object(record, body,
                                   role=provenance.ROLE_ARMATURE)
        for obj in weighted:
            if obj is not mh.cage:          # the cage is ours; it goes with us
                provenance.snapshot_object(record, obj,
                                           role=provenance.ROLE_BODY,
                                           weights=True)
        saved_armature = provenance.store_armature(record, body)
        if not saved_armature:
            self.report(
                {'WARNING'},
                f"Could not save '{body.name}' to this character's .mhfrt "
                "file before merging - check the Character File Folder in the "
                "add-on preferences. The merge went ahead; Remove can still "
                "rebuild the bones from the record in this .blend")

        # Refresh the cage from the target's latest body weight paint before
        # the two modifiers collapse into one armature.
        cage_verts, _cage_groups = transfer_body_weights_to_cage(
            mh.cage, mh.target, body, facial)
        if not cage_verts:
            self.report(
                {'ERROR'},
                "Could not transfer the Head Target's body weights to the cage",
            )
            return {'CANCELLED'}

        # Everything that can still fail happens BEFORE the first real change:
        # a merge that stops halfway would leave the character half-rebound.
        # `object.join` needs both armatures really selectable, which a hidden
        # object - or a hidden collection ABOVE it - is not; the character's own
        # rig is usually hidden one of those ways.
        with organization.shown_for_edit(facial, body,
                                         view_layer=context.view_layer):
            context.view_layer.update()
            for obj in list(context.selected_objects):
                obj.select_set(False)
            try:
                facial.select_set(True)
                body.select_set(True)
            except RuntimeError:
                self.report(
                    {'ERROR'},
                    "Both armatures must be visible and selectable to merge")
                return {'CANCELLED'}
            context.view_layer.objects.active = body

            rid = str(facial[RIG_ID_PROP])
            props = _addon_props(facial)
            gui_coll = facial.get(RIG_GUI_COLL_PROP)
            gui_objects = set()
            if isinstance(gui_coll, bpy.types.Collection):
                gui_objects = {obj.name for obj in gui_coll.all_objects}
            # Retarget the deform bindings while BOTH armatures are alive: that
            # is the only moment the facial modifier is unambiguous.  Blender
            # keeps a joined-away armature as a zero-user orphan for as long as
            # anything still points at it, so "the object went away" is not a
            # usable signal.
            target = mh.target
            bound_meshes = [obj for obj in rig_meshes(rid, (facial,))
                            if retarget_armature_modifiers(obj, (facial,), body)]

            # Both rigs go to Rest Position first.  The facial armature usually
            # FOLLOWS the head bone, so joining while the body is posed would
            # bake the current animation frame into the facial rest pose.
            pose_backup = [(arm.data, arm.data.pose_position)
                           for arm in (facial, body)]
            for data, _position in pose_backup:
                data.pose_position = 'REST'
            context.view_layer.update()

            factor = merge_scale(facial.matrix_world, body.matrix_world)
            children = [(obj, obj.matrix_world.copy())
                        for obj in bpy.data.objects if obj.parent == facial]

            # Hand the panel's skeleton pointer over BEFORE the join: a live
            # PointerProperty is a user, and a referenced armature is only
            # unlinked, not deleted, leaving a tagged ghost rig behind.  The
            # saved character record owns another such pointer, so both must
            # be released before object.join.
            mh.switching_character = True
            try:
                mh.skeleton = None
                if record is not None and record.skeleton == facial:
                    record.skeleton = None
            finally:
                mh.switching_character = False

            try:
                # Unparent the facial rig at its rest world transform: the join
                # reads object matrices, and an ambiguous bone-parented chain
                # has no business influencing where the bones land.
                world = facial.matrix_world.copy()
                facial.parent = None
                facial.matrix_world = world
                context.view_layer.update()

                context.view_layer.objects.active = body
                result = bpy.ops.object.join()
                if 'FINISHED' not in result:
                    mh.switching_character = True
                    try:
                        mh.skeleton = facial
                    finally:
                        mh.switching_character = False
                    if record is not None:
                        record.skeleton = facial
                    self.report({'ERROR'},
                                "Blender could not join the armatures")
                    return {'CANCELLED'}
            finally:
                for data, position in pose_backup:
                    try:
                        data.pose_position = position
                    except (ReferenceError, AttributeError):
                        pass    # the facial armature is gone: that is the point

        # ---- the merged armature becomes this character's skeleton -------
        _apply_props(body, props)
        body[RIG_ID_PROP] = rid
        body[RIG_MERGED_PROP] = True
        body[RIG_MERGE_BONE_PROP] = bone_name
        body[RIG_DESTINATION_PROP] = 'EXISTING'
        body[RIG_DEFORM_BONE_PROP] = deform_name
        if RIG_BODY_ARMATURE_PROP in body:
            del body[RIG_BODY_ARMATURE_PROP]
        # stamp the cage too: the panel reads the choice back from whichever of
        # the two it finds (a character can be reopened before its skeleton is)
        _store_connection(body, mh.cage, 'EXISTING', deform_bone=deform_name)

        # Complete the registry side of the ownership handoff immediately,
        # before any list refresh or organization pass can inspect the file.
        if record is not None:
            record.skeleton = body
            registry.claim(record, body)
            registry.write_card(record)

        # Blender re-parents the joined object's children to the survivor but
        # keeps their old parent inverse, so restore the transforms ourselves.
        #
        # Everything goes under the OBJECT, the board included.  Bone-parenting
        # the board used to be how it followed the head, but it was a follow the
        # artist could never switch off, and the board ships with a switch for
        # exactly this: `install_follow_head` below turns it into a driven
        # Armature constraint, so the panel follows the head when
        # CTRL_faceGUIfollowHead is up and holds still when it is down - the
        # same control MetaHuman has.
        for obj, world in children:
            try:
                _reparent_keep_world(obj, body, "")
            except ReferenceError:
                continue
        from . import op_rig as _op_rig
        for name in gui_objects:
            obj = bpy.data.objects.get(name)
            if obj is not None and boardmod.is_board_armature(obj):
                # Keeping the world transform through a parent change leaves a
                # transform on the object; push it back into the rest so the
                # board stays at 0/0/0, 1/1/1 the way it was built. Does not
                # move the panel, and restates its units if the survivor of the
                # join is at a different scale.
                _op_rig.reflatten_board(body, obj)
                _op_rig.install_follow_head(body, obj)

        parented = parent_facial_roots(context, body, bone_name)
        _rescale_manual_tuning(body, factor)
        meshes, verts, unpainted = rebind_meshes(context, body, bound_meshes,
                                                 claim_for=target)

        mh.switching_character = True
        try:
            mh.skeleton = body
            mh.merge_body_armature = body
            mh.merge_head_bone = bone_name
            mh.merge_deform_bone = deform_name
            mh.rig_destination = 'EXISTING'
        finally:
            mh.switching_character = False

        organization.ensure_character_collections(
            context,
            cage=mh.cage,
            target=mh.target,
            skel=body,
            gui_coll=(body.get(RIG_GUI_COLL_PROP)
                      if isinstance(body.get(RIG_GUI_COLL_PROP),
                                    bpy.types.Collection) else None),
            root=record.root if record is not None else None,
            record=record,
        )
        context.view_layer.update()
        op_rig.bump_rig_topology()
        op_rig.invalidate()

        # A clip imported before the merge would otherwise be replayed on the
        # character itself by FBX export (see op_anim.ensure_export_safety).
        try:
            from .op_anim import ensure_export_safety
            ensure_export_safety(body)
        except (AttributeError, RuntimeError):
            pass

        # The character file now knows this rig lives inside the artist's own
        # armature, and holds the copy of that armature from before it did.
        try:
            from ..core import sidecar
            sidecar.touch(context.scene, record)
        except Exception:                    # noqa: BLE001 - never fail a merge
            pass

        # Rebuild the rest bases: joining re-derives every bone roll, so the
        # DNA conversion matrices must be recomputed against the real bones.
        refit = bpy.ops.mhfrt.fit_skeleton()
        if 'FINISHED' not in refit:
            self.report({'WARNING'},
                        f"Merged into '{body.name}', but Update Rig did not "
                        "finish - run it once manually")
            return {'FINISHED'}

        if unpainted:
            self.report(
                {'WARNING'},
                f"Merged into '{body.name}', but {unpainted} vertices the face "
                "drives carry no usable head-deform weight, so they keep "
                "following the bones you painted them to and lose their "
                f"expression (facial parent: '{bone_name}')")
        else:
            self.report(
                {'INFO'},
                f"Merged into '{body.name}': one armature, {parented} facial "
                f"root(s) under '{bone_name}', {meshes} mesh(es) rebound"
                + (f", {verts} vertex weights reconciled" if verts else ""))
        return {'FINISHED'}

    def _refresh_merged(self, context, mh, skel):
        """Already merged: re-run the maintenance half (idempotent)."""
        from .op_live import stop_running

        stop_running()
        _mode_object(context)
        rid = str(skel.get(RIG_ID_PROP) or "")
        bone_name = merged_parent_bone(skel)
        if not bone_name:
            bone_name = _resolve_parent_bone(skel, mh.merge_head_bone)
            if bone_name:
                skel[RIG_MERGE_BONE_PROP] = bone_name
        deform_name = merged_deform_bone(skel)
        if not deform_name:
            requested = str(mh.merge_deform_bone or "")
            deform = skel.data.bones.get(requested)
            if deform is not None and deform.use_deform:
                deform_name = requested
                skel[RIG_DEFORM_BONE_PROP] = deform_name
        if not deform_name:
            self.report(
                {'ERROR'},
                "Choose the Head Deform Bone before refreshing the merged rig",
            )
            return {'CANCELLED'}
        skel[RIG_DESTINATION_PROP] = 'EXISTING'
        if mh.cage and mh.target:
            transfer_body_weights_to_cage(
                mh.cage, mh.target, skel, skel)
        parented = parent_facial_roots(context, skel, bone_name)
        bound = []
        for obj in rig_meshes(rid, (skel,)):
            claimed = retarget_armature_modifiers(
                obj, (), skel, adopt_empty=True)
            drives = any(m.type == 'ARMATURE' and m.object == skel
                         for m in obj.modifiers)
            if claimed or drives:
                bound.append(obj)
        meshes, verts, unpainted = rebind_meshes(context, skel, bound,
                                                 claim_for=mh.target)
        organization.ensure_character_collections(
            context,
            cage=mh.cage,
            target=mh.target,
            skel=skel,
            gui_coll=(skel.get(RIG_GUI_COLL_PROP)
                      if isinstance(skel.get(RIG_GUI_COLL_PROP),
                                    bpy.types.Collection) else None),
        )
        refit = bpy.ops.mhfrt.fit_skeleton()
        if 'FINISHED' not in refit:
            self.report({'WARNING'}, "Update Rig did not finish - run it once")
            return {'FINISHED'}
        self.report(
            {'INFO'},
            f"'{skel.name}' is already one armature - refreshed "
            f"({parented} root(s) re-parented, {meshes} mesh(es) rebound"
            + (f", {verts} weights" if verts else "")
            + (f", {unpainted} verts without head-deform weight"
               if unpainted else "")
            + ")")
        return {'FINISHED'}


# ----------------------------------------------------------------- unmerge ---

def _belongs_to_rig(obj, rid, skel):
    """Is this mesh THIS character's to strip facial weights from?

    Two characters can share one body armature - one body, two faces - and
    then every mesh that armature deforms answers to ``rig_meshes``, including
    the OTHER character's head.  Its facial groups are named after the very
    same MetaHuman joints, so stripping by name alone deleted a second
    character's entire bind when the first one was removed.

    A mesh is ours when it carries our rig id, or when it carries none and no
    other facial rig drives it.  Anything else stays exactly as it is.
    """
    if obj is None:
        return False
    own = str(obj.get(RIG_ID_PROP) or "")
    if own:
        return own == str(rid)
    for modifier in obj.modifiers:
        if modifier.type != 'ARMATURE' or modifier.object is None:
            continue
        other = modifier.object
        if other == skel:
            continue
        if str(other.get(RIG_ID_PROP) or "") not in ("", str(rid)):
            return False                # another character's rig drives it
    return True


def strip_facial_from_armature(context, skel):
    """Take the facial rig back out of a merged armature, keeping the body.

    Used by Remove Rig: the merged skeleton is the artist's OWN body armature
    and must survive.  The facial bones, the add-on's bone collections and all
    add-on properties go; the body bones, their collections and the body
    animation stay untouched.
    """
    if skel is None or skel.type != 'ARMATURE':
        return 0
    facial_names = facial_bone_names(skel)
    body_names = _body_bone_names(skel)
    deform_bone = merged_deform_bone(skel)
    rid = str(skel.get(RIG_ID_PROP) or "")

    # Give the body its deform share back before its partner bones go.
    for obj in rig_meshes(rid, (skel,)):
        if not _belongs_to_rig(obj, rid, skel):
            continue
        release_weights_from_merged_rig(
            obj,
            facial_names,
            weight_carrier(
                obj, deform_bone, body_names, facial_names, skel),
        )
        for name in list(facial_names):
            group = obj.vertex_groups.get(name)
            if group is not None:
                obj.vertex_groups.remove(group)

    _mode_object(context)
    previous_active = context.view_layer.objects.active
    removed = 0
    with organization.shown_for_edit(
        skel, view_layer=context.view_layer):
        context.view_layer.update()
        try:
            context.view_layer.objects.active = skel
            bpy.ops.object.mode_set(mode='EDIT')
            for bone in list(skel.data.edit_bones):
                if bone.name in facial_names:
                    skel.data.edit_bones.remove(bone)
                    removed += 1
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass
        finally:
            if previous_active is not None and \
                    context.view_layer.objects.get(previous_active.name):
                context.view_layer.objects.active = previous_active

    for coll in list(skel.data.collections_all):
        if coll.get(organization.BONE_COLL_PROP) and not coll.bones:
            try:
                skel.data.collections.remove(coll)
            except (RuntimeError, ReferenceError):
                pass
    return removed


_classes = (
    MHFRT_OT_confirm_standalone,
    MHFRT_OT_merge_body_rig,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
