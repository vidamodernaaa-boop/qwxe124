"""One-click attachment of accessory meshes to facial rig bones.

Eyes and teeth are rigid, the tongue uses Blender automatic weights, and
eyelashes copy their roots from the head before those weights are propagated
through the lash mesh's own topology.  Each attach is a clean re-bind: previous
armature modifiers and vertex groups on the mesh are replaced.

Beyond the fixed slots the artist can name their own - brows, peach fuzz, a
piercing, stubble cards.  A custom part is bound exactly like an eyelash
(head-surface roots + topology propagation), because that is the binding that
works for any mesh sitting ON the face without a bone of its own.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ..core import bind as bindmod
from ..core import organization, topology_weights
from ..props import RIG_ID_PROP

ATTACH_PART_PROP = "mhfrt_attached_part"

# (identifier, UI label, bones)
#   str   -> rigid: one bone, every vertex at 100%
#   tuple -> eyelashes: topology transfer from the head surface
#   None  -> automatic weights restricted to the tongue bone region
PART_SPECS = (
    ('EYE_L', "Eye Left", "FACIAL_L_Eye"),
    ('EYE_R', "Eye Right", "FACIAL_R_Eye"),
    ('TEETH_UPPER', "Teeth Upper", "FACIAL_C_TeethUpper"),
    ('TEETH_LOWER', "Teeth Lower", "FACIAL_C_TeethLower"),
    ('LASH_UPPER_R', "Lash Upper Right", ("FACIAL_R_EyelidUpperA1",
                                          "FACIAL_R_EyelidUpperA2",
                                          "FACIAL_R_EyelidUpperA3")),
    ('LASH_UPPER_L', "Lash Upper Left", ("FACIAL_L_EyelidUpperA1",
                                         "FACIAL_L_EyelidUpperA2",
                                         "FACIAL_L_EyelidUpperA3")),
    ('LASH_LOWER_R', "Lash Lower Right", ("FACIAL_R_EyelidLowerA1",
                                          "FACIAL_R_EyelidLowerA2",
                                          "FACIAL_R_EyelidLowerA3")),
    ('LASH_LOWER_L', "Lash Lower Left", ("FACIAL_L_EyelidLowerA1",
                                         "FACIAL_L_EyelidLowerA2",
                                         "FACIAL_L_EyelidLowerA3")),
    ('TONGUE', "Tongue", None),
)

PART_LABELS = {part: label for part, label, _bone in PART_SPECS}
PART_BONES = {part: bone for part, _label, bone in PART_SPECS}

# A custom slot is stored on the mesh as its own part id so it behaves like any
# built-in one everywhere else (attached_objects, organization, the Morphs list
# and re-attaching all key off the same string).  The artist's name rides along
# after the prefix rather than in a second property: one id, one lookup, and a
# part slot cannot end up half-named on a mesh that was appended or copied.
CUSTOM_PART_PREFIX = "CUSTOM:"


def clean_part_name(name):
    """The artist's typed name, whitespace collapsed and length-capped."""
    return " ".join(str(name or "").split())[:48]


def custom_part_id(name):
    return CUSTOM_PART_PREFIX + clean_part_name(name)


def custom_part_name(part):
    """The artist's name for a custom slot, or "" for the built-in slots."""
    part = str(part or "")
    if not part.startswith(CUSTOM_PART_PREFIX):
        return ""
    return part[len(CUSTOM_PART_PREFIX):]


def part_label(part):
    """UI label for any part id - built-in slot or custom name."""
    part = str(part or "")
    if part in PART_LABELS:
        return PART_LABELS[part]
    return custom_part_name(part) or part


def custom_parts(mh):
    """Custom slot ids already on this rig, in first-attached order."""
    out = []
    for obj in attached_objects(mh):
        part = str(obj.get(ATTACH_PART_PROP) or "")
        if part.startswith(CUSTOM_PART_PREFIX) and part not in out:
            out.append(part)
    return out


def _rig_id(mh):
    skel = mh.skeleton
    return skel.get(RIG_ID_PROP) if skel is not None else None


def attached_objects(mh, part=None):
    """Meshes attached to the active rig, optionally for one part slot."""
    rid = _rig_id(mh)
    if not rid:
        return []
    return [obj for obj in bpy.data.objects
            if obj.type == 'MESH'
            and obj.get(ATTACH_PART_PROP)
            and str(obj.get(RIG_ID_PROP) or "") == str(rid)
            and (part is None or obj.get(ATTACH_PART_PROP) == part)]


def parts_done(mh):
    return bool(attached_objects(mh))


def _tongue_bone_names(skel):
    return [bone.name for bone in skel.data.bones
            if organization._bone_region(bone.name) == "tongue"]


def _influencing_armatures(skel):
    """Armature datablocks whose pose can move the part visually: the facial
    skeleton itself plus every armature above it (a merged body rig moves the
    facial skeleton object through its parent bone)."""
    out = []
    seen = set()
    obj = skel
    while obj is not None:
        if obj.type == 'ARMATURE' and obj.data.as_pointer() not in seen:
            seen.add(obj.data.as_pointer())
            out.append(obj.data)
        obj = obj.parent
    return out


def _mode_object(context):
    if context.object and context.object.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass


def _remove_armature_modifiers(obj):
    for mod in [m for m in obj.modifiers if m.type == 'ARMATURE']:
        obj.modifiers.remove(mod)


def _parent_keep_transform(obj, skel):
    matrix = obj.matrix_world.copy()
    obj.parent = skel
    obj.parent_type = 'OBJECT'
    obj.matrix_parent_inverse = skel.matrix_world.inverted()
    obj.matrix_world = matrix


def _release_from_character(obj):
    """Take a detached mesh out of the character's filing entirely.

    ``restore_collections`` LINKS an object back into the collections it came
    from but never unlinks it from ours - in the Remove Rig path ours are about
    to be deleted, so it never had to.  Detaching one part deletes nothing, so
    the mesh would be left living in both places, still stamped as belonging to
    this character, and would come back as a part on the next organize.

    A collection is only unlinked when the mesh has somewhere else to live: an
    object in no collection at all is invisible and effectively lost, which is
    a far worse outcome than an untidy one.
    """
    from ..core import provenance

    ours = [coll for coll in obj.users_collection if provenance.made_by(coll)]
    for coll in ours:
        if len(obj.users_collection) <= 1:
            break                       # never leave it homeless
        try:
            coll.objects.unlink(obj)
        except (RuntimeError, ReferenceError):
            continue
    for key in (ATTACH_PART_PROP, RIG_ID_PROP,
                organization.CHARACTER_COLL_PROP,
                organization.OBJECT_ROLE_PROP,
                provenance.ORIGIN_KEY_PROP):
        if key in obj:
            del obj[key]


def _unparent_keep_transform(obj):
    """Free an object from its parent, leaving it exactly where it sits."""
    matrix = obj.matrix_world.copy()
    obj.parent = None
    obj.parent_type = 'OBJECT'
    obj.parent_bone = ""
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_world = matrix


def _finish_modifier(obj, skel):
    from .op_weights import ARM_MOD_NAME
    mods = [m for m in obj.modifiers if m.type == 'ARMATURE']
    mod = mods[0] if mods else obj.modifiers.new(ARM_MOD_NAME, 'ARMATURE')
    mod.name = ARM_MOD_NAME
    mod.object = skel
    mod.show_in_editmode = True
    mod.show_on_cage = True
    for extra in mods[1:]:
        obj.modifiers.remove(extra)
    return mod


def _attach_single_bone(obj, skel, bone_name):
    _remove_armature_modifiers(obj)
    _parent_keep_transform(obj, skel)
    obj.vertex_groups.clear()
    group = obj.vertex_groups.new(name=bone_name)
    group.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
    _finish_modifier(obj, skel)


def _attach_auto_weights(context, obj, skel, bone_names):
    """Parent with automatic weights restricted to `bone_names`.

    Blender's ``parent_set(type='ARMATURE_AUTO')`` operator normally preserves
    the child's world matrix, but that guarantee depends on the parent's world
    transform being sane at the moment of the call.  When the skeleton has been
    moved, rotated, or scaled away from the origin the operator can still leave
    the child at the wrong world position (older Blender revisions do it every
    time; recent ones only in degenerate contexts).  We snapshot the world
    matrix ourselves and restore it explicitly, so the part sits exactly where
    the artist placed it regardless of the character's world transform.
    """
    _remove_armature_modifiers(obj)
    obj.vertex_groups.clear()

    world_matrix = obj.matrix_world.copy()
    allowed = set(bone_names)
    deform_backup = {bone.name: bone.use_deform for bone in skel.data.bones}
    skel_hidden = skel.hide_get()
    skel_hide_viewport = skel.hide_viewport
    try:
        for bone in skel.data.bones:
            bone.use_deform = deform_backup[bone.name] and bone.name in allowed
        try:
            skel.hide_set(False)
        except RuntimeError:
            pass
        skel.hide_viewport = False
        for other in list(context.selected_objects):
            other.select_set(False)
        obj.select_set(True)
        skel.select_set(True)
        context.view_layer.objects.active = skel
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    finally:
        for bone in skel.data.bones:
            bone.use_deform = deform_backup.get(bone.name, True)
        skel.hide_viewport = skel_hide_viewport
        try:
            skel.hide_set(skel_hidden)
        except RuntimeError:
            pass
        # Reassert world position exactly where the artist placed it: with a
        # non-identity skeleton transform Blender's parent operator recomputes
        # parent_inverse in a way that can leave the child slightly off.
        obj.matrix_parent_inverse = skel.matrix_world.inverted()
        obj.matrix_world = world_matrix

    _finish_modifier(obj, skel)
    return sum(1 for g in obj.vertex_groups if g.name in allowed)


def _world_coordinates(obj, current_shape=False):
    """Object vertices in world space, optionally with shape sliders mixed."""
    if current_shape:
        from .op_weights import _current_local
        local = _current_local(obj)
    else:
        local = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", local)
        local = local.reshape(-1, 3)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.hstack([local, np.ones((len(local), 1))])
    return (homogeneous @ matrix.T)[:, :3]


def _weighted_edges(mesh, world_coordinates):
    edges = []
    for edge in mesh.edges:
        a, b = edge.vertices
        length = float(np.linalg.norm(
            world_coordinates[int(a)] - world_coordinates[int(b)]))
        edges.append((int(a), int(b), length))
    return edges


def _surface_weight_row(hit, head_tris, head_world, head_weights):
    """Barycentrically sample one normalized skin row from a head BVH hit."""
    location, triangle_index = hit
    triangle = head_tris[triangle_index]
    a, b, c = (head_world[int(index)] for index in triangle)
    from .op_weights import _barycentric
    barycentric = _barycentric(np.asarray(location, dtype=np.float64), a, b, c)
    row = {}
    for corner, factor in zip(triangle, barycentric):
        if factor <= 0.0:
            continue
        for name, weight in head_weights[int(corner)].items():
            row[name] = row.get(name, 0.0) + float(factor) * weight
    if len(row) > 8:
        row = dict(sorted(
            row.items(), key=lambda item: (-item[1], item[0]))[:8])
    total = sum(row.values())
    return ({name: value / total for name, value in row.items()}
            if total > 1.0e-12 else {})


def _topology_weight_assignments(obj, head, skel):
    """Build one normalized topology-propagated weight row per part vertex.

    Only roots query skin weights from the head.  Non-root vertices receive
    those rows exclusively through the part's own edge graph, so a long lash
    tip does not lose strength merely because it is far from the face.
    """
    if head is None or head.type != 'MESH':
        raise RuntimeError("Set a mesh as Head Target first")
    if not obj.data.vertices:
        raise RuntimeError("the mesh has no vertices")

    head_world = _world_coordinates(head, current_shape=True)
    head_tris = bindmod.triangles_of(head.data)
    if not len(head_tris):
        raise RuntimeError("the Head Target has no faces to sample")
    head_bvh = bindmod.build_bvh(head_world, head_tris)

    deform_names = {
        bone.name for bone in skel.data.bones if bone.use_deform
    }
    head_group_names = {
        group.index: group.name for group in head.vertex_groups
        if group.name in deform_names
    }
    if not head_group_names:
        raise RuntimeError(
            "the Head Target has no skin weights matching deform bones on the rig")
    head_weights = []
    for vertex in head.data.vertices:
        head_weights.append({
            head_group_names[element.group]: float(element.weight)
            for element in vertex.groups
            if element.group in head_group_names and element.weight > 0.0
        })

    lash_world = _world_coordinates(obj)
    edges = _weighted_edges(obj.data, lash_world)
    hits = []
    distances = []
    for point in lash_world:
        location, _normal, triangle_index, distance = head_bvh.find_nearest(
            Vector(point))
        if location is None or triangle_index is None:
            raise RuntimeError("could not find the closest Head Target surface")
        hits.append((location, int(triangle_index)))
        distances.append(float(distance))

    candidate_roots = topology_weights.select_root_vertices(distances, edges)
    sampled_roots = {}
    for root in candidate_roots:
        row = _surface_weight_row(
            hits[root], head_tris, head_world, head_weights)
        if row:
            sampled_roots[root] = row

    # A malformed/unweighted patch on the head must not leave a whole separate
    # card or shell with no skin.  Fall back to that island's next closest vertex
    # which actually samples a weight, while still sampling only a root.
    components = topology_weights.connected_components(len(lash_world), edges)
    missing_components = 0
    for component in components:
        if any(root in sampled_roots for root in component):
            continue
        for root in sorted(component, key=distances.__getitem__):
            row = _surface_weight_row(
                hits[root], head_tris, head_world, head_weights)
            if row:
                sampled_roots[root] = row
                break
        else:
            missing_components += 1
    if missing_components:
        raise RuntimeError(
            f"the closest head surface has no usable skin weights for "
            f"{missing_components} mesh island(s)")

    assignments = topology_weights.propagate_root_weights(
        len(lash_world), edges, sampled_roots)
    if any(not row for row in assignments):
        raise RuntimeError(
            "some vertices are not connected to a weighted root")
    return assignments, len(sampled_roots), len(components)


def _attach_topology_weights(obj, head, skel, prepared=None):
    """Attach a mesh with root surface samples + topology propagation."""
    if prepared is None:
        prepared = _topology_weight_assignments(obj, head, skel)
    assignments, root_count, island_count = prepared

    # Compute successfully before making the clean re-bind destructive.
    _remove_armature_modifiers(obj)
    _parent_keep_transform(obj, skel)
    obj.vertex_groups.clear()
    groups = {}
    for vertex_index, row in enumerate(assignments):
        for name, weight in row.items():
            group = groups.get(name)
            if group is None:
                group = obj.vertex_groups.new(name=name)
                groups[name] = group
            group.add([vertex_index], float(weight), 'REPLACE')
    _finish_modifier(obj, skel)
    return root_count, island_count, len(groups)


class MHFRT_OT_attach_part(bpy.types.Operator):
    bl_idname = "mhfrt.attach_part"
    bl_label = "Attach Part"
    bl_description = ("Bind the selected mesh to its facial bone: replaces "
                      "armature modifiers, parents to the facial skeleton, "
                      "and rebuilds the vertex groups")
    bl_options = {'REGISTER', 'UNDO'}

    part: bpy.props.EnumProperty(items=[
        (part, label, "") for part, label, _bone in PART_SPECS
    ] + [('CUSTOM', "Custom Part", "A part slot you name yourself")])
    custom_name: bpy.props.StringProperty(
        name="Part Name",
        description="Name for the custom part slot (brows, fuzz, stubble...)",
        default="",
    )

    @classmethod
    def description(cls, context, properties):
        if properties.part == 'CUSTOM':
            label = clean_part_name(properties.custom_name) or "custom part"
            return (f"Bind the selected mesh as \"{label}\": sample skin "
                    "weights at its head-facing roots, then spread them "
                    "through the mesh by topology distance - the same binding "
                    "the eyelashes use")
        label = PART_LABELS[properties.part]
        bones = PART_BONES[properties.part]
        if bones is None:
            return (f"Bind the selected mesh as the {label}: parent it to the "
                    "facial skeleton with automatic weights from the tongue "
                    "bones only")
        if isinstance(bones, str):
            return (f"Bind the selected mesh as the {label}: parent it to the "
                    f"facial skeleton and weight every vertex 100% to {bones}")
        return (f"Bind the selected mesh as the {label}: sample skin weights "
                "at its head-facing roots, then spread them through the "
                "eyelash mesh by topology distance")

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        return bool(mh and mh.skeleton is not None
                    and mh.skeleton.get(RIG_ID_PROP))

    def _candidates(self, context, mh):
        from .op_rig import _control_map
        out = []
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            if obj in (mh.target, mh.cage):
                continue
            if obj.name in _control_map:
                continue
            out.append(obj)
        return out

    def execute(self, context):
        from .op_live import stop_running

        mh = context.scene.mhfrt
        skel = mh.skeleton
        rid = skel.get(RIG_ID_PROP) if skel else None
        if not rid:
            self.report({'ERROR'}, "Build the rig first")
            return {'CANCELLED'}

        is_custom = self.part == 'CUSTOM'
        auto_bones = []
        bone_name = None
        # Custom slots have no bone list of their own: they bind off the head's
        # own skin weights, so whatever deforms the face at that spot is what
        # ends up driving the part.
        if is_custom:
            name = clean_part_name(self.custom_name)
            if not name:
                self.report({'ERROR'}, "Name the custom part first")
                return {'CANCELLED'}
            label = name
            part_id = custom_part_id(name)
            use_topology = True
        else:
            label = PART_LABELS[self.part]
            part_id = self.part
            spec = PART_BONES[self.part]
            use_topology = self.part.startswith('LASH_')
            bone_name = spec if isinstance(spec, str) else None
            if bone_name is not None:
                if skel.data.bones.get(bone_name) is None:
                    self.report({'ERROR'},
                                f"Bone '{bone_name}' not found on the rig")
                    return {'CANCELLED'}
            elif spec is None:
                auto_bones = _tongue_bone_names(skel)
                if not auto_bones:
                    self.report({'ERROR'}, "No tongue bones found on the rig")
                    return {'CANCELLED'}
            else:
                auto_bones = [name for name in spec
                              if skel.data.bones.get(name) is not None]
                if not auto_bones:
                    self.report({'ERROR'},
                                f"No {label} bones found on the rig "
                                f"({', '.join(spec)})")
                    return {'CANCELLED'}
        if use_topology and (mh.target is None or mh.target.type != 'MESH'):
            self.report({'ERROR'},
                        f"Set the Head Target before attaching the {label}")
            return {'CANCELLED'}

        objects = self._candidates(context, mh)
        if not objects:
            self.report(
                {'ERROR'},
                f"Select the mesh you made for the {label} "
                "(head target, cage, and controls do not count)")
            return {'CANCELLED'}

        stop_running()
        _mode_object(context)

        # Exactly the manual recipe: switch every involved armature (facial
        # skeleton AND the existing body rig) to Rest Position, bind the mesh
        # right where the artist placed it, then bring the pose back.  The
        # placement is therefore the part's rest-pose home; restoring the pose
        # snaps it correctly into the posed character.
        # Attaching wipes the part's vertex groups and its armature modifiers -
        # they are the artist's eyes, teeth or tongue, and that is a change
        # that has to be written down before it happens, not after.
        from ..core import registry as _registry, provenance as _provenance
        _record = (_registry.for_object(context.scene, skel)
                   or _registry.active(context.scene))
        for obj in objects:
            _provenance.snapshot_object(_record, obj,
                                        role=_provenance.ROLE_PART,
                                        weights=True, adopt=True)

        pose_backup = [(data, data.pose_position)
                       for data in _influencing_armatures(skel)]
        for data, _position in pose_backup:
            data.pose_position = 'REST'
        context.view_layer.update()

        weighted = 0
        topology_stats = []
        try:
            # Validate and calculate every selected lash first.  A bad second
            # mesh should not leave the first one destructively half-attached.
            prepared_lashes = {}
            if use_topology:
                try:
                    prepared_lashes = {
                        obj: _topology_weight_assignments(obj, mh.target, skel)
                        for obj in objects
                    }
                except RuntimeError as exc:
                    self.report({'ERROR'},
                                f"Could not attach the {label}: {exc}")
                    return {'CANCELLED'}
            for obj in objects:
                try:
                    if bone_name is not None:
                        _attach_single_bone(obj, skel, bone_name)
                    elif use_topology:
                        topology_stats.append(_attach_topology_weights(
                            obj, mh.target, skel, prepared_lashes[obj]))
                        weighted += 1
                    else:
                        weighted += _attach_auto_weights(
                            context, obj, skel, auto_bones)
                except RuntimeError as exc:
                    self.report({'ERROR'},
                                f"Could not attach '{obj.name}': {exc}")
                    return {'CANCELLED'}
                obj[ATTACH_PART_PROP] = part_id
                obj[RIG_ID_PROP] = str(rid)
        finally:
            for data, position in pose_backup:
                data.pose_position = position
            context.view_layer.update()

        # leave the artist on their mesh, organized into the character
        for other in list(context.selected_objects):
            other.select_set(False)
        for obj in objects:
            obj.select_set(True)
        context.view_layer.objects.active = objects[0]
        organization.organize_current(context, parts=attached_objects(mh))

        from . import op_rig
        op_rig.invalidate()

        names = ", ".join(obj.name for obj in objects)
        if bone_name is not None:
            self.report({'INFO'}, f"{label}: '{names}' -> {bone_name} (100%)")
        elif use_topology:
            roots = sum(item[0] for item in topology_stats)
            islands = sum(item[1] for item in topology_stats)
            groups = len({group.name for obj in objects
                          for group in obj.vertex_groups})
            self.report(
                {'INFO'},
                f"{label}: '{names}' -> topology weights from {roots} "
                f"head-facing root(s) across {islands} island(s), "
                f"{groups} bone group(s)")
        elif weighted == 0:
            self.report(
                {'WARNING'},
                f"{label}: '{names}' parented, but automatic weights "
                "produced no groups (check the mesh for loose geometry)")
        else:
            self.report(
                {'INFO'},
                f"{label}: '{names}' -> automatic weights from "
                f"{len(auto_bones)} bone(s)")
        return {'FINISHED'}


class MHFRT_OT_detach_part(bpy.types.Operator):
    """Take a mesh back out of its part slot and hand it back untouched.

    The mirror of Attach, and it uses the same ledger: attaching snapshots the
    mesh first (:func:`provenance.snapshot_object`), so removing it is not
    "undo the things we can think of" - it is the recorded before-state being
    put back, weights and parenting and modifiers and all.

    A CUSTOM slot has no existence beyond the meshes in it - ``custom_parts``
    derives the list from what is attached - so detaching the last mesh IS
    deleting the slot, and there is nothing else to clean up.  Nothing is ever
    deleted here: the artist's mesh is theirs, and Remove means "stop owning
    it", not "throw it away".
    """

    bl_idname = "mhfrt.detach_part"
    bl_label = "Remove Part"
    bl_options = {'REGISTER', 'UNDO'}

    part: bpy.props.StringProperty(name="Part", default="",
                                   options={'HIDDEN'})
    object_name: bpy.props.StringProperty(
        name="Object", default="", options={'HIDDEN'},
        description="Detach only this mesh; empty means the whole slot")

    @classmethod
    def description(cls, context, properties):
        label = part_label(properties.part) or "this part"
        return (f"Remove the {label} slot: the mesh is handed back exactly as "
                "you gave it - its own weights, parent and modifiers restored "
                "- and simply stops being part of this character. Nothing is "
                "deleted")

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        return bool(mh and mh.skeleton is not None
                    and mh.skeleton.get(RIG_ID_PROP))

    def execute(self, context):
        from ..core import provenance, registry
        from .op_live import stop_running
        from . import op_rig

        mh = context.scene.mhfrt
        skel = mh.skeleton
        objects = [obj for obj in attached_objects(mh, self.part or None)
                   if not self.object_name or obj.name == self.object_name]
        if not objects:
            self.report({'ERROR'}, "That part is not attached to this rig")
            return {'CANCELLED'}

        stop_running()
        _mode_object(context)
        record = (registry.for_object(context.scene, skel)
                  or registry.active(context.scene))
        ledger = provenance.read(record)
        our_bones = [bone.name for bone in skel.data.bones]
        label = part_label(self.part)

        freed, unrecorded, dropped = [], [], []
        for obj in objects:
            entry = provenance.entry_for(record, obj, ledger)
            if entry is None:
                # Attached before the ledger existed. The recorded state is not
                # there to put back, so undo what attaching is KNOWN to do and
                # say plainly that it was the shallow version.
                _remove_armature_modifiers(obj)
                if obj.parent is skel:
                    _unparent_keep_transform(obj)
                unrecorded.append(obj.name)
            else:
                # `doomed=(skel,)` is what tells the restore that this mesh is
                # parting company with our skeleton: it is the flag both
                # restore_modifiers and restore_parent read to decide whether a
                # binding is OURS to take off, and without it the Armature
                # modifier we added survives, aimed at a rig the mesh is no
                # longer part of.
                provenance.restore_object(
                    context, record, obj, entry, our_names=our_bones,
                    doomed=(skel,), data=ledger)
                dropped.append(provenance.origin_key(obj, create=False))
            _release_from_character(obj)
            freed.append(obj.name)

        # Forget the entries for the meshes we just handed back. Leaving them
        # would have a later Remove Rig "restore" a mesh that is no longer part
        # of this character, over whatever the artist has done to it since.
        if dropped:
            entries = ledger.get("objects") or {}
            for key in dropped:
                entries.pop(key, None)
            provenance.save(record, ledger)

        # Re-file what is STILL attached. The detached meshes are already out
        # of attached_objects, so this cannot pull them back in.
        organization.organize_current(context, parts=attached_objects(mh))
        op_rig.invalidate()

        names = ", ".join(freed)
        if unrecorded:
            self.report(
                {'WARNING'},
                f"{label}: '{names}' detached, but there was no record of how "
                "it arrived - its bindings were cleared rather than restored")
        else:
            self.report({'INFO'},
                        f"{label}: '{names}' handed back as it was")
        return {'FINISHED'}


_classes = (MHFRT_OT_attach_part, MHFRT_OT_detach_part)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
