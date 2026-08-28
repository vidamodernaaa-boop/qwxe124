"""On-demand MetaHuman morph shape keys driven by the live rig.

Picking a RigLogic channel creates one empty Basis clone for the artist to
sculpt; channels that are never picked do not add Shape Keys to the mesh.
"""

import re

import bpy
import numpy as np
from bpy.app.handlers import persistent

from ..core import riglogic
from ..core import board
from ..core import organization
from ..props import RIG_ID_PROP

MORPH_DONE_PROP = "mhfrt_morphs_ready"
MORPH_EXTRA_PROP = "mhfrt_extra_morph_object"
AUTO_KEYS_PROP = "mhfrt_auto_keys"     # picker-created keys, eligible for auto-removal
MORPH_ACTIVE_OBJECT_PROP = "mhfrt_active_morph_object"
MORPH_INPUT_MAP_PROP = "mhfrt_morph_input_map"
DNA_CHANNEL_MAP_PROP = "dna_channel_map"
LEGACY_MORPH_RIG_IDS_PROP = "mhfrt_morph_rig_ids"

# Per-channel bone intensity - how strongly one morph channel is allowed to
# drive the facial BONES.  1.0 is the untouched rig; 0.0 keeps that channel's
# joints at rest while its shape key still follows the pose, so the channel
# runs on its sculpted shape alone.  Stored on the skeleton so it survives
# file save/load and each character keeps its own.  Nothing on screen moves:
# the controllers stay wherever the artist posed them and the RigLogic
# evaluator simply scales the matching DNA GUI channels - see op_rig._read_gui.
# The first two key names date from the binary isolate toggle and are kept so
# channels muted in older files stay muted.
CHANNEL_NAMES_PROP = "mhfrt_muted_morph"              # \x1f-joined names
CHANNEL_GUI_INDICES_PROP = "mhfrt_muted_gui_indices"  # GUI channels to scale
CHANNEL_WEIGHTS_PROP = "mhfrt_channel_weights"        # aligned with the names
# The next three are aligned with the indices: the scale factor, and the value
# window on that channel it applies to (see _channel_windows_for).
CHANNEL_GUI_FACTORS_PROP = "mhfrt_muted_gui_factors"
CHANNEL_GUI_LOWS_PROP = "mhfrt_channel_gui_lows"
CHANNEL_GUI_HIGHS_PROP = "mhfrt_channel_gui_highs"
VALUE_EPS = 1e-4
GEOMETRY_EPS = 1e-7
_MORPH_UI_NAMES = None

_LEGACY_LONG_KEY = (
    "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtre"
)
_LEGACY_LONG_SUFFIX = re.compile(
    r"^McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenE\.\d{3,}$"
)


def _mode_object(context):
    if context.object and context.object.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass


def _enable_morph_edit_display(obj):
    """Match Blender's Shape Key/Modifier edit-display checkboxes."""
    obj.use_shape_key_edit_mode = True
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            modifier.show_in_editmode = True


def _ensure_basis(obj):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    return obj.data.shape_keys


def _driven_name_set():
    return set(riglogic.morph_input_by_key_name(head_only=True).keys())


def morph_key_name(source_name):
    """Public UI helper: full DNA channel -> Blender-safe KeyBlock name."""
    return riglogic.morph_key_name(source_name)


def existing_morph_key(obj, source_name):
    """Return the real KeyBlock represented by one add-on morph row."""
    if not source_name or obj is None or obj.type != 'MESH':
        return None
    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return None
    source_name = riglogic.morph_source_name(source_name)
    return shape_keys.key_blocks.get(morph_key_name(source_name))


def _is_legacy_driven_name(name):
    return name == _LEGACY_LONG_KEY or bool(_LEGACY_LONG_SUFFIX.fullmatch(name))


def _is_driven_key_name(name):
    return name in _driven_name_set() or _is_legacy_driven_name(name)


def morphs_done(mh):
    target = mh.target
    shape_keys = target.data.shape_keys if target and target.type == 'MESH' else None
    return bool(shape_keys and shape_keys.get(MORPH_DONE_PROP))


def _map_lines(entries, field):
    return "\n".join(
        f"{morph_key_name(entry['name'])}\t{int(entry[field])}"
        for entry in entries
    )


def _rig_id(mh):
    skel = mh.skeleton
    return skel.get(RIG_ID_PROP) if skel is not None else None


def _valid_extra_object(mh, obj):
    # The Head Cage is allowed as an extra morph object: it already carries the
    # rig id and rides the facial rig (armature modifier added in setup_rig), so
    # the artist can add it to the Morphs list and sculpt driven shapes on it.
    # It stays add-on data everywhere else (still deleted with the rig, still
    # filed in the Head Cage collection) - see op_rig._purge_rig and
    # organization.ensure_character_collections.
    return bool(obj and obj.type == 'MESH'
                and obj != mh.target)


def _clear_legacy_owner_list(obj):
    if obj and LEGACY_MORPH_RIG_IDS_PROP in obj:
        del obj[LEGACY_MORPH_RIG_IDS_PROP]


def _extra_owner_rig_id(obj):
    if not obj or not obj.get(MORPH_EXTRA_PROP):
        return ""
    return str(obj.get(RIG_ID_PROP) or "")


def _assign_extra_owner(obj, rig_id):
    if not obj or not rig_id:
        return False
    owner = _extra_owner_rig_id(obj)
    if owner and owner != str(rig_id):
        return False
    obj[MORPH_EXTRA_PROP] = True
    obj[RIG_ID_PROP] = str(rig_id)
    _clear_legacy_owner_list(obj)
    # A morph object is the artist's mesh to sculpt on, whatever it was made
    # from.  Duplicating one of ours to start it (the cage is the obvious
    # candidate) would otherwise leave our made-by stamp on it, and removal
    # deletes what carries that stamp.  The cage itself keeps its own claim:
    # it is registered as a morph object without ever becoming the artist's.
    scene = getattr(bpy.context, "scene", None)
    mh = getattr(scene, "mhfrt", None) if scene is not None else None
    if mh is None or obj != mh.cage:
        from ..core import provenance, registry
        provenance.snapshot_object(registry.active(scene) if scene else None,
                                   obj, role="morph", adopt=True)
    return owner != str(rig_id)


def _remove_extra_owner(obj, rig_id):
    if not obj or not rig_id:
        return False
    if _extra_owner_rig_id(obj) != str(rig_id):
        return False
    if MORPH_EXTRA_PROP in obj:
        del obj[MORPH_EXTRA_PROP]
    if RIG_ID_PROP in obj:
        del obj[RIG_ID_PROP]
    _clear_legacy_owner_list(obj)
    return True


def object_belongs_to_rig(obj, rig_id):
    return bool(obj and rig_id and _extra_owner_rig_id(obj) == str(rig_id))


def _remember_extra_item(mh, obj):
    for item in mh.morph_extra_objects:
        if item.obj == obj:
            return False
    item = mh.morph_extra_objects.add()
    item.obj = obj
    return True


def extra_morph_objects(mh):
    """Extra morph meshes that belong to the active rig."""
    out = []
    seen = set()
    rid = _rig_id(mh)
    if not rid:
        return out

    def add(obj):
        if not _valid_extra_object(mh, obj):
            return
        if not obj.get(MORPH_EXTRA_PROP):
            return
        if not object_belongs_to_rig(obj, rid):
            return
        key = obj.as_pointer()
        if key in seen:
            return
        seen.add(key)
        out.append(obj)

    for item in mh.morph_extra_objects:
        add(item.obj)

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            add(obj)
    return out


def active_extra_object(mh, context=None):
    obj = active_morph_object(mh, context)
    return obj if obj is not None and obj != mh.target else None


def morph_objects(mh):
    out = []
    if mh.target and mh.target.type == 'MESH':
        out.append(mh.target)
    out.extend(extra_morph_objects(mh))
    return out


def morph_ui_index(mh, source_name):
    """Stable backing-collection index for a full DNA morph channel name."""
    for index, item in enumerate(mh.morph_ui_items):
        if item.key_name == source_name:
            return index
    return -1


def _morph_ui_names():
    global _MORPH_UI_NAMES
    if _MORPH_UI_NAMES is None:
        _MORPH_UI_NAMES = tuple(
            entry["name"] for entry in
            riglogic.morph_entries(head_only=True)
        )
    return _MORPH_UI_NAMES


def _ensure_morph_ui_items(mh):
    """Populate the static, non-saved channel rows outside panel draw code."""
    names = _morph_ui_names()
    items = mh.morph_ui_items
    current_ok = (len(items) == len(names)
                  and (not names
                       or (items[0].key_name == names[0]
                           and items[-1].key_name == names[-1])))
    if current_ok:
        return False
    items.clear()
    for name in names:
        item = items.add()
        item.key_name = name
    mh.morph_ui_active_index = -1
    return True


def _sync_morph_object_ui_items(mh):
    """Mirror head + owned extras for the Blender template-list view."""
    objects = morph_objects(mh)
    items = mh.morph_object_ui_items
    current = tuple(
        item.obj.as_pointer() if item.obj is not None else 0
        for item in items
    )
    wanted = tuple(obj.as_pointer() for obj in objects)
    if current != wanted:
        items.clear()
        for obj in objects:
            item = items.add()
            item.obj = obj

    active = active_morph_object(mh)
    active_index = objects.index(active) if active in objects else 0
    _set_object_row(mh, active_index)
    return current != wanted


def sync_morph_ui_state(mh):
    """Refresh both UIList view models; safe from handlers/operators only."""
    _ensure_morph_ui_items(mh)
    _sync_morph_object_ui_items(mh)


def prune_morph_selection(mh, context=None):
    """Drop the Morphs list selection once its row is no longer in the list.

    The list only ever shows what the current pose fires, and no row is pinned
    just for being selected.  So the selection has to follow the rows: reset
    the controllers, or simply pose somewhere else, and the row the artist
    picked goes away - leaving it highlighted (and its Channels Intensity
    slider live) would point at something no longer on screen.  An empty list
    with no selection is a normal resting state, not a gap to fill.

    Channels dialled down or previewed without their bones keep their rows
    (see ``picker_items``), so those selections survive an empty pose.

    Returns True when it cleared the selection.  Writes a property, so it must
    run from a handler or an operator - never from panel draw code."""
    if int(mh.morph_ui_active_index) < 0:
        return False
    obj = active_morph_object(mh, context)
    source = selected_channel_name(mh)
    if obj is not None and source and source in listed_morph_names(mh, obj):
        return False
    mh.morph_ui_active_index = -1
    return True


def morph_object_index(mh, obj):
    for index, candidate in enumerate(morph_objects(mh)):
        if candidate == obj:
            return index
    return -1


# The Objects row index is written from both directions - the artist clicking a
# row, and the viewport sync answering a click in the 3D view.  Only the first
# should reach back out and change the viewport selection, so every write the
# add-on makes itself passes through _set_object_row and is ignored by the
# property's update callback (see morph_object_row_changed).
_syncing_object_row = False


def _set_object_row(mh, index):
    """Write the Objects list row without echoing back to the viewport."""
    global _syncing_object_row
    if mh.morph_object_ui_active_index == index:
        return False
    _syncing_object_row = True
    try:
        mh.morph_object_ui_active_index = index
    finally:
        _syncing_object_row = False
    return True


def activate_morph_object(context, mh, obj):
    """Make one registered mesh the morph target, in the panel AND the viewport.

    Panel and viewport always say the same thing: picking a row here selects
    that mesh in the 3D view, exactly as clicking the mesh in the 3D view
    selects its row."""
    if obj is None or morph_object_index(mh, obj) < 0:
        return False
    if context.view_layer.objects.get(obj.name) is None:
        return False
    track_viewport_mesh(mh, obj)
    remember_active_morph_object(mh, obj)
    sync_morph_ui_state(mh)
    if context.view_layer.objects.active != obj:
        _mode_object(context)
    select_only(context, obj)
    return True


_pending_row_object = ""


def _apply_row_selection():
    """Run the queued row pick once the click that made it has finished."""
    global _pending_row_object
    from ..core import render_state
    if render_state.is_rendering():
        return 0.5      # changing mode or active object mid-render: never
    name = _pending_row_object
    _pending_row_object = ""
    if not name:
        return None
    context = bpy.context
    scene = getattr(context, "scene", None)
    mh = getattr(scene, "mhfrt", None) if scene is not None else None
    obj = bpy.data.objects.get(name)
    if mh is not None and obj is not None:
        activate_morph_object(context, mh, obj)
    return None


def morph_object_row_changed(mh, context):
    """Objects list row -> viewport selection.

    Lives on the property rather than in the row's button so that every native
    way of moving through the list - clicking the row, arrow keys - selects the
    mesh in the viewport too.

    The selection itself is queued on a timer instead of run here: this is a
    property update callback, and changing the mode or the active object from
    inside one runs while Blender is still applying the click.  A zero-delay
    timer lands it a moment later, on solid ground - the same reason
    op_rig._apply_pending_switch exists."""
    global _pending_row_object
    if _syncing_object_row or context is None:
        return
    items = mh.morph_object_ui_items
    index = int(mh.morph_object_ui_active_index)
    if not (0 <= index < len(items)):
        return
    obj = items[index].obj
    if obj is None:
        return
    if obj == context.view_layer.objects.active:
        # Already the one in the viewport: record the pick, touch nothing else,
        # so clicking the row you are sculpting never drops you out of the mode.
        track_viewport_mesh(mh, obj)
        remember_active_morph_object(mh, obj)
        return
    _pending_row_object = obj.name
    if not bpy.app.timers.is_registered(_apply_row_selection):
        bpy.app.timers.register(_apply_row_selection, first_interval=0.0)


def remember_active_morph_object(mh, obj):
    """Persist one registered morph object as this rig's sculpt target."""
    index = morph_object_index(mh, obj)
    skel = mh.skeleton
    if index < 0 or skel is None:
        return False
    changed = False
    if skel.get(MORPH_ACTIVE_OBJECT_PROP) != obj.name:
        skel[MORPH_ACTIVE_OBJECT_PROP] = obj.name
        changed = True
    if mh.morph_extra_active_index != index:
        mh.morph_extra_active_index = index
        changed = True
    # A registered pick is also the last mesh clicked, so the two never
    # disagree: choosing a row in the panel clears any stray-mesh warning.
    if mh.morph_last_mesh != obj:
        mh.morph_last_mesh = obj
        changed = True
    changed = _set_object_row(mh, index) or changed
    return changed


# --------------------------------------------- which mesh the step is on ---
# The Morphs step sculpts exactly ONE mesh, and the artist says which by
# clicking it.  Only MESH clicks count: armatures (the control board), curves
# (landmarks) and everything else are not sculptable, so clicking them leaves
# the choice alone - that is what keeps "pose a control, then sculpt" working.
# A mesh that is NOT in the list counts too, and that is the whole point: it is
# recorded as the last click so the panel can say "this one is not in the list"
# instead of quietly handing the sculpt to the head (see unregistered_mesh).


def track_viewport_mesh(mh, obj):
    """Record one viewport click, if it landed on a mesh.  Returns True when
    the recorded mesh changed."""
    if obj is None or obj.type != 'MESH':
        return False
    if mh.morph_last_mesh == obj:
        return False
    mh.morph_last_mesh = obj
    return True


def reset_mesh_tracking(mh, context=None):
    """Re-read the last-clicked mesh from the viewport, for step entry.

    Entering the step should judge what is selected NOW, not whatever mesh was
    clicked three steps ago: with a control (or nothing) active, there is no
    stray mesh to warn about and the remembered morph object stands."""
    obj = context.view_layer.objects.active if context is not None else None
    obj = obj if obj is not None and obj.type == 'MESH' else None
    if mh.morph_last_mesh != obj:
        mh.morph_last_mesh = obj
        return True
    return False


def tracked_mesh(mh):
    obj = mh.morph_last_mesh
    return obj if obj is not None and obj.type == 'MESH' else None


def unregistered_mesh(mh, context=None):
    """The last-clicked mesh, when it is NOT one of this character's morph
    objects - the state the panel warns about.  None means all is well."""
    obj = tracked_mesh(mh)
    if obj is None or morph_object_index(mh, obj) >= 0:
        return None
    if context is not None and context.view_layer.objects.get(obj.name) is None:
        return None
    return obj


def active_morph_object(mh, context=None):
    """Current morph mesh: the last registered mesh the artist clicked.

    Clicking a registered head/extra mesh in the viewport immediately selects
    its row in the Morph step.  Armatures, curves and unregistered meshes are
    ignored here so they do not erase the artist's last morph choice - an
    unregistered mesh is reported separately by unregistered_mesh, and the
    panel refuses to offer a sculpt while one is on screen.

    Read-only so panel draw code can call it; recording the viewport choice
    happens in _selection_sync_handler and the operators, never here.
    """
    objects = morph_objects(mh)
    if not objects:
        return None
    obj = tracked_mesh(mh)
    if obj is not None and obj in objects:
        return obj
    if context is not None:
        viewport_obj = context.view_layer.objects.active
        if viewport_obj in objects:
            return viewport_obj
    skel = mh.skeleton
    name = skel.get(MORPH_ACTIVE_OBJECT_PROP) if skel else ""
    for obj in objects:
        if obj.name == name:
            return obj
    return objects[0]


_last_active_ptr = 0
_in_selection_sync = False


def _animation_playing():
    """The Morphs list freezes during playback (see panel._picker_live), so its
    selection has to freeze with it - otherwise every frame would clear it."""
    screen = getattr(bpy.context, "screen", None)
    return bool(screen is not None and screen.is_animation_playing)


@persistent
def _selection_sync_handler(scene, depsgraph=None):
    """Record viewport mesh picks, and keep the Morphs list selection honest.

    Keeps the Morph step row on the artist's last chosen mesh after they move
    on to a face control: select morph mesh, pose a control, then Sculpt/Edit
    the same mesh from the panel.  Runs outside draw code because recording
    writes rig properties, which Blender forbids during panel draw.

    Clicking a mesh that is NOT registered is recorded just the same, so the
    panel can say so; only meshes are recorded at all, so clicking a control
    board bone or a landmark curve changes nothing (see track_viewport_mesh).

    The selection prune runs on every update, not just when the active object
    changes, because it tracks the POSE: moving a controller (or resetting them
    all) is what empties the list out from under the selected row.  Only while
    the Morphs step is the one on screen, though - every other step pays
    nothing for it, and stepping back into Morphs prunes on the way in
    (panel.MHFRT_OT_set_tab).
    """
    global _last_active_ptr, _in_selection_sync
    # Everything below writes scene properties, and every write retags the
    # scene - which can bring the depsgraph straight back here before this call
    # has returned.  One pass at a time.
    if _in_selection_sync:
        return
    from ..core import render_state
    # A render has no selection to track, and this runs on the render job's
    # thread, where reading ``bpy.context`` is not safe (core.render_state).
    if render_state.is_rendering() \
            or getattr(depsgraph, "mode", 'VIEWPORT') == 'RENDER':
        return
    view_layer = getattr(bpy.context, "view_layer", None)
    if view_layer is None:
        return
    _in_selection_sync = True
    try:
        mh = getattr(scene, "mhfrt", None)
        if (mh is not None and mh.ui_tab == 'MORPHS'
                and not _animation_playing()):
            # Same context the panel's filter uses, so the prune judges
            # membership against exactly the rows the list is about to draw.
            prune_morph_selection(mh, bpy.context)
        obj = view_layer.objects.active
        ptr = obj.as_pointer() if obj is not None else 0
        if ptr == _last_active_ptr:
            return
        _last_active_ptr = ptr
        if mh is None:
            return
        if obj is not None and obj.type == 'MESH':
            track_viewport_mesh(mh, obj)
        sync_morph_ui_state(mh)
        if obj is None or obj.type != 'MESH':
            return
        if morph_object_index(mh, obj) < 0:
            return
        remember_active_morph_object(mh, obj)
    finally:
        _in_selection_sync = False


@persistent
def _morph_ui_load_handler(_dummy=None):
    global _last_active_ptr
    _last_active_ptr = 0
    scenes = getattr(bpy.data, "scenes", None)
    if scenes is None:
        return
    for scene in scenes:
        mh = getattr(scene, "mhfrt", None)
        if mh is not None:
            sync_morph_ui_state(mh)


def _deferred_morph_ui_init():
    """Initialize after Blender leaves its restricted add-on install state."""
    if getattr(bpy.data, "scenes", None) is None:
        return 0.1
    _morph_ui_load_handler()
    return None


def managed_morph_objects(mh, context=None):
    obj = active_morph_object(mh, context)
    return [obj] if obj and obj.type == 'MESH' else []


def _read_key_data(key, n):
    data = np.empty(n * 3, dtype=np.float64)
    key.data.foreach_get("co", data)
    return data


def _write_key_data(key, coords):
    key.data.foreach_set("co", np.ascontiguousarray(coords))


def _driven_keys(obj):
    if not obj or obj.type != 'MESH' or obj.data.shape_keys is None:
        return []
    names = _driven_name_set()
    return [key for key in obj.data.shape_keys.key_blocks[1:]
            if key.name in names or _is_legacy_driven_name(key.name)]


def _key_is_untouched(obj, key, basis_data=None):
    n = len(obj.data.vertices)
    if n == 0:
        return True
    basis = basis_data if basis_data is not None else _read_key_data(key.relative_key, n)
    co = _read_key_data(key, n)
    return bool(np.max(np.abs(co - basis)) <= GEOMETRY_EPS)


def _tag_ready(obj, ready):
    if obj.data.shape_keys:
        obj.data.shape_keys[MORPH_DONE_PROP] = bool(ready)
    obj[MORPH_DONE_PROP] = bool(ready)


def _has_driven_keys(obj):
    return bool(_driven_keys(obj))


def ensure_single_morph(context, obj, name, *, rig_id=None, extra=False):
    """Create ONE driven morph key on demand.

    Picking a morph from the panel creates just that key, so the mesh only
    ever carries the shapes the artist actually sculpts."""
    entry = next((e for e in riglogic.morph_entries(head_only=True)
                  if e["name"] == name), None)
    if entry is None:
        return None
    _mode_object(context)
    shape_keys = _ensure_basis(obj)
    key_name = morph_key_name(name)
    key = shape_keys.key_blocks.get(key_name)
    if key is None:
        key = obj.shape_key_add(name=key_name, from_mix=False)
        key.value = 0.0
    key.slider_min = 0.0
    key.slider_max = 1.0
    entries = riglogic.morph_entries(head_only=True)
    shape_keys[DNA_CHANNEL_MAP_PROP] = _map_lines(entries, "channel")
    shape_keys[MORPH_INPUT_MAP_PROP] = _map_lines(entries, "input")
    shape_keys[MORPH_DONE_PROP] = True
    obj[MORPH_DONE_PROP] = True
    if extra:
        if rig_id:
            _assign_extra_owner(obj, rig_id)
        obj[MORPH_EXTRA_PROP] = True
    elif rig_id:
        obj[RIG_ID_PROP] = rig_id
    obj.data.update()
    return key


def _auto_key_names(shape_keys):
    return {n for n in shape_keys.get(AUTO_KEYS_PROP, "").split("\n") if n}


def _set_auto_key_names(shape_keys, names):
    if names:
        shape_keys[AUTO_KEYS_PROP] = "\n".join(sorted(names))
    elif AUTO_KEYS_PROP in shape_keys:
        del shape_keys[AUTO_KEYS_PROP]


def mark_auto_key(obj, name):
    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return
    names = _auto_key_names(shape_keys)
    if name not in names:
        names.add(name)
        _set_auto_key_names(shape_keys, names)


def _remove_keys(obj, names):
    """Remove exact keys plus their baked drivers and auto-key bookkeeping."""
    from . import op_drivers

    shape_keys = obj.data.shape_keys
    if shape_keys is None or not names:
        return 0
    bookkeeping_names = set(names)
    for name in names:
        bookkeeping_names.add(riglogic.morph_source_name(name))
        if _is_legacy_driven_name(name):
            bookkeeping_names.update(
                source_name for source_name in
                riglogic.morph_input_by_name(head_only=True)
                if riglogic.legacy_morph_key_names(source_name)
            )
    op_drivers.unbake_keys(obj, bookkeeping_names)
    removed = 0
    for name in names:
        key = shape_keys.key_blocks.get(name)
        if key is not None:
            obj.shape_key_remove(key)
            removed += 1
    shape_keys = obj.data.shape_keys
    if shape_keys is not None:
        _set_auto_key_names(shape_keys,
                            _auto_key_names(shape_keys) - bookkeeping_names)
    obj.data.update()
    _tag_ready(obj, _has_driven_keys(obj))
    return removed


def auto_cleanup(obj, keep=None):
    """Remove picker-created keys that were never sculpted.

    Keys materialize when picked and clean themselves up when 'not needed'
    (still identical to Basis). ONLY keys the picker created are eligible -
    an imported full key set (old files) is never auto-removed."""
    shape_keys = obj.data.shape_keys if obj and obj.type == 'MESH' else None
    if shape_keys is None:
        return 0
    auto = _auto_key_names(shape_keys)
    if not auto:
        return 0
    from . import op_drivers
    n = len(obj.data.vertices)
    basis_cache = {}
    dropped = set()
    for name in sorted(auto):
        key = shape_keys.key_blocks.get(name)
        if key is None:
            dropped.add(name)          # stale bookkeeping
            continue
        if name == keep:
            continue
        basis = key.relative_key
        if basis is None or basis == key:
            continue
        if basis.name not in basis_cache:
            basis_cache[basis.name] = _read_key_data(basis, n)
        if _key_is_untouched(obj, key, basis_cache[basis.name]):
            op_drivers.unbake_keys(obj, [name])
            obj.shape_key_remove(key)
            dropped.add(name)
    if dropped:
        _set_auto_key_names(shape_keys, auto - dropped)
        obj.data.update()
    return len(dropped)


def rig_input_values(mh):
    """Live RigLogic input vector of the active rig (None until it has
    evaluated once). Lets the picker rank ALL morph channels by how strongly
    the current control pose fires them - even channels with no key yet.

    Uses the UNSCALED inputs so dialling a channel's Bone Intensity down never
    changes which rows appear or their order - the list holds still.  Falls
    back to the scaled vector on older caches that predate the clean
    snapshot."""
    from . import op_rig
    skel = mh.skeleton
    if skel is None:
        return None
    cache = op_rig._caches.get(skel.get(RIG_ID_PROP))
    if not cache:
        return None
    return cache.get("last_inputs_unmuted", cache.get("last_inputs"))


def picker_items(mh, obj):
    """[(key_name, value, exists)] - activated morph channels in DNA order.

    Filtering is driven by ``mh.morph_display_threshold`` (default 0.35) so
    only morphs at or above that level pass through.  The list preserves the
    DNA entry order so rows never jump around under the artist mid-selection;
    the picker used to sort by value, which the artist experienced as the
    list shuffling with every pose change.

    Ranking uses the UNSCALED input values (see ``rig_input_values``), so
    dialling a morph's bones down never drops it or its neighbours out of the
    list - the membership and order hold still.

    The SELECTED row is deliberately NOT pinned: the list only ever shows what
    the current pose fires, so resetting the controllers empties it and the
    selection goes with it (op_morphs.prune_morph_selection).  Only channels
    switched off by hand are pinned, below."""
    entries = riglogic.morph_entries(head_only=True)
    keys = (obj.data.shape_keys.key_blocks
            if obj and obj.type == 'MESH' and obj.data.shape_keys else None)
    values = rig_input_values(mh)
    # A channel the artist dialled down, or is previewing without its bones,
    # stays in the list whatever the pose does - otherwise it could vanish
    # while switched off and be impossible to switch back on.  That is a
    # deliberate rig/viewing state, not the pose, so it is the one thing that
    # survives an empty pose.
    dialled = set(channel_weights(mh.skeleton)) | preview_muted(mh.skeleton)
    threshold = max(0.0, float(getattr(mh, "morph_display_threshold", 0.35)))
    # A tiny floor keeps morphs at exact rest (0.0) hidden even when the
    # slider is dragged to 0 - otherwise 800+ rows would flood the list.
    effective_min = max(threshold, VALUE_EPS)
    items = []
    for entry in entries:
        name = entry["name"]
        key_name = morph_key_name(name)
        exists = bool(keys and key_name in keys)
        idx = int(entry["input"])
        if values is not None and idx < len(values):
            value = float(values[idx])
        elif exists:
            value = float(keys[key_name].value)
        else:
            value = 0.0
        if value < effective_min and name not in dialled:
            continue
        items.append((name, value, exists))
    return items


def listed_morph_names(mh, obj):
    """Channel names the Morphs list is showing for this object right now."""
    return {name for name, _value, _exists in picker_items(mh, obj)}


def active_morph_items(obj):
    """[(key_name, value)] sorted by active RigLogic value, strongest first."""
    if not obj or obj.type != 'MESH' or obj.data.shape_keys is None:
        return []
    name_set = set(riglogic.morph_input_by_key_name())
    out = []
    for key in obj.data.shape_keys.key_blocks[1:]:
        if key.name not in name_set:
            continue
        value = float(key.value)
        if value > VALUE_EPS:
            out.append((riglogic.morph_source_name(key.name), value))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def basis_key_name(obj):
    if not obj or obj.type != 'MESH' or obj.data.shape_keys is None:
        return ""
    keys = obj.data.shape_keys.key_blocks
    return keys[0].name if keys else ""


class MHFRT_OT_pick_morph(bpy.types.Operator):
    bl_idname = "mhfrt.pick_morph"
    bl_label = "Select Morph"
    bl_description = ("Select this active morph. Its shape key is created "
                      "automatically; keys you never sculpt remove "
                      "themselves again")
    bl_options = {'REGISTER', 'UNDO'}

    key_name: bpy.props.StringProperty()
    object_name: bpy.props.StringProperty(default="")
    action: bpy.props.EnumProperty(items=[
        ('SELECT', "Select", "Select this morph (key auto-created)"),
        ('SCULPT', "Sculpt", "Select this morph and enter Sculpt mode"),
        ('EDIT', "Edit", "Select this morph and enter Edit mode"),
    ], default='SELECT')

    def execute(self, context):
        mh = context.scene.mhfrt
        obj = (bpy.data.objects.get(self.object_name) if self.object_name
               else active_morph_object(mh, context))
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a morph object first")
            return {'CANCELLED'}
        if morph_object_index(mh, obj) < 0:
            self.report({'ERROR'},
                        "That object is not registered for this character's morphs")
            return {'CANCELLED'}
        rid = _rig_id(mh)
        if not rid:
            self.report({'ERROR'}, "Build the rig first")
            return {'CANCELLED'}

        _mode_object(context)
        # picker keys are transient until sculpted: switching selection
        # sweeps away any auto-created key that is still identical to Basis
        actual_name = morph_key_name(self.key_name)
        for morph_obj in morph_objects(mh):
            auto_cleanup(morph_obj,
                         keep=actual_name if morph_obj == obj else None)

        existed = bool(obj.data.shape_keys
                       and actual_name in obj.data.shape_keys.key_blocks)
        key = ensure_single_morph(context, obj, self.key_name,
                                  rig_id=rid, extra=(obj != mh.target))
        if key is None:
            self.report({'ERROR'}, f"Unknown morph channel '{self.key_name}'")
            return {'CANCELLED'}
        if not existed:
            mark_auto_key(obj, key.name)
        if obj != mh.target:
            _remember_extra_item(mh, obj)
        from . import op_rig
        op_rig.invalidate()
        sync_morph_ui_state(mh)
        ui_index = morph_ui_index(mh, self.key_name)
        if ui_index >= 0:
            mh.morph_ui_active_index = ui_index
        # Direct call, never bpy.ops: see apply_morph_key_selection.  The keys
        # and drivers above were just rebuilt, and a nested operator would flush
        # the depsgraph right here, between the surgery and the selection.
        level, message = apply_morph_key_selection(
            context, mh, obj, self.key_name, self.action)
        self.report({level}, message)
        return {'FINISHED'} if level == 'INFO' else {'CANCELLED'}


def select_only(context, obj):
    """Make obj the one selected, visible, active object - writing NOTHING that
    is already true.

    Every one of these writes lands on the object's view-layer Base, and each
    one retags the base flags for the next depsgraph evaluation.  Rewriting a
    flag that already holds its value is not free, so the redundant writes are
    skipped rather than issued and thrown away."""
    if context.view_layer.objects.get(obj.name) is None:
        return False
    try:
        if obj.hide_get():
            obj.hide_set(False)
    except RuntimeError:
        pass
    if obj.hide_viewport:
        obj.hide_viewport = False
    for other in list(context.selected_objects):
        if other != obj:
            other.select_set(False)
    if not obj.select_get():
        obj.select_set(True)
    if context.view_layer.objects.active != obj:
        context.view_layer.objects.active = obj
    return True


def apply_morph_key_selection(context, mh, obj, key_name, action):
    """Select one driven morph key, optionally entering Sculpt/Edit or Solo.

    A plain function, deliberately.  pick_morph used to finish by running
    ``bpy.ops.mhfrt.select_morph_key`` - and EVERY bpy.ops call from Python
    flushes the whole depsgraph before it runs.  Landing that flush in the
    middle of an operator that has just added and removed shape keys and their
    drivers is asking for trouble; calling this directly keeps the shape-key
    surgery and the selection inside one evaluation, with the only remaining
    flush the mode_set at the very end.

    Returns (level, message) - level 'INFO' on success, 'ERROR' on failure."""
    if obj is None or obj.type != 'MESH' or obj.data.shape_keys is None:
        return 'ERROR', "Morph shape keys not found"
    if morph_object_index(mh, obj) < 0:
        return 'ERROR', "That object is not registered for this character's morphs"
    if context.view_layer.objects.get(obj.name) is None:
        return 'ERROR', f"'{obj.name}' is excluded from the current View Layer"
    source_name = riglogic.morph_source_name(key_name)
    actual_name = morph_key_name(source_name)
    index = obj.data.shape_keys.key_blocks.find(actual_name)
    if index < 0:
        return 'ERROR', f"Shape key '{source_name}' not found"

    from . import op_rig

    was_soloed = bool(
        action == 'SOLO'
        and obj.show_only_shape_key
        and obj.active_shape_key_index == index
    )

    # Solo is a pure display toggle.  When the morph object is already the
    # active object (e.g. you are sculpting it), flip the solo preview in
    # place and STAY in the current mode instead of dropping to Object -
    # so clicking the solo icon never kicks you out of Sculpt/Edit.
    if action == 'SOLO' and context.view_layer.objects.active == obj:
        obj.active_shape_key_index = index
        op_rig.prepare_shape_key_preview(obj)
        obj.show_only_shape_key = not was_soloed
        state = "Soloed" if obj.show_only_shape_key else "Unsoloed"
        return 'INFO', f"{state} '{source_name}'"

    _mode_object(context)
    select_only(context, obj)
    obj.active_shape_key_index = index
    _enable_morph_edit_display(obj)
    if action == 'SOLO':
        obj.show_only_shape_key = not was_soloed
    op_rig.prepare_shape_key_preview(obj)
    op_rig.invalidate()
    track_viewport_mesh(mh, obj)
    remember_active_morph_object(mh, obj)
    ui_index = morph_ui_index(mh, source_name)
    if ui_index >= 0:
        mh.morph_ui_active_index = ui_index

    if action in {'SCULPT', 'EDIT'}:
        # A dialled-down channel still drives its shape key, so sculpting
        # a muted morph previews normally - the intensity is left exactly
        # as the artist set it.  Solo is left to the artist too (the solo
        # button toggles it without leaving the mode).
        mode = 'SCULPT' if action == 'SCULPT' else 'EDIT'
        try:
            bpy.ops.object.mode_set(mode=mode)
        except RuntimeError as exc:
            return 'ERROR', f"Could not enter {mode.title()} mode: {exc}"
        return 'INFO', f"{mode.title()} mode on '{source_name}'"
    if action == 'SOLO':
        state = "Soloed" if obj.show_only_shape_key else "Unsoloed"
        return 'INFO', f"{state} '{source_name}'"
    return 'INFO', f"Selected '{source_name}'"


class MHFRT_OT_select_morph_key(bpy.types.Operator):
    bl_idname = "mhfrt.select_morph_key"
    bl_label = "Select Morph Key"
    bl_description = ("Select a driven morph shape key, toggle its solo "
                      "preview, or enter Sculpt/Edit mode")
    bl_options = {'REGISTER', 'UNDO'}

    key_name: bpy.props.StringProperty()
    object_name: bpy.props.StringProperty(default="")
    action: bpy.props.EnumProperty(items=[
        ('SELECT', "Select", "Select this shape key"),
        ('SCULPT', "Sculpt", "Select this shape key and enter Sculpt mode"),
        ('EDIT', "Edit", "Select this shape key and enter Edit mode"),
        ('SOLO', "Solo", "Select this shape key and toggle its solo preview"),
    ], default='SELECT')

    def execute(self, context):
        mh = context.scene.mhfrt
        obj = (bpy.data.objects.get(self.object_name) if self.object_name
               else active_morph_object(mh, context))
        level, message = apply_morph_key_selection(
            context, mh, obj, self.key_name, self.action)
        self.report({level}, message)
        return {'FINISHED'} if level == 'INFO' else {'CANCELLED'}


class MHFRT_OT_add_morph_objects(bpy.types.Operator):
    bl_idname = "mhfrt.add_morph_objects"
    bl_label = "Add Selected Objects"
    bl_description = ("Add selected mesh objects as separate morph targets. "
                      "A shape key is created only when its morph is picked")
    bl_options = {'REGISTER', 'UNDO'}

    # Named object: the "not in the list" warning adds exactly the mesh it
    # names, so the artist never has to re-select it first.
    object_name: bpy.props.StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        mh = context.scene.mhfrt
        if mh.skeleton is None or not mh.skeleton.get(RIG_ID_PROP):
            self.report({'ERROR'}, "Build the rig first")
            return {'CANCELLED'}
        rid = mh.skeleton[RIG_ID_PROP]
        added = 0
        if self.object_name:
            named = bpy.data.objects.get(self.object_name)
            selected = [named] if _valid_extra_object(mh, named) else []
        else:
            selected = [obj for obj in context.selected_objects
                        if _valid_extra_object(mh, obj)]
        if not selected:
            self.report({'ERROR'}, "Select one or more extra mesh objects")
            return {'CANCELLED'}
        owned_elsewhere = [
            obj.name for obj in selected
            if _extra_owner_rig_id(obj) and _extra_owner_rig_id(obj) != str(rid)
        ]
        if owned_elsewhere:
            self.report(
                {'ERROR'},
                "Already belongs to another character: "
                + ", ".join(owned_elsewhere[:3]),
            )
            return {'CANCELLED'}

        for obj in selected:
            if _assign_extra_owner(obj, rid):
                added += 1
            _remember_extra_item(mh, obj)
            remember_active_morph_object(mh, obj)

        extras = extra_morph_objects(mh)
        organization.organize_current(context, extras=extras)
        sync_morph_ui_state(mh)
        from . import op_rig
        op_rig.invalidate()
        self.report(
            {'INFO'},
            f"Added {added} extra morph object(s)",
        )
        return {'FINISHED'}


class MHFRT_OT_remove_morph_object(bpy.types.Operator):
    bl_idname = "mhfrt.remove_morph_object"
    bl_label = "Remove Object"
    bl_description = "Remove this object from the additional morph list"
    bl_options = {'REGISTER', 'UNDO'}

    object_name: bpy.props.StringProperty(default="")

    def execute(self, context):
        mh = context.scene.mhfrt
        rid = _rig_id(mh)
        removed = False
        for i in range(len(mh.morph_extra_objects) - 1, -1, -1):
            obj = mh.morph_extra_objects[i].obj
            if obj and obj.name == self.object_name:
                mh.morph_extra_objects.remove(i)
                removed = True
        obj = bpy.data.objects.get(self.object_name)
        if obj is not None:
            was_cage = obj == mh.cage
            removed = _remove_extra_owner(obj, rid) or removed
            # The cage stays part of the rig even after it leaves the morph
            # list: restore the rig id _remove_extra_owner strips (that stripping
            # is meant for user meshes, which fully detach). Its armature
            # modifier and RIG_CAGE_PROP link are untouched, so it keeps riding
            # the facial rig.
            if was_cage and rid and RIG_ID_PROP not in obj:
                obj[RIG_ID_PROP] = str(rid)
        if mh.skeleton and mh.skeleton.get(MORPH_ACTIVE_OBJECT_PROP) == self.object_name:
            if mh.target:
                mh.skeleton[MORPH_ACTIVE_OBJECT_PROP] = mh.target.name
                mh.morph_extra_active_index = 0
            elif MORPH_ACTIVE_OBJECT_PROP in mh.skeleton:
                del mh.skeleton[MORPH_ACTIVE_OBJECT_PROP]
                mh.morph_extra_active_index = 0
        from . import op_rig
        op_rig.invalidate()
        sync_morph_ui_state(mh)
        level = {'INFO'} if removed else {'WARNING'}
        message = "Removed extra morph object" if removed else "Object was not in the list"
        self.report(level, message)
        return {'FINISHED'}


class MHFRT_OT_set_active_morph_object(bpy.types.Operator):
    bl_idname = "mhfrt.set_active_morph_object"
    bl_label = "Set Active Object"
    bl_description = "Make this additional object the active morph sculpt target"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty(default=0)
    object_name: bpy.props.StringProperty(default="")

    def execute(self, context):
        mh = context.scene.mhfrt
        obj = bpy.data.objects.get(self.object_name) if self.object_name else None
        if obj is None and self.index == 0:
            obj = mh.target
        if obj is None or morph_object_index(mh, obj) < 0:
            self.report({'ERROR'}, "Morph object is no longer registered for this character")
            return {'CANCELLED'}
        if context.view_layer.objects.get(obj.name) is None:
            self.report({'ERROR'}, f"'{obj.name}' is excluded from the current View Layer")
            return {'CANCELLED'}
        activate_morph_object(context, mh, obj)
        return {'FINISHED'}


class MHFRT_OT_remove_morph_key(bpy.types.Operator):
    bl_idname = "mhfrt.remove_morph_key"
    bl_label = "Remove Selected Morph"
    bl_description = ("Remove the real shape key represented by the selected "
                      "add-on morph row")
    bl_options = {'REGISTER', 'UNDO'}

    source_name: bpy.props.StringProperty(options={'HIDDEN'})
    object_name: bpy.props.StringProperty(options={'HIDDEN'})

    def execute(self, context):
        mh = context.scene.mhfrt
        selected_index = int(mh.morph_ui_active_index)
        if not (0 <= selected_index < len(mh.morph_ui_items)):
            self.report({'WARNING'}, "No add-on morph row is selected")
            return {'CANCELLED'}
        selected_source = mh.morph_ui_items[selected_index].key_name
        if selected_source != self.source_name:
            self.report({'WARNING'}, "The selected add-on morph row changed")
            return {'CANCELLED'}

        obj = bpy.data.objects.get(self.object_name)
        if obj is None or morph_object_index(mh, obj) < 0:
            self.report({'ERROR'}, "Selected morph object is not available")
            return {'CANCELLED'}
        key = existing_morph_key(obj, self.source_name)
        if key is None:
            self.report({'WARNING'},
                        "This add-on morph has no real shape key to remove")
            return {'CANCELLED'}
        if not _is_driven_key_name(key.name):
            self.report({'ERROR'}, f"'{key.name}' is not a driven morph key")
            return {'CANCELLED'}

        _mode_object(context)
        remember_active_morph_object(mh, obj)
        actual_name = key.name
        active_name = (obj.active_shape_key.name
                       if obj.active_shape_key is not None
                       and obj.active_shape_key != key else "")
        if _remove_keys(obj, [actual_name]) != 1:
            self.report({'WARNING'}, "The selected shape key no longer exists")
            return {'CANCELLED'}
        if active_name and obj.data.shape_keys is not None:
            active_index = obj.data.shape_keys.key_blocks.find(active_name)
            if active_index >= 0:
                obj.active_shape_key_index = active_index
        from . import op_rig
        op_rig.invalidate()
        self.report(
            {'INFO'},
            f"Removed '{self.source_name}' from {obj.name} - pick it again "
            "to recreate it empty",
        )
        return {'FINISHED'}


_ACTIVE_CONTROLLER_TEMPLATES = None


def _active_controller_templates():
    """Templates that actually feed RigLogic - anything else on the board (the
    frame itself, CTRL_faceGUI panel-position control, background props) is
    NOT an expression driver and must never be silently reset.
    """
    global _ACTIVE_CONTROLLER_TEMPLATES
    if _ACTIVE_CONTROLLER_TEMPLATES is None:
        templates = set()
        for name in riglogic.meta().get("gui_names", ()):
            template, _, _channel = name.rpartition(".")
            if template:
                templates.add(template)
        _ACTIVE_CONTROLLER_TEMPLATES = frozenset(templates)
    return _ACTIVE_CONTROLLER_TEMPLATES


# The look-at handles are posed like any other control but carry no DNA
# channel of their own (op_rig solves them back into CTRL_L_eye / CTRL_R_eye),
# so ``gui_names`` does not list them and a reset would leave the eyes aimed
# wherever they were last dragged.  Named explicitly rather than by loosening
# the filter: the rest of the eye-aim chain is helpers and switches, and
# CTRL_lookAtSwitch in particular is a deliberate setting, not part of the pose.
# Shared with board.rest_pose, which has to make the same call about what
# counts as "the face" - two lists would drift apart on the first change.
_AIM_CONTROLLER_TEMPLATES = board.AIM_CONTROLS


def _rig_controllers(rid, expressions_only=True):
    """(template_name -> control) for one rig's board controls.

    A control is a board pose bone, or an Object on a pre-3.0 loose-object
    board; both expose ``.location`` and custom properties, so callers do not
    have to care which. ``expressions_only`` (default) filters to templates
    that appear in the DNA ``gui_names`` list - plus the eye-aim handles - so
    utility board pieces like CTRL_faceGUI (which only slides the whole panel
    around) never end up zeroed by a reset.
    """
    controls = board.controls_for_rig(RIG_ID_PROP, rid)
    if not expressions_only:
        return controls
    active = _active_controller_templates()
    return {template: control for template, control in controls.items()
            if template in active or template in _AIM_CONTROLLER_TEMPLATES}


_CHANNEL_WINDOWS_CACHE = {}


def _channel_windows_for(morph_source_name):
    """[(gui channel, lo, hi)] - the board channels that feed one morph, each
    with the window of channel values that actually reaches it.

    Directly usable to scale ``_read_gui``'s output vector inside the RigLogic
    evaluator.  A board control such as CTRL_C_jaw hosts several morphs on
    separate channels: dialling ``jaw_open`` down leaves ``jaw_left`` alone
    because they sit on different channels.  A left/right pair is subtler -
    mouth_left and mouth_right share ONE channel, split by sign ([0, 1] feeds
    left, [-1, 0] feeds right), so the window is what keeps mouth_right at full
    strength while mouth_left is muted.  Combination morphs driven by the SAME
    direction of a control (the brow correctives) cannot be separated this way
    and do come down together.  The bundled DNA never changes inside a session,
    so the lookup is memoized."""
    cached = _CHANNEL_WINDOWS_CACHE.get(morph_source_name)
    if cached is not None:
        return cached
    entry = next((e for e in riglogic.morph_entries(head_only=True)
                  if e["name"] == morph_source_name), None)
    windows = []
    if entry is not None:
        raws = {int(r) for r in
                riglogic.raw_inputs_feeding(int(entry["input"]))}
        segments = riglogic.gui_segments()
        for index in sorted(int(g) for g in
                            riglogic.gui_channels_feeding_raws(raws)):
            spans = [(float(lo), float(hi))
                     for raw_out, lo, hi, _slope, _cut
                     in segments.get(index, ())
                     if int(raw_out) in raws]
            if spans:
                windows.append((index,
                                min(span[0] for span in spans),
                                max(span[1] for span in spans)))
    windows = tuple(windows)
    _CHANNEL_WINDOWS_CACHE[morph_source_name] = windows
    return windows


# The dialled-down channels are stored on the skeleton as an ASCII-31
# (\x1f)-joined name string plus a parallel float list, so they round-trip
# through .blend saves without needing a PropertyGroup.
_CHANNEL_SEP = "\x1f"


# --- preview mute -----------------------------------------------------------
# Two different things can hold a channel's bones still, and they must not be
# confused with each other:
#
# * Channels Intensity (the slider) is the REAL setting. It is part of how the
#   character is rigged, it is what gets saved and exported, and dialling it to
#   0 is a deliberate statement that this channel drives no bones.
# * Preview mute (the bone icon on a row) is a LOOKING aid. It silences the
#   channel's bones so the artist can see the sculpted shape by itself, and it
#   is meant to be switched on and off constantly while working.
#
# Sharing one number for both made the icon indistinguishable from a slider at
# zero: the "below full" count moved either way, and there was no way to tell a
# deliberate 0 from something switched off a second ago to look at it. They are
# stored apart now and only combined when the rows the evaluator reads are
# written (see _rebuild_channel_rows).
PREVIEW_MUTED_PROP = "mhfrt_preview_muted"


def preview_muted(skel):
    """Channel names currently silenced by the bone icon, as a set."""
    if skel is None:
        return set()
    stored = skel.get(PREVIEW_MUTED_PROP, "")
    return {name for name in str(stored).split(_CHANNEL_SEP) if name}


def is_preview_muted(skel, source_name):
    return bool(source_name) and source_name in preview_muted(skel)


def set_preview_muted(skel, source_name, muted):
    """Silence or un-silence one channel's bones for previewing. True if changed."""
    if skel is None or not source_name:
        return False
    current = preview_muted(skel)
    if muted == (source_name in current):
        return False
    if muted:
        current.add(source_name)
    else:
        current.discard(source_name)
    _store_preview_muted(skel, current)
    _rebuild_channel_rows(skel)
    _reevaluate_rig(skel)
    return True


def _store_preview_muted(skel, names):
    if names:
        skel[PREVIEW_MUTED_PROP] = _CHANNEL_SEP.join(sorted(names))
    elif PREVIEW_MUTED_PROP in skel:
        del skel[PREVIEW_MUTED_PROP]


def clear_preview_mutes(skel):
    """Un-silence every previewed channel. Returns how many were cleared."""
    if skel is None:
        return 0
    count = len(preview_muted(skel))
    if count:
        _store_preview_muted(skel, ())
        _rebuild_channel_rows(skel)
        _reevaluate_rig(skel)
    return count


def clear_channel_weights(skel):
    """Put every channel of this character back to full bone strength.

    Preview mutes are left alone - they are a viewing state, not a rig setting.
    """
    if skel is None:
        return
    for prop in (CHANNEL_NAMES_PROP, CHANNEL_WEIGHTS_PROP):
        if prop in skel:
            del skel[prop]
    _rebuild_channel_rows(skel)
    # Strip the legacy per-controller snapshot map from files saved with the
    # older "move the controller to rest" implementation so upgrading doesn't
    # leave dead metadata behind.
    legacy = "mhfrt_muted_morph_state"
    if legacy in skel:
        del skel[legacy]


def channel_drives_bones(source_name):
    """True when this channel's bone strength can be dialled at all.

    False for channels such as head_turn*, which no board control feeds."""
    return bool(source_name) and bool(_channel_windows_for(source_name))


def _apply_channel_weights(skel, weights):
    """Persist {morph name: weight} - the artist's real Channels Intensity.

    Channels at (or above) full strength are dropped, so the stored set is
    exactly the channels deliberately dialled down.  The rows the evaluator
    reads are rebuilt afterwards, because they also depend on the preview
    mutes."""
    kept = {}
    for name in sorted(weights):
        weight = min(1.0, max(0.0, float(weights[name])))
        if weight >= 1.0 - VALUE_EPS or not channel_drives_bones(name):
            continue
        kept[name] = weight
    if kept:
        names = sorted(kept)
        skel[CHANNEL_NAMES_PROP] = _CHANNEL_SEP.join(names)
        skel[CHANNEL_WEIGHTS_PROP] = [kept[name] for name in names]
    else:
        for prop in (CHANNEL_NAMES_PROP, CHANNEL_WEIGHTS_PROP):
            if prop in skel:
                del skel[prop]
    _rebuild_channel_rows(skel)


def _rebuild_channel_rows(skel):
    """Write the per-window scale rows op_rig evaluates, from BOTH sources.

    A channel's effective bone strength is its Channels Intensity, or zero
    while it is preview-muted - whichever is lower.  Where two morphs share a
    window the smallest wins, so dialling one down is never undone by a
    neighbour still at full strength.
    """
    if skel is None:
        return
    effective = dict(channel_weights(skel))
    for name in preview_muted(skel):
        if channel_drives_bones(name):
            effective[name] = 0.0
    rows = {}
    for name, weight in effective.items():
        if weight >= 1.0 - VALUE_EPS:
            continue
        for window in _channel_windows_for(name):
            rows[window] = min(rows.get(window, 1.0), weight)
    if not rows:
        for prop in (CHANNEL_GUI_INDICES_PROP, CHANNEL_GUI_FACTORS_PROP,
                     CHANNEL_GUI_LOWS_PROP, CHANNEL_GUI_HIGHS_PROP):
            if prop in skel:
                del skel[prop]
        return
    windows = sorted(rows)
    skel[CHANNEL_GUI_INDICES_PROP] = [window[0] for window in windows]
    skel[CHANNEL_GUI_LOWS_PROP] = [window[1] for window in windows]
    skel[CHANNEL_GUI_HIGHS_PROP] = [window[2] for window in windows]
    skel[CHANNEL_GUI_FACTORS_PROP] = [rows[window] for window in windows]


def channel_weights(skel):
    """{morph source name: weight} for every channel below full bone strength."""
    if skel is None:
        return {}
    stored = skel.get(CHANNEL_NAMES_PROP, "")
    if not stored:
        return {}
    names = [name for name in str(stored).split(_CHANNEL_SEP) if name]
    values = skel.get(CHANNEL_WEIGHTS_PROP)
    out = {}
    for i, name in enumerate(names):
        # Files written by the older binary isolate toggle carry names with no
        # weight list at all: those channels were fully muted.
        weight = 0.0
        if values is not None and i < len(values):
            weight = min(1.0, max(0.0, float(values[i])))
        out[name] = weight
    return out


def channel_weight(skel, source_name):
    if not source_name:
        return 1.0
    return channel_weights(skel).get(source_name, 1.0)


def set_channel_weight(skel, source_name, weight):
    """Dial one channel's bone contribution.  True when something changed."""
    if skel is None or not source_name:
        return False
    weight = min(1.0, max(0.0, float(weight)))
    current = channel_weights(skel)
    if abs(current.get(source_name, 1.0) - weight) <= VALUE_EPS:
        return False
    if weight >= 1.0 - VALUE_EPS:
        current.pop(source_name, None)
    else:
        current[source_name] = weight
    _apply_channel_weights(skel, current)
    _reevaluate_rig(skel)
    return True


def is_muted(skel, source_name):
    """True when this channel moves no bones at all (intensity dialled to 0)."""
    return channel_weight(skel, source_name) <= VALUE_EPS


def selected_channel_name(mh):
    """The morph channel the Morphs list row is parked on, "" for none."""
    items = mh.morph_ui_items
    index = int(mh.morph_ui_active_index)
    if 0 <= index < len(items):
        return items[index].key_name
    return ""


def selected_channel_weight(mh):
    return channel_weight(mh.skeleton, selected_channel_name(mh))


def set_selected_channel_weight(mh, weight):
    return set_channel_weight(mh.skeleton, selected_channel_name(mh), weight)


def _reevaluate_rig(skel):
    """Push a weight change through the always-on evaluator without a rescan.

    ``_read_gui`` reads the channel factors from the skeleton on every call,
    so we just have to nudge the rig to re-run once; the sparse joint-delta
    path recomputes only what changed.  Because no board controller actually
    moved, Blender's depsgraph wouldn't otherwise notice anything worth
    redrawing - so we also poke the view layer and tag every 3D viewport, so
    the new bone pose is visible the moment the slider moves."""
    from . import op_rig
    cache = op_rig._caches.get(skel.get(RIG_ID_PROP))
    if cache is None:
        op_rig.invalidate()
    else:
        op_rig._evaluate_guarded([cache])
    try:
        bpy.context.view_layer.update()
    except (AttributeError, RuntimeError):
        pass
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except (AttributeError, RuntimeError):
        pass


class MHFRT_OT_toggle_morph_isolate(bpy.types.Operator):
    bl_idname = "mhfrt.toggle_morph_isolate"
    bl_label = "Preview Without Bones"
    bl_description = ("Silence this channel's BONE movement so you can see its "
                      "sculpted shape by itself. RigLogic evaluates as if the "
                      "channel's board controls sat at rest; the shape key "
                      "keeps deforming and the controllers do NOT move.\n"
                      "This is a VIEWING aid - it does not change the "
                      "channel's value or its Channels Intensity, and the row "
                      "keeps its place in the list. Press again to bring the "
                      "bones back")
    bl_options = {'REGISTER', 'UNDO'}

    key_name: bpy.props.StringProperty()

    def execute(self, context):
        mh = context.scene.mhfrt
        skel = mh.skeleton
        if skel is None or not skel.get(RIG_ID_PROP):
            self.report({'ERROR'}, "Build the rig first")
            return {'CANCELLED'}
        source_name = riglogic.morph_source_name(self.key_name)

        if is_preview_muted(skel, source_name):
            set_preview_muted(skel, source_name, False)
            remaining = len(preview_muted(skel))
            suffix = "" if remaining == 0 \
                else f" ({remaining} still previewing without bones)"
            self.report({'INFO'}, f"Bones back on '{source_name}'{suffix}")
            return {'FINISHED'}

        if not channel_drives_bones(source_name):
            self.report(
                {'WARNING'},
                f"No board channel of its own drives '{source_name}'",
            )
            return {'CANCELLED'}

        set_preview_muted(skel, source_name, True)
        self.report(
            {'INFO'},
            f"Previewing '{source_name}' without its bones - its shape key "
            "still deforms, and its Channels Intensity is unchanged",
        )
        return {'FINISHED'}


class MHFRT_OT_clear_morph_previews(bpy.types.Operator):
    bl_idname = "mhfrt.clear_morph_previews"
    bl_label = "Bring Every Preview Back"
    bl_description = ("Bring the bones back on every channel you silenced with "
                      "the bone icon. Channels Intensity settings are left "
                      "exactly as they are")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        skel = mh.skeleton if mh else None
        return bool(skel is not None and preview_muted(skel))

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        count = clear_preview_mutes(skel)
        self.report({'INFO'}, f"Bones back on {count} channel(s)")
        return {'FINISHED'}


class MHFRT_OT_restore_morph_channels(bpy.types.Operator):
    bl_idname = "mhfrt.restore_morph_channels"
    bl_label = "Restore All Channels"
    bl_description = ("Put every morph channel of this character back to full "
                      "bone strength, clearing all Channels Intensity edits")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        skel = mh.skeleton if mh else None
        return bool(skel is not None and channel_weights(skel))

    def execute(self, context):
        skel = context.scene.mhfrt.skeleton
        count = len(channel_weights(skel))
        clear_channel_weights(skel)
        _reevaluate_rig(skel)
        self.report({'INFO'}, f"Restored {count} channel(s) to full strength")
        return {'FINISHED'}


class MHFRT_OT_reset_all_controllers(bpy.types.Operator):
    bl_idname = "mhfrt.reset_all_controllers"
    bl_label = "Reset All Controllers"
    bl_description = ("Return every facial-board controller of this character "
                      "to its rest position, the eye-aim handles included, "
                      "giving a neutral pose and a straight-ahead gaze. "
                      "Undoable with Ctrl+Z")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mh = getattr(context.scene, "mhfrt", None)
        return bool(mh and mh.skeleton
                    and mh.skeleton.get(RIG_ID_PROP))

    def execute(self, context):
        mh = context.scene.mhfrt
        skel = mh.skeleton
        rid = skel[RIG_ID_PROP]
        # Bone Intensity is a rig setting, not part of the pose: a neutral
        # pose leaves every dialled-down channel exactly as the artist set it
        # (Restore All Channels in the Morphs step clears those).
        n = sum(board.rest_pose(
                    control, orientation=template in _AIM_CONTROLLER_TEMPLATES)
                for template, control in _rig_controllers(rid).items())
        from . import op_rig
        op_rig.invalidate()
        # Push the neutral pose through the evaluator here rather than waiting
        # for the next depsgraph tick, so the Morphs list is already empty -
        # and its selection already gone - by the time the panel redraws.
        _reevaluate_rig(skel)
        prune_morph_selection(mh, context)
        self.report({'INFO'}, f"Reset {n} controller(s) to neutral")
        return {'FINISHED'}


class MHFRT_OT_manage_morph_keys(bpy.types.Operator):
    bl_idname = "mhfrt.manage_morph_keys"
    bl_label = "Manage Morph Keys"
    bl_description = "Compact or remove driven morph shape keys on the selected morph object"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(items=[
        ('COMPACT', "Compact Untouched", "Remove driven keys that still match Basis"),
        ('REMOVE_ALL', "Remove All Morphs", "Remove every driven morph key from the object"),
    ], default='COMPACT')

    @classmethod
    def description(cls, context, properties):
        return {
            'COMPACT': "Delete untouched driven morph keys on the selected morph object (their baked drivers are removed with them)",
            'REMOVE_ALL': "Remove EVERY driven morph key (and its baked driver) from the selected morph object. Pick a morph again to recreate it empty",
        }.get(properties.action, cls.bl_description)

    def invoke(self, context, event):
        if self.action == 'REMOVE_ALL':
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def _objects(self, context):
        mh = context.scene.mhfrt
        objects = managed_morph_objects(mh, context)
        if not objects:
            self.report({'ERROR'}, "No selected morph object")
        else:
            remember_active_morph_object(mh, objects[0])
        return mh, objects

    def execute(self, context):
        mh, objects = self._objects(context)
        if not objects:
            return {'CANCELLED'}
        _mode_object(context)

        if self.action == 'COMPACT':
            removed, kept = self._compact(objects)
            self._invalidate()
            name = objects[0].name
            self.report(
                {'INFO'},
                f"Compacted {removed} untouched key(s) on {name}; kept {kept} sculpted key(s)",
            )
            return {'FINISHED'}

        if self.action == 'REMOVE_ALL':
            removed = self._remove_driven_keys(objects)
            self._invalidate()
            self.report(
                {'INFO'},
                f"Removed {removed} driven morph key(s) from {objects[0].name}"
                f" - pick a morph to recreate it",
            )
            return {'FINISHED'}

        return {'CANCELLED'}

    def _compact(self, objects):
        removed = 0
        kept = 0
        for obj in objects:
            keys = obj.data.shape_keys
            if keys is None:
                continue
            basis_cache = {}
            remove = []
            for key in _driven_keys(obj):
                basis = key.relative_key
                if basis is None:
                    kept += 1
                    continue
                cache_key = basis.name if basis else ""
                if cache_key not in basis_cache:
                    basis_cache[cache_key] = _read_key_data(basis, len(obj.data.vertices))
                if _key_is_untouched(obj, key, basis_cache[cache_key]):
                    remove.append(key.name)
                else:
                    kept += 1
            if remove:
                # a deleted key must not leave its baked driver behind
                from . import op_drivers
                op_drivers.unbake_keys(obj, remove)
            for name in remove:
                key = obj.data.shape_keys.key_blocks.get(name)
                if key is not None:
                    obj.shape_key_remove(key)
                    removed += 1
            if obj.data.shape_keys:
                obj.data.update()
        return removed, kept

    def _remove_driven_keys(self, objects):
        removed = 0
        for obj in objects:
            names = [key.name for key in _driven_keys(obj)]
            removed += _remove_keys(obj, names)
        return removed

    @staticmethod
    def _invalidate():
        from . import op_rig
        op_rig.invalidate()


_classes = (
    MHFRT_OT_pick_morph,
    MHFRT_OT_select_morph_key,
    MHFRT_OT_add_morph_objects,
    MHFRT_OT_remove_morph_object,
    MHFRT_OT_set_active_morph_object,
    MHFRT_OT_remove_morph_key,
    MHFRT_OT_toggle_morph_isolate,
    MHFRT_OT_clear_morph_previews,
    MHFRT_OT_restore_morph_channels,
    MHFRT_OT_reset_all_controllers,
    MHFRT_OT_manage_morph_keys,
)

_MORPH_UI_REFRESH_EVENTS = ("load_post", "undo_post", "redo_post")


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    if _selection_sync_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_selection_sync_handler)
    for event in _MORPH_UI_REFRESH_EVENTS:
        handlers = getattr(bpy.app.handlers, event)
        if _morph_ui_load_handler not in handlers:
            handlers.append(_morph_ui_load_handler)
    if not bpy.app.timers.is_registered(_deferred_morph_ui_init):
        bpy.app.timers.register(_deferred_morph_ui_init, first_interval=0.0)


def unregister():
    global _pending_row_object
    if _selection_sync_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_selection_sync_handler)
    for event in _MORPH_UI_REFRESH_EVENTS:
        handlers = getattr(bpy.app.handlers, event)
        if _morph_ui_load_handler in handlers:
            handlers.remove(_morph_ui_load_handler)
    if bpy.app.timers.is_registered(_deferred_morph_ui_init):
        bpy.app.timers.unregister(_deferred_morph_ui_init)
    # A queued row pick must never fire into a disabled add-on.
    _pending_row_object = ""
    if bpy.app.timers.is_registered(_apply_row_selection):
        bpy.app.timers.unregister(_apply_row_selection)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
