"""Monochrome studio viewport - a reversible TOGGLE, not a one-way button.

Enabling styles every 3D view (matcap, cavity, dark backdrop, object-color
tints: light grey target, translucent graphite wireframe cage) after
snapshotting the previous shading per view space; disabling restores exactly
what the user had. The cage's studio look has its own toggle (mh.cage_studio)
so it can be inspected with its normal viewport display at any time.
"""

import bpy

TARGET_COLOR = (0.80, 0.80, 0.80, 1.0)   # light paper grey, opaque
CAGE_COLOR = (0.07, 0.07, 0.07, 0.55)    # translucent graphite
CAGE_NORMAL_COLOR = (1.0, 1.0, 1.0, 1.0)
BG_COLOR = (0.17, 0.17, 0.17)

# Snapshot/restore set, in restore order (modes first, dependents after).
_ATTRS = ("type", "light", "studio_light", "color_type", "show_object_outline",
          "show_cavity", "cavity_type", "cavity_ridge_factor",
          "cavity_valley_factor", "background_type", "background_color",
          "wireframe_color_type")
# Used only if the snapshot is gone (addon reloaded while styled).
_FALLBACK = {
    "type": 'SOLID', "light": 'STUDIO', "color_type": 'MATERIAL',
    "show_object_outline": True, "show_cavity": False,
    "background_type": 'THEME',
}
_saved = {}   # space.as_pointer() -> {attr: value}


def _view3d_spaces(context):
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            yield area.spaces.active


def _snapshot(sh):
    d = {}
    for a in _ATTRS:
        if hasattr(sh, a):
            v = getattr(sh, a)
            if not isinstance(v, (str, bool, int, float)):
                v = tuple(v)
            d[a] = v
    return d


def _apply(sh, d):
    for a in _ATTRS:                     # ordered: type/light before values
        if a in d and hasattr(sh, a):
            try:
                setattr(sh, a, d[a])
            except (TypeError, ValueError):
                pass


def _style(sh):
    sh.type = 'SOLID'
    sh.light = 'MATCAP'
    for matcap in ("basic_grey.exr", "basic_1.exr"):  # 4.5+/older names
        try:
            sh.studio_light = matcap
            break
        except (TypeError, ValueError):
            continue
    else:
        sh.light = 'STUDIO'
    sh.color_type = 'OBJECT'
    sh.show_object_outline = True
    sh.show_cavity = True
    sh.cavity_type = 'WORLD'
    sh.cavity_ridge_factor = 0.25
    sh.cavity_valley_factor = 1.0
    sh.background_type = 'VIEWPORT'
    sh.background_color = BG_COLOR
    if hasattr(sh, "wireframe_color_type"):
        try:
            sh.wireframe_color_type = 'OBJECT'
        except (TypeError, ValueError):
            pass


def tint_objects(context):
    """Object colors + wire display. The cage honors mh.cage_studio: graphite
    wireframe glass, or its normal viewport look."""
    mh = getattr(context.scene, "mhfrt", None)
    if mh is None:
        return
    if mh.target:
        mh.target.color = TARGET_COLOR
        mh.target.show_wire = False
    cage = mh.cage
    if cage and cage is not mh.target:
        if mh.cage_studio:
            cage.color = CAGE_COLOR
            cage.show_wire = True
            cage.show_all_edges = True
        else:
            cage.color = CAGE_NORMAL_COLOR
            cage.show_wire = False
            cage.show_all_edges = False
        cage.show_in_front = mh.cage_in_front


def apply_studio_state(context, enabled):
    """Update target of mh.studio_shading: style every 3D view (snapshotting
    the user's shading first) or restore what each view had before."""
    for space in _view3d_spaces(context):
        sh = space.shading
        key = space.as_pointer()
        if enabled:
            if key not in _saved:
                _saved[key] = _snapshot(sh)
            _style(sh)
        else:
            _apply(sh, _saved.pop(key, dict(_FALLBACK)))
    tint_objects(context)


def register():
    pass


def unregister():
    _saved.clear()
