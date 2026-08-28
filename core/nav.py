"""Which mouse buttons the artist uses to fly the viewport.

The modal tools (landmark curves, brush, live session) own LMB and RMB while
they run. That is fine on Blender's own keymap, where navigation lives on the
middle mouse button, but it locks out anyone working Maya/Max style - Alt+LMB
to orbit, Alt+RMB to zoom - which is exactly what Blender's *Industry
Compatible* preset and the *Emulate 3 Button Mouse* preference bind.

Rather than hard-code Alt, the navigation bindings are read out of the keymap
the artist actually has loaded, so a hand-rolled binding works too and a
default-keymap user never loses a plain click to a modifier they happened to
be holding.
"""

import bpy

# The mouse-driven half of the 3D View navigation set.
NAV_OPERATORS = {
    "view3d.rotate",
    "view3d.move",
    "view3d.zoom",
    "view3d.dolly",
}

MOUSE_TYPES = {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'}

_bindings = None


def _collect():
    """{(type, ctrl, alt, shift, oskey)} that navigates the 3D view."""
    found = set()
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return found

    configs = wm.keyconfigs
    for kc in (configs.user, configs.active, configs.default):
        if kc is None:
            continue
        for km in kc.keymaps:
            if km.space_type != 'VIEW_3D':
                continue
            for kmi in km.keymap_items:
                if not kmi.active or kmi.idname not in NAV_OPERATORS:
                    continue
                if kmi.type not in MOUSE_TYPES:
                    continue
                found.add((kmi.type, bool(kmi.ctrl), bool(kmi.alt),
                           bool(kmi.shift), bool(kmi.oskey)))

    # Three-button emulation rewrites the event before any keymap sees it, so
    # it leaves no keymap item behind to find.
    inputs = bpy.context.preferences.inputs
    if getattr(inputs, "use_mouse_emulate_3_button", False):
        modifier = getattr(inputs, "mouse_emulate_3_button_modifier", 'ALT')
        alt = modifier == 'ALT'
        found.add(('LEFTMOUSE', False, alt, False, not alt))

    return found


def refresh():
    """Re-read the keymap. Called when a modal tool starts."""
    global _bindings
    try:
        _bindings = _collect()
    except Exception:                  # noqa: BLE001 - never break a tool
        _bindings = set()


def is_navigation(event):
    """True when this event is the artist's own orbit / pan / zoom, and the
    tool should hand it to the viewport untouched."""
    if event.type not in MOUSE_TYPES:
        return False
    if _bindings is None:
        refresh()
    return (event.type, bool(event.ctrl), bool(event.alt),
            bool(event.shift), bool(event.oskey)) in _bindings
