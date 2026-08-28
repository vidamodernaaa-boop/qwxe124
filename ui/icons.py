"""Custom monochrome icon set for the N-panel.

PNGs live in ui/icons/ (generated SDF glyphs, soft white on transparency, one
``*_done`` variant per workflow tab with a check badge). Loaded lazily into a
single preview collection; ``icon(name)`` returns an icon_value usable in any
layout call, 0 if the file is missing so the UI degrades gracefully.
"""

import os

import bpy

_pcoll = None


def _load():
    global _pcoll
    if _pcoll is None:
        import bpy.utils.previews
        _pcoll = bpy.utils.previews.new()
        d = os.path.join(os.path.dirname(__file__), "icons")
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".png"):
                    _pcoll.load(f[:-4], os.path.join(d, f), 'IMAGE')
    return _pcoll


def icon(name):
    """icon_value for layout(..., icon_value=...); 0 = no icon."""
    try:
        pc = _load()
        return pc[name].icon_id if name in pc else 0
    except Exception:
        return 0


def tab_icon(step_id, done=False):
    return icon(f"tab_{step_id.lower()}" + ("_done" if done else ""))


def register():
    pass  # lazy


def unregister():
    global _pcoll
    if _pcoll is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_pcoll)
        _pcoll = None
