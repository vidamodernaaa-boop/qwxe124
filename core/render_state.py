"""Is a render running, and what must the live rig not do while it is.

The rig is Python.  Every frame, ``frame_change_post`` reads the board, runs
RigLogic and writes ~838 bone matrices onto the ORIGINAL armature, then tags
it so the frame being rendered picks the new pose up.  During an ANIMATION
RENDER that callback runs on the render job's own thread, and the tag it ends
with is not local to the render: ``DEG_id_tag_update`` writes into every
depsgraph that holds the armature, the VIEWPORT one included - and the
viewport graph belongs to the main thread, which is still running its event
loop while the job renders.

Two threads, one graph, one of them adding entry tags while the other flushes
them.  A few frames in, the flush walks a node whose owner is already gone:

    EXCEPTION_ACCESS_VIOLATION (write) at 0x10A
    blender::deg::deg_graph_flush_updates
    blender::wm_event_do_depsgraph          <- the frame that checks the lock
    blender::wm_event_do_notifiers

Blender's own guard for exactly this is the interface lock.
``wm_event_do_depsgraph`` opens with "the whole idea of locked interface is to
prevent viewport and whatever thread from modifying the same data" and returns
immediately when it is set, so with ``scene.render.use_lock_interface`` on the
main thread does not touch the graph for the duration of the job and there is
no race left to lose.

Reported by Souhail (2026-08-16): a 1090-frame EEVEE render of a live rig, a
handful of frames written, then Blender gone - and the same file rendering to
the end with the add-on disabled, because with no add-on nothing tags anything
from a render thread.

So, two jobs:

* :func:`ensure_lock` turns that lock on for a scene that drives a live rig.
  It has to happen BEFORE the render: the flag is read once, when the job
  starts, so a ``render_init`` handler would already be too late.
* :func:`is_rendering` lets everything that is only about the viewport - the
  selection syncs, the panel bookkeeping, the deferred ``bpy.ops`` timers -
  stand down until the render is over.  None of that has any business running
  on a render thread, where reading ``bpy.context`` is not safe to begin with.

The rig evaluation itself keeps running: the render depsgraph is handed to the
handler and the pose it writes is what gets rendered.  Only the viewport half
steps aside.
"""

import threading

import bpy
from bpy.app.handlers import persistent


# Set on the render thread, read from everywhere. A plain bool: the GIL makes
# the store atomic, and nothing here needs more than "a job is running".
_rendering = False


def _on_main_thread():
    """False inside a render job's handler callbacks."""
    return threading.current_thread() is threading.main_thread()


def is_rendering():
    """True while a render job owns the scene.

    Also self-heals: a job that dies without firing ``render_complete`` or
    ``render_cancel`` (a cancelled background task, an add-on raising inside
    another handler) would otherwise leave the flag stuck and the viewport rig
    frozen for the rest of the session.  The cross-check only runs on the main
    thread - ``is_job_running`` reads the window manager's job list, which is
    the main thread's to change.
    """
    if not _rendering:
        return False
    if _on_main_thread() and not bpy.app.background:
        try:
            if not bpy.app.is_job_running('RENDER'):
                _set(False)
                return False
        except (AttributeError, TypeError):
            pass
    return True


def _set(value):
    global _rendering
    _rendering = bool(value)


@persistent
def _render_begin(*_args):
    _set(True)


@persistent
def _render_end(*_args):
    _set(False)


def lock_wanted():
    """Is the add-on allowed to switch Lock Interface on by itself?"""
    try:
        prefs = bpy.context.preferences.addons[__package__.rsplit(".", 1)[0]]
        return bool(prefs.preferences.lock_interface_for_render)
    except (KeyError, AttributeError, TypeError):
        return True                     # default on; prefs may not be up yet


def lock_needed(scene):
    """True when this scene renders a live rig with the lock still off."""
    if scene is None or bpy.app.background:
        return False
    render = getattr(scene, "render", None)
    try:
        return render is not None and not render.use_lock_interface
    except (AttributeError, ReferenceError):
        return False


def ensure_lock(scene):
    """Switch Lock Interface on for `scene`. True when this call did it.

    Called from the rig runtime whenever it has live rigs in hand, which is
    every depsgraph tick a character exists for - so the flag is on long before
    the artist reaches for F12.  Writing it costs a notifier, not a depsgraph
    tag, and it is written only when it is off.
    """
    # Cheapest test first: after the first time this fires, every later call is
    # one attribute read, which matters because the rig runtime asks on every
    # depsgraph tick.  ``_rendering`` raw rather than :func:`is_rendering`: a
    # blocking render (a script calling ``bpy.ops.render.render``) runs its
    # callbacks on the main thread, and writing render settings underneath a
    # render in progress is the one thing this module exists to avoid.  Callers
    # ask ``is_rendering`` first, which is what un-sticks a flag left set by a
    # job that died without saying so.
    if not lock_needed(scene) or _rendering or not lock_wanted():
        return False
    try:
        scene.render.use_lock_interface = True
    except (AttributeError, RuntimeError, ReferenceError):
        return False
    return True


_BEGIN_EVENTS = ("render_init", "render_pre")
# NOT render_post: in an animation render that fires per FRAME, and clearing
# the flag between frames would hand the gap back to the race this exists to
# close.
_END_EVENTS = ("render_complete", "render_cancel")


def register():
    for event in _BEGIN_EVENTS:
        hlist = getattr(bpy.app.handlers, event)
        if _render_begin not in hlist:
            hlist.append(_render_begin)
    for event in _END_EVENTS:
        hlist = getattr(bpy.app.handlers, event)
        if _render_end not in hlist:
            hlist.append(_render_end)
    _set(False)


def unregister():
    for event in _BEGIN_EVENTS:
        hlist = getattr(bpy.app.handlers, event)
        if _render_begin in hlist:
            hlist.remove(_render_begin)
    for event in _END_EVENTS:
        hlist = getattr(bpy.app.handlers, event)
        if _render_end in hlist:
            hlist.remove(_render_end)
    _set(False)
