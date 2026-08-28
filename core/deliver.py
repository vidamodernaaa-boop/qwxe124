"""Writing a delivered character to its own .blend.

``bpy.data.libraries.write`` is the only way to write a chosen SUBSET of the
current session, but what it produces carries no window layout, so Blender
classifies it as a library: whoever double-clicks it is greeted with "Library
file, loading empty scene" and an empty viewport. Everything is in there - it
just looks like the export failed.

So the subset goes to a scratch library first, and a second Blender appends it
into an empty file and saves THAT as a real main file. There is no in-process
alternative: ``save_as_mainfile`` would write the artist's whole open session
into the deliverable, and ``read_homefile`` would throw their work away to make
room for it.

The same second pass is where the copies get their real names back. They only
wore a suffix so they could coexist with the artist's own objects during the
bake; in the delivered file the originals are gone and the name is free.
"""

import os
import subprocess
import tempfile

import bpy

# Appending a character into an empty file is a seconds-long job; the ceiling
# is only here so a wedged child process cannot hang the artist.
TIMEOUT = 600

DISPLACED_MARK = "MHFR-DISPLACED:"

_SCRIPT = '''
import sys
import bpy

lib, out, coll, name, start, end, suffix = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_homefile(use_empty=True)
with bpy.data.libraries.load(lib, link=False) as (data_from, data_to):
    data_to.collections = [coll]
appended = bpy.data.collections.get(coll)
if appended is None:
    raise SystemExit("collection %r missing from %r" % (coll, lib))

# Give the delivered copies their real names back. A name can still be held by
# something that came along as a dependency (a body rig the clothing is skinned
# to, say), so that one is moved aside rather than left to win: the character
# the artist asked for is what should carry the plain name.
displaced = []
if suffix:
    groups = (bpy.data.objects, bpy.data.armatures, bpy.data.meshes,
              bpy.data.materials, bpy.data.shape_keys, bpy.data.collections)
    for group in groups:
        for datablock in list(group):
            if not datablock.name.endswith(suffix):
                continue
            wanted = datablock.name[:-len(suffix)]
            if not wanted:
                continue
            other = group.get(wanted)
            if other is not None and other is not datablock:
                other.name = wanted + "_original"
                displaced.append(wanted)
            datablock.name = wanted

scene = bpy.context.scene
scene.collection.children.link(appended)
scene.name = name
scene.frame_start = int(start)
scene.frame_end = int(end)

# Cut the delivered file loose from the add-on entirely.
#
# The bake strips its own copies, but a delivered file also holds datablocks it
# never copied: the artist's originals that came along as dependencies, the
# board's shared handle widgets, the organization collections. Those still wore
# the tags - including the registry CARD, which is enough on its own to make a
# character appear in the Rig list of whoever opens this file. A delivered rig
# runs on drivers and constraints alone, so nothing here needs a tag to work.
#
# Done in THIS pass rather than during the bake because this is the only point
# that sees the final set, hitchhikers included.
PREFIXES = ("mhfrt", "dna_", "mhfa")
freed = 0


def _strip(datablock):
    global freed
    try:
        names = [k for k in datablock.keys() if str(k).startswith(PREFIXES)]
    except (TypeError, AttributeError):
        return
    for key in names:
        try:
            del datablock[key]
            freed += 1
        except (KeyError, TypeError, AttributeError):
            pass


for group in (bpy.data.scenes, bpy.data.collections, bpy.data.objects,
              bpy.data.armatures, bpy.data.meshes, bpy.data.materials,
              bpy.data.shape_keys, bpy.data.actions, bpy.data.images,
              bpy.data.node_groups):
    for datablock in group:
        _strip(datablock)
for armature in bpy.data.armatures:
    for bone in armature.bones:
        _strip(bone)
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE' and obj.pose is not None:
        for pose_bone in obj.pose.bones:
            _strip(pose_bone)
    for slot in obj.material_slots:
        if slot.material is not None:
            _strip(slot.material)

bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
print("MHFR-FREED:%d" % freed)
if displaced:
    print("MHFR-DISPLACED:" + ",".join(sorted(set(displaced))))
'''


def write_blend(filepath, collection, scene_name, frame_start, frame_end,
                strip_suffix=""):
    """Write `collection` to `filepath` as a file that OPENS, not a library.

    `scene_name` and the frame range are what the second pass gives the scene
    it builds around the appended collection.

    Returns (reason, displaced). `reason` is "" on success, or why the second
    pass failed - in which case the library file is left at the destination, so
    an export never comes back with nothing. `displaced` names datablocks that
    already held a delivered name and were moved aside.
    """
    scratch = f"{os.path.splitext(filepath)[0]}.mhfr-scratch.blend"
    # The COLLECTION is what goes in, never a Scene wrapped around it.  A
    # written scene would be dead weight - the second pass builds its own and
    # appends only the collection - and on Blender 5.2 it is fatal: writing a
    # subset now deep-copies every datablock it is handed, and copying a scene
    # that exists only to carry this write crashes Blender outright inside
    # BKE_view_layer_copy_data.  Nothing is lost: that scene held this one
    # collection and nothing else, and dependencies are followed from here.
    bpy.data.libraries.write(scratch, {collection}, path_remap='ABSOLUTE',
                             fake_user=True, compress=True)
    script = None
    reason = ""
    displaced = []
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8")
        with handle:
            handle.write(_SCRIPT)
            script = handle.name
        completed = subprocess.run(
            [bpy.app.binary_path, "-b", "--factory-startup",
             "--python-exit-code", "1", "--python", script, "--",
             scratch, filepath, collection.name, scene_name,
             str(frame_start), str(frame_end), strip_suffix],
            capture_output=True, text=True, timeout=TIMEOUT)
        if completed.returncode != 0 or not os.path.exists(filepath):
            tail = (completed.stderr or completed.stdout or "").strip()
            reason = (tail.splitlines()[-1] if tail
                      else "the second Blender pass did not finish")
        else:
            for line in (completed.stdout or "").splitlines():
                if line.startswith(DISPLACED_MARK):
                    displaced = [n for n in
                                 line[len(DISPLACED_MARK):].split(",") if n]
    except (subprocess.TimeoutExpired, OSError) as error:
        reason = str(error) or "the second Blender pass could not be started"
    finally:
        if script and os.path.exists(script):
            try:
                os.remove(script)
            except OSError:
                pass
    if reason:
        try:                              # hand over the library file instead
            os.replace(scratch, filepath)
        except OSError:
            pass
    elif os.path.exists(scratch):
        try:
            os.remove(scratch)
        except OSError:
            pass
    return reason, displaced
