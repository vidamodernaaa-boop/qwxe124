"""Legacy cleanup for baked shape-key drivers.

Driver baking shipped in v0.7.0-0.7.3 and was removed at the user's request.
Only the cleanup path remains: files saved by those versions may still carry
native drivers on morph keys, and when such a key is deleted or reset the
driver and its bookkeeping must go with it. The live rig also keeps skipping
any key that has a driver (op_rig._shape_key_targets), so legacy-baked files
never end up with two owners for one key.
"""

import bpy

BAKED_LIST_PROP = "mhfrt_baked_keys"     # on the shape_keys ID (legacy)


def unbake_keys(obj, names):
    """Remove any (legacy) baked driver + bookkeeping for specific keys."""
    sk = obj.data.shape_keys if obj and obj.type == 'MESH' else None
    if sk is None:
        return 0
    removed = 0
    for name in names:
        path = f'key_blocks["{name}"].value'
        ad = sk.animation_data
        if ad and any(fc.data_path == path for fc in ad.drivers):
            sk.driver_remove(path)
            removed += 1
    baked = {n for n in sk.get(BAKED_LIST_PROP, "").split("\n") if n}
    baked -= set(names)
    if baked:
        sk[BAKED_LIST_PROP] = "\n".join(sorted(baked))
    elif BAKED_LIST_PROP in sk:
        del sk[BAKED_LIST_PROP]
    return removed


def register():
    pass


def unregister():
    pass
