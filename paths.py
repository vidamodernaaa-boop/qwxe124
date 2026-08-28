"""Filesystem paths for bundled data.

``AdaHeadsv3.blend`` is the source character: every cage LOD (lod0..lod5) and
the skeleton the rig is transferred FROM. It is a straight Character DNA import
of Ada, so the bones this add-on hands a character are the same bones - same
positions, same axes, same rolls - that MetaHuman's own runtime drives, which is
what keeps the transferred anatomy right. It is resolved from ``data/`` first;
the project root is a dev fallback.

The control board comes out of the same file: the ``Ada_Head_face_gui``
armature whose pose bones are the controls, bundled under GPL-3.0 (see
``data/LICENSE-OpenRigLogic.txt``). It used to live in a separate
``face_board.blend``; that was a byte-for-byte equivalent second copy of the
same board - same 435 bones, same handle shapes, same Limit Location frames -
so it was dropped and the board is now taken from the character itself.
"""

import os

ADDON_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ADDON_DIR, "data")
_ROOT_DIR = os.path.dirname(ADDON_DIR)

# Preference order: the current source character, then the pre-3.2 asset whose
# skeleton was rebuilt rather than imported (kept only so a dev tree that still
# has it keeps working; it is not shipped).
_CANDIDATES = ("AdaHeadsv3.blend", "AdaHeads.blend")


def _resolve_blend():
    for d in (DATA_DIR, _ROOT_DIR):
        for name in _CANDIDATES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return os.path.join(DATA_DIR, _CANDIDATES[0])


BUNDLED_BLEND = _resolve_blend()


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR
