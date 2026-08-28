"""Surface brushes for refining the wrapped cage - CLO3D-grab style.

Two modes of one modal operator:

* SLIDE - grab the topology and drag it; moved vertices are re-projected onto
  the target head every frame, so the quads slide ALONG the surface instead of
  lifting off it. The affected set is anchored at stroke start.
* SMOOTH - a true smoothing brush: while dragging, vertices under the moving
  brush continuously relax toward their neighbor average, then re-project onto
  the surface. Holding Shift in slide mode smooths too.

Open mesh borders (neck seam, eye/mouth openings) can be pinned so strokes
never pull the boundary out of place (panel toggle, or B in-tool).

Performance design (no viewport freezes):
* the target BVH is built ONCE on entry,
* affected vertices come from KD-tree radius queries - per-mouse-move work is
  proportional to the brush size, not the mesh,
* only affected vertices are re-projected; Region_Mask verts (inner lips)
  move with the brush but are never snapped to the surface.
"""

import numpy as np
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from bpy_extras import view3d_utils

from ..core import bvh as bvhmod
from ..core import nav
from ..core import wrap as wrapmod
from ..ui import gizmo_draw as gd
from .op_wrap import WRAPPED_KEY, region_mask_frozen, pose_offset_local

UNDO_MAX = 30
SMOOTH_RATE = 0.35   # relax amount per mouse-move at full strength/falloff


def _apply_affine(M, pts):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ np.asarray(M).T)[:, :3]


def _group_weights(obj, group_name):
    vg = obj.vertex_groups.get(group_name)
    if vg is None:
        return None
    weights = np.zeros(len(obj.data.vertices), dtype=float)
    gi = vg.index
    for i, v in enumerate(obj.data.vertices):
        for g in v.groups:
            if g.group == gi:
                weights[i] = max(0.0, min(1.0, float(g.weight)))
                break
    return weights


# ----------------------------------------------------------------- drawing ---

def _draw_hud(op):
    ctx = bpy.context
    region = ctx.region
    if region is None or region.type != 'WINDOW' or region != op._region:
        return
    mh = getattr(ctx.scene, "mhfrt", None)
    if mh is None:
        return
    cx = region.width * 0.5
    y = region.height - 54
    title = "SMOOTH BRUSH" if op._smoothing() else "SLIDE BRUSH"
    gd.text(cx, y, title, 17, gd.WHITE, align='CENTER')
    w = max(gd.text_width(title, 17), 320.0)
    gd.stroke([(cx - w * 0.5, y - 9), (cx + w * 0.5, y - 9)], gd.DIM, 1.0)
    what = ("drag = smooth the quads on the surface" if op._smoothing()
            else "drag = slide topology on the surface      Shift = smooth")
    gd.text(cx, y - 26,
            f"{what}      borders {'PINNED' if mh.brush_pin_boundary else 'free'}"
            f"      strokes: {op.strokes_done}",
            12, gd.MID, align='CENTER')
    gd.text(cx, y - 44,
            "F size   ·   Shift+F strength   ·   [ ] / Ctrl+Wheel size"
            "   ·   B pin borders   ·   Ctrl+Z undo stroke   ·   Esc / Enter done",
            11, gd.DIM, align='CENTER')
    if op.msg:
        gd.text(cx, y - 64, op.msg, 12, gd.WHITE, align='CENTER')
    gd.finish()


def _draw_ring(op):
    ctx = bpy.context
    if ctx.region != op._region:
        return
    center, normal = op.brush_center, op.brush_normal
    if center is None:
        return
    color = gd.WHITE if op.stroke is not None else gd.INK
    gd.ring_3d(center, normal, op.radius, color, 1.8)
    if op._smoothing():
        gd.ring_3d(center, normal, op.radius * 0.7, color, 1.2)
    gd.finish()


# ---------------------------------------------------------------- operator ---

class MHFRT_OT_slide_brush(bpy.types.Operator):
    bl_idname = "mhfrt.slide_brush"
    bl_label = "Refine Brush"
    bl_description = ("Refine the wrapped cage on the head surface. Slide mode grabs "
                      "and slides the topology (Shift = smooth); Smooth mode relaxes "
                      "the quads under the brush. Ctrl+Z undoes strokes; Esc/Enter "
                      "finishes")
    bl_options = {'REGISTER'}

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('SLIDE', "Slide", "Grab and slide the topology along the surface"),
            ('SMOOTH', "Smooth", "Relax the quads under the moving brush"),
        ],
        default='SLIDE',
    )
    cleanup_feature: bpy.props.EnumProperty(items=[
        ('NONE', "Neutral Wrap", "Refine the neutral cage wrap"),
        ('MOUTH', "Mouth Cleanup", "Brush the mouth cleanup key"),
        ('EYES', "Eyes Cleanup", "Brush the eyes cleanup key"),
    ], default='NONE', options={'SKIP_SAVE'})

    def invoke(self, context, event):
        from .op_live import stop_running
        stop_running()          # brushes and the live session share the key
        nav.refresh()           # honour the artist's own navigation bindings
        mh = context.scene.mhfrt
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run this from the 3D Viewport")
            return {'CANCELLED'}

        setup = self._brush_setup(context)
        if setup is None:
            return {'CANCELLED'}
        edit_obj, snap_obj, hit_obj, write_key, no_project, frozen, label = setup
        me = edit_obj.data
        wk = me.shape_keys.key_blocks.get(write_key) if me.shape_keys else None
        if wk is None:
            self.report({'ERROR'}, f"Missing '{write_key}' shape key")
            return {'CANCELLED'}
        wk.value = 1.0
        me.update()

        n = len(me.vertices)
        co = np.empty(n * 3)
        wk.data.foreach_get("co", co)
        self.edit_obj = edit_obj
        self.snap_obj = snap_obj
        self.hit_obj = hit_obj
        self.write_key = write_key
        self.scope_label = label
        self.mw = np.array(edit_obj.matrix_world)
        self.mwi = np.array(edit_obj.matrix_world.inverted())
        # brush the POSED surface (what the artist sees): other shape keys at
        # their current values - e.g. the Weight Cleanup mouth/eyes pose -
        # are added here and subtracted again on every write-back
        self.P = _apply_affine(
            self.mw,
            co.reshape(-1, 3) + pose_offset_local(edit_obj,
                                                  exclude={write_key}))

        self.topo = wrapmod.topology(me)
        self.bvh = bvhmod.build_world_bvh(snap_obj, context.evaluated_depsgraph_get())
        self.no_project = no_project
        self.frozen = frozen
        self.boundary = wrapmod.boundary_verts(me)
        self.kd = None
        self.size = max(max(snap_obj.dimensions), 1e-6)
        self.radius = max(mh.brush_radius * self.size, 1e-9)

        self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self.brush_center = None
        self.brush_normal = None
        self.stroke = None
        self.shift_smooth = False
        self.adjust = None      # sculpt-style F / Shift+F interactive adjust
        self.undo = []
        self.strokes_done = 0
        self.msg = ""
        self._region = context.region

        self._hud = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (self,), 'WINDOW', 'POST_PIXEL')
        self._ring = bpy.types.SpaceView3D.draw_handler_add(
            _draw_ring, (self,), 'WINDOW', 'POST_VIEW')
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # -------------------------------------------------------------- helpers --

    def _brush_setup(self, context):
        mh = context.scene.mhfrt
        cage, target = mh.cage, mh.target
        if not cage or not target or cage == target:
            self.report({'ERROR'}, "Set Head Cage and Head Target first")
            return None

        if self.cleanup_feature == 'NONE':
            me = cage.data
            wk = me.shape_keys.key_blocks.get(WRAPPED_KEY) if me.shape_keys else None
            if wk is None:
                self.report({'ERROR'}, "Wrap the cage first")
                return None
            n = len(me.vertices)
            return (
                cage,
                target,
                cage,
                WRAPPED_KEY,
                region_mask_frozen(cage),
                np.zeros(n, dtype=bool),
                "WRAP",
            )

        from .op_mouth import ensure_cleanup_pose
        spec = ensure_cleanup_pose(context, self, self.cleanup_feature,
                                   reset_amount=False)
        if spec is None:
            return None
        weights = _group_weights(target, spec["group"])
        if weights is None or not np.any(weights > 1e-4):
            self.report(
                {'ERROR'},
                f"Paint some vertices into '{spec['group']}' before brushing cleanup",
            )
            return None
        frozen = weights <= 1e-4
        return (
            target,
            cage,
            target,
            spec["key"],
            frozen.copy(),
            frozen,
            f"{spec['label'].upper()} CLEANUP",
        )

    def _smoothing(self):
        return self.mode == 'SMOOTH' or self.shift_smooth

    def _movable(self, mh, indices):
        """Filter a vertex index array by the pin-borders option."""
        if len(indices):
            indices = indices[~self.frozen[indices]]
        if mh.brush_pin_boundary and len(indices):
            indices = indices[~self.boundary[indices]]
        return indices

    def _ensure_kd(self):
        if self.kd is None:
            kd = KDTree(len(self.P))
            for i, p in enumerate(self.P):
                kd.insert(Vector(p), i)
            kd.balance()
            self.kd = kd

    def _hit(self, context):
        """Brush anchor: raycast the editable cleanup mesh / wrapped cage."""
        region, rv3d = context.region, context.region_data
        if region is None or rv3d is None:
            return None
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse)
        obj = self.hit_obj
        mwi = obj.matrix_world.inverted()
        ro = mwi @ origin
        rd = (mwi.to_3x3() @ direction)
        if rd.length < 1e-12:
            return None
        rd.normalize()
        try:
            ok, loc, nor, _ = obj.ray_cast(
                ro, rd, depsgraph=context.evaluated_depsgraph_get())
        except RuntimeError:
            return None
        if not ok:
            return None
        mw = obj.matrix_world
        return (mw @ loc, (mw.to_3x3() @ nor).normalized() if nor else None)

    def _write(self, context):
        edit_obj = self.edit_obj
        wk = edit_obj.data.shape_keys.key_blocks.get(self.write_key)
        if wk is None:
            return
        # strokes edit the POSED surface; store key data without the pose
        local = (_apply_affine(self.mwi, self.P)
                 - pose_offset_local(edit_obj, exclude={self.write_key}))
        wk.data.foreach_set("co", np.ascontiguousarray(local.reshape(-1)))
        edit_obj.data.update()
        if context.area:
            context.area.tag_redraw()

    def _project(self, indices):
        """Snap moved verts back onto the head - this is what makes it a slide."""
        limit = self.radius * 3.0
        for i in indices:
            if self.no_project[i]:
                continue
            loc, _n, _i, _d = self.bvh.find_nearest(Vector(self.P[i]), limit)
            if loc is not None:
                self.P[i] = (loc.x, loc.y, loc.z)

    def _scale_radius(self, mh, fac):
        mh.brush_radius = min(0.5, max(0.005, mh.brush_radius * fac))
        self.radius = mh.brush_radius * self.size

    # -------------------------------------- sculpt-style F / Shift+F adjust --

    def _adjust_begin(self, mh, kind):
        """F = size, Shift+F = strength: drag right/left, click to confirm,
        Esc cancels - same habit as sculpt mode."""
        self.adjust = {"kind": kind, "mouse0": self.mouse.copy(),
                       "value0": (mh.brush_radius if kind == 'RADIUS'
                                  else mh.brush_strength)}
        self._adjust_update(mh)

    def _adjust_update(self, mh):
        adj = self.adjust
        fac = 1.005 ** (self.mouse.x - adj["mouse0"].x)
        if adj["kind"] == 'RADIUS':
            mh.brush_radius = min(0.5, max(0.005, adj["value0"] * fac))
            self.radius = mh.brush_radius * self.size
            self.msg = f"Size  {mh.brush_radius:.3f}"
        else:
            mh.brush_strength = min(1.0, max(0.1, adj["value0"] * fac))
            self.msg = f"Strength  {mh.brush_strength:.2f}"

    def _adjust_end(self, mh, cancel):
        if cancel:
            adj = self.adjust
            if adj["kind"] == 'RADIUS':
                mh.brush_radius = adj["value0"]
                self.radius = mh.brush_radius * self.size
            else:
                mh.brush_strength = adj["value0"]
        self.adjust = None
        self.msg = ""

    # -------------------------------------------------------------- strokes --

    def _stroke_begin(self, context, shift):
        hit = self._hit(context)
        if hit is None:
            self.msg = "Start the stroke on the mesh"
            return
        pos, normal = hit
        self.shift_smooth = shift and self.mode == 'SLIDE'
        self._ensure_kd()

        if len(self.undo) >= UNDO_MAX:
            self.undo.pop(0)
        self.undo.append(self.P.copy())

        mh = context.scene.mhfrt
        stroke = {"smooth": self._smoothing()}
        if not stroke["smooth"]:
            found = self.kd.find_range(pos, self.radius)
            raw_indices = np.array([i for _co, i, _d in found], dtype=np.int64)
            raw_distances = np.array([d for _co, _i, d in found], dtype=float)
            keep = np.ones(len(raw_indices), dtype=bool)
            if len(raw_indices):
                keep &= ~self.frozen[raw_indices]
                if mh.brush_pin_boundary:
                    keep &= ~self.boundary[raw_indices]
            aff = raw_indices[keep]
            if not len(aff):
                self.undo.pop()
                return
            d = raw_distances[keep]
            w = ((1.0 - (d / self.radius) ** 2) ** 2) * mh.brush_strength
            region, rv3d = context.region, context.region_data
            press = view3d_utils.region_2d_to_location_3d(region, rv3d,
                                                          self.mouse, pos)
            stroke.update(aff=aff, w=w[:, None], P0=self.P[aff].copy(),
                          anchor=pos.copy(), press=np.array(press))
        self.stroke = stroke
        self.brush_center = pos
        self.brush_normal = normal
        self.strokes_done += 1
        self.msg = ""

    def _drag_slide(self, context):
        st = self.stroke
        region, rv3d = context.region, context.region_data
        cur = view3d_utils.region_2d_to_location_3d(
            region, rv3d, self.mouse, Vector(st["anchor"]))
        delta = np.array(cur) - st["press"]
        aff, w = st["aff"], st["w"]
        self.P[aff] = st["P0"] + delta[None, :] * w
        self._project(aff)
        self.brush_center = Vector(st["anchor"]) + Vector(delta)
        self._write(context)

    def _drag_smooth(self, context):
        """Continuous relax under the MOVING brush (true smoothing brush)."""
        hit = self._hit(context)
        if hit is None:
            return
        pos, normal = hit
        self.brush_center, self.brush_normal = pos, normal
        mh = context.scene.mhfrt
        found = self.kd.find_range(pos, self.radius)
        if not found:
            return
        aff = self._movable(mh, np.array([i for _co, i, _d in found],
                                         dtype=np.int64))
        if not len(aff):
            return
        d = np.linalg.norm(self.P[aff] - np.array(pos), axis=1)
        w = ((1.0 - np.clip(d / self.radius, 0.0, 1.0) ** 2) ** 2
             ) * mh.brush_strength * SMOOTH_RATE
        avg = wrapmod.neighbor_average(self.P, self.topo)
        self.P[aff] += (avg[aff] - self.P[aff]) * w[:, None]
        self._project(aff)
        self._write(context)

    def _stroke_end(self):
        self.stroke = None
        self.shift_smooth = False
        self.kd = None  # positions changed; rebuild lazily at next stroke

    # ---------------------------------------------------------------- modal --

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as e:
            self.report({'ERROR'}, f"Brush error: {e}")
            self._finish(context)
            return {'CANCELLED'}

    def _modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        mh = context.scene.mhfrt

        if (event.type == 'MIDDLEMOUSE' or event.type.startswith('NUMPAD')
                or (event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and not event.ctrl)):
            return {'PASS_THROUGH'}

        # Alt+LMB / Alt+RMB and friends belong to the viewport, not the brush
        if (self.stroke is None and self.adjust is None
                and nav.is_navigation(event)):
            return {'PASS_THROUGH'}

        # interactive F / Shift+F adjust owns all events until confirmed
        if self.adjust is not None:
            if event.type == 'MOUSEMOVE':
                self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                self._adjust_update(mh)
            elif (event.value == 'PRESS'
                    and event.type in {'LEFTMOUSE', 'RET', 'SPACE', 'F'}):
                self._adjust_end(mh, cancel=False)
            elif (event.value == 'PRESS'
                    and event.type in {'ESC', 'RIGHTMOUSE'}):
                self._adjust_end(mh, cancel=True)   # cancels the adjust only
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            if self.stroke is not None:
                if self.stroke["smooth"]:
                    self._drag_smooth(context)
                else:
                    self._drag_slide(context)
            else:
                hit = self._hit(context)
                if hit is not None:
                    self.brush_center, self.brush_normal = hit
                else:
                    self.brush_center = None
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.ctrl:
            self._scale_radius(mh, 1.15 if event.type == 'WHEELUPMOUSE' else 1 / 1.15)
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS':
            if event.type == 'LEFT_BRACKET':
                self._scale_radius(mh, 1 / 1.15)
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHT_BRACKET':
                self._scale_radius(mh, 1.15)
                return {'RUNNING_MODAL'}

            if event.type == 'F' and self.stroke is None:
                self._adjust_begin(mh, 'STRENGTH' if event.shift else 'RADIUS')
                return {'RUNNING_MODAL'}

            if event.type == 'B':
                mh.brush_pin_boundary = not mh.brush_pin_boundary
                return {'RUNNING_MODAL'}

            if event.type == 'Z' and event.ctrl:
                if self.undo:
                    self.P = self.undo.pop()
                    self.kd = None
                    self.strokes_done = max(0, self.strokes_done - 1)
                    self._write(context)
                    self.msg = ""
                else:
                    self.msg = "Nothing to undo"
                return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE':
                self._stroke_begin(context, shift=event.shift)
                return {'RUNNING_MODAL'}

            if event.type in {'ESC', 'RIGHTMOUSE', 'RET', 'SPACE'}:
                self._finish(context)
                return {'FINISHED'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.stroke is not None:
                self._stroke_end()
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _finish(self, context):
        if getattr(self, "_hud", None):
            bpy.types.SpaceView3D.draw_handler_remove(self._hud, 'WINDOW')
            self._hud = None
        if getattr(self, "_ring", None):
            bpy.types.SpaceView3D.draw_handler_remove(self._ring, 'WINDOW')
            self._ring = None
        self._write(context)
        context.window.cursor_modal_restore()
        try:
            bpy.ops.ed.undo_push(message="Refine brush")
        except Exception:
            pass
        if context.area:
            context.area.tag_redraw()
        self.report({'INFO'}, f"Brush: {self.strokes_done} stroke(s)")


_classes = (MHFRT_OT_slide_brush,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
