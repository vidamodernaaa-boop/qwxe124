"""Shared monochrome GPU drawing kit.

One visual language for the whole add-on: white/grey ink over a dark outline,
no rainbow colors. 2D helpers take region pixel coordinates (POST_PIXEL);
ring_3d takes world coordinates (POST_VIEW). Every stroke is drawn with the
anti-aliased polyline shader so gizmos stay crisp at any DPI.
"""

import math

import blf
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

# --- monochrome ink palette -------------------------------------------------
WHITE = (1.00, 1.00, 1.00, 1.00)     # active / emphasis
INK = (0.92, 0.92, 0.92, 1.00)       # normal markers
MID = (0.65, 0.65, 0.65, 1.00)       # secondary text
DIM = (0.48, 0.48, 0.48, 0.95)       # incomplete / hints
FAINT = (0.80, 0.80, 0.80, 0.30)     # link lines
OUTLINE = (0.00, 0.00, 0.00, 0.85)   # dark underlay for contrast on any surface

# --- landmark-curve accent: light cyan, the one colored ink -------------------
CYAN = (0.45, 0.90, 1.00, 1.00)      # live stroke / active / hover emphasis
CYAN_SOFT = (0.42, 0.80, 0.95, 0.65) # resting curve bodies
CYAN_FAINT = (0.45, 0.85, 1.00, 0.30)  # mirror previews / merge links

# --- state ink: ON / OFF, and nothing else ----------------------------------
#
# The one place colour carries MEANING rather than emphasis, so it is kept to
# exactly two words in the HUD. Desaturated well below a pure red/green: over a
# lit viewport, full-saturation text vibrates and reads as an error rather than
# as a state, and these have to be legible against skin, hair and a grey
# background all at once.
ON = (0.44, 0.88, 0.52, 1.00)        # enabled
OFF = (0.95, 0.44, 0.44, 1.00)       # disabled


def state_color(enabled):
    return ON if enabled else OFF


def _polyline():
    return gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')


def _uniform():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


def _to3(p):
    return (p[0], p[1], p[2] if len(p) > 2 else 0.0)


def _draw_poly(pts, color, width, prim):
    vp = gpu.state.viewport_get()
    sh = _polyline()
    sh.bind()
    sh.uniform_float("viewportSize", (vp[2], vp[3]))
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("color", color)
    batch_for_shader(sh, prim, {"pos": pts}).draw(sh)


def stroke(points, color, width=1.5, closed=False):
    """Anti-aliased polyline (2D px or 3D world coordinates)."""
    if len(points) < 2:
        return
    gpu.state.blend_set('ALPHA')
    pts = [_to3(p) for p in points]
    if closed:
        pts.append(pts[0])
    _draw_poly(pts, color, width, 'LINE_STRIP')


def segments(points, color, width=1.5):
    """Anti-aliased independent line segments (flat list of point pairs)."""
    if len(points) < 2:
        return
    gpu.state.blend_set('ALPHA')
    _draw_poly([_to3(p) for p in points], color, width, 'LINES')


def circle_points(center, radius, segs=28):
    cx, cy = center[0], center[1]
    step = 2.0 * math.pi / segs
    return [(cx + math.cos(i * step) * radius,
             cy + math.sin(i * step) * radius) for i in range(segs)]


def ring(center, radius, color, width=1.6, outline=True, segs=28):
    pts = circle_points(center, radius, segs)
    if outline:
        stroke(pts, OUTLINE, width + 2.4, closed=True)
    stroke(pts, color, width, closed=True)


def dot(center, radius, color, segs=16):
    gpu.state.blend_set('ALPHA')
    pts = [(center[0], center[1], 0.0)]
    pts += [_to3(p) for p in circle_points(center, radius, segs)]
    pts.append(pts[1])
    sh = _uniform()
    sh.bind()
    sh.uniform_float("color", color)
    batch_for_shader(sh, 'TRI_FAN', {"pos": pts}).draw(sh)


# --- batched primitives -------------------------------------------------------
#
# Every batch_for_shader call builds a vertex buffer, so drawing one marker at a
# time is what actually made a face with a few hundred landmarks crawl: 400
# points meant 800 buffers per viewport per frame, twice over in the split
# view.  dots() draws any number of same-coloured dots from ONE buffer, with
# the same tessellation dot() uses - the picture is identical, only the number
# of draw calls changes.

_UNIT_CIRCLE = {}


def _unit_circle(segs):
    ring_pts = _UNIT_CIRCLE.get(segs)
    if ring_pts is None:
        step = 2.0 * math.pi / segs
        ring_pts = [(math.cos(i * step), math.sin(i * step))
                    for i in range(segs)]
        _UNIT_CIRCLE[segs] = ring_pts
    return ring_pts


def dot_tris(centers, radius, segs=16):
    """The triangle soup :func:`dots` draws - one fan per centre, flattened.

    Split out from the drawing so it can be checked without a GPU: Blender in
    background mode refuses every gpu call, which makes this the one part of
    the add-on that no headless test can execute.
    """
    ring_pts = _unit_circle(segs)
    verts = []
    for center in centers:
        cx, cy = center[0], center[1]
        prev = (cx + ring_pts[-1][0] * radius,
                cy + ring_pts[-1][1] * radius, 0.0)
        for ux, uy in ring_pts:
            cur = (cx + ux * radius, cy + uy * radius, 0.0)
            verts.append((cx, cy, 0.0))
            verts.append(prev)
            verts.append(cur)
            prev = cur
    return verts


def dots(centers, radius, color, segs=16):
    """Every dot in one triangle buffer.

    Curve BODIES are deliberately not batched the same way: they carry alpha,
    and merging them into one LINES buffer double-blends every segment join
    into a visible bead.  Bodies are per-curve (a handful); dots are per point
    (hundreds), which is the count that actually needed fixing.
    """
    if not centers:
        return
    verts = dot_tris(centers, radius, segs)
    if not verts:
        return
    gpu.state.blend_set('ALPHA')
    sh = _uniform()
    sh.bind()
    sh.uniform_float("color", color)
    batch_for_shader(sh, 'TRIS', {"pos": verts}).draw(sh)


def diamond(center, radius, color, width=1.6, outline=True):
    cx, cy = center[0], center[1]
    pts = [(cx, cy + radius), (cx + radius, cy), (cx, cy - radius), (cx - radius, cy)]
    if outline:
        stroke(pts, OUTLINE, width + 2.4, closed=True)
    stroke(pts, color, width, closed=True)


def cross(center, radius, color, width=1.4):
    cx, cy = center[0], center[1]
    segments([(cx - radius, cy), (cx + radius, cy),
              (cx, cy - radius), (cx, cy + radius)], color, width)


def dashed(a, b, color, width=1.2, dash=7.0, gap=5.0):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    pts = []
    d = 0.0
    while d < length:
        e = min(d + dash, length)
        pts += [(ax + ux * d, ay + uy * d, 0.0), (ax + ux * e, ay + uy * e, 0.0)]
        d = e + gap
    segments(pts, color, width)


def ring_3d(center, normal, radius, color, width=2.0, segs=40):
    """World-space circle oriented to a surface normal (brush/landmark cursor)."""
    n = Vector(normal) if normal is not None else Vector((0.0, 0.0, 1.0))
    if n.length < 1e-8:
        n = Vector((0.0, 0.0, 1.0))
    n.normalize()
    a = n.orthogonal().normalized()
    b = n.cross(a)
    c = Vector(center)
    step = 2.0 * math.pi / segs
    pts = [tuple(c + (a * math.cos(i * step) + b * math.sin(i * step)) * radius)
           for i in range(segs)]
    stroke(pts, color, width, closed=True)


# --- landmark curves ----------------------------------------------------------

def curve_stroke(points, color, width=2.0, outline=True):
    """Curve body: dark underlay for contrast, ink line on top (2D px)."""
    if len(points) < 2:
        return
    if outline:
        stroke(points, OUTLINE, width + 2.6)
    stroke(points, color, width)


def dashed_polyline(points, color, width=1.2, dash=6.0, gap=4.0):
    """Dashed run through consecutive 2D points (mirror-preview curves)."""
    for a, b in zip(points, points[1:]):
        dashed(a, b, color, width, dash, gap)


def arrow_head(p, direction, size, color, width=1.8, outline=True):
    """Open-V arrowhead with its tip at p, pointing along `direction` (2D)."""
    dx, dy = direction[0], direction[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    dx, dy = dx / n, dy / n
    px, py = -dy, dx
    a = (p[0] - (dx - px * 0.75) * size, p[1] - (dy - py * 0.75) * size)
    b = (p[0] - (dx + px * 0.75) * size, p[1] - (dy + py * 0.75) * size)
    pts = [a, (p[0], p[1]), b]
    if outline:
        stroke(pts, OUTLINE, width + 2.4)
    stroke(pts, color, width)


def tick(p, size, color, width=1.4):
    """Small vertical bar - marks a centre-merged (unmirrored) point."""
    pts = [(p[0], p[1] - size), (p[0], p[1] + size)]
    segments(pts, OUTLINE, width + 2.2)
    segments(pts, color, width)


def smooth_polyline(points, subdiv=6, closed=False):
    """Catmull-Rom curve through 2D points - display smoothing only, the
    underlying landmark data stays a plain polyline.

    ``closed`` makes it PERIODIC rather than merely appending the first point
    again: every span, the seam included, sees its true neighbours, so a loop
    is one continuous curve with no privileged joint.

    The distinction is not cosmetic.  Catmull-Rom needs a control point on each
    side of a span, and an open curve invents the two missing ones by repeating
    its ends.  Closing the ring by hand - ``pts + [pts[0]]`` - keeps those
    invented controls, which does two visible things to a loop: the seam gets a
    corner, because the tangent at the shared point is computed from a different
    pair on each side, and the last span leaves the ring tangentially instead of
    curving back into it, which reads as a straight line stuck onto the curve.
    Wrapping the controls removes both.
    """
    n = len(points)
    if n < 3 or subdiv < 2:
        # Below three points there is no curvature to find; a "loop" of two is
        # still closed, and has to come back.
        return list(points) + [points[0]] if closed and n > 1 else list(points)
    if closed:
        pts = [points[-1]] + list(points) + [points[0], points[1]]
        spans = n                       # the seam span closes the ring
        last = points[0]
    else:
        pts = [points[0]] + list(points) + [points[-1]]
        spans = n - 1
        last = points[-1]
    out = []
    for i in range(1, spans + 1):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(subdiv):
            t = j / subdiv
            t2 = t * t
            t3 = t2 * t
            out.append((
                0.5 * ((2.0 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3),
                0.5 * ((2.0 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3),
            ))
    out.append((last[0], last[1]))
    return out


# --- markers (composite, one look everywhere) --------------------------------

def marker_src(p, r=6.5, color=INK, width=1.6, fill=False):
    """Cage-side landmark: ring + center dot (filled ring while dragging)."""
    ring(p, r, color, width)
    dot(p, (r - 2.5) if fill else 2.0, color)


def marker_tgt(p, r=6.5, color=INK, width=1.6, fill=False):
    """Target-side landmark: diamond (dot appears while dragging)."""
    diamond(p, r + 1.0, color, width)
    if fill:
        dot(p, r - 2.5, color)


# --- text ---------------------------------------------------------------------

def text(x, y, s, size=12, color=INK, shadow=True, align='LEFT'):
    f = 0
    if shadow:
        blf.enable(f, blf.SHADOW)
        blf.shadow(f, 3, 0.0, 0.0, 0.0, 0.9)
    blf.size(f, size)
    blf.color(f, *color)
    if align != 'LEFT':
        w = blf.dimensions(f, s)[0]
        x -= w if align == 'RIGHT' else w * 0.5
    blf.position(f, x, y, 0)
    blf.draw(f, s)
    if shadow:
        blf.disable(f, blf.SHADOW)


def text_width(s, size=12):
    blf.size(0, size)
    return blf.dimensions(0, s)[0]


def text_runs(x, y, runs, size=12, shadow=True, align='LEFT'):
    """One line of text whose pieces carry their own colours.

    ``runs`` is ``[(string, colour), ...]``.  Laid out by measuring each piece
    and advancing, rather than drawn as one string, so a HUD line can say
    "symmetry ON" with only the ON in green - the state is then readable at a
    glance instead of having to be read as a word.

    Returns the total width, so a caller can rule a line under it.
    """
    blf.size(0, size)
    widths = [blf.dimensions(0, s)[0] for s, _c in runs]
    total = sum(widths)
    if align == 'CENTER':
        x -= total * 0.5
    elif align == 'RIGHT':
        x -= total
    for (string, color), width in zip(runs, widths):
        text(x, y, string, size, color, shadow=shadow)
        x += width
    return total


def finish():
    """Restore GPU state after a draw callback."""
    gpu.state.blend_set('NONE')
