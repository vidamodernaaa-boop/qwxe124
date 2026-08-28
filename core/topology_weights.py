"""Topology-distance propagation for attachment skin weights.

This module deliberately has no Blender dependency.  The Parts operator does
the surface queries and supplies one normalized skin-weight row per root; the
functions here only decide which vertices are roots and spread those rows over
the accessory's own edge graph.
"""

from heapq import heappop, heappush
from math import isfinite
from statistics import median


ROOT_BAND_EDGE_FACTOR = 0.5
MAX_NEAR_ROOTS = 8
INVERSE_DISTANCE_POWER = 2.0
EPSILON = 1.0e-12


def _adjacency(vertex_count, weighted_edges):
    graph = [[] for _ in range(vertex_count)]
    for edge in weighted_edges:
        if len(edge) != 3:
            raise ValueError("weighted edges must be (vertex_a, vertex_b, length)")
        a, b, length = int(edge[0]), int(edge[1]), float(edge[2])
        if not (0 <= a < vertex_count and 0 <= b < vertex_count):
            raise ValueError("edge vertex index is outside the mesh")
        if a == b:
            continue
        # Duplicate/coincident vertices are legal in Blender meshes.  Give a
        # zero-length edge a tiny cost so Dijkstra remains deterministic.
        cost = max(length, EPSILON)
        graph[a].append((b, cost))
        graph[b].append((a, cost))
    return graph


def connected_components(vertex_count, weighted_edges):
    """Return edge-connected vertex components, including isolated verts."""
    graph = _adjacency(vertex_count, weighted_edges)
    seen = set()
    components = []
    for start in range(vertex_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor, _length in graph[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def select_root_vertices(surface_distances, weighted_edges,
                         band_factor=ROOT_BAND_EDGE_FACTOR):
    """Select the head-facing/root band independently per mesh island.

    A vertex belongs to the root band when its head-surface distance is within
    half a typical edge length of the closest vertex in its connected island.
    Per-island selection is important for separate lash cards or hair strands:
    a slightly closer card must not prevent every other card from getting a
    root.  Isolated vertices naturally become one-vertex root islands.
    """
    distances = [float(value) for value in surface_distances]
    if any(value < 0.0 or not isfinite(value) for value in distances):
        raise ValueError("surface distances must be finite and non-negative")
    if band_factor < 0.0:
        raise ValueError("root band factor must be non-negative")

    vertex_count = len(distances)
    components = connected_components(vertex_count, weighted_edges)
    # A card can have very long lateral edges and short root-to-tip edges.
    # Using every edge's median lets those long spans make the root band swallow
    # a second row.  One shortest incident edge per vertex measures the local
    # sampling scale and remains stable on both cards and individual strands.
    local_edge_lengths = [dict() for _ in components]
    component_of = {}
    for component_index, component in enumerate(components):
        for vertex in component:
            component_of[vertex] = component_index
    for a, b, length in weighted_edges:
        component_index = component_of[int(a)]
        value = float(length)
        if component_index == component_of[int(b)] and value > EPSILON:
            local = local_edge_lengths[component_index]
            local[int(a)] = min(value, local.get(int(a), value))
            local[int(b)] = min(value, local.get(int(b), value))

    roots = []
    for component_index, component in enumerate(components):
        nearest = min(distances[vertex] for vertex in component)
        lengths = list(local_edge_lengths[component_index].values())
        band = band_factor * median(lengths) if lengths else 0.0
        cutoff = nearest + max(band, EPSILON)
        component_roots = [vertex for vertex in component
                           if distances[vertex] <= cutoff]
        if not component_roots:  # defensive against future tolerance changes
            component_roots = [min(component, key=distances.__getitem__)]
        roots.extend(component_roots)
    return sorted(roots)


def nearest_root_distances(vertex_count, weighted_edges, roots,
                           max_roots=MAX_NEAR_ROOTS):
    """Return up to ``max_roots`` closest roots per vertex by graph distance."""
    if max_roots < 1:
        raise ValueError("max_roots must be at least one")
    graph = _adjacency(vertex_count, weighted_edges)
    root_ids = sorted({int(root) for root in roots})
    if any(root < 0 or root >= vertex_count for root in root_ids):
        raise ValueError("root vertex index is outside the mesh")

    found = [dict() for _ in range(vertex_count)]
    heap = []
    for root in root_ids:
        found[root][root] = 0.0
        heappush(heap, (0.0, root, root))

    while heap:
        distance, vertex, root = heappop(heap)
        if found[vertex].get(root) != distance:
            continue
        for neighbor, edge_length in graph[vertex]:
            candidate = distance + edge_length
            rows = found[neighbor]
            previous = rows.get(root)
            if previous is not None and candidate >= previous:
                continue
            if previous is None and len(rows) >= max_roots:
                worst_root, worst_distance = max(
                    rows.items(), key=lambda item: (item[1], item[0]))
                if candidate >= worst_distance:
                    continue
                del rows[worst_root]
            rows[root] = candidate
            heappush(heap, (candidate, neighbor, root))
    return found


def propagate_root_weights(vertex_count, weighted_edges, root_weights,
                           max_roots=MAX_NEAR_ROOTS,
                           distance_power=INVERSE_DISTANCE_POWER,
                           max_influences=8):
    """Spread normalized root skins through a mesh without amplitude falloff.

    Root vertices retain their sampled rows exactly.  Other vertices blend the
    nearest roots using inverse topology distance.  Contributions to the same
    bone use a max merge, so a later/weaker proposal can never overwrite a
    stronger one, and every result row is normalized only after propagation.
    """
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    if max_influences < 1:
        raise ValueError("max_influences must be at least one")

    cleaned = {}
    for raw_root, row in root_weights.items():
        root = int(raw_root)
        values = {str(name): max(0.0, float(value))
                  for name, value in row.items() if float(value) > 0.0}
        total = sum(values.values())
        if total > EPSILON:
            cleaned[root] = {name: value / total
                             for name, value in values.items()}
    if not cleaned:
        return [dict() for _ in range(vertex_count)]

    distances = nearest_root_distances(
        vertex_count, weighted_edges, cleaned, max_roots=max_roots)
    result = []
    for vertex, root_distances in enumerate(distances):
        if vertex in cleaned:
            result.append(dict(cleaned[vertex]))
            continue
        if not root_distances:
            result.append({})
            continue

        scale = min(root_distances.values())
        scale = max(scale, EPSILON)
        raw_influence = {
            root: (scale / max(distance, EPSILON)) ** distance_power
            for root, distance in root_distances.items()
        }
        influence_total = sum(raw_influence.values())
        merged = {}
        for root, raw_value in raw_influence.items():
            influence = raw_value / influence_total
            for bone, root_weight in cleaned[root].items():
                proposal = influence * root_weight
                if proposal > merged.get(bone, 0.0):
                    merged[bone] = proposal

        if len(merged) > max_influences:
            merged = dict(sorted(
                merged.items(), key=lambda item: (-item[1], item[0])
            )[:max_influences])
        total = sum(merged.values())
        result.append({bone: value / total for bone, value in merged.items()}
                      if total > EPSILON else {})
    return result
