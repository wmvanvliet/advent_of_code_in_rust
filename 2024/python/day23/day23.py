start_nodes = set()
edges = dict()
with open("day23.txt") as f:
    for line in f:
        node_from, node_to = line.strip().split("-")
        if node_from.startswith("t"):
            start_nodes.add(node_from)
        if node_from not in edges:
            edges[node_from] = {node_to}
        else:
            edges[node_from].add(node_to)

        if node_to.startswith("t"):
            start_nodes.add(node_to)
        if node_to not in edges:
            edges[node_to] = {node_from}
        else:
            edges[node_to].add(node_from)

# Cliques of size 3
cliques = set()
for start_node in start_nodes:
    for hop1 in edges[start_node]:
        for hop2 in edges[hop1]:
            if start_node in edges[hop2]:
                cliques.add(tuple(sorted([start_node, hop1, hop2])))
print("part 1:", len(cliques))

def bron_kerbosch(clique, to_visit, visited, max_cliques):
    """Find all maximum cliques.

    Parameters
    ----------
    clique : set
        R: The vertices already determined to be part of the clique.
    to_visit : set
        P: The candidate vertices that could be part of the clique.
    visited : set
        X: The vertices already determined not to be part of the clique.
    max_cliques : list
        The list of maximum cliques found so far.

    Returns
    -------
    max_cliques : set
        For each clique that was determined to be a maximum clique, The
        vertices belonging to that clique (sorted).
    """
    if len(to_visit) == 0 and len(visited) == 0:
        max_cliques.append(sorted(clique))
        return
    cliques = list()
    pivot = max(to_visit | visited, key=lambda x: len(edges[x]))
    for vertex in to_visit - edges[pivot]:
        neighbours = edges[vertex]
        bron_kerbosch(clique | {vertex}, to_visit & neighbours, visited & neighbours, max_cliques)
        to_visit = to_visit - {vertex}
        visited = visited | {vertex}
    return max_cliques

max_cliques = bron_kerbosch(set(), set(edges.keys()), set(), list())
largest_clique = max(max_cliques, key=len)
print("part 2:", ",".join(largest_clique))


