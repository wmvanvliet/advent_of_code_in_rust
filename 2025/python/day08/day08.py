from math import sqrt
from collections import defaultdict

junctions = list()
with open("input.txt") as f:
    for line in f:
        x, y, z = line.strip().split(",")
        junctions.append((int(x), int(y), int(z)))
print(len(junctions))

distances = list()
for i, j1 in enumerate(junctions):
    for j, j2 in enumerate(junctions[i + 1:], i + 1):
        distances.append((i, j, sqrt((j1[0] - j2[0]) ** 2 + (j1[1] - j2[1]) ** 2 + (j1[2] - j2[2]) ** 2)))

connections = defaultdict(set)
for i, j, _ in sorted(distances, key=lambda x: x[2])[:1000]:
    connections[i].add(j)
    connections[j].add(i)

graphs = list()
to_visit = set(range(len(junctions)))
while len(to_visit) > 0:
    junction = to_visit.pop()
    graph = {junction}
    edges = connections[junction]
    while len(edges) > 0:
        neighbour = edges.pop()
        if neighbour in to_visit:
            to_visit.remove(neighbour)
            graph.add(neighbour)
            edges |= connections[neighbour]
    graphs.append(graph)

graphs = sorted(graphs, key=len)[::-1]
print("part 1:", len(graphs[0]) * len(graphs[1]) * len(graphs[2]))
