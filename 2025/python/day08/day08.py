from itertools import combinations

# Read in the coordinates of all junction boxes.
coords = list()
with open("input.txt") as f:
    for line in f:
        x, y, z = line.strip().split(",")
        coords.append((int(x), int(y), int(z)))


def straight_line_dist(points):
    """Compute squared straight line distance between two points."""
    p1, p2 = points
    return sum((j1 - j2) ** 2 for j1, j2 in zip(coords[p1], coords[p2]))


# Compute the order in which we are going to connect junction boxes.
connections = sorted(combinations(range(len(coords)), 2), key=straight_line_dist)

# At first, each junction box is its own network.
networks = [{i} for i in range(len(coords))]

# Start connecting boxes together, meging networks.
for step, (j1, j2) in enumerate(connections):
    if step == 1000:
        # We have connected 1000 junction boxes. Time to compute part 1.
        networks = sorted(networks, key=len)[::-1]
        print("part 1:", len(networks[0]) * len(networks[1]) * len(networks[2]))

    # For each of the two junction boxes that we are going to connect, find the index of
    # the networks they belong to.
    for i, g in enumerate(networks):
        if j1 in g:
            j1_net = i
            break
    for i, g in enumerate(networks):
        if j2 in g:
            j2_net = i
            break

    # If the two junction boxes already belong to the same network, nothing happens.
    if j1_net == j2_net:
        continue

    # Merge the networks.
    networks[j1_net] |= networks[j2_net]
    del networks[j2_net]

    # If there's only one network left, it means all the junction boxes are now
    # connected. Time to compute part 2.
    if len(networks) == 1:
        print("part 2:", coords[j1][0] * coords[j2][0])
        break  # no point in continuing to connect boxes.
