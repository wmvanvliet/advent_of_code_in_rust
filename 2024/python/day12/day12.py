NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


def neighbours(y, x):
    if y > 0:
        yield (y - 1, x)
    if x < width - 1:
        yield (y, x + 1)
    if y < height - 1:
        yield (y + 1, x)
    if x > 0:
        yield (y, x - 1)


def location_in_direction(y, x, direction):
    if direction == NORTH:
        return y - 1, x
    elif direction == EAST:
        return y, x + 1
    elif direction == SOUTH:
        return y + 1, x
    elif direction == WEST:
        return y, x - 1


def walk_edge(area):
    # Start on the left side facing north
    n_edges = 0
    circumference = 0
    location = start_location = min(area, key=lambda x: x[1])
    direction = start_direction = NORTH

    neighbour_types = set()
    while True:
        left = (direction - 1) % 4
        right = (direction + 1) % 4
        on_our_left = location_in_direction(*location, left)
        straight_ahead = location_in_direction(*location, direction)
        # print(f"{location=}, {direction=} {on_our_left=} {straight_ahead=} {n_edges=} {circumference=}")
        if on_our_left not in area:
            neighbour_types.add(garden.get(on_our_left, "."))
        if on_our_left in area:
            # Can we turn left?
            n_edges += 1
            circumference += 1
            location = on_our_left
            direction = left
        elif straight_ahead in area:
            # Can we go straight?
            location = straight_ahead
            circumference += 1
        else:
            # Guess we'll have to turn right
            n_edges += 1
            circumference += 1
            direction = right

        # Are we back where we started?
        if location == start_location and direction == start_direction:
            break  # done!

    if len(neighbour_types) == 1 and "." not in neighbour_types:
        n_edges *= 2
    return circumference, n_edges, neighbour_types


def count_edges(area):
    n_edges = 0
    for y, x in area:
        nia = [n in area for n in [(y - 1, x), (y, x + 1), (y + 1, x), (y, x - 1)]]
        match nia:
            case [False, False, False, False]:
                #   .
                #  .#.
                #   .
                n_edges += 4
            case [True, False, False, False]:
                #   #
                #  .#.
                #   .
                n_edges += 2
            case [False, True, False, False]:
                #   .
                #  .##
                #   .
                n_edges += 2
            case [False, False, True, False]:
                #   .
                #  .#.
                #   #
                n_edges += 2
            case [False, False, False, True]:
                #   .
                #  ##.
                #   .
                n_edges += 2
            case [True, True, False, False]:
                #   #
                #  .##
                #   .
                n_edges += 1
            case [False, True, True, False]:
                #   .
                #  .##
                #   #
                n_edges += 1
            case [False, False, True, True]:
                #   .
                #  ##.
                #   #
                n_edges += 1
            case [True, False, False, True]:
                #   #
                #  ##.
                #   .
                n_edges += 1
    return n_edges


garden = dict()
with open("day12.txt") as f:
    for y, line in enumerate(f):
        for x, c in enumerate(line.strip()):
            garden[(y, x)] = c
height = y + 1
width = len(line.strip())

# for y in range(height):
#    for x in range(width):
#        print(garden[(y, x)], end="")
#    print()

all_n_neighbours = dict()
to_visit = set(garden.keys())
# visited = set()
areas = list()
while len(to_visit) > 0:
    y, x = next(iter(to_visit))
    area = set([(y, x)])
    to_visit_next = [(y, x)]
    while len(to_visit_next) > 0:
        y, x = to_visit_next.pop(0)
        n_neighbours = 0
        for neighbour in neighbours(y, x):
            if garden[neighbour] != garden[(y, x)]:
                continue
            n_neighbours += 1
            if neighbour not in area:
                area.add(neighbour)
                to_visit_next.append(neighbour)
        all_n_neighbours[(y, x)] = n_neighbours
        to_visit.remove((y, x))
        # print("Visiting:", (y, x), garden[(y, x)], n_neighbours)
        # print("To visit next:", to_visit_next)
        # print("Area so far:", area)
        # print()
    areas.append(area)

part1 = 0
part2 = 0
for area in areas:
    circumference = sum(4 - all_n_neighbours[a] for a in area)
    # #print(garden[area[0]], len(area), "*", circumference, "=", len(area) * circumference)
    # part1 += len(area) * circumference
    # circumference, n_edges, neighbour_types = walk_edge(area)
    n_edges = count_edges(area)
    part1 += circumference * len(area)
    part2 += n_edges * len(area)
    print(garden[next(iter(area))], len(area), circumference, n_edges)

print("part 1:", part1)
print("part 2:", part2)
