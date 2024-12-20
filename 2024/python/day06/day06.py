# Parse the grid.
rows = []
cols = []
for i, line in enumerate(open("day06.txt")):
    if i == 0:
        cols = [[] for _ in range(len(line))]
    row = []
    for j, c in enumerate(list(line)):
        if c == "#":
            row.append(j)
            cols[j].append(i)
        if c == "^":
            start_position = (i, j)
    rows.append(row)

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
start_direction = NORTH


def turn(direction):
    """Change the direction in a clockwise manner."""
    return (direction + 1) % 4


def first_larger(xs, val):
    """Return the first value in the list that is larger than the given value."""
    return next(x for x in xs if x > val)


def first_smaller(xs, val):
    """Return the first value in the list that is smaller than the given value."""
    return next(x for x in xs[::-1] if x < val)


def try_barrier(barrier_y, barrier_x):
    """Try placing a barrier and see if this results in a loop."""
    # Small sanity check first. We don't want to place barriers outside the map.
    if not (0 <= barrier_y < len(rows) and 0 <= barrier_x <= len(cols)):
        return False

    # Modify the cols/rows lists in-place for speed
    cols[barrier_x].append(barrier_y)
    rows[barrier_y].append(barrier_x)
    cols[barrier_x].sort()
    rows[barrier_y].sort()

    # Start stimulating the route of the guard again, but this time we don't have to
    # track every single step and instead "jump" from barrier to barrier.
    is_loop = False
    visited = set()
    y, x = start_position
    direction = start_direction
    while True:
        if (y, x, direction) in visited:
            is_loop = True  # loop found
            break
        else:
            visited.add((y, x, direction))

        try:
            if direction == NORTH:
                y = first_smaller(cols[x], y) + 1
            elif direction == EAST:
                x = first_larger(rows[y], x) - 1
            elif direction == SOUTH:
                y = first_larger(cols[x], y) - 1
            elif direction == WEST:
                x = first_smaller(rows[y], x) + 1
            direction = turn(direction)
        except StopIteration:
            # We have walked off the map.
            break

    # Unto the modifications we made to the cols/rows lists
    cols[barrier_x].remove(barrier_y)
    rows[barrier_y].remove(barrier_x)
    return is_loop


# Start simulating the guard walking. After each step, see if placing a barrier there
# would cause a loop.
visited = set()
barrier_possibilities = set()
y, x = start_position
direction = start_direction
while (0 <= y < len(rows)) and (0 <= x < len(cols)):
    visited.add((y, x))

    # Figure out the location we would visit next if there were no barrier.
    if direction == NORTH:
        next_y, next_x = y - 1, x
    elif direction == EAST:
        next_y, next_x = y, x + 1
    elif direction == SOUTH:
        next_y, next_x = y + 1, x
    elif direction == WEST:
        next_y, next_x = y, x - 1

    # If there is a barrier, turn, otherwise walk.
    if next_y in cols[x] or next_x in rows[y]:
        direction = turn(direction)
    else:
        if try_barrier(next_y, next_x):
            barrier_possibilities.add((next_y, next_x))
        y = next_y
        x = next_x


print("part 1:", len(visited))
print("part 2:", len(barrier_possibilities))
