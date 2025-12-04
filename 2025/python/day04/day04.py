# Prepare for 2D grid puzzle!
grid = set()
directions = [complex(0, -1), complex(1, -1), complex(1, 0), complex(1, 1), complex(0, 1), complex(-1, 1), complex(-1, 0), complex(-1, -1)]

# Load the grid
with open("input.txt") as f:
    for y, line in enumerate(f):
        for x, char in enumerate(line.strip()):
            if char == "@":
                grid.add(complex(x, y))

part1 = 0
part2 = 0

# Start removing rolls of paper. We break when no more rolls can be removed.
step = 1
while True:
    to_remove = set()

    # For each roll, check along each direction to see if there's a neighbour.
    for roll in grid:
        n_neighbours = sum((roll + direction in grid) for direction in directions)
        if n_neighbours < 4:
            to_remove.add(roll)

    # Are we done yet?
    if len(to_remove) == 0:
        break

    # Part 1 is the number of rolls we removed during the first step.
    if step == 1:
        part1 = len(to_remove)

    # Part 2 is the total number of rolls we removed.
    part2 += len(to_remove)

    # Remove the rolls from the grid.
    grid -= to_remove
    step += 1

print("part 1:", part1)
print("part 2:", part2)
