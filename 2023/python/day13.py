with open("day13_input.txt") as f:
    grids = [grid.split("\n") for grid in f.read().strip().split("\n\n")]


def find_mirror(grid, allow_smudge=False):
    """Scan for a mirror."""
    smudge = False  # whether we've already taken a smudge into account

    def cmp(line1, line2):
        global smudge
        if len(line1) != len(line2):
            return False
        if line1 == line2:
            return True
        if not allow_smudge or smudge:
            return False
        for char1, char2 in zip(line1, line2):
            if char1 != char2:
                if not smudge:
                    global smudge = True
                else:
                    return False

    for i, (line1, line2) in enumerate(zip(grid[:-1], grid[1:])):
        # Possible mirror found, verify
        if line1 == line2:
            # Mirror at the edge
            if i == 0 or i == len(grid) - 2:
                return i + 1

            # Mirror not at the edge
            for line3, line4 in zip(grid[i - 1 :: -1], grid[i + 2 :]):
                print(i, line3, line4)
                if line3 != line4:
                    # Not a mirror
                    break
            else:
                return i + 1
    return None  # no mirror found


verticals = 0
horizontals = 0
for j, grid in enumerate(grids):
    print("Grid", j)
    if i := find_mirror(grid):
        print("Vertical mirror found at row:", i)
        verticals += i
        continue
    # Transpose grid
    grid = ["".join([g[i] for g in grid]) for i in range(len(grid[0]))]
    if i := find_mirror(grid):
        print("Horizontal mirror found at column:", i)
        horizontals += i
        continue
    raise ValueError("No mirror found")
print(100 * verticals + horizontals)
