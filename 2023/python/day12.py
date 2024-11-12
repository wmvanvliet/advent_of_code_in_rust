from functools import cache

@cache
def place(grid, groups_to_place, depth=0):
    """Count the number of ways the groups can be placed."""
    # pre = " " * depth
    # print(f"{pre}place({grid=}, {groups_to_place=})")

    if len(groups_to_place) == 0:
        if "#" not in grid:
            return 1  # we're done
        else:
            return 0  # not possible to fulfill grid
    possibilities = 0  # number of possible ways to place the group
    i = 0  # our position in the grid
    group = groups_to_place[0]  # group we are currently trying to place
    while i < len(grid):
        if len(grid) < (i + sum(groups_to_place) + len(groups_to_place) - 1):
            return possibilities  # not enough space to place all groups
        g = grid[i]
        if g == ".":  # cannot place a group on space known to be empty
            i += 1
            continue

        # Scan ahead to see if we can place the group
        can_place = "." not in grid[i + 1 : i + group] and (
            len(grid) == (i + group) or grid[i + group] != "#"
        )
        if g == "#":
            # The group must go here.
            if not can_place:
                return possibilities
            possibilities += place(grid[i + group + 1 :], groups_to_place[1:], depth + 1)
            return possibilities
        if g == "?":
            if can_place:
                # print(f"{pre}trying {i}")
                possibilities += place(
                    grid[i + group + 1 :], groups_to_place[1:], depth + 1
                )
                # print(f"{pre}{possibilities}"))
        i += 1

    return possibilities


with open("day12_input.txt") as f:
    ans_part1 = 0
    ans_part2 = 0
    for line in f:
        grid, groups = line.split()
        groups = tuple([int(x) for x in groups.split(",")])
        a = place(grid, groups)
        print(grid, groups, a)
        ans_part1 += a

        # Unfold for part 2
        grid = "?".join([grid] * 5)
        groups = groups * 5
        a = place(grid, groups)
        print(grid, groups, a)
        ans_part2 += a
    print("Part 1:", ans_part1)
    print("Part 2:", ans_part2)
