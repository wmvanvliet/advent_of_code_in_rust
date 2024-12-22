with open("day15.txt") as f:
    grid, directions = f.read().strip().split("\n\n")
contents_part1 = dict()
contents_part2 = dict()
for y, line in enumerate(grid.split("\n")):
    for x, c in enumerate(line.strip()):
        contents_part1[(y, x)] = c
        if c == "@":
            robot_part1 = y, x
            robot_part2 = y, 2 * x
        if c == "O":
            contents_part2[(y, x * 2)] = "["
            contents_part2[(y, x * 2 + 1)] = "]"
        elif c == "@":
            contents_part2[(y, x * 2)] = c
            contents_part2[(y, x * 2 + 1)] = "."
        else:
            contents_part2[(y, x * 2)] = c
            contents_part2[(y, x * 2 + 1)] = c
directions = directions.replace("\n", "")
height = y + 1
width_part1 = len(line.strip())
width_part2 = 2 * len(line.strip())

def apply_direction(position, direction, n_steps=1):
    if direction == "^":
        return position[0] - n_steps, position[1]
    elif direction == ">":
        return position[0], position[1] + n_steps
    elif direction == "v":
        return position[0] + n_steps, position[1]
    elif direction == "<":
        return position[0], position[1] - n_steps
    else:
        raise ValueError(f"Unknown direction: {direction}")

def can_move(position, direction, contents):
    #print(f"can_move({position=} {direction=} {contents[position]=}")
    c = contents[position]
    new_position = apply_direction(position, direction)
    if c == "#":
        return False
    elif c == ".":
        return True
    elif c in ["@", "O"]:
        return new_position == "." or can_move(new_position, direction, contents)
    elif c == "[":
        if direction == "<":
            return new_position == "." or can_move(new_position, direction, contents)
        elif direction == ">":
            new_position = apply_direction(new_position, direction)
            return new_position == "." or can_move(new_position, direction, contents)
        else:
            new_neighbour_position = apply_direction(new_position, ">")
            return (new_position == "." and new_neighbour_position == ".") or (can_move(new_position, direction, contents) and can_move(new_neighbour_position, direction, contents))
    elif c == "]":
        if direction == ">":
            return new_position == "." or can_move(new_position, direction, contents)
        elif direction == "<":
            new_position = apply_direction(new_position, direction)
            return new_position == "." or can_move(new_position, direction, contents)
        else:
            new_neighbour_position = apply_direction(new_position, "<")
            return (new_position == "." and new_neighbour_position == ".") or (can_move(new_position, direction, contents) and can_move(new_neighbour_position, direction, contents))
    raise RuntimeError(f"{position=} {c=}")

    
def move(position, direction, contents):
    c = contents[position]
    if c == "#":
        raise RuntimeError("Cannot move a wall.")
    elif c == ".":
        return contents, position

    new_position = apply_direction(position, direction)
    #print(f"move({position=} {direction=} {c=} {contents[new_position]=}")
    if c in ["@", "O"]:
        # Thing of width 1
        if can_move(new_position, direction, contents):
            move(new_position, direction, contents)
            contents[new_position] = c
            contents[position] = "."
            return contents, new_position
        else:
            return contents, position

    # We are a thing of width 2
    if c == "[":
        neighbour_position = apply_direction(position, ">")
        new_neighbour_position = apply_direction(new_position, ">")
        if direction == "<":
            if can_move(new_position, direction, contents):
                move(new_position, direction, contents)
            else:
                return contents, new_position
        elif direction == ">":
            if can_move(new_neighbour_position, direction, contents):
                move(new_neighbour_position, direction, contents)
            else:
                return contents, new_position
        else: # ^ or v
            if can_move(new_position, direction, contents) and can_move(new_neighbour_position, direction, contents):
                move(new_position, direction, contents)
                move(new_neighbour_position, direction, contents)
            else:
                return contents, new_position
        contents[position] = "."
        contents[neighbour_position] = "."
        contents[new_position] = "["
        contents[new_neighbour_position] = "]"
        return contents, new_position
    elif c == "]":
        neighbour_position = apply_direction(position, "<")
        new_neighbour_position = apply_direction(new_position, "<")
        if direction == ">":
            if can_move(new_position, direction, contents):
                move(new_position, direction, contents)
            else:
                return contents, new_position
        elif direction == "<":
            if can_move(new_neighbour_position, direction, contents):
                move(new_neighbour_position, direction, contents)
            else:
                return contents, new_position
        else: # ^ or v
            if can_move(new_position, direction, contents) and can_move(new_neighbour_position, direction, contents):
                move(new_position, direction, contents)
                move(new_neighbour_position, direction, contents)
            else:
                return contents, new_position
        contents[position] = "."
        contents[neighbour_position] = "."
        contents[new_position] = "]"
        contents[new_neighbour_position] = "["
        return contents, new_position

def print_map(contents, part1=True):
    if part1:
        width = width_part1
    else:
        width = width_part2
    for y in range(height):
        for x in range(width):
            print(contents[(y, x)], end="")
        print()

def gps(contents, part1=True):
    if part1:
        width = width_part1
    else:
        width = width_part2
    ans = 0
    for y in range(height):
        for x in range(width):
            if contents[(y, x)] == "O" or contents[(y, x)] == "[":
                ans += y * 100 + x
    return ans

#print_map(contents_part2, part1=False)
for direction in directions:
    #print("Move", direction)
    contents_part1, robot_part1 = move(robot_part1, direction, contents_part1)
    contents_part2, robot_part2 = move(robot_part2, direction, contents_part2)
    #print_map(contents_part2, part1=False)
print("part 1:", gps(contents_part1, part1=True))
print("part 2:", gps(contents_part2, part1=False))
