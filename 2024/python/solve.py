import re
from functools import cmp_to_key
from tqdm import tqdm

import numpy as np


def is_safe(levels):
    differences = np.diff(levels)
    return (
        len(np.unique(np.sign(differences))) == 1
        and abs(differences).min() >= 1
        and abs(differences).max() <= 3
    )


n_safe = 0
n_safe_with_dampner = 0
reports = [
    [int(level) for level in report.split()]
    for report in open("day2.txt").read().split("\n")
]
for levels in reports:
    if is_safe(levels):
        n_safe += 1
        continue
    for i in range(len(levels)):
        if is_safe(levels[:i] + levels[i + 1 :]):
            n_safe_with_dampner += 1
            break

print("Day 2, part 1:", n_safe)
print("Day 2, part 1:", n_safe + n_safe_with_dampner)

matches = re.findall(
    r"(mul\((\d\d?\d?),(\d\d?\d?)\)|do\(\)|don't\(\))", open("day3.txt").read()
)
ans_part1 = 0
ans_part2 = 0
do = True
for instr, arg1, arg2 in matches:
    match instr.split("(", 1)[0]:
        case "mul":
            ans_part1 += int(arg1) * int(arg2)
            if do:
                ans_part2 += int(arg1) * int(arg2)
        case "don't":
            do = False
        case "do":
            do = True
        case _:
            print(instr, arg1, arg2)

print("Day 3, part 1:", ans_part1)
print("Day 3, part 2:", ans_part2)


lines = open("day4.txt").read().strip().split("\n")
n_lines = len(lines)

n_xmas = 0
for line in lines:
    n_xmas += len(re.findall("XMAS", line))
    n_xmas += len(re.findall("SAMX", line))

lines2 = ["".join(line) for line in zip(*[list(line) for line in lines])]
for line in lines2:
    n_xmas += len(re.findall("XMAS", line))
    n_xmas += len(re.findall("SAMX", line))

lines3 = [
    "".join([lines[i + j][i] for i in range(max(-j, 0), min(n_lines - j, n_lines))])
    for j in range(-n_lines, n_lines)
]
for line in lines3:
    n_xmas += len(re.findall("XMAS", line))
    n_xmas += len(re.findall("SAMX", line))

lines4 = [
    "".join(
        [lines[i + j][-(i + 1)] for i in range(max(-j, 0), min(n_lines - j, n_lines))]
    )
    for j in range(-n_lines, n_lines)
]
for line in lines4:
    n_xmas += len(re.findall("XMAS", line))
    n_xmas += len(re.findall("SAMX", line))

print("Day 4, part 1:", n_xmas)

n_xmas = 0
for y in range(1, n_lines - 1):
    for x in range(1, n_lines - 1):
        mas1 = "".join([lines[y + i][x + i] for i in range(-1, 2)])
        mas2 = "".join([lines[y - i][x + i] for i in range(-1, 2)])
        if (mas1 == "MAS" or mas1 == "SAM") and (mas2 == "MAS" or mas2 == "SAM"):
            n_xmas += 1
print("Day 4, part 2:", n_xmas)

str_rules, str_updates = open("day5.txt").read().strip().split("\n\n")
rules = dict()
for rule in str_rules.split("\n"):
    x, y = rule.split("|")
    rules[(x, y)] = -1
    rules[(y, x)] = 1
updates = [u.split(",") for u in str_updates.split("\n")]
key_func = cmp_to_key(lambda x, y: rules.get((x, y), 0))
updates_sorted = [sorted(u, key=key_func) for u in updates]
part1 = sum(int(u[len(u) // 2]) for u, us in zip(updates, updates_sorted) if u == us)
part2 = sum(int(us[len(us) // 2]) for u, us in zip(updates, updates_sorted) if u != us)
print("Day 5, part 1:", part1)
print("Day 5, part 2:", part2)

grid = dict()
with open("day6.txt") as f:
    for y, line in enumerate(f):
        for x, c in enumerate(line.strip()):
            if c == '#':
                grid[(x, y)] = c
            if c == '^':
                start_position = (x, y)
                start_direction = (0, -1)
height = y + 1
width = len(line)
print(height, width)

def walk(grid, position, direction, visited):
    visited = set(visited)
    visited.add(position + direction)
    route = list()
    route.append(position + direction)
    while position[0] >= 0 and position[0] < width and position[1] >= 0 and position[1] <= height:
        next_position = tuple([p + d for p, d in zip(position, direction)])
        #print(position, direction, next_position, grid.get(next_position, '.'))
        if grid.get(next_position, '.') == '#':
            if direction == (0, -1):
                direction = (1, 0)
            elif direction == (0, 1):
                direction = (-1, 0)
            elif direction == (1, 0):
                direction = (0, 1)
            elif direction == (-1, 0):
                direction = (0, -1)
        else:
            if (next_position + direction) in visited:
                return "loop" 
            visited.add(next_position + direction)
            route.append(next_position + direction)
            position = next_position
    return set([v[:2] for v in visited]), route

normal_visited, normal_route = walk(grid, start_position, start_direction, [])
print("Day 6, part 1:", len(normal_visited) - 1)

# visited = set()
# route = list()
# for pos_x, pos_y, dir_x, dir_y in normal_route:
#     pos = (pos_x, pos_y)
#     if pos not in visited:
#         visited.add(pos)
#         route.append((pos_x, pos_y, dir_x, dir_y))
# ans = 0
# for i, (pos_x, pos_y, dir_x, dir_y) in tqdm(enumerate(route[1:], 1), total=len(route)):
#     next_position = (pos_x, pos_y)
#     next_direction = (dir_x, dir_y)
#     if next_position in grid:
#         raise RuntimeError("should not happen")
#         continue
#     grid[next_position] = '#'
#     position = route[i - 1][:2]
#     direction = route[i - 1][2:]
#     if walk(grid, position, direction, route[:i]) == "loop":
#         ans += 1
#     del grid[next_position]
# print("Day 6, part 2:", ans)

rows = []
cols = []
for i, line in enumerate(open("day6_test.txt")):
    if i == 0:
        cols = [[] for _ in range(len(line))]
    row = []
    for j, c in enumerate(list(line)):
        if c == '#':
            row.append(j)
            cols[j].append(i)
        if c == '^':
            start_pos = (i, j)
    rows.append(row)
print(rows)
print(cols)
print(start_pos)

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

def turn(d):
    return (d + 1) % 4

def first_larger(xs, val):
    return next(x for x in xs if x > val)

def first_smaller(xs, val):
    return next(x for x in xs[::-1] if x < val)

n_loops = 0
visited = set()
direction = NORTH
y, x = start_pos
while (0 <= y < len(rows)) and (0 <= x < len(cols)):
    visited.add((y, x))
    if direction == NORTH:
        if y - 1 in cols[x]:
            direction = turn(direction)
        else:
            try:
                #  #
                #  ^.#
                # #..
                #   #
                p1 = (y - 1, x)
                p2 = (y, first_larger(rows[y], x))
                p3 = (first_larger(cols[p2[1] - 1], y), p2[1] - 1)
                p4 = (p3[0] - 1, x - 1)
                if p4[1] in rows[p4[0]]:
                    print(p1, p2, p3, p4)
                    n_loops += 1
            except StopIteration:
                pass
            y -= 1
    elif direction == EAST:
        if x + 1 in rows[y]:
            direction = turn(direction)
        else:
            try:
                #  #
                #  .>#
                # #..
                #   #
                p1 = (y, x + 1)
                p2 = (first_larger(cols[x], y), x)
                p3 = (p2[0] - 1, first_smaller(rows[p2[0] - 1], x))
                p4 = (y - 1, p3[1] + 1)
                if p4[1] in rows[p4[0]]:
                    print(p1, p2, p3, p4)
                    n_loops += 1
            except StopIteration:
                pass
            x += 1
    elif direction == SOUTH:
        if y + 1 in cols[x]:
            direction = turn(direction)
        else:
            try:
                #  #
                #  ..#
                # #.v
                #   #
                p1 = (y + 1, x)
                p2 = (y, first_smaller(rows[y], x))
                p3 = (first_smaller(cols[p2[1] + 1], y), p2[1] + 1)
                p4 = (p3[0] + 1, x + 1)
                if p4[1] in rows[p4[0]]:
                    print(p1, p2, p3, p4)
                    n_loops += 1
            except StopIteration:
                pass
            y += 1
    elif direction == WEST:
        if x - 1 in rows[y]:
            direction = turn(direction)
        else:
            try:
                #  #
                #  ..#
                # #<.
                #   #
                p1 = (y, x - 1)
                p2 = (first_smaller(cols[x], y), x)
                p3 = (p2[0] + 1, first_larger(rows[p2[0] + 1], x))
                p4 = (y + 1, p3[1] - 1)
                if p4[1] in rows[p4[0]]:
                    print(p1, p2, p3, p4)
                    n_loops += 1
            except StopIteration:
                pass
            x -= 1
#print(visited)
print("Day 6, part 1:", len(visited))
print("Day 6, part 2:", n_loops)
