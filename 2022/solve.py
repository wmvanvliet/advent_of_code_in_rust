import numpy as np
from collections import deque

## Day 1
# This datastructure keeps track of the current top-3 elfs with most calories.
top_three = deque([0, 0, 0], maxlen=3)


def evaluate_elf(calories):
    """Compare the calories carried by an elf with the top 3.
    Update the top 3 if necessary."""
    if calories > top_three[2]:
        top_three.append(calories)
    elif calories > top_three[1]:
        top_three.popleft()
        top_three.insert(1, calories)
    elif calories > top_three[0]:
        top_three[0] = calories


# This keeps track of the calories carried by the elf we are currently
# counting.
calories = 0

# Go through the puzzle input line-by-line
with open('input_day1.txt') as f:
    for line in f:
        # If the line is empty, we start counting for the next elf.
        # Evaluate the calories carried by the current elf before
        # starting counting for the next elf.
        if line == '\n':
            evaluate_elf(calories)
            calories = 0
        else:
            # We are still counting calories for the current elf.
            calories += int(line)
    # Don't forget the final elf!
    evaluate_elf(calories)
print('Day 1, part 1:', top_three[2])
print('Day 1, part 2:', sum(top_three))

## Day 2
points = np.array([[3, 6, 0], [0, 3, 6], [6, 0, 3]])
strat = np.loadtxt('input_day2.txt', dtype='c').view('uint8') - [ord('A'), ord('X')]
print('Day 2, part 1:', (points[(strat[:, 0], strat[:, 1])] + strat[:, 1] + 1).sum())

choose = np.array([[3, 1, 2], [1, 2, 3], [2, 3, 1]])
print('Day 2, part 2:', (choose[(strat[:, 0], strat[:, 1])] + strat[:, 1] * 3).sum())

## Day 3
def bitmask(s):
    mask = 0
    for char in s:
        if ord(char) & 0b100000:
            # Lower case
            mask |= 1 << (ord(char) - ord('a'))
        else:
            # Upper case
            mask |= 1 << (ord(char) - ord('A') + 26)
    return mask

answer = 0
with open('input_day3.txt') as f:
    for line in f:
        line = line.strip()
        compartment1 = bitmask(line[:len(line)//2])
        compartment2 = bitmask(line[len(line)//2:])
        common_item = compartment1 & compartment2
        answer += common_item.bit_length()
print('Day 3, part 1:', answer)

answer = 0
with open('input_day3.txt') as f:
    for group in zip(*[iter(f)] * 3):
        badge_options = 0b1111111111111111111111111111111111111111111111111111
        for rucksack in group:
            badge_options &= bitmask(rucksack.strip())
        answer += badge_options.bit_length()
print('Day 3, part 1:', answer)
print('Day 3, part 1:', answer)


## Day 4
answer1 = 0
answer2 = 0
with open('input_day4.txt') as f:
    for line in f:
        [(from1, to1), (from2, to2)] = [allotment.split('-') for allotment in line.strip().split(',')]
        from1 = int(from1)
        to1 = int(to1)
        from2 = int(from2)
        to2 = int(to2)
        if (from1 >= from2 and to1 <= to2) or (from2 >= from1 and to2 <= to1):
            answer1 += 1
        if (from1 <= 2 and to1 >= from2) or (from2 <= to1 and to2 >= from1):
            answer2 += 1
print('Day 4, part 1:', answer1)
print('Day 4, part 1:', answer2)

## Day 5
def init_stacks(f):
    stacks = [[] for _ in range(9)]
    for _ in range(8):
        for stack, crate in zip(stacks, f.readline()[1::4]):
            if crate != ' ':
                stack.insert(0, crate)
    f.readline()
    f.readline()
    return stacks

with open('input_day5.txt') as f:
    stacks = init_stacks(f)
    while line := f.readline():
        _, amount, _, stack_from, _, stack_to = line.split()
        for _ in range(int(amount)):
            stacks[int(stack_to) - 1].append(stacks[int(stack_from) - 1].pop())
print('Day 5, part 1:', ''.join([stack[-1] for stack in stacks]))

with open('input_day5.txt') as f:
    stacks = init_stacks(f)
    while line := f.readline():
        _, amount, _, stack_from, _, stack_to = line.split()
        stacks[int(stack_to) - 1].extend(stacks[int(stack_from) - 1][-int(amount):])
        stacks[int(stack_from) - 1] = stacks[int(stack_from) - 1][:-int(amount)]
print('Day 5, part 2:', ''.join([stack[-1] for stack in stacks]))


## Day 6
with open('input_day6.txt') as f:
    buffer = f.readline().strip()

def find_marker(marker_len):
    marker_start = 0
    for marker_end in range(len(buffer)):
        i = marker_end - 1
        while i >= marker_start and buffer[i] != buffer[marker_end]:
            i -= 1
        marker_start = i + 1
        if marker_end - marker_start == marker_len:
            return marker_end

print('Day 6, part 1:', find_marker(4))
print('Day 6, part 2:', find_marker(14))


## Day 7
dir_sizes = dict()
dir_stack = list()
total_size = 0
with open('input_day7.txt') as f:
    for line in f:
        match line.strip().split():
            case ['$', 'cd', '/']:
                dir_stack = list()
            case ['$', 'cd', '..']:
                dir_stack.pop()
            case ['$', 'cd', dir_name]:
                dir_stack.append(dir_name)
            case ['$', 'ls']:
                pass
            case ['dir', dir_name]:
                pass
            case [size, file_name]:
                dir_path = ''
                for dir_name in dir_stack:
                    dir_path += '/' + dir_name
                    dir_sizes[dir_path] = dir_sizes.get(dir_path, 0) + int(size)
                total_size += int(size)
size_needed = 30_000_000 - (70_000_000 - total_size)
print('Day 7, part 1:', sum([size for size in dir_sizes.values() if size <= 100_000]))
print('Day 7, part 2:', min([size for size in dir_sizes.values() if size >= size_needed]))


## Day 8
import numpy as np

with open('input_day8.txt') as f:
    forest = np.array([[int(x) for x in list(line.strip())] for line in f])

def look_along(x):
    return x > np.hstack((-1, np.maximum.accumulate(x)[:-1]))

is_visible = np.apply_along_axis(look_along, 0, forest)
is_visible |= np.apply_along_axis(look_along, 1, forest)
is_visible |= np.apply_along_axis(look_along, 0, forest[::-1, :])[::-1, :]
is_visible |= np.apply_along_axis(look_along, 1, forest[:, ::-1])[:, ::-1]
print('Day 8, part 1:', is_visible.sum())

def compute_scenic_score(candidate_tree):
    height = forest[candidate_tree]
    row, col = candidate_tree
    if row == 0 or col == 0 or row == forest.shape[0] - 1 or col == forest.shape[1] - 1:
        return 0

    score = (np.maximum.accumulate(forest[row, col + 1:-1]) < height).sum() + 1
    score *= (np.maximum.accumulate(forest[row + 1:-1, col]) < height).sum() + 1
    score *= (np.maximum.accumulate(forest[row, col - 1:0:-1]) < height).sum() + 1
    score *= (np.maximum.accumulate(forest[row - 1:0:-1, col]) < height).sum() + 1
    return score

scenic_scores = [compute_scenic_score(tree) for tree in zip(*np.nonzero(is_visible))]
print('Day 8, part 2:', np.max(scenic_scores))


## Day 9
rope_len = 10
pos = [[0, 0] for _ in range(rope_len)]
pos_hist = [[tuple(p) for p in pos]]
with open('input_day9.txt') as f:
    for stepno, line in enumerate(f):
        direction, amount = line.split()
        amount = int(amount)
        for _ in range(amount):
            # Update the position of the head
            match direction:
                case 'U':
                    pos[0][1] += 1
                case 'D':
                    pos[0][1] -= 1
                case 'L':
                    pos[0][0] -= 1
                case 'R':
                    pos[0][0] += 1

            # Each knot follows the previous knot
            for knot in range(1, rope_len):
                # Only update position of a knot if the distance to predecessor is 2
                if (abs(pos[knot - 1][1] - pos[knot][1]) == 2) or (abs(pos[knot - 1][0] - pos[knot][0]) == 2):
                    # Update postition of a know
                    pos[knot][0] += min(max(pos[knot - 1][0] - pos[knot][0], -1), 1)
                    pos[knot][1] += min(max(pos[knot - 1][1] - pos[knot][1], -1), 1)

            # Keep track of the history of the positions of the knots
            pos_hist.append([tuple(p) for p in pos])

# Compute number of unique positions
print('Day 9, part 1:', len(set([p[1] for p in pos_hist])))
print('Day 9, part 2:', len(set([p[-1] for p in pos_hist])))

def print_pos(pos):
    for y in range(4, -1, -1):
        for x in range(6):
            for i, p in enumerate(pos):
                if p[0] == x and p[1] == y:
                    if i == 0:
                        print('H', end='')
                    else:
                        print(str(i), end='')
                    break
            else:
                print('.', end='')
        print()
    print()


## Day 10
def cpu():
    """Generates the value of the x register for each cycle."""
    x = 1
    with open('input_day10.txt') as f:
        for line in f:
            instruction, *param = line.strip().split()
            match instruction, param:
                case 'noop', []:
                    yield x
                case 'addx', [val]:
                    yield x
                    yield x
                    x += int(val)


print('Day 10, part 1:', sum([cycle * x for cycle, x in enumerate(cpu(), 1)
                              if cycle in {20, 60, 100, 140, 180, 220}]))
print('Day 10, part 2:')
output = iter(cpu())
for y in range(6):
    for x in range(40):
        if (x - 1) <= next(output) <= (x + 1):
            print('#', end='')
        else:
            print('.', end='')
    print()


## Day 11
from collections import deque
import numpy as np

items = []
operations = []
divisible_by = []
if_divisible_throw_to_monkey = []
if_not_divisible_throw_to_monkey = []

with open('input_day11.txt') as f:
    try:
        while True:
            assert next(f).startswith('Monkey')
            items.append(deque([int(x) for x in next(f).split('  Starting items: ')[1].split(', ')]))
            operation_str = next(f).split('  Operation: new = ')[1].strip()
            operations.append(np.frompyfunc(eval(f'lambda old: {operation_str}'), 1, 1))
            divisible_by.append(int(next(f).split('  Test: divisible by ')[1]))
            if_divisible_throw_to_monkey.append(int(next(f).split('    If true: throw to monkey ')[1]))
            if_not_divisible_throw_to_monkey.append(int(next(f).split('    If false: throw to monkey ')[1]))
            next(f)
    except StopIteration:
        pass

# At this point, each item is represented by its worry level. However, this
# representation will not work for part 2. We convert the worry value into a
# tuple that contains the remainder for each value we want to test the worry
# level against.
for monkey_items in items:
    for i in range(len(monkey_items)):
        monkey_items[i] = np.array([monkey_items[i] % val for val in divisible_by])

n_monkeys = len(items)
n_items_inspected = np.zeros(n_monkeys, dtype='int64')
n_rounds = 10_000
for _ in range(n_rounds):
    for monkey in range(n_monkeys):
        while len(items[monkey]) > 0:
            # Monkey inspects an item
            item = items[monkey].popleft()

            # This causes the worry level of the item to be modified
            item = operations[monkey](item)

            # After the operation performed by the monkey, we can reduce the
            # item representation down again to the remainder for each value we
            # want to test against.
            item %= divisible_by

            # Performing the test is now a lookup
            if item[monkey] == 0:
                items[if_divisible_throw_to_monkey[monkey]].append(item)
            else:
                items[if_not_divisible_throw_to_monkey[monkey]].append(item)

            # Computing the answer to the puzzle along the way
            n_items_inspected[monkey] += 1

print('Day 11, part 2:', np.multiply.reduce(np.sort(n_items_inspected)[-2:]))


## Day 12
import numpy as np
from collections import deque


# Read the puzzle input and make a matrix containing the height for each
# location.
height = []
with open('input_day12.txt') as f:
    for y, line in enumerate(f):
        height.append([ord(x) - ord('a') for x in line.strip()])
        if 'S' in line:
            x = line.index('S')
            start_loc = (y, x)
        if 'E' in line:
            x = line.index('E')
            end_loc = (y, x)
height = np.array(height)

# Start and end locations have pre-defined heights
height[start_loc] = 0
height[end_loc] = 25


def find_route(start_loc, slope_criterion, stop_criterion):
    """Find a route that fulfills the given criteria.

    Parameters
    ----------
    start_loc : (int, int)
        The y, x locations of the start of the route.
    slope_criterion : function (int -> bool)
        Function that checks whether the slope of the land is acceptable.
    stop_criterion : function (int, int -> bool)
        Function that checks whether we've reached a suitable destination.

    Returns
    -------
    dist : int
        The distance travelled.
    """
    # These are the locations we need to evaluate next, along with the
    # distances from the starting location.
    to_eval = deque([(start_loc, 0)])

    # These are the locations we have already evaluated
    seen = set([start_loc])

    # These are the potential neighbours of a location
    dirs = [(-1, 0), (+1, 0), (0, -1), (0, +1)]

    curr_loc = start_loc
    while not stop_criterion(curr_loc):
        # Grab the next location to evaluate
        curr_loc, dist = to_eval.popleft()

        # Check all the neighbours
        for d in dirs:
            new_loc = (curr_loc[0] + d[0], curr_loc[1] + d[1])

            # Check bounds
            if new_loc[0] < 0 or new_loc[0] >= height.shape[0]:
                continue
            if new_loc[1] < 0 or new_loc[1] >= height.shape[1]:
                continue

            # Don't re-visit locations we've already seen.
            if new_loc in seen:
                continue

            # Check whether the slope is ok
            if not slope_criterion(height[new_loc] - height[curr_loc]):
                continue

            # Ok, let's evaluate this location
            to_eval.append((new_loc, dist + 1))

            # Mark this location so we never re-visit it.
            seen.add(new_loc)

    return dist

print('Day 12, part 1:', find_route(start_loc, lambda slope: slope <= 1, lambda loc: loc == end_loc))
print('Day 12, part 2:', find_route(end_loc, lambda slope: slope >= -1, lambda loc: height[loc] == 0))


## Day 13
from functools import cmp_to_key

packets = []
with open('input_day13.txt') as f:
    for line in f:
        if len(line.strip()) > 0:
            packets.append(eval(line))


def cmp(a, b):
    for left_val, right_val in zip(a, b):
        if type(left_val) == int and type(right_val) == int:
            ans = left_val - right_val
        elif type(left_val) == list and type(right_val) == list:
            ans = cmp(left_val, right_val)
        elif type(left_val) == list:
            ans = cmp(left_val, [right_val])
        else:
            ans = cmp([left_val], right_val)
        if ans != 0:
            return ans
    return len(a) - len(b)

as_pairs = zip(packets[:-1:2], packets[1::2])
print('Day 13, part 1:', sum([i for i, (a, b) in enumerate(as_pairs, 1)
                              if cmp(a, b) < 0]))

dividers = ([[2]], [[6]])
packets.extend(dividers)
packets = sorted(packets, key=cmp_to_key(cmp))
print('Day 13, part 2:', (packets.index(dividers[0]) + 1) * (packets.index(dividers[1]) + 1))


## Day 14
import numpy as np

# This is the map of where walls are and sand is. Walls will be marked with a 1
# and sand with a 2, so we can make pretty plots later.
grid = np.zeros((1000, 1000), dtype='uint8')

# Parse the puzzle input, draw the walls inside the grid
with open('input_day14.txt') as f:
    for line in f:
        coords = [tuple([int(x) for x in coord.split(',')]) for coord in line.strip().split(' -> ')]
        start = coords[0]
        for end in coords[1:]:
            grid[min(start[1], end[1]):max(start[1], end[1]) + 1, min(start[0], end[0]):max(start[0], end[0]) + 1] = 1
            start = end

# Compute where the abyss starts
abyss_start = np.nonzero(grid)[0].max() + 1

# Keep track of the number of grains of sand that have come to rest
n_rest = 0

# Start dropping grains of sand.
# We'll work on a copy of the grid so we can re-use it for part 2.
grid_with_sand = grid.copy()

# Instead of dropping grains from the very top all of the time, we drop the
# next grain from the last position the previous grain was still
# falling/rolling. Hence, we keep track of the route the current grain of sand
# is taking.
route = [(500, 0)]
for _ in range(10_000):
    x, y = route.pop()
    while y < abyss_start:
        if grid_with_sand[y + 1, x] == 0:
            # Falling down
            y += 1
        elif grid_with_sand[y + 1, x - 1] == 0:
            # Rolling down and left
            y += 1
            x -= 1
        elif grid_with_sand[y + 1, x + 1] == 0:
            # Rolling down and right
            y += 1
            x += 1
        else:
            # Sand has come to rest
            grid_with_sand[y, x] = 2
            n_rest += 1
            route.pop()
            break
        route.append((x, y))
    else:
        # Sand is falling into the abyss
        break

print('Day 14, part 1:', n_rest)

# For part 2, lay down a floor
floor = abyss_start + 1
grid[floor, :] = 1

# The initial pile of sand is a big triangle
sand = np.zeros_like(grid)
for y in range(floor):
    sand[y, 500-y:500+y+1] = 2

# Go line by line, chipping away at the big pile of sand
no_sand = np.zeros_like(grid[0])
for y in range(floor):
    # Shrink the ranges of no sand by one from each end,
    # so:
    #    #####   #####
    # becomes:
    #     ###     ###
    for start, end in np.flatnonzero(np.diff(no_sand)).reshape(-1, 2):
        no_sand[start + 1] = 0
        no_sand[end] = 0

    # Add any walls on top of that
    no_sand[grid[y] > 0] = 1

    # Carve the no_sand out of the big heap of sand
    sand[y, no_sand > 0] = 0

print('Day 14, part 2:', int(sand.sum() / 2))


## Day 15
import re
from tqdm import tqdm
from numba import jit

def range_union(ranges):
    """Collapse a list of ranges into a list of disjoint ranges."""
    union = []
    for r in sorted(ranges, key=lambda r: r.start):
        if len(union) > 0 and union[-1].stop >= r.start:
            union[-1] = range(union[-1].start, max(union[-1].stop, r.stop))
        else:
            union.append(r)
    return union

def positions(sensor_loc, beacon_loc, y):
    """Given a sensor-beacon pair, compute the range of x-coords seen along the
    given y coordinate."""
    radius = abs(beacon_loc[0] - sensor_loc[0]) + abs(beacon_loc[1] - sensor_loc[1])
    radius -= abs(y - sensor_loc[1])
    if radius >= 0:
        return range(sensor_loc[0] - radius, sensor_loc[0] + radius + 1)
    else:
        return range(sensor_loc[0], sensor_loc[0])

sensor_beacon_pairs = []
with open('input_day15.txt') as f:
    for line in f:
        sensor_x, sensor_y, beacon_x, beacon_y = re.match(r'Sensor at x=(-?\d+), y=(-?\d+): closest beacon is at x=(-?\d+), y=(-?\d+)', line).groups()
        sensor_beacon_pairs.append(((int(sensor_x), int(sensor_y)), (int(beacon_x), int(beacon_y))))

y = 2_000_000
ranges_seen = range_union([positions(sensor_loc, beacon_loc, y=y)
                           for sensor_loc, beacon_loc in sensor_beacon_pairs])

# Apparently, sensors and beacons along the y-coordinate don't count?
other_stuff = []
other_stuff = set([beacon_loc[0] for _, beacon_loc in sensor_beacon_pairs if beacon_loc[1] == y])
other_stuff |= set([sensor_loc[0] for _, sensor_loc in sensor_beacon_pairs if sensor_loc[1] == y])

print('Day 15, part 1:', sum([r.stop - r.start for r in ranges_seen]) - len(other_stuff))

# Wait for a loooong time
for y in tqdm(range(4_000_000), ncols=80):
    ranges_seen = [positions(sensor_loc, beacon_loc, y=y)
                   for sensor_loc, beacon_loc in sensor_beacon_pairs]
    ranges_seen = [r for r in ranges_seen if r.start <= 4_000_000 and r.stop >= 0]
    ranges_seen = range_union(ranges_seen)
    if len(ranges_seen) > 0 and ranges_seen[0].stop <= 4_000_000:
        print('Day 15, part 2:', ranges_seen[0].stop * 4_000_000 + y)
        break


## Day 16
import re
from math import inf

flow_rates = dict()
tunnels = dict()
with open('input_day16.txt') as f:
    for line in f:
        room, flow_rate, connections = re.match(r'Valve ([A-Z][A-Z]) has flow rate=(\d+); tunnels? leads? to valves? ([A-Z, ]+)', line).groups()
        flow_rate = int(flow_rate)
        # Valves with a flow rate of 0 don't count
        if flow_rate > 0:
            flow_rates[room] = flow_rate
        tunnels[room] = connections.split(', ')
caves = tunnels.keys()
valves = set(flow_rates.keys())

# Compute all-to-all distances between the valves
D = {start_cave: {to_cave: 0 if to_cave == start_cave else inf for to_cave in caves} for start_cave in caves}
for start, destinations in tunnels.items():
    for d in destinations:
        D[start][d] = 1
for cave1 in caves:
    for cave2 in caves:
        for cave3 in caves:
            D[cave2][cave3] = min(D[cave2][cave3], D[cave2][cave1] + D[cave1][cave3])

cache = dict()
def best_route(my_pos, valves_left, time_left=30, elephant_present=False):
    if len(valves_left) == 0 or time_left <= 0:
        return 0

    situation = (my_pos, tuple(valves_left), time_left, elephant_present)
    if situation in cache:
        return cache[situation]

    scores = []

    # Either we go and turn on another valve...
    for next_valve in valves_left:
        time_spent = D[my_pos][next_valve] + 1
        score = max(0, flow_rates[next_valve] * (time_left - time_spent))
        score += best_route(next_valve, valves_left - set([next_valve]), time_left - time_spent, elephant_present)
        scores.append(score)

    # ...or we let the elephant take it from here
    if elephant_present:
        scores.append(best_route('AA', valves_left, time_left=26, elephant_present=False))

    cache[situation] = max(scores)
    return max(scores)

print('Day 16, part 1:', best_route('AA', set(valves), time_left=30))
print('Day 16, part 2:', best_route('AA', set(valves), time_left=26, elephant_present=True))


## Day 17
import numpy as np
from itertools import cycle
from time import sleep


def print_field(field):
    for y in range(np.nonzero(field)[0].min(), len(field)):
        for x in range(7):
            if field[y, x] == 1:
                print('█', end='')
            elif field[y, x] == 2:
                print('▒', end='')
            else:
                print(' ', end='')
        print()
    print()
    print()
    print()

rock_types = [
    np.array([[1, 1, 1, 1]]),
    np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]]),
    np.array([[0, 0, 1],
              [0, 0, 1],
              [1, 1, 1]]),
    np.array([[1],
              [1],
              [1],
              [1]]),
    np.array([[1, 1],
              [1, 1]]),
]

# The rock types and the jet directions cycle
rock_types = cycle([t.astype('bool') for t in rock_types])
with open('input_day17.txt') as f:
    jet_directions = cycle(list(f.readline().strip()))

# This is the tetris field we'll be filling up.
# Whenever we run out of space, we will double its size.
field = np.zeros((100, 7), dtype='bool')

# Lay down a floor so the rocks have something to fall onto.
field[-1, :] = True

def hits(rock, y, x):
    '''Does the given rock shape at the given coordinates overlap with another
    rock (or the floor)?'''
    return np.any(field[y:y + rock.shape[0], x:x + rock.shape[1]] & rock)

# Lots of bookkeeping to be done during the main loop
n_rocks_thrown = 0
field_height = 0
height_after_each_rock = list()
n_rocks_thrown_to_reach_height = dict()

# Start throwing down rocks
while True:
    rock = next(rock_types)

    # The rock starts 3 lines above the current field. The y-coordinates are a
    # bit wonky in this simulation, as they also serve as indexes in the field
    # array, hence grow larger downwards.
    y = len(field) - 1 - field_height - 3 - rock.shape[0]

    # The rock starts 2 spots to the right of the left wall
    x = 2

    # Check if we have run out of space in the field. 
    if y < 0:
        # Double the field size
        y += len(field)
        field = np.vstack((np.zeros_like(field), field))

    # The rock is falling down
    while True:
        # Move the rock sideways if it doesn't hit anything
        match next(jet_directions):
            case '<':
                next_x = max(0, x - 1)
            case '>':
                next_x = min(7 - rock.shape[1], x + 1)
        if not hits(rock, y, next_x):
            x = next_x

        # Move the rock downwards if it doesn't hit anything
        if not hits(rock, y + 1, x):
            y += 1
        else:
            # Rock has come to rest
            break

    # Fill in the shape of the rock in the field
    field[y:y + rock.shape[0], x:x + rock.shape[1]] |= rock

    # Update bookkeeping stuff
    n_rocks_thrown += 1
    field_height = len(field) - np.nonzero(field)[0].min() - 1
    height_after_each_rock.append(field_height)

    # A rock can add multiple rows at once. First repeat the last value until
    # we reach the current height...
    if len(n_rocks_thrown_to_reach_height) > 0:
        for j in range(list(n_rocks_thrown_to_reach_height.keys())[-1], field_height):
            n_rocks_thrown_to_reach_height[j] = list(n_rocks_thrown_to_reach_height.values())[-1]
    # ...then add the value for the current height
    n_rocks_thrown_to_reach_height[field_height] = n_rocks_thrown

    if n_rocks_thrown == 2022:
        print('Day 17, part 1:', field_height)

    # Cycle detection using Floyd's hare and tortoise algorithm.
    # Only kicks in after we've collected enough rows.
    if n_rocks_thrown > 1000:
        # The "hare" position is the row directly under the rock that was just
        # placed.
        hare = y + rock.shape[0]
        hare_height = len(field) - 1 - hare

        # The "tortoise" position at half the height of the "hare"
        tortoise_height = hare_height // 2
        tortoise = len(field) - 1 - tortoise_height

        # This is the magic of the algorithm: if the area surrounding the hare
        # position matches the area surrounding the tortoise position, we have
        # detected a cycle! Here, we use an area of size 20 (probably overfill).
        if np.array_equal(field[hare:hare+20], field[tortoise:tortoise+20]):
            break  # We found the cycle. That's all we need from the simulation.

# Number of rocks the elephants asked us to simulate
target_n_rocks = 1_000_000_000_000

# Length of the cycle in terms of number of rocks
cycle_n_rocks = n_rocks_thrown_to_reach_height[hare_height] - n_rocks_thrown_to_reach_height[tortoise_height]

# Length of the cycle in terms of rows in the playing field
cycle_n_rows = tortoise - hare

# To compute the answer to part 2, we first see how many complete cycles could
# be completed
part2_ans = target_n_rocks // cycle_n_rocks * cycle_n_rows
# The number of rocks needed before the cycling part and after the cycling part
# can be conveniently computed using a single modulo. See how many rows they
# produce.
part2_ans += height_after_each_rock[target_n_rocks % cycle_n_rocks - 1]
print('Day 17, part 2:', part2_ans)

## Day 18
from collections import deque
import numpy as np

deltas = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

def count_open_sides(voxels):
    cubes = set(zip(*np.where(voxels == 1)))
    open_sides = 0
    for cube in cubes:
        for delta in deltas:
            if tuple(np.add(cube, delta)) not in cubes:
                open_sides += 1
    return open_sides

voxels = np.zeros((22, 22, 22), dtype='int')
with open('input_day18.txt') as f:
    for line in f:
        x, y, z = line.strip().split(',')
        voxels[int(x) + 1, int(y) + 1, int(z) + 1] = 1

print('Day 18, part 1:', count_open_sides(voxels))

to_flood = deque([(0, 0, 0)])
voxels[(0, 0, 0)] = 2
while len(to_flood) > 0:
    current_voxel = to_flood.popleft()
    for delta in deltas:
        neighbour_voxel = tuple(np.clip(np.add(current_voxel, delta), 0, 21))
        if voxels[neighbour_voxel] == 0:
            voxels[neighbour_voxel] = 2
            to_flood.append(neighbour_voxel)
voxels[voxels == 0] = 1

print('Day 18, part 2:', count_open_sides(voxels))


## Day 19
import re

ORE = 0
CLAY = 1
OBSIDIAN = 2
GEODE = 3

def progress_state(resources, robots, recipe, time=1):
    factory_making = None

    # If we can make a geode robot, do so
    if all([r >= c for r, c in zip(resources, recipe[GEODE])]):
        factory_making = GEODE
        resources = [r - c for r, c in zip(resources, recipe[GEODE])]
    # If we need an obsidian robot, make one if we can
    elif all([r >= c for r, c in zip(resources, recipe[OBSIDIAN])]) and recipe[GEODE][OBSIDIAN] / (recipe[GEODE][ORE] + recipe[OBSIDIAN][ORE] + recipe[CLAY][ORE]) > robots[OBSIDIAN] / robots[ORE]:
        factory_making = OBSIDIAN
        resources = [r - c for r, c in zip(resources, recipe[OBSIDIAN])]
    # If we need a clay robot, make one if we can
    elif all([r >= c for r, c in zip(resources, recipe[CLAY])]) and recipe[OBSIDIAN][CLAY] / (recipe[OBSIDIAN][ORE] + recipe[CLAY][ORE]) > robots[CLAY] / robots[ORE]:
        factory_making = CLAY
        resources = [r - c for r, c in zip(resources, recipe[CLAY])]
    elif all([r >= c for r, c in zip(resources, recipe[ORE])]):
        # Make an ore robot
        factory_making = ORE
        resources = [r - c for r, c in zip(resources, recipe[ORE])]

    # Robots collect resources
    resources = [r + i for r, i in zip(resources, robots)]

    # Factory finishes producing the robot
    if factory_making is not None:
        robots = list(robots)
        robots[factory_making] += 1

    return resources, robots, recipe, time + 1

recipes = list()
with open('input_day19_test.txt') as f:
    for line in f:
        (ore_robot_ore_cost, clay_robot_ore_cost,
         obsidian_robot_ore_cost, obsidian_robot_clay_cost,
         geode_robot_ore_cost, geode_robot_obsidian_cost) = re.match(
             r'Blueprint \d+: Each ore robot costs (\d+) ore. '
             r'Each clay robot costs (\d+) ore. Each obsidian robot '
             r'costs (\d+) ore and (\d+) clay. Each geode robot costs '
             r'(\d+) ore and (\d+) obsidian.', line).groups()
        recipes.append([
            [int(ore_robot_ore_cost), 0, 0, 0],
            [int(clay_robot_ore_cost), 0, 0, 0],
            [int(obsidian_robot_ore_cost), int(obsidian_robot_clay_cost), 0, 0],
            [int(geode_robot_ore_cost), 0, int(geode_robot_obsidian_cost), 0],
        ])

resources = [0, 0, 0, 0]
robots = [1, 0, 0, 0]
recipe = recipes[0]
time = 0
for i in range(1, 25):
    resources, robots, recipe, time = progress_state(resources, robots, recipe, time)
    print(time, resources, robots)


## Day 19
import numpy as np
import re
from dataclasses import dataclass

recipes = list()
with open('input_day19.txt') as f:
    for line in f:
        (a, b, c, d, e, f) = re.match(
             r'Blueprint \d+: Each ore robot costs (\d+) ore. '
             r'Each clay robot costs (\d+) ore. Each obsidian robot '
             r'costs (\d+) ore and (\d+) clay. Each geode robot costs '
             r'(\d+) ore and (\d+) obsidian.', line).groups()
        recipes.append([
            [int(a), 0, 0, 0],
            [int(b), 0, 0, 0],
            [int(c), int(d), 0, 0],
            [int(e), 0, int(f), 0],
        ])
recipes = np.array(recipes)

@dataclass
class State:
    resources: np.ndarray
    robots: np.ndarray

    def score(self):
        return (self.robots @ np.array([1, 2, 500, 10000]) +
                self.resources @ np.array([1, 10, 50, 10000]))

    def __lt__(self, other):
        return self.score() < other.score()

    def __eq__(self, other):
        return self.score() == other.score()

# Branch out the possibilities based on the robot we decide to build.
def max_geodes(recipe, max_time):
    states_to_explore = [State(resources=np.array([0, 0, 0, 0]),
                               robots=np.array([1, 0, 0, 0]))]
    for time in range(max_time):
        next_states = list()
        for state in list(states_to_explore):
            new_resources = state.resources + state.robots
            for i, cost in enumerate(recipe):
                if np.all(state.resources >= cost):
                    new_robots = state.robots.copy()
                    new_robots[i] += 1
                    next_states.append(State(new_resources - cost, new_robots))
            next_states.append(State(new_resources, state.robots))
        states_to_explore = sorted(next_states)[-1000:]
    return max(states_to_explore, key=lambda state: state.resources[3]).resources[3]

print('Day 19, part 1:', sum([i * max_geodes(r, max_time=24) for i, r in enumerate(recipes, 1)]))
print('Day 19, part 2:', np.prod([max_geodes(r, max_time=32) for r in recipes[:3]]))


## Day 20
import numpy as np
numbers = np.loadtxt('input_day20.txt', dtype='int64')

def mix(numbers, n_times):
    # Instead of shuffling around the numbers, we will be shuffling around
    # these indices.
    indices = list(range(len(numbers)))
    for i in list(range(len(numbers))) * n_times:
        j = indices.index(i)
        indices.pop(j)
        indices.insert((j + numbers[i]) % len(indices), i)
    return indices

indices = mix(numbers, n_times=1)
zero_location = indices.index(np.flatnonzero(numbers == 0)[0])
print('Day 20, part 1:', sum(numbers[indices[(zero_location + x) % len(numbers)]] for x in [1000, 2000, 3000]))

numbers *= 811589153
indices = mix(numbers, n_times=10)
zero_location = indices.index(np.flatnonzero(numbers == 0)[0])
print('Day 20, part 2:', sum(numbers[indices[(zero_location + x) % len(numbers)]] for x in [1000, 2000, 3000]))


## Day 21
import operator
operators = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.floordiv}

# For each monkey, the operation they are performing
operations = dict()

# For each monkey not yelling a number, the left and right arguments for their
# operation
arguments = dict()

# For each monkey not yelling a number, whether the human is in their left (0) or
# right (1) argument.
human_in_argument = dict()

# Parse the input
with open('input_day21.txt') as f:
    for line in f:
        monkey, operation = line.strip().split(': ')
        try:
            number = int(operation)
            operations[monkey] = number
        except ValueError:
            left, symbol, right = operation.split()
            arguments[monkey] = (left, right)
            operations[monkey] = operators[symbol]

def resolve(monkey='root'):
    '''Compute what the given monkey is yelling.'''
    operation = operations[monkey]
    if type(operation) == int:
        return operation
    else:
        left, right = arguments[monkey]
        return operation(resolve(left), resolve(right))

def find_the_human(monkey='root'):
    '''Determine whether the human is needed for the given monkey to know what
    number to yell. Keeps track of this in the human_in_argument dict along the
    way.'''
    if monkey == 'humn':
        return True
    if monkey not in arguments:
        return False
    left, right = arguments[monkey]
    if find_the_human(left):
        human_in_argument[monkey] = 0
        return True
    elif find_the_human(right):
        human_in_argument[monkey] = 1
        return True
    else:
        return False

def human_needs_to_yell(monkey='root', target=None):
    '''Compute the value the human needs to yell. The `target` parameter is set
    to what the monkey one level up wants to hear.'''
    if monkey == 'humn':
        return target

    operation = operations[monkey]
    args = arguments[monkey]
    human_arg = human_in_argument[monkey]
    monkey_arg = 1 - human_arg

    # The trick here is to first resolve the argument for which there is no
    # human in the loop. Then you known enough to pass along the desired target
    # to argument that has the human in the loop.
    if operation is operator.eq:
        target = resolve(args[monkey_arg])
    elif operation is operator.add:
        target = target - resolve(args[monkey_arg])
    elif operation is operator.mul:
        target = target // resolve(args[monkey_arg])
    elif operation is operator.sub:
        if human_arg == 0:
            target = target + resolve(args[monkey_arg])
        else:
            target = resolve(args[monkey_arg]) - target
    elif operation is operator.floordiv:
        if human_arg == 0:
            target = target * resolve(args[monkey_arg])
        else:
            target = resolve(args[monkey_arg]) / target
    return human_needs_to_yell(args[human_arg], target)


print('Day 21, part 1:', resolve())

# Make the requested modifications
operations['root'] = operator.eq
del operations['humn']

# Play hide and seek
find_the_human()

print('Day 22, part 2:', human_needs_to_yell())
