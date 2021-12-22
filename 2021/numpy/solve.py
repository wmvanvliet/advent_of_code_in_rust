import numpy as np
import pandas as pd
import string
from collections import deque, Counter, defaultdict
from copy import deepcopy
from itertools import chain
from scipy.signal import convolve2d
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib.pyplot as plt
import heapq
import numba
from sklearn.linear_model import LinearRegression
from numpy.lib.stride_tricks import sliding_window_view


## Day 1
depths = np.loadtxt('day1_input.txt')
print('Day 1, Part 1:', np.sum(np.diff(depths) > 0))
print('Day 1, Part 2:', np.sum(np.diff(depths[3:] - depths[:-3]) > 0))


## Day 2
instructions = pd.read_csv('day2_input.txt', sep=' ', header=None, names=['instruction', 'amount'])
down, forward, up = instructions.groupby('instruction').sum().amount
print('Day 2, Part 1:', forward * (down - up))

instructions['aim'] = instructions.query('instruction == "down"').amount
instructions['aim'] = instructions['aim'].combine_first(-instructions.query('instruction == "up"').amount)
instructions['aim'] = instructions['aim'].fillna(0).cumsum()
instructions = instructions.query('instruction == "forward"')
forward = sum(instructions.amount)
depth = sum(instructions.amount * instructions.aim)
print('Day 2, Part 2:', int(forward * depth))


## Day 3
readings = np.loadtxt('day3_input.txt', encoding='utf8', converters={0: list}, dtype='int')
n_readings, n_bits = readings.shape

def bits_to_int(x):
    """Convert a list of 1's and 0's to an integer number"""
    return (x << np.arange(len(x))[::-1]).sum()

gamma = readings.sum(axis=0) >= len(readings) / 2
epsilon = ~gamma
print('Day 3, Part 1:', bits_to_int(gamma) * bits_to_int(epsilon))

oxygen = readings.copy()
for i in range(n_bits):
    one_is_most_common = oxygen[:, i].sum() >= len(oxygen) / 2
    oxygen = oxygen[oxygen[:, i] == one_is_most_common]
    if len(oxygen) == 1:
        break

co2 = readings.copy()
for i in range(n_bits):
    one_is_least_common = co2[:, i].sum() < len(co2) / 2
    co2 = co2[co2[:, i] == one_is_least_common]
    if len(co2) == 1:
        break

print('Day 3, Part 2:', bits_to_int(oxygen.flat) * bits_to_int(co2.flat))


## Day 4
bingo_numbers = np.loadtxt('day4_input.txt', delimiter=',', max_rows=1, dtype='int')
charts = np.loadtxt('day4_input.txt', skiprows=2, dtype='int').reshape(-1, 5, 5)
marks = np.zeros_like(charts)
someone_already_won = False
for num in bingo_numbers:
    marks[charts == num] = 1
    winning_chart_ind = np.append(np.where(marks.sum(axis=1) == 5)[0],
                                  np.where(marks.sum(axis=2) == 5)[0])
    if len(winning_chart_ind) > 0:
        winning_chart = charts[winning_chart_ind[0]]
        winning_marks = marks[winning_chart_ind[0]]
        if not someone_already_won:
            print('Day 4, Part 1:', winning_chart[winning_marks == 0].sum() * num)
            someone_already_won = True
        charts = np.delete(charts, winning_chart_ind, axis=0)
        marks = np.delete(marks, winning_chart_ind, axis=0)
    if len(charts) == 0:
        print('Day 4, Part 2:', winning_chart[winning_marks == 0].sum() * num)
        break


## Day 5
def split_on_comma(x):
    return x.split(',')

lines = np.loadtxt('day5_input.txt', delimiter=' -> ', converters={0: split_on_comma, 1: split_on_comma}, encoding='utf8', dtype='int')
width, height = lines.max(axis=(0, 1)) + 1
field_part1 = np.zeros((height, width), dtype='int')
field_part2 = np.zeros_like(field_part1)

for [x1, y1], [x2, y2] in lines:
    xmin, xmax = min(x1, x2), max(x1, x2) + 1
    ymin, ymax = min(y1, y2), max(y1, y2) + 1
    xdir = 1 if x1 < x2 else -1
    ydir = 1 if y1 < y2 else -1

    if x1 == x2 or y1 == y2:  # horizontal or vertical line
        field_part1[ymin:ymax, xmin:xmax] += 1
        field_part2[ymin:ymax, xmin:xmax] += 1
    else:  # Diagonal line
        field_part2[(np.arange(ymin, ymax)[::ydir], np.arange(xmin, xmax)[::xdir])] += 1

print('Day 5, part 1:', np.sum(field_part1 >= 2))
print('Day 5, part 12', np.sum(field_part2 >= 2))


## Day 6
ages = [0] * 9
with open('day6_input.txt') as f:
    for age in f.read().split(','):
        ages[int(age)] += 1
for day in range(1, 257):
    n_births = ages.pop(0)
    ages.append(n_births)
    ages[6] += n_births
    if day == 80:
        print('Day 6, part 1:', sum(ages))
print('Day 6, part 2:', sum(ages))


## Day 7
pos = np.loadtxt('day7_input.txt', delimiter=',')
print('Day 7, part 1:', int(np.sum(np.abs(pos - np.median(pos)))))

def cost(to_pos):
    """Compute the total cost of moving all the crabs to the given pos."""
    distance = np.abs(pos - to_pos)
    individual_cost = (distance + 1) * (distance / 2)  # = sum(1..distance)
    return int(np.sum(individual_cost))

print('Day 7, part 2:', min(cost(np.floor(pos.mean())), cost(np.ceil(pos.mean()))))


## Day 8
digits = [
    'abcefg',  # 0
    'cf',      # 1
    'acdeg',   # 2
    'acdfg',   # 3
    'bcdf',    # 4
    'abdfg',   # 5
    'abdefg',  # 6
    'acf',     # 7
    'abcdefg', # 8
    'abcdfg',  # 9
]
seg_freqs = Counter(chain(*digits))
freq_to_digit = {sum([seg_freqs[seg] for seg in digit]): i for i, digit in enumerate(digits)}

part1_ans = 0
part2_ans = 0
for line in open('day8_input.txt'):
    inp, out = line.strip().split(' | ')
    inp = inp.split()
    out = out.split()
    inp_seg_freqs = Counter(chain(*inp))
    out_freqs = []
    for out_digit in out:
        if len(out_digit) in [2, 3, 4, 7]:
            part1_ans += 1
        out_freqs.append(sum([inp_seg_freqs[seg] for seg in out_digit]))
    translated_out_digits = [freq_to_digit[freq] for freq in out_freqs]
    part2_ans += int(''.join([str(digit) for digit in translated_out_digits]))
print('Day 8, part 1:', part1_ans)
print('Day 8, part 2:', part2_ans)


## Day 8 long solution
numbers = [
    set('abcefg'),  # 0
    set('cf'),      # 1
    set('acdeg'),   # 2
    set('acdfg'),   # 3
    set('bcdf'),    # 4
    set('abdfg'),   # 5
    set('abdefg'),  # 6
    set('acf'),     # 7
    set('abcdefg'), # 8
    set('abcdfg'),  # 9
]

# Keep track of possible connections
possibilities = {'a': set('abcdefg'),
                 'b': set('abcdefg'),
                 'c': set('abcdefg'),
                 'd': set('abcdefg'),
                 'e': set('abcdefg'),
                 'f': set('abcdefg'),
                 'g': set('abcdefg')}

def decode(possibilities, inp, numbers):
    """Decode a list of patterns into a segment<->segment translation table."""
    if len(inp) == 0 or len(numbers) == 0:
        # We're done decoding
        return possibilities

    # Decode first pattern
    pattern = inp[0]
    possible_numbers = [num for num in numbers if len(pattern) == len(num)]
    if len(possible_numbers) == 0:
        return None  # Decoding failed

    for number in possible_numbers:
        new_possibilities = deepcopy(possibilities)
        for segment in 'abcdefg':
            # Update possible connections
            if segment in number:
                new_possibilities[segment] &= pattern
            else:
                new_possibilities[segment] -= pattern

        # Check if any segment was no possible connections
        if any([len(v) == 0 for v in new_possibilities.values()]):
            continue  # Decoding failed, try next possible number

        # Try decoding the rest of the numbers (recursive call)
        new_possibilities = decode(new_possibilities, inp[1:], [n for n in numbers if n != number])
        if new_possibilities is not None:
            # Decoding succeeded!
            return new_possibilities
        # Decoding failed. Try next possible number.

    # If we reach this point, there were no possible numbers that could be matched to the pattern.
    return None

part2_ans = 0
for line in open('day8_input.txt'):
    inp, out = line.strip().split(' | ')
    inp = [set(p) for p in inp.split()]
    # Try unique number lengths first
    inp.sort(key=lambda x: len(x) not in {2, 3, 4, 7})
    translation_table = decode(possibilities, inp, numbers)
    translation_table = {v.pop(): k for k, v in translation_table.items()}  # Flip keys and values
    num = ''
    for digit in out.split():
        segments = set([translation_table[segment] for segment in digit])
        num += str([i for i, num in enumerate(numbers) if num == segments][0])
    part2_ans += int(num)
print('Day 8, part 2:', part2_ans)


## Day 9
heightmap = np.loadtxt('day9_input.txt', encoding='utf8', converters={0: list}, dtype='int')
is_lowpoint = np.ones_like(heightmap, dtype='bool')
is_lowpoint[1:] &= (heightmap[:-1] - heightmap[1:]) > 0
is_lowpoint[:-1] &= (heightmap[1:] - heightmap[:-1]) > 0
is_lowpoint[:, 1:] &= (heightmap[:, :-1] - heightmap[:, 1:]) > 0
is_lowpoint[:, :-1] &= (heightmap[:, 1:] - heightmap[:, :-1]) > 0
print('Day 9, part 1:', np.sum(1 + heightmap[is_lowpoint]))

map_height, map_width = heightmap.shape
seeds = list(zip(*np.where(is_lowpoint)))
basins = list()

for seed in seeds:
    basin = set()
    to_check = deque([seed])
    i = 0
    while len(to_check) > 0 and i < 100000:
        point = to_check.popleft()
        if heightmap[point] == 9:
            continue

        basin.add(point)
        row, col = point
        neighbours = set()
        if row > 0:
            neighbours.add((row - 1, col))
        if row < map_height - 1:
            neighbours.add((row + 1, col))
        if col > 0:
            neighbours.add((row, col - 1))
        if col < map_width - 1:
            neighbours.add((row, col + 1))
        to_check.extend(neighbours - basin)
        i += 1
    basins.append(basin)
print('Day 9, part2:', np.product(sorted([len(b) for b in basins])[-3:]))


## Day 10
parens = {'(': ')', '[': ']', '{': '}', '<': '>'}
class Corrupt(Exception):
    @property
    def score(self):
        scores = {')': 3, ']': 57, '}': 1197, '>': 25137}
        return scores[self.args[0]] 

class Incomplete(Exception):
    @property
    def score(self):
        autocomplete_score = 0
        scores = {'(': 1, '[': 2, '{': 3, '<': 4}
        for char in reversed(self.args[0]):
            autocomplete_score *= 5
            autocomplete_score += scores[char]
        return autocomplete_score

part1_ans = 0
part2_scores = []
for line in open('day10_input.txt'):
    stack = list()
    try:
        for char in list(line.strip()):
            if char in parens.keys():
                stack.append(char)
            elif char != parens[stack.pop()]:
                raise Corrupt(char)
        if len(stack) > 0:
            raise Incomplete(stack)
    except Corrupt as e:
        part1_ans += e.score
    except Incomplete as e:
        part2_scores.append(e.score)
    
print('Day 10, part1:', part1_ans)
print('Day 10, part2:', int(np.median(part2_scores)))


## Day 11
octopuses = np.loadtxt('day11_input.txt', encoding='utf8', converters={0: list}, dtype='int')
radius = np.ones((3, 3))
radius[1, 1] = 0
total_num_flashes = 0
step = 1
while True:
    flashes = np.zeros_like(octopuses, dtype='bool')
    prev_num_flashes = 0
    after_flash = octopuses + 1
    while True:
        flashes = np.logical_or(flashes, after_flash > 9)
        affected = convolve2d(flashes, radius, mode='same').astype(int)
        after_flash = octopuses + 1 + affected
        num_flashes = flashes.sum()
        if num_flashes == prev_num_flashes:
            break
        prev_num_flashes = num_flashes
    after_flash[flashes] = 0
    octopuses = after_flash
    total_num_flashes += num_flashes
    if step == 100:
        print('Day 11, part1:', total_num_flashes)
    if np.all(flashes):
        print('Day 11, part2:', step)
        break
    step += 1


## Day 12
connections = defaultdict(list)  # cave -> [connecting caves]
with open('day12_input.txt') as f:
    for line in f:
        start, end = line.strip().split('-')
        connections[start].append(end)
        connections[end].append(start)

def num_paths(start_cave, visited, twice):
    if start_cave == 'end':
        return 1

    if start_cave == start_cave.lower():
        visited.add(start_cave)

    n = 0
    for cave in connections[start_cave]:
        if cave not in visited:
            n += num_paths(cave, visited=set(visited), twice=twice)
        elif twice is None and cave not in ['start', 'end']:
            n += num_paths(cave, visited=set(visited), twice=cave)
    return n

print('Day 12, part 1:', num_paths('start', visited=set(), twice='disabled'))
print('Day 12, part 2:', num_paths('start', visited=set(), twice=None))


## Day 13
dots = np.loadtxt('day13_input.txt', delimiter=',', comments='fold along', dtype='int')
step = 1
with open('day13_input.txt') as f:
    for line in f:
        if not line.startswith('fold along'):
            continue
        direction, position = line.split('=')
        direction, position = int(direction[-1] == 'y'), int(position)
        fold = dots[:, direction] > position 
        dots[fold, direction] = position + (position - dots[fold, direction])
        if step == 1:
            print('Day 13, part 1:', len(np.unique(dots, axis=0)))
        step += 1
print('Day 13, part 2:')
grid = np.zeros(np.ptp(dots, axis=0) + 1).T
grid[tuple((dots - dots.min(axis=0)).T[[1, 0]])] = True
for row in grid:
    print(''.join(['#' if col else ' ' for col in row]))


## Day 14
with open('day14_input.txt') as f:
    template = f.readline().strip()
    pairs = Counter(zip(template[:-1], template[1:]))
    f.readline()
    rules = dict()
    for line in f:
        pair, between = line.strip().split(' -> ')
        rules[tuple(pair)] = ((pair[0], between), (between, pair[1]))

def element_count_difference(pairs):
    counts = defaultdict(int)
    for (element, _), count in pairs.items():
        counts[element] += count
    counts[template[-1]] += 1
    counts = sorted(counts.values())
    return counts[-1] - counts[0]

for step in range(1, 41):
    next_pairs = defaultdict(int)
    for pair, count in pairs.items():
        for new_pair in rules.get(pair, tuple()):
            next_pairs[new_pair] += count
    pairs = next_pairs
    if step == 10:
        print('Day 14, part 1:', element_count_difference(pairs))
part2_ans = element_count_difference(pairs)
print('Day 14, part 2:', element_count_difference(pairs))


## Day 15
risk_level1 = np.loadtxt('day15_input.txt', encoding='utf', converters={0: list}, dtype='int')

a = np.array([
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
]).repeat(risk_level1.shape[0], axis=0).repeat(risk_level1.shape[1], axis=1)
risk_level2 = np.tile(risk_level1, (5, 5)) + a
risk_level2[risk_level2 > 9] -= 9

def solve_grid(risk_level):
    n_rows, n_cols = risk_level.shape
    start = (0, 0)
    target = (n_rows - 1, n_cols - 1)
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    visited = dict()
    to_visit = [(0, start)]

    def in_bounds(node):
        return node[0] >= 0 and node[0] < n_rows and node[1] >= 0 and node[1] < n_cols

    for i in range(n_rows * n_cols):
        cost, node = heapq.heappop(to_visit)
        if node == target:
            return cost  # Done!
        visited[node] = cost

        for neighbor in [(node[0] + dy, node[1] + dx) for (dy, dx) in directions]:
            neighbor = tuple(neighbor)
            if not in_bounds(neighbor):
                continue
            neighbor_cost = cost + risk_level[neighbor]
            if neighbor not in visited or visited[neighbor] > neighbor_cost:
                visited[neighbor] = neighbor_cost
                heapq.heappush(to_visit, (neighbor_cost, neighbor))
    else:
        print('No solution found yet.')
        return -1

print('Day 15, part 1:', solve_grid(risk_level1))
print('Day 15, part 2:', solve_grid(risk_level2))


## Day 16
from dataclasses import dataclass, field
@dataclass
class ValuePacket:
    version: int
    typ: int
    value: int

@dataclass
class OperatorPacket:
    version: int
    typ: int
    length_type_id: int
    length: int

def parse_packets(bits):
    """Parses a stream of bits into a stream of packets."""
    offset = 0
    while offset < (len(bits) - 8):
        version = int(bits[offset : offset + 3], 2)
        offset += 3

        typ = int(bits[offset : offset + 3], 2)
        offset += 3

        if typ == 4:
            value = ''
            next_group = True
            while next_group:
                next_group = int(bits[offset], 2)
                offset += 1
                value += bits[offset : offset + 4]
                offset += 4
            value = int(value, 2)
            yield ValuePacket(version, typ, value), offset
        else:
            length_type_id = int(bits[offset : offset + 1], 2)
            offset += 1
            if length_type_id == 0:
                length = int(bits[offset : offset + 15], 2)
                offset += 15
            else:
                length = int(bits[offset : offset + 11], 2)
                offset += 11
            yield OperatorPacket(version, typ, length_type_id, length), offset

def evaluate_next(packets):
    """Evaluate the next packet including possible subpackets."""
    packet, offset = next(packets)
    if isinstance(packet, ValuePacket):
        return packet.value, offset
    else:  # OperatorPacket
        subpacket_values = []
        if packet.length_type_id == 0:
            start_offset = offset
            while (offset - start_offset) < packet.length:
                subpacket_value, offset = evaluate_next(packets)
                subpacket_values.append(subpacket_value)
        else:
            for i in range(packet.length):
                subpacket_value, offset = evaluate_next(packets)
                subpacket_values.append(subpacket_value)
        if packet.typ == 0:
            value = sum(subpacket_values)
        elif packet.typ == 1:
            value = subpacket_values[0]
            for subpacket_value in subpacket_values[1:]:
                value *= subpacket_value
        elif packet.typ == 2:
            value = min(subpacket_values)
        elif packet.typ == 3:
            value = max(subpacket_values)
        elif packet.typ == 5:
            assert len(subpacket_values) == 2
            value = int(subpacket_values[0] > subpacket_values[1])
        elif packet.typ == 6:
            assert len(subpacket_values) == 2
            value = int(subpacket_values[0] < subpacket_values[1])
        elif packet.typ == 7:
            assert len(subpacket_values) == 2
            value = int(subpacket_values[0] == subpacket_values[1])
        else:
            raise ValueError(f'Invalid packet type {packet}')
        return value, offset

part1_ans = 0
with open('day16_input.txt') as f:
    hex_input = f.read()
bits = ''.join([f'{byte:08b}' for byte in bytes.fromhex(hex_input)])
packets = parse_packets(bits)
print('Day 16, part 1:', sum([packet.version for packet, _ in packets]))
print('Day 16, part 2:', evaluate_next(parse_packets(bits))[0])


## Day 17
with open('day17_input.txt') as f:
    x, y = f.readline().split(', ')
    target_x = x.split('=')[1].split('..')
    target_y = y.split('=')[1].split('..')
target_x = int(target_x[0]), int(target_x[1])
target_y = int(target_y[0]), int(target_y[1])

def pos_at_zero_vel(init_vel):
    """Compute the position at which the velocity reaches zero."""
    return (init_vel ** 2 + init_vel) / 2

def init_vel_x(target_x):
    """Compute the required initial x-velocity in order to hit the target x-position."""
    return (np.sqrt(1 + 8 * target_x) - 1) / 2

def hits_target(vel_x, vel_y):
    t = 0
    x, y = 0, 0
    while x <= target_x[1] and y >= target_y[0]:
        t += 1
        x += vel_x
        y += vel_y
        vel_x = max(vel_x - 1, 0)
        vel_y -= 1
        #print(x, y)
        if target_x[0] <= x <= target_x[1] and target_y[0] <= y <= target_y[1]:
            return True
    return False

min_vel_y = target_y[0]
max_vel_y = -target_y[0] - 1
min_vel_x = int(init_vel_x(target_x[0]))
max_vel_x = target_x[1]

part2_ans = 0
for vel_y in range(min_vel_y, max_vel_y + 1):
    for vel_x in range(min_vel_x, max_vel_x + 1):
        if hits_target(vel_x, vel_y):
            part2_ans += 1

print('Day 17, part 1:', int(pos_at_zero_vel(max_vel_y)))
print('Day 17, part 2:', part2_ans)  # 4748


## Day 18
def tokenize(s):
    tokens = list()
    num = ''
    for ch in s.strip():
        if ch in string.digits:
            tokens.append(int(ch))
        elif ch != ',':
            tokens.append(ch)
    return tokens

def explode(number):
    depth = 0
    first_num_to_the_left = None
    for i, ch in enumerate(number):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        elif type(ch) == int:
            first_num_to_the_left = (i, ch)

        if depth == 5:
            left, right = number[i + 1], number[i + 2]
            if first_num_to_the_left is not None:
                res = number[:first_num_to_the_left[0]]
                res.append(first_num_to_the_left[1] + left)
                res += number[first_num_to_the_left[0] + 1:i]
            else:
                res = number[:i]

            res.append(0)

            for j in range(i + 3, len(number)):
                if type(number[j]) == int:
                    first_num_to_the_right = (j, number[j])
                    break
            else:
                first_num_to_the_right = None

            if first_num_to_the_right is not None:
                res += number[i + 4:first_num_to_the_right[0]]
                res.append(first_num_to_the_right[1] + right)
                res += number[first_num_to_the_right[0] + 1:]
            else:
                res += number[i + 4:]

            return res
    return number

def split(number):
    for i, n in enumerate(number):
        if type(n) == int and n > 9:
            left = int(np.floor(n / 2))
            right = int(np.ceil(n / 2))
            return number[:i] + ['[', left, right, ']'] + number[i + 1:]
    return number

def add(a, b):
    return reduce(['['] + a + b + [']'])

def reduce(number):
    while True:
        after_explode = explode(number)
        if after_explode != number:
            number = after_explode
            continue
        after_split = split(number)
        if after_split != number:
            number = after_split
            continue
        return number

def magnitude(number):
    stack = list()
    for ch in number:
        if type(ch) == int:
            stack.append(ch)
        elif ch == ']':
            right = stack.pop()
            left = stack.pop()
            stack.append(3 * left + 2 * right)
    return stack.pop()

with open('day18_input.txt') as f:
     numbers = [tokenize(line) for line in f.readlines()]
num = add(numbers[0], numbers[1])
for number in numbers[2:]:
    num = add(num, number)
print('Day 18, part 1:', magnitude(num))

mags = []
for i, a in enumerate(numbers):
    for j, b in enumerate(numbers):
        if i == j:
            continue
        mags.append(magnitude(add(a, b)))
print('Day 18, part 2:', max(mags))


## Day 19
scanners = []
with open('day19_input.txt') as f:
    for line in f:
        if line.startswith('--- scanner '):
            scanner = []
        elif line == '\n':
            scanners.append(np.array(scanner))
        else:
            scanner.append([int(i) for i in line.split(',')])
    scanners.append(np.array(scanner))

def find_scanner_transform(a, b):
    """Find the transform from scanner b relative to scanner a."""
    d_a = distance.squareform(distance.pdist(a, metric='cityblock'))
    d_a = [set(d) for d in d_a]
    d_b = distance.squareform(distance.pdist(b, metric='cityblock'))
    d_b = [set(d) for d in d_b]

    # Match beacons by distance pattern
    scores = np.array([[len(a) - len(d1 - d2) for d2 in d_b] for d1 in d_a])
    i_a, i_b = np.where(scores >= 12)
    if len(i_a) < 12 or len(i_b) < 12:
        return None  # Not enough beacons match

    # Determine transform of scanner b relative to scanner a
    return LinearRegression().fit(b[i_b], a[i_a])

transformed_scanners = {0: scanners[0]}
transforms = dict()
while len(transformed_scanners) < len(scanners):
    for i, scanner in enumerate(scanners):
        if i in transformed_scanners:
            continue
        for j, beacons in transformed_scanners.items():
            m = find_scanner_transform(beacons, scanner)
            if m is None:
                continue
            transforms[i] = m
            transformed_scanners[i] = m.predict(scanner).round(0).astype('int')
            break
all_beacons = np.unique(np.vstack(list(transformed_scanners.values())), axis=0)

scanner_positions = np.array([m.intercept_ for m in transforms.values()]).round(0).astype('int')
scanner_positions = np.vstack((scanner_positions, [[0, 0, 0]]))

print('Day 19, part 1:', len(all_beacons))
print('Day 19, part 2:', int(distance.pdist(scanner_positions, metric='cityblock').max()))


## Day 20
with open('day20_input.txt') as f:
    algorithm, _, *img = f.readlines()
    algorithm = np.array([int(ch == '#') for ch in algorithm.strip()])
    img = np.array([[int(ch == '#') for ch in row.strip()] for row in img])

bin2dec = (2**np.arange(9)).reshape(3, 3)

def enhance(img, n):
    img = np.pad(img, 2 * n)
    for _ in range(n):
        img = algorithm[convolve2d(img, bin2dec, mode='valid')]
    return img

print('Day 20, part 1:', enhance(img, 2).sum())
print('Day 20, part 2:', enhance(img, 50).sum())


## Day 21
with open('day21_input.txt') as f:
    player1_pos = int(f.readline().split(': ')[1])
    player2_pos = int(f.readline().split(': ')[1])
pos = [player1_pos, player2_pos]
score = [0, 0]

def die_gen():
    while True:
        for i in range(1, 101):
            yield i

die = die_gen()

n_rolls = 0
turn = 0
while score[0] < 1000 and score[1] < 1000:
    pos[turn] = ((pos[turn] - 1 + next(die) + next(die) + next(die)) % 10) + 1
    score[turn] += pos[turn]
    turn = int(not turn)
    n_rolls += 3

print('Day 21, part 1:', min(score) * n_rolls)

unfinished_games = {
    # pos1, score1, pos2, score2 -> copies of this game
    ((pos[0] - 1, pos[1] - 1), (0, 0)): 1,
}
n_wins = [0, 0]
turn = 0

while len(unfinished_games) > 0:
    still_unfinished = dict()
    for (pos, score), n in unfinished_games.items():
        for die1 in [1, 2, 3]:
            for die2 in [1, 2, 3]:
                for die3 in [1, 2, 3]:
                    new_pos = (pos[turn] + die1 + die2 + die3) % 10
                    new_score = score[turn] + new_pos + 1
                    if new_score >= 21:
                        # Game finished
                        n_wins[turn] += n
                        continue # No longer track these games
                    else:
                        # Game remains unfinished.
                        if turn == 0:
                            new_state = ((new_pos, pos[1]), (new_score, score[1]))
                        else:
                            new_state = ((pos[0], new_pos), (score[0], new_score))
                        still_unfinished[new_state] = still_unfinished.get(new_state, 0) + n
    unfinished_games = still_unfinished
    turn = int(not turn)
print('Day 21, part 2:', max(n_wins))


## Day 22
def x_overlaps(r1, r2):
    return (r1[0] >= r2[0] or r1[1] >= r2[0]) and (r1[0] <= r2[0] or r1[1] <= r2[1])

def y_overlaps(r1, r2):
    return (r1[2] >= r2[2] or r1[3] >= r2[2]) and (r1[2] <= r2[3] or r1[3] <= r2[3])

def z_overlaps(r1, r2):
    return (r1[4] >= r2[4] or r1[5] >= r2[4]) and (r1[4] <= r2[5] or r1[5] <= r2[5])

def independent(r1, r2):
    return not (x_overlaps(r1, r2) and y_overlaps(r1, r2) and z_overlaps(r1, r2))

def encompasses(r1, r2):
    return (r1[0] <= r2[0] and r1[1] >= r2[1] and
            r1[2] <= r2[2] and r1[3] >= r2[3] and
            r1[4] <= r2[4] and r1[5] >= r2[5])

from itertools import combinations
from pprint import pprint

def remove_range(ranges, r):
    step = 0
    print('Step:', step)
    #pprint(ranges)

    while True:
        step += 1
        for i, r2 in enumerate(ranges):
            if independent(r, r2):
                continue
            if encompasses(r, r2):
                print('Removing range', i)
                del ranges[i]
                break
        else:
            print('Done!')
            break
    return ranges

def normalize_ranges(ranges):
    step = 0
    print('Step:', step)
    pprint(ranges)

    while True:
        step += 1
        print('\nStep:', step)
        for i_r1, i_r2 in combinations(range(len(ranges)), 2):
            r1, r2 = ranges[i_r1], ranges[i_r2]
            if independent(r1, r2):
                continue
            elif encompasses(r1, r2):
                print('Merge', i_r2, 'into', i_r1)
                del ranges[i_r2]
                break
            elif encompasses(r2, r1):
                print('Merge', i_r1, 'into', i_r2)
                del ranges[i_r1]
                break
            else:
                print('Splitting', i_r1, 'and', i_r2)
                del ranges[i_r1]
                del ranges[i_r2 - 1]
                xs = np.unique(r1[:2] + r2[:2])
                ys = np.unique(r1[2:4] + r2[2:4])
                zs = np.unique(r1[4:] + r2[4:])
                print('xs', xs)
                print('ys', ys)
                print('zs', zs)
                if len(xs) == 1:
                    xs = [xs[0], xs[0]]
                if len(ys) == 1:
                    ys = [ys[0], ys[0]]
                if len(zs) == 1:
                    zs = [zs[0], zs[0]]
                for x1, x2 in list(zip(xs[:-2], xs[1:-1])) + [(xs[-2], xs[-1] + 1)]:
                    for y1, y2 in list(zip(ys[:-2], ys[1:-1])) + [(ys[-2], ys[-1] + 1)]:
                        for z1, z2 in list(zip(zs[:-2], zs[1:-1])) + [(zs[-2], zs[-1] + 1)]:
                            ranges.append([x1, x2 - 1, y1, y2 - 1, z1, z2 - 1])
                break
        else:
            print('Done!')
            break
        pprint(ranges)
    return ranges
                
import re
day22_input_pat = re.compile(r'^(.+) x=(-?\d+)\.\.(-?\d+),y=(-?\d+)\.\.(-?\d+),z=(-?\d+)\.\.(-?\d+)$')
ranges = []
cubes = np.zeros((101, 101, 101), 'bool')
with open('day22_test.txt') as f:
    for i, line in enumerate(f):
        if i == 2:
            break
        toggle, *r = day22_input_pat.match(line).groups()
        r = [int(x) for x in r]
        if not all([-50 <= x <= 50 for x in r]):
            print('discarding', r)
            continue
        if toggle == 'on':
            cubes[r[0] + 50:r[1] + 51, r[2] + 50:r[3] + 51, r[4] + 50:r[5] + 51] = True
            ranges.append(r)
            ranges = normalize_ranges(ranges)
        elif toggle == 'off':
            print('Stopping for now.')
            break
            #remove_range(ranges, r)
            #ranges = normalize_ranges(ranges)
            #cubes[r[0] + 50:r[1] + 51, r[2] + 50:r[3] + 51, r[4] + 50:r[5] + 51] = False
print('Day 22, part 1:', cubes.sum())

cubes2 = np.zeros((101, 101, 101), 'bool')
for r in ranges:
    cubes2[r[0] + 50:r[1] + 51, r[2] + 50:r[3] + 51, r[4] + 50:r[5] + 51] = True
