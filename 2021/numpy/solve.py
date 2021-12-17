import numpy as np
import pandas as pd
import string
from collections import deque, Counter, defaultdict
from copy import deepcopy
from itertools import chain
from scipy.signal import convolve2d
from tqdm import tqdm
import matplotlib.pyplot as plt
import heapq
import numba

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
with open('day14_test.txt') as f:
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
