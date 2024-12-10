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


# Parse the grid.
rows = []
cols = []
for i, line in enumerate(open("day6.txt")):
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


print("Day 6, part 1:", len(visited))
print("Day 6, part 2:", len(barrier_possibilities))


def is_solvable(answer, terms, allow_concat=False):
    if len(terms) == 1:
        return terms[0] == answer
    elif terms[0] >= answer:
        return False
    else:  # len(terms) >= 2
        *first_terms, final_term = terms

        if allow_concat:
            # Check if the final digits of the answer match the final term
            final_term_str = str(final_term)
            answer_str = str(answer)
            if (
                len(answer_str) > len(final_term_str)
                and answer_str[-len(final_term_str) :] == final_term_str
            ):
                # Concatenation is an option for the solution. Check the rest.
                if is_solvable(
                    int(answer_str[: -len(final_term_str)]), first_terms, allow_concat
                ):
                    return True

        # Multiplication is only an option if the answer is dividable by the final term.
        if answer % final_term == 0:
            if is_solvable(answer // final_term, first_terms, allow_concat):
                return True

        # Addition is only an option if the answer is more than the final term.
        if answer > final_term:
            if is_solvable(answer - final_term, first_terms, allow_concat):
                return True

        # No options remain.
        return False


equations = list()
with open("day7.txt") as f:
    for line in f:
        answer, terms = line.split(": ", 1)
        answer = int(answer)
        terms = [int(t) for t in terms.split(" ")]
        equations.append((answer, terms))

part1 = sum(answer for answer, terms in equations if is_solvable(answer, terms, False))
part2 = sum(answer for answer, terms in equations if is_solvable(answer, terms, True))
print("Day 7, part 1:", part1)
print("Day 7, part 2:", part2)


antennae = dict()
with open("day8.txt") as f:
    for y, line in enumerate(f):
        for x, freq in enumerate(line.strip()):
            if freq != ".":
                freq_antennae = antennae.get(freq, set())
                freq_antennae.add((y, x))
                antennae[freq] = freq_antennae
height = y + 1
width = len(line.strip())
print(height, width)

antinodes = set()
for freq in antennae.keys():
    for a1 in antennae[freq]:
        for a2 in antennae[freq]:
            if a1 == a2:
                continue
            dy = a1[0] - a2[0]
            dx = a1[1] - a2[1]
            antinode = (a1[0], a1[1])
            while 0 <= antinode[0] < height and 0 <= antinode[1] < width:
                antinodes.add(antinode)
                antinode = (antinode[0] + dy, antinode[1] + dx)

print("Day 8, part 1:", len(antinodes))

##
import numpy as np

disk_map = np.fromregex("day9.txt", r"(\d)", dtype=[("num", "int")])["num"].tolist()
chunks, empty = disk_map[::2], disk_map[1::2]
chunks = [(i, size) for i, size in enumerate(chunks)]

insert_point = 1
space_available = empty.pop(0)
to_place_id, to_place_size = chunks.pop(-1)
while insert_point < len(chunks):
    # print(files)
    # print(insert_point)
    space_to_take = min(space_available, to_place_size)
    chunks.insert(insert_point, (to_place_id, space_to_take))
    if to_place_size > 0:
        insert_point += 1
    to_place_size -= space_to_take
    if to_place_size == 0:
        to_place_id, to_place_size = chunks.pop(-1)
    space_available -= space_to_take
    if space_available == 0:
        space_available = empty.pop(0)
        insert_point += 1
chunks.append((to_place_id, to_place_size))

offsets = [0] + np.cumsum([chunk_size for _, chunk_size in chunks[:-1]]).tolist()

checksum = 0
for (chunk_id, chunk_size), offset in zip(chunks, offsets):
    checksum += chunk_id * (sum(range(chunk_size)) + chunk_size * offset)
print(checksum)

##
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional


@dataclass
class Chunk:
    size: int
    pos: int
    prev: Optional["Empty"]
    next: Optional["Empty"]

    def insert_after(self, chunk):
        chunk.next = self.next
        chunk.prev = self
        chunk.pos = self.pos + 1
        if self.next:
            self.next.prev = chunk
            self.next.pos = chunk.pos + 1
        self.next = chunk

    def detach(self):
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        self.prev = None
        self.next = None
        self.pos = -1

    def print_chain(self):
        n = self
        while n is not None:
            print(f"{n} ", end="")
            n = n.next
        print()

    def __iter__(self):
        n = self
        while n:
            yield n
            n = n.next


@dataclass
class File(Chunk):
    id: int

    def __repr__(self):
        return f"({self.pos}, {self.id}, {self.size})"


@dataclass
class Empty(Chunk):
    def __repr__(self):
        return f"({self.pos}, _, {self.size})"


disk_map = np.fromregex("day9_test.txt", r"(\d)", dtype=[("num", "int")])["num"].tolist()
files, empties = disk_map[::2], disk_map[1::2]
files = [(i, size) for i, size in enumerate(files)]
chunks = File(id=files[0][0], size=files[0][1], pos=0, prev=None, next=None)
last_chunk = chunks
for empty_size, (file_id, file_size) in zip(empties, files[1:]):
    if empty_size > 0:
        e = Empty(size=empty_size, pos=-1, prev=None, next=None)
        last_chunk.insert_after(e)
        last_chunk = e
    c = File(id=file_id, size=file_size, pos=-1, prev=None, next=None)
    last_chunk.insert_after(c)
    last_chunk = c

files_by_size = defaultdict(list)
for chunk in iter(chunks):
    if isinstance(chunk, File):
        files_by_size[chunk.size].insert(0, chunk)

chunks.print_chain()
# print(files_by_size)
for chunk in chunks:
    assert chunk.pos > -1

done = False
empty = chunks
last_file = last_chunk
while True:
    # Find the next empty chunk.
    while not isinstance(empty, Empty):
        empty = empty.next
        if empty is None:
            print("No more empties")
            done = True
            break
        assert empty.pos > -1
    if done:
        break

    # Grab the last file
    while not isinstance(last_file, File):
        if last_file.prev is None:
            print("No more last files")
            done = True
            break
        last_file = last_file.prev
        if last_file.next == empty:
            print(">", empty, last_file)
            empty.detach()
            done = True
            break
        if last_file.size > empty.size:
            fast_file = last_file.prev
        last_file.next.detach()
        assert last_file.pos > -1
    if done:
        break

    # Grab the next file that will fit in the empty space


    file, last_file = last_file, last_file.prev
    assert file.pos > -1
    assert last_file.pos > -1

    chunks.print_chain()
    print(empty, file)

    file.detach()
    if file.size == empty.size:
        # Perfect fit. Discard the Empty.
        empty.prev.insert_after(file)
        empty.detach()
        empty = file.next
    elif file.size < empty.size:
        # Shrink available space in the Empty.
        empty.prev.insert_after(file)
        empty.size -= file.size
    elif file.size > empty.size:
        # Break up the file
        last_file.insert_after(
            File(id=file.id, size=file.size - empty.size, pos=-1, prev=None, next=None)
        )
        last_file = last_file.next
        file.size = empty.size
        empty.prev.insert_after(file)
        empty.detach()
        empty = file.next

chunks.print_chain()

checksum = 0
offset = 0
for chunk in iter(chunks):
    assert isinstance(chunk, File)
    checksum += chunk.id * (sum(range(chunk.size)) + chunk.size * offset)
    offset += chunk.size
print(checksum)
