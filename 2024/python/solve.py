import re

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

n_xmas = 0
lines = open("day4_test.txt").read().strip().split("\n")
for line in lines:
    # Forwards
    n_xmas += len(re.findall("XMAS", line))
    # Backwards
    n_xmas += len(re.findall("XMAS", line[::-1]))
# Move lines vertical
lines = ["".join(l) for l in zip(*[list(line) for line in lines])]
for line in lines:
    # Downwards
    n_xmas += len(re.findall("XMAS", line))
    # Upwards
    n_xmas += len(re.findall("XMAS", line[::-1]))
print("Day 4, part 1:", n_xmas)

n_xmas = 0
lines = open("day4_test.txt").read().strip().split("\n")
height = len(lines)
width = len(lines[0])
for y in range(height):
    for x in range(width):
        # Forwards
            lines[y][x:]
