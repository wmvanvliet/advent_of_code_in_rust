import re
from functools import cmp_to_key
from operator import add, mul

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
