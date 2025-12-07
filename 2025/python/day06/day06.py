import numpy as np
from functools import reduce
from operator import add, mul

with open("example.txt") as f:
    lines = f.read().split("\n")[:-1]

problems = list()
for line in lines[:-1]:
    problem = [int(x) for x in line.split()]
    problems.append(problem)
problems = list(zip(*problems))
operators = lines[-1].split()

def do_homework(problems, operators):
    total = 0
    for problem, operator in zip(problems, operators):
        if operator == "+":
            total += reduce(add, problem)
        elif operator == "*":
            total += reduce(mul, problem)
    return total

print("part 1:", do_homework(problems, operators))

# Transpose the lines
lines = list(zip(*lines[:-1]))[::-1]
lines = [''.join(line).strip() for line in lines]
problems = list()
problem = list()
for line in lines:
    if len(line) == 0:
        problems.append(problem)
        problem = list()
    else:
        problem.append(int(line))
problems.append(problem)

print("part 2:", do_homework(problems, operators[::-1]))
