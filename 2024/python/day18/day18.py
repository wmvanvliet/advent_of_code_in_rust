import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

to_fall = list()
with open("day18.txt") as f:
    for brick in f:
        x, y = brick.split(",")
        to_fall.append((int(y), int(x)))
start_pos = (0, 0)
end_pos = 70, 70
width = 71
height = 71

def plot_map(corrupted, visited=[]):
    g = np.zeros((height, width), dtype="int")
    for block in corrupted:
        g[block] = 2
    for block in visited:
        g[block] = 1
    plt.imshow(g)

def neighbours(y, x):
    if y > 0:
        yield y - 1, x
    if x < width - 1:
        yield y, x + 1
    if y < height - 1:
        yield y + 1, x
    if x > 0:
        yield y, x - 1


def solve(corrupted, plot=False):
    to_visit = PriorityQueue()
    to_visit.put((end_pos[0] + end_pos[1], 0, start_pos))
    visited = {start_pos}

    iterations = 0
    pos = None
    while pos != end_pos:
        if to_visit.empty():
            return -1, iterations
        heuristic, steps, pos = to_visit.get()
        for n in neighbours(*pos):
            if n not in corrupted and n not in visited:
                to_visit.put((steps + end_pos[0] + end_pos[1] + 1 - n[0] - n[1], steps + 1, n))
                visited.add(n)
        iterations += 1
    if plot:
        plot_map(corrupted, visited)
    return steps, iterations

print("part 1:", solve(set(to_fall[:1024]))[0])

for i in range(len(to_fall), 0, -1):
    steps, iterations = solve(set(to_fall[:i]))
    if steps != -1:
        break
print("part 2:", f"{to_fall[i][1]},{to_fall[i][0]}")

# solve(to_fall[:i], plot=True)
