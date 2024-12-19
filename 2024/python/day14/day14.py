import numpy as np
import matplotlib.pyplot as plt
pos = []
vel = []
with open("day14.txt") as f:
    for line in f:
        p, v = line.split()
        p = [int(x) for x in p[2:].split(",")]
        v = [int(x) for x in v[2:].split(",")]
        pos.append(p)
        vel.append(v)
pos = np.array(pos)
vel = np.array(vel)

width = 101
height = 103
# width = 11
# height = 7

pos2 = pos + 100 * vel
pos2[:, 0] = pos2[:, 0] % width
pos2[:, 1] = pos2[:, 1] % height

part1 = (
    np.sum((pos2[:, 0] < (width // 2)) & (pos2[:, 1] < (height // 2))) *
    np.sum((pos2[:, 0] > (width // 2)) & (pos2[:, 1] < (height // 2))) *
    np.sum((pos2[:, 0] < (width // 2)) & (pos2[:, 1] > (height // 2))) *
    np.sum((pos2[:, 0] > (width // 2)) & (pos2[:, 1] > (height // 2)))
)
print("part 1:", part1)

def plot(pos):
    grid = np.zeros((height, width), dtype="int")
    for x, y in pos:
        grid[y, x] += 1
    plt.clf()
    plt.imshow(grid)
    plt.draw()

# Vertical
# 99
# 200
# Horizontal
# 145
# 248
fig = plt.figure()
for i in range(width * height):
    if (i + 2) % width == 0 and (i - 42) % height == 0:
        pos2 = pos + i * vel
        pos2[:, 0] = pos2[:, 0] % width
        pos2[:, 1] = pos2[:, 1] % height
        plot(pos2)
        break
print("part 2:", i)
