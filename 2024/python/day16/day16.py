from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np

grid = dict()
with open("day16.txt") as f:
    for y, line in enumerate(f):
        for x, c in enumerate(line.strip()):
            if c == "S":
                start_position = (y, x)
                grid[(y, x)] = "."
            elif c == "E":
                end_position = (y, x)
                grid[(y, x)] = "."
            else:
                grid[(y, x)] = c
width = len(line.strip())
height = y + 1

def neighbours(y, x):
    if y > 0:
        yield  y - 1, x
    if x < width - 1:
        yield  y, x + 1
    if y < height - 1:
        yield y + 1, x
    if x > 0:
        yield y, x - 1

to_explore = PriorityQueue()
to_explore.put((0, start_position, (0, 1)))
visited = {(start_position, (0, 1)): 0}

step = 0
position = start_position
while position != end_position:
    cost, position, direction = to_explore.get()
    #print(cost, position, direction)
    for n in neighbours(*position):
        #print("    ", n, grid[n])
        if grid[n] == ".":
            next_cost = cost + 1
            required_direction = n[0] - position[0], n[1] - position[1]
            if required_direction != direction:
                next_cost += 1000
            #print("        ", next_cost, n, required_direction)
            if (n, required_direction) in visited and visited[(n, required_direction)] <= next_cost:
                continue
            visited[(n, required_direction)] = next_cost
            to_explore.put((next_cost, n, required_direction))
            #print("        explore")
    #if step == 14:
    #    break
    step += 1

print("part 1:", cost)
end_direction = direction

route = dict()
for (position, direction), cost in visited.items():
    if position not in route:
        route[position] = [(cost, direction)]
    else:
        route[position].append((cost, direction))

def trace(route, position, direction, g, level=0):
    if position == start_position:
        return
    options = route[position]
    options = [(c - 1000, d) if d == direction else (c, d) for c, d in options]
    options.sort()
    #print(level * " ", position, direction, options)
    min_cost, _ = options[0]
    for cost, (dy, dx) in options:
        #print(level * " " + "  ", cost, dy, dx)
        if cost == min_cost:
            next_position = position[0] - dy, position[1] - dx
            g[next_position] += 1
            trace(route, next_position, (dy, dx), g, level + 1)
        else:
            break

g = np.zeros((height, width), dtype="int")
for p, c in grid.items():
    g[p] = -1 if c == "#" else 0
g[end_position] = 1
trace(route, end_position, end_direction, g)

#plt.imshow(g)
print("part 2:", g[g > 0].size)
