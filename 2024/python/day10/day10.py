grid = dict()
starting_points = set()
with open("day10.txt") as f:
    for y, line in enumerate(f):
        for x, elevation in enumerate(line.strip()):
            elevation = int(elevation)
            if elevation == 9:
                grid[(y, x)] = [elevation, set([(y, x)]), 1]
            else:
                grid[(y, x)] = [elevation, set(), 0]
            if elevation == 0:
                starting_points.add((y, x))
height = y + 1
width = len(line.strip())


# def print_elevation_map():
#     for y in range(height):
#         for x in range(width):
#             print(grid[(y, x)][0], end="")
#         print()
#     print()
#
#
# def print_reachable_map():
#     for y in range(height):
#         for x in range(width):
#             print(f"{len(grid[(y, x)][1]):02d} ", end="")
#         print()
#     print()
#
#
# def print_rating_map():
#     for y in range(height):
#         for x in range(width):
#             print(f"{grid[(y, x)][2]:02d} ", end="")
#         print()
#     print()
#
#
def neighbours(y, x):
    if y > 0:
        yield (y - 1, x)
    if x < width - 1:
        yield (y, x + 1)
    if y < height - 1:
        yield (y + 1, x)
    if x > 0:
        yield (y, x - 1)


#
#
# print_elevation_map()
# print_reachable_map()
# print_rating_map()

for _ in range(9):
    for y in range(height):
        for x in range(width):
            elevation, reachable, rating = grid[(y, x)]
            if elevation == 9:
                continue
            rating = 0
            for neighbour in neighbours(y, x):
                elevation2, reachable2, rating2 = grid[neighbour]
                if elevation2 - elevation == 1:
                    reachable.update(reachable2)
                    rating += rating2
            grid[(y, x)][2] = rating

# print_reachable_map()
# print_rating_map()

print("part 1:", sum(len(grid[p][1]) for p in starting_points))
print("part 2:", sum(grid[p][2] for p in starting_points))
