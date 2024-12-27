import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label

part1 = 0
part2 = 0

grid = np.array([list(l.strip()) for l in open("day12.txt")])
abs_sum = lambda x: abs(x).sum()
for patches, n in [label(grid == x) for x in np.unique(grid)]:
    for i in range(n):
        patch = (patches == (i + 1)).astype("int")
        area = np.sum(patch)
        h = abs_sum(convolve2d(patch, [[1, -1]]))
        v = abs_sum(convolve2d(patch, [[1], [-1]]))
        x = abs_sum(convolve2d(patch, [[-1, 1], [1, -1]]))
        part1 += area * (h + v)
        part2 += area * x

print("part 1:", part1)
print("part 2:", part2)

