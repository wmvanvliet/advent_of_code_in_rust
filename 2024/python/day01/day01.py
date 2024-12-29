import numpy as np
from collections import Counter

left, right = np.loadtxt("day01.txt", dtype="int").T
left.sort()
right.sort()
print("part 1:", np.sum(np.abs(left - right)))

right_counts = Counter(right)
print("part 2:", sum(l * right_counts.get(l, 0) for l in left))
