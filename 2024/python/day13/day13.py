import re
import numpy as np

part1 = 0
with open("day13.txt") as f:
    try:
        while True:
            x, y = re.match(r"Button A: X([\+-]\d+), Y([\+-]\d+)", next(f)).groups()
            A = (int(x), int(y))
            x, y = re.match(r"Button B: X([\+-]\d+), Y([\+-]\d+)", next(f)).groups()
            B = (int(x), int(y))
            x, y = re.match(r"Prize: X=(\d+), Y=(\d+)", next(f)).groups()
            #prize = (int(x) + 10000000000000, int(y) + 10000000000000)
            prize = (int(x), int(y))
            next(f)
            n_a, n_b = np.linalg.solve(np.vstack((A, B)).T, np.array(prize))
            if abs(n_a - round(n_a)) < 1e-1 and abs(n_b - round(n_b)) < 1e-1:
                print(n_a, n_b, int(round(n_a * 3 + n_b)))
                part1 += int(round(n_a * 3 + n_b))
            else:
                print(n_a, n_b, "no solution")
    except StopIteration:
        pass
print("part 1:", part1)
