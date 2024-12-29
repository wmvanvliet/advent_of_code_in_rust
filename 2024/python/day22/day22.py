import numpy as np
from numba import njit

secret_number = 123


@njit
def pseudo_random(secret_number: int):
    """Generate all 2000 secret numbers for a monkey."""
    numbers = np.empty(2000, dtype="int32")
    prices = np.empty(2001, dtype="int8")
    prices[0] = secret_number % 10
    for i in range(2000):
        secret_number ^= secret_number << 6 & 0xFFFFFF
        secret_number ^= secret_number >> 5 & 0xFFFFFF
        secret_number ^= secret_number << 11 & 0xFFFFFF
        numbers[i] = secret_number
        prices[i + 1] = secret_number % 10
    return numbers, prices


part1 = 0
monkeys = [int(x) for x in open("day22.txt")]
all_patterns = dict()
for monkey in monkeys:
    patterns = dict()
    numbers, prices = pseudo_random(monkey)
    part1 += numbers.sum(dtype="int64")
    differences = np.diff(prices)
    for i in range(len(differences) - 3):
        pattern = tuple(differences[i : i + 4])
        if pattern not in patterns:
            patterns[pattern] = prices[i + 4]

    for pattern, price in patterns.items():
        if pattern in all_patterns:
            all_patterns[pattern] += price
        else:
            all_patterns[pattern] = price.astype("int64")

best_pattern, part2 = max(all_patterns.items(), key=lambda x: x[1])
print("part 1:", part1)
print("part 2:", part2)
