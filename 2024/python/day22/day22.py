secret_number = 123
from collections import deque

def pseudo_random(secret_number):
    prev_price = secret_number % 10
    pattern = deque(maxlen=4)
    while True:
        secret_number ^= secret_number << 6
        secret_number &= 0b111111111111111111111111
        secret_number ^= secret_number >> 5
        secret_number &= 0b111111111111111111111111
        secret_number ^= secret_number << 11
        secret_number &= 0b111111111111111111111111
        price = secret_number % 10
        pattern.append(price - prev_price)
        if len(pattern) == 4:
            yield secret_number, price, tuple(pattern)
        else:
            yield secret_number, 0, None
        prev_price = price

part1 = 0
monkeys = [int(x) for x in open("day22.txt")]
all_patterns = dict()
for monkey in monkeys:
    patterns = dict()
    number_iter = pseudo_random(monkey)
    for _ in range(2000):
        x, price, pattern = next(number_iter)
        if pattern not in patterns:
            patterns[pattern] = price
            if pattern in all_patterns:
                all_patterns[pattern] += price
            else:
                all_patterns[pattern] = price
    part1 += x
part2 = max(all_patterns.items(), key=lambda x: x[1])[1]

print("part 1:", part1)
print("part 2:", part2)
