secret_number = 123

def pseudo_random(secret_number):
    while True:
        secret_number ^= secret_number << 6
        secret_number &= 0b111111111111111111111111
        secret_number ^= secret_number >> 5
        secret_number &= 0b111111111111111111111111
        secret_number ^= secret_number << 11
        secret_number &= 0b111111111111111111111111
        yield secret_number, secret_number % 10

part1 = 0
monkeys = [int(x) for x in open("day22.txt")]
all_patterns = dict()
for monkey in monkeys:
    patterns = dict()
    prices = [monkey % 10]
    number_iter = pseudo_random(monkey)
    for _ in range(2000):
        x, price = next(number_iter)
        prices.append(price)
    part1 += x

    for p0, p1, p2, p3, p4 in zip(prices[:-4], prices[1:-3], prices[2:-2], prices[3:-1], prices[4:]):
        pattern = (p1 - p0, p2 - p1, p3 - p2, p4 - p3)
        if pattern not in patterns:
            patterns[pattern] = p4

    for pattern, price in patterns.items():
        if pattern in all_patterns:
            all_patterns[pattern] += price
        else:
            all_patterns[pattern] = price

part2 = 0
best_pattern = None
for pattern, price in all_patterns.items():
    if price > part2:
        part2 = price
        best_pattern = pattern
print("part 1:", part1)
print("part 2:", part2)
