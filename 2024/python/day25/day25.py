from itertools import product

with open("day25.txt") as f:
    schemas = [schema.strip().split("\n") for schema in f.read().split("\n\n")]

locks = list()
keys = list()
for schema in schemas:
    n_pins = [pins.count("#") - 1 for pins in zip(*schema)]
    if schema[0][0] == "#":
        locks.append(n_pins)
    else:
        keys.append(n_pins)

part1 = 0
for key, lock in product(keys, locks):
    if all(n1 + n2 < 6 for n1, n2 in zip(key, lock)):
        part1 += 1

print("part 1:", part1)
