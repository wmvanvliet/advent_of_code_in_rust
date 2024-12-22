from queue import PriorityQueue

dirpad = {(0, 1): "^", (0, 2): "A", (1, 0): "<", (1, 1): "v" , (1, 2): ">"}
numpad = {(0, 0): "7", (0, 1): "8", (0, 2): "9", (1, 0): "4", (1, 1): "5", (1, 2): "6", (2, 0): "1", (2, 1): "2", (2, 2): "3", (3, 1): "0", (3, 2): "A"}

santa = {(fro, to): 1 for fro in dirpad.values() for to in dirpad.values()}

def print_cost(keys, cost):
    print(" " + "".join([f"{k:>3}" for k in (" " + keys)]))
    for fro in keys:
        print(f"{fro:>3} ", end="")
        for to in keys:
            print(f"{cost.get((fro, to), ' '):>3}", end="")
        print()
    print()

def neighbours(y, x, pad):
    for pos, key in zip([(y - 1, x), (y, x + 1), (y + 1, x), (y, x - 1)], "^>v<"):
        if pos in pad:
            yield pos, pad[pos], key

def compute_cost(pad, operator, verbose=False):
    cost = dict()
    for (start_y, start_x), start_key in pad.items():
        start_dir = "A"
        if verbose:
            print(f"{start_y=}, {start_x=}, {start_key=} {start_dir=}")
        to_visit = PriorityQueue()
        to_visit.put((0, start_y, start_x, start_dir, start_dir))
        cost[(start_key, start_key)] = 1
        while not to_visit.empty():
            fro_cost, fro_y, fro_x, fro_key, fro_dir = to_visit.get()
            if verbose:
                print(f"  {fro_y=}, {fro_x=}, {fro_cost=}, {fro_dir=}")
            for (to_y, to_x), to_key, to_dir in neighbours(fro_y, fro_x, pad):
                to_cost = fro_cost + operator[(fro_dir, to_dir)]
                total_cost = to_cost + operator[(to_dir, "A")]
                if (start_key, to_key) not in cost or cost[(start_key, to_key)] > total_cost:
                    if verbose:
                        print(f"    {to_y=}, {to_x=}, {to_key=}, {to_dir=}, {to_cost=}, {total_cost=}")
                    cost[(start_key, to_key)] = total_cost
                    to_visit.put((to_cost, to_y, to_x, to_key, to_dir))
                    if verbose:
                        print("    ", cost)
    return cost

operators = [santa]
for _ in range(25):
    operators.append(compute_cost(dirpad, operators[-1]))
code_part1 = compute_cost(numpad, operators[2])
code_part2 = compute_cost(numpad, operators[-1])

# print_cost("A^>v<", operators[1])
# print_cost("A^>v<", operators[2])
# print_cost("A0123456789", code_part1)

part1 = 0
part2 = 0
with open("day21.txt") as f:
    for line in f:
        fro_key = "A"
        n_keys_part1 = 0
        n_keys_part2 = 0
        for to_key in line.strip():
            n_keys_part1 += code_part1[(fro_key, to_key)]
            n_keys_part2 += code_part2[(fro_key, to_key)]
            fro_key = to_key
        part1 += n_keys_part1 * int(line[:-2])
        part2 += n_keys_part2 * int(line[:-2])
print("part 1:", part1)
print("part 2:", part2)

