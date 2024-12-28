from collections import defaultdict

wires_values = dict()
gates = dict()
wires = defaultdict(set)

with open("day24.txt") as f:
    wires_lines, gates_lines = f.read().split("\n\n")
    for wire in wires_lines.strip().split("\n"):
        name, value = wire.split(": ")
        wires_values[name] = int(value)
    for gate in gates_lines.strip().split("\n"):
        in1, op, in2, _, out = gate.split()
        gates[out] = (op, in1, in2)
        wires[in1].add(out)
        wires[in2].add(out)


def compute_value(wire):
    """Compute value of a wire."""
    if wire in wires_values:
        return wires_values[wire]

    op, in1, in2 = gates[wire]
    in1 = compute_value(in1)
    in2 = compute_value(in2)
    if op == "AND":
        return in1 & in2
    elif op == "OR":
        return in1 | in2
    else:  # XOR
        return in1 ^ in2


part1 = 0
for i in range(45, -1, -1):
    part1 = (part1 << 1) + compute_value(f"z{i:02d}")
print("part 1:", part1)


wrong_wires = set()
for out, (op, *ins) in gates.items():
    connecting_gates = {gates[g][0] for g in wires[out]}

    # XOR/AND-Gates of the first adder, all of which are good
    if all(i in ("x00", "y00") for i in ins):
        continue

    # OR-gate of the final adder, which is good
    elif op == "OR" and out == "z45":
        continue

    # XOR-gates should either be connected to an input or an output
    elif op == "XOR":
        if all(i[0] in "xy" for i in ins):
            # XOR-gates connected to an input should connect to an XOR and AND gate
            if connecting_gates != {"XOR", "AND"}:
                wrong_wires.add(out)
        elif out[0] != "z":
            wrong_wires.add(out)

    # AND-gates should be connected to a single OR-gate
    elif op == "AND" and connecting_gates != {"OR"}:
        wrong_wires.add(out)

    # OR-gates are the carry-out, which should connect to an XOR and AND gate
    elif op == "OR" and connecting_gates != {"XOR", "AND"}:
        wrong_wires.add(out)

print("part 2:", ",".join(sorted(wrong_wires)))
