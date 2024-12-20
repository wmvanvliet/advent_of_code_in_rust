with open("day17.txt") as f:
    a = int(next(f).strip().split(": ")[1])
    b = int(next(f).strip().split(": ")[1])
    c = int(next(f).strip().split(": ")[1])
    next(f)
    program = next(f).strip().split(": ")[1]
    program = [int(x) for x in program.split(",")]
instr = 0


def parse_combo(operand, a, b, c):
    if operand == 4:
        return a
    elif operand == 5:
        return b
    elif operand == 6:
        return c
    else:
        return operand


step = 0
output = []
while 0 <= instr < len(program) - 1:
    opcode = program[instr]
    operand = program[instr + 1]
    if opcode == 0:  # adv
        a = a >> parse_combo(operand, a, b, c)
    elif opcode == 1:  # bxl
        b = int(b ^ operand)
    elif opcode == 2:  # bst
        b = parse_combo(operand, a, b, c) & 7
    elif opcode == 3:  # jnz
        if a != 0:
            instr = operand
            step += 1
            continue
    elif opcode == 4:  # bxc
        b = b ^ c
    elif opcode == 5:  # out
        operand = parse_combo(operand, a, b, c) & 7
        output.append(operand)
    elif opcode == 6:  # bdv
        b = a >> parse_combo(operand, a, b, c)
    elif opcode == 7:  # cdv
        c = a >> parse_combo(operand, a, b, c)
    else:
        raise ValueError(f"Unknown opcode: {opcode} ({operand})")
    instr += 2

print("part 1:", ",".join([str(o) for o in output]))


def solve(program, part2):
    if len(program) == 0:
        return part2
    target = program[-1]
    options = []
    for a_part in range(8):
        a = (part2 << 3) + a_part
        b = a & 0b111
        b = b ^ 0b001
        c = a >> b
        b = b ^ 0b101
        b = b ^ c
        if (b & 0b111) == target:
            options.append(a_part)
    for option in options:
        sol = solve(program[:-1], (part2 << 3) + option)
        if sol >= 0:
            return sol
    return -1


print("part 2:", solve(program, 0))
