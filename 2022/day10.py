def cpu():
    """Generates the value of the x register for each cycle."""
    x = 1
    with open('input_day10.txt') as f:
        for line in f:
            instruction, *param = line.strip().split()
            if instruction == 'noop':
                yield x
            elif instruction == 'addx':
                yield x
                yield x
                x += int(param[0])

output = iter(cpu())
for y in range(6):
    for x in range(40):
        if (x - 1) <= next(output) <= (x + 1):
            print('█', end='')
        else:
            print(' ', end='')
    print()
