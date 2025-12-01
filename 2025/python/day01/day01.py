dial_pos = 50
n_zeros = 0
total_n_crossings = 0
with open("input.txt") as f:
    for line in f:
        direction, n_steps = line[0], int(line[1:])
        if direction == "L":
            n_steps *= -1

        n_crossings, new_pos = divmod(dial_pos + n_steps, 100)
        n_crossings = abs(n_crossings)

        if dial_pos == 0 and n_steps < 0:
            n_crossings -= 1
        if new_pos == 0 and n_steps > 0:
            n_crossings -= 1

        if new_pos == 0:
            n_zeros += 1

        total_n_crossings += n_crossings
        dial_pos = new_pos

print("part 1:", n_zeros)
print("part 2:", n_zeros + total_n_crossings)
