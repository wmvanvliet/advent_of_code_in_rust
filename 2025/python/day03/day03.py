def k_max(bank, k):
    """Determine which k batteries to turn on for maximum joltage."""
    joltages = list()
    min_ind = 0  # minimum index to start the search for largest voltage.
    for n in range(k, 0, -1):
        max_ind = len(bank) - (n - 1)  # need space for selecting the other batteries
        bank_selection = bank[min_ind:max_ind]
        bank_indices = range(min_ind, max_ind)
        joltage, ind = max(zip(bank_selection, bank_indices), key=lambda x: x[0])
        min_ind = ind + 1  # restrict future search to batteries after the chosen one
        joltages.append(joltage)
    return joltages


part1 = 0
part2 = 0
with open("input.txt") as f:
    for line in f:
        bank = [int(battery) for battery in line.strip()]

        # Part 1
        turn_on = k_max(bank, k=2)
        part1 += int("".join(str(joltage) for joltage in turn_on))

        # Part 2
        turn_on = k_max(bank, k=12)
        part2 += int("".join(str(joltage) for joltage in turn_on))

print("part 1:", part1)
print("part 2:", part2)
