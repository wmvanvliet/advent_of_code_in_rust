# Read in the file with the ranges.
ranges = [range_str.split("-") for range_str in open("input.txt").read().strip().split(",")]

part1 = 0
for start_str, stop_str in ranges:
    start_int, stop_int = int(start_str), int(stop_str)

    # The smallest number (to repeat twice to make the full pattern) is the first half
    # if the starting number of the range.
    if len(start_str) == 1:
        number = 0
    else:
        number = int(start_str[:len(start_str) // 2])

    # This is the first pattern to try
    pattern = int(str(number) * 2)

    # Try all patterns by incrementing the number until we reach the end of the range.
    while pattern <= stop_int:
        if pattern >= start_int:
            part1 += pattern
        number += 1
        pattern = int(str(number) * 2)
print("part 1:", part1)

part2 = 0
for start_str, stop_str in ranges:
    wrong_ids = set()
    start_int, stop_int = int(start_str), int(stop_str)

    # The pattern needs to repeat at least twice, so we don't have to check numbers
    # longer than half the length of stop_int.
    max_number = int('9' * (len(stop_str) // 2))

    # Try all numbers to use as a base for a repeating pattern.
    for num in range(max_number + 1):
        num_str = str(num)

        # Can we make a pattern that fits the length of the start number?
        if len(start_str) >= 2 and len(start_str) % len(num_str) == 0:
            # Repeat the number until we reach the length of the start number.
            pattern = int(num_str * (len(start_str) // len(num_str)))
            if start_int <= pattern <= stop_int:
                wrong_ids.add(pattern)

        # Can we make a pattern that fits the length of the stop number?
        if len(stop_str) >= 2 and len(stop_str) % len(num_str) == 0:
            pat = int(num_str * (len(stop_str) // len(num_str)))
            if start_int <= pattern <= stop_int:
                wrong_ids.add(pattern)

    part2 += sum(wrong_ids)
print("part 2:", part2)
