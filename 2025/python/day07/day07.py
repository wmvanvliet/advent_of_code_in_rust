from collections import defaultdict

beams = defaultdict(int)
part1 = 0
with open("input.txt") as f:
    start_position = next(f).index("S")
    beams[start_position] = 1
    for line in f:
        splits = [i for i, c in enumerate(line) if (c == "^" and i in beams)]
        part1 += len(splits)
        for pos in splits:
            # split all the beams
            n = beams.pop(pos)
            beams[pos - 1] += n
            beams[pos + 1] += n

print("part 1:", part1)
print("part 2:", sum(beams.values()))
