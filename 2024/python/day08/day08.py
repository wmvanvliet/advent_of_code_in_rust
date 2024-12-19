antennae = dict()
with open("day08.txt") as f:
    for y, line in enumerate(f):
        for x, freq in enumerate(line.strip()):
            if freq != ".":
                freq_antennae = antennae.get(freq, set())
                freq_antennae.add((y, x))
                antennae[freq] = freq_antennae
height = y + 1
width = len(line.strip())

antinodes_part1 = set()
antinodes_part2 = set()
for freq in antennae.keys():
    for a1 in antennae[freq]:
        for a2 in antennae[freq]:
            if a1 == a2:
                continue
            dy = a1[0] - a2[0]
            dx = a1[1] - a2[1]
            antinode = (a1[0], a1[1])
            part1_done = False
            while 0 <= antinode[0] < height and 0 <= antinode[1] < width:
                if not part1_done:
                    antinodes_part1.add(antinode)
                    part1_done = True
                antinodes_part2.add(antinode)
                antinode = (antinode[0] + dy, antinode[1] + dx)

print("part 1:", len(antinodes_part1))
print("part 2:", len(antinodes_part2))
