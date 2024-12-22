from tqdm import tqdm

grid = dict()
with open("day20.txt") as f:
    for y, line in enumerate(f):
        for x, c in enumerate(line.strip()):
            if c == "S":
                start_pos = y, x
                c = "."
            elif c == "E":
                end_pos = y, x
                c = "."
            grid[(y, x)] = c

track = dict()
to_visit = [(start_pos, 0)]
pos = None
while pos != end_pos and len(to_visit) > 0:
    pos, dist = to_visit.pop(0)
    track[pos] = dist
    y, x = pos
    for n in [(y - 1, x), (y, x + 1), (y + 1, x), (y, x - 1)]:
        if grid.get(n, "#") == "." and n not in track:
            to_visit.append((n, dist + 1))


def count_shortcuts(max_length, min_time_saved):
    shortcuts = list()
    for pos, dist in tqdm(track.items()):
        y, x = pos
        for ty, tx in track.keys():
            td = abs(ty - y) + abs(tx - x)
            if td <= max_length:
                saved = track[(ty, tx)] - dist - td
                if saved >= min_time_saved:
                    shortcuts.append(saved)
                    # print(saved, len(shortcuts), flush=True)
    return len(shortcuts)


print("\npart 1:", count_shortcuts(max_length=2, min_time_saved=100))
print("\npart 2:", count_shortcuts(max_length=20, min_time_saved=100))
