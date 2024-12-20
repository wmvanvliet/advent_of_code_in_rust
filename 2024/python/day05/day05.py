from functools import cmp_to_key

str_rules, str_updates = open("day05.txt").read().strip().split("\n\n")
rules = dict()
for rule in str_rules.split("\n"):
    x, y = rule.split("|")
    rules[(x, y)] = -1
    rules[(y, x)] = 1
updates = [u.split(",") for u in str_updates.split("\n")]
key_func = cmp_to_key(lambda x, y: rules.get((x, y), 0))
updates_sorted = [sorted(u, key=key_func) for u in updates]
part1 = sum(int(u[len(u) // 2]) for u, us in zip(updates, updates_sorted) if u == us)
part2 = sum(int(us[len(us) // 2]) for u, us in zip(updates, updates_sorted) if u != us)
print("part 1:", part1)
print("part 2:", part2)
