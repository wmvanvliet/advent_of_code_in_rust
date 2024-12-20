from collections import Counter
stones = Counter([int(x) for x in open("day11.txt").read().strip().split()])

def count_stones(stones, n):
    for i in range(n):
        new_stones = dict()
        for stone, num in stones.items():
            if stone == 0:
                new_stones[1] = new_stones.get(1, 0) + num
                continue
            stone_str = str(stone)
            if len(stone_str) % 2 == 0:
                first_half = int(stone_str[:len(stone_str) // 2])
                second_half = int(stone_str[len(stone_str) // 2:])
                new_stones[first_half] = new_stones.get(first_half, 0) + num
                new_stones[second_half] = new_stones.get(second_half, 0) + num
            else:
                prod = stone * 2024
                new_stones[prod] = new_stones.get(prod, 0) + num
        stones = new_stones
    return sum(stones.values())
print("part 1:", count_stones(stones, 25))
print("part 2:", count_stones(stones, 75))
