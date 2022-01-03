#include <stdio.h>
#include <string.h>

int part1() {
	FILE* f = fopen("day2_input.txt", "r");
	int depth = 0;
	int horizontal_pos = 0;

	char direction[8];
	int amount;
	while(fscanf(f, "%s %d", direction, &amount) != EOF) {
		if(!strcmp(direction, "forward")) {
			horizontal_pos += amount;
		} else if (!strcmp(direction, "up")) {
			depth -= amount;
		} else if (!strcmp(direction, "down")) {
			depth += amount;
		}
	}
	return horizontal_pos * depth;
}

int part2() {
	FILE* f = fopen("day2_input.txt", "r");
	int aim = 0;
	int depth = 0;
	int horizontal_pos = 0;

	char direction[8];
	int amount;
	while(fscanf(f, "%s %d", direction, &amount) != EOF) {
		if(!strcmp(direction, "forward")) {
			horizontal_pos += amount;
			depth += aim * amount;
		} else if (!strcmp(direction, "up")) {
			aim -= amount;
		} else if (!strcmp(direction, "down")) {
			aim += amount;
		}
	}
	return horizontal_pos * depth;
}

int main(int argc, char** argv) {
	printf("Day 2, part 1: %d\n", part1());
	printf("Day 2, part 2: %d\n", part2());
}
