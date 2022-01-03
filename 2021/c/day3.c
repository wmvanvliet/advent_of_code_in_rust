#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int part1() {
	FILE* f = fopen("day3_input.txt", "r");

	char num[13]; // leave some room for possible \r
	int sums[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	while(fscanf(f, "%s\n", num) != EOF) {
		for(int i=0; i<12; i++)
			sums[i] += num[i] - '0';
	}

	int gamma = 0;
	int epsilon = 0;
	for(int i=0; i<12; i++) {
		if(sums[11 - i] > 500)
			gamma += 1 << i;
		else
			epsilon += 1 << i;
	}
	return gamma * epsilon;
}

int compare_ints(const void* a, const void* b) {
	return *(int*)a - *(int*)b;
}

unsigned int binsearch(unsigned int* nums, int from, int to, unsigned int bitmask) {
	if(to - from <= 1) return to;
	int midpoint = from + (to - from) / 2;
	if(nums[midpoint] & bitmask) {
		if((nums[midpoint - 1] & bitmask) == 0) return midpoint;
		else return binsearch(nums, from, midpoint, bitmask);
	} else {
		if((nums[midpoint + 1] & bitmask) == 1) return midpoint + 1;
		else return binsearch(nums, midpoint, to, bitmask);
	}
}

#define N_NUMS 1000

int part2() {
	FILE* f = fopen("day3_input.txt", "r");
	unsigned int nums[N_NUMS];
	char line[100];
	for(unsigned int* cur_num = nums; fscanf(f, "%s\n", line) != EOF; cur_num++) {
		*cur_num = strtol(line, NULL, 2);
	};

	qsort(nums, N_NUMS, sizeof(unsigned int), compare_ints);

	int from = 0;
	int to = N_NUMS;
	for(unsigned int mask = 1 << 4; mask > 0; mask >>= 1) {
		int transition_point = binsearch(nums, from, to, mask);
		if(transition_point > (from + (to - from) / 2)) {
			to = transition_point;
		} else {
			from = transition_point;
		}
	}
	unsigned int epsilon = nums[from];

	from = 0;
	to = N_NUMS;
	for(unsigned int mask = 1 << 4; mask > 0; mask >>= 1) {
		int transition_point = binsearch(nums, from, to, mask);
		if(transition_point <= (from + (to - from) / 2)) {
			to = transition_point;
		} else {
			from = transition_point;
		}
	}
	unsigned int gamma = nums[from - 1];

	printf("gamma point %d\n", from);
	printf("epsilon: %u, gamma: %u\n", epsilon, gamma);
	return epsilon * gamma;
}

int main(int argc, char** argv) {
	printf("Day 3, part 1: %d\n", part1());
	printf("Day 3, part 2: %d\n", part2());
	return 0;
}
