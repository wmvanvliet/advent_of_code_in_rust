#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int num_lines(FILE* f) {
	fseek(f, 0, SEEK_SET);
	int count = 0;
	for(char c = getc(f); c != EOF; c = getc(f))
        count += c == '\n';
	fseek(f, 0, SEEK_SET);
	return count;
}

int part1() {
	FILE* f = fopen("day1_input.txt", "r");
	int prev_depth = INT_MAX;
	int curr_depth;
	int part1_ans = 0;
	while(fscanf(f, "%d", &curr_depth) != EOF) {
		part1_ans += curr_depth > prev_depth;
		prev_depth = curr_depth;
	}
	return part1_ans;
}

int part2() {
	FILE* f = fopen("day1_input.txt", "r");
	int n_depths = num_lines(f);
	int* buf = malloc(n_depths * sizeof(int));
	int* buf_head = buf;
	int depth;
	while(fscanf(f, "%d", &depth) != EOF)
		*buf_head++ = depth;
	int part2_ans = 0;
	int prev_depth = INT_MAX;
	for(int i=0; i<n_depths-3; i++) {
		int curr_depth = buf[i] + buf[i+3];
		part2_ans += curr_depth > prev_depth;
		prev_depth = curr_depth;
	}
	free(buf);
	return part2_ans;
}

int main(int argc, char** argv) {
	printf("Day 1, part 1: %d\n", part1());
	printf("Day 1, part 2: %d\n", part2());
}
