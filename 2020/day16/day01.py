"""
Advent of code 2020, day 1
--------------------------

The Elves in accounting just need you to fix your expense report (your puzzle
input); apparently, something isn't quite adding up.

Specifically, they need you to find the two entries that sum to 2020 and then
multiply those two numbers together.
"""


def solve_part1(puzzle_input):
    """Solve part 1 of today's puzzle."""

    # We need quick lookups, so we store the expenses as a set.
    expense_report = set(int(entry) for entry in puzzle_input.split())

    for entry1 in expense_report:
        # This is the entry that needs to be in the expense report if the two
        # are to sum to 2020.
        entry2 = 2020 - entry1
        if entry2 in expense_report:
            return entry1 * entry2


def solve_part2(puzzle_input):
    """Solve part 2 of today's puzzle."""
    return 2


if __name__ == '__main__':
    with open('day1_input.txt') as f:
        puzzle_input = f.read()
        print('Part 1:', solve_part1(puzzle_input))
        print('Part 2:', solve_part2(puzzle_input))


def test_part1():
    """Run the test cases for part1 given in the puzzle description."""

    # In this list, the two entries that sum to 2020 are 1721 and 299.
    # Multiplying them together produces 1721 * 299 = 514579, so the correct
    # answer is 514579.
    assert solve_part1('''1721
                          979
                          366
                          299
                          675
                          1456''') == 514579
