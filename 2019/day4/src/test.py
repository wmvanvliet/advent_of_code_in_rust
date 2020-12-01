"""
Stupid brute force solution to https://adventofcode.com/2019/day/4
in Python. I use this to test a more elegant analytical solution in Rust.
"""
from math import log10
from scipy.special import comb


def _get_digits(n):
    """Generate individual digits for a number."""
    if n == 0:
        yield 0
        return

    n_digits = int(log10(n))
    for i in range(n_digits, -1, -1):
        digit = n // (10 ** i)
        yield digit
        n = n - digit * (10 ** i)


def brute_force(start, end, restriction=None, verbose=False):
    """Generate all valid passwords in the given range.

    Parameters
    ----------
    start : int
        Minimum value (inclusive) of valid passwords
    end : int
        Maximum value (includive) of valid passwords
    restriction : None | 'part1' | 'part2'
        In addition to the restriction of increasing digits, what kind of
        additional restrictions to place.
        None - no additional restrictions
        'part1' - must have at least 2 adjacent equal numbers (puzzle, part 1)
        'part2' - must have at least 2 adjacent equal numbers that are not part
                  of a larger cluster (puzzle, part 2)
    verbose : bool
        Whether to display all the numbers that are tried (True),
        or be silent (False).
    """
    # Try all numbers in the given range
    for n in range(start, end + 1):
        if verbose: print(n, end=' ', flush=True)
        valid = True

        digits = list(_get_digits(n))

        # Make sure digits are increasing
        for x, y in zip(digits[:-1], digits[1:]):
            if y < x:
                valid = False
                break
        if not valid:
            if verbose: print('not increasing', flush=True)
            continue

        if restriction == 'part1':
            # Make sure there is at lease one number repeated
            for x, y in zip(digits[:-1], digits[1:]):
                if y == x:
                    valid = True
                    break
            else:
                # No adjacent equal numbers found
                valid = False
                if verbose: print('no two adjacent equal numbers', flush=True)

        elif restriction == 'part2':
            # Make sure there is a cluster of 2 repeated numbers, no more and
            # no less. This is accomplished by using a sliding window. Padding
            # is added to deal with boundary conditions.
            padded = ['#', *digits, '#']
            for w, x, y, z in zip(padded[:-3], padded[1:-2], padded[2:-1], padded[3:]):
                if w != x == y != z:
                    valid = True
                    break
            else:
                # No adjacent equal numbers found
                valid = False
                if verbose: print('no two adjacent equal numbers', flush=True)

        if valid:
            if verbose: print('valid', flush=True)
            yield n


all_digits = list(range(0, 10))
def num_inc_comb(allowed_digits, n_digits):
    return comb(len(allowed_digits), n_digits, exact=True, repetition=True)

def part1(allowed_first_digits, n_digits):
    total = 0
    for first_digit in allowed_first_digits:
        total += comb(10 - first_digit, n_digits - 1, exact=True, repetition=True)
        total -= comb(10 - first_digit - 1, n_digits - 1, exact=True, repetition=False)
    return int(total)

def part2(allowed_digits, n_digits):
    total = 0

    for duplicated_digit in all_digits:
        total += comb(10 - 1, n_digits - 2, exact=True, repetition=True)
    return total


if __name__ == '__main__':
    print('Part 1:', part1([4, 5, 6, 7], 6))
