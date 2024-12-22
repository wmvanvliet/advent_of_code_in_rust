from functools import cache

with open("day19.txt") as f:
    available_towels = next(f).strip().split(", ")
    next(f)
    designs = f.read().strip().split("\n")


@cache
def count_possibilities(design):
    """Count the number of ways to make the design out of the available towels.

    Parameters
    ----------
    design : str
        The design to create.

    Returns
    -------
    n_possibilities : int
        The number of ways to make the design out of the available towels.
    """
    # When designing a recursive function, always do the end conditions first.
    if len(design) == 0:
        return 1

    # Try all possible towels to see if they would be a good beginning for the design.
    return sum(
        count_possibilities(design.removeprefix(towel))
        for towel in available_towels
        if design.startswith(towel)
    )


print("part 1:", sum(min(count_possibilities(design), 1) for design in designs))
print("part 2:", sum(count_possibilities(design) for design in designs))
