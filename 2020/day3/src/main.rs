use std::io::{self, Read};

/**
 * This reads in the puzzle input from stdin. So you would call this program like:
 *     cat input | cargo run
 * It then feeds the input as a string to the functions that solve both parts of the puzzle.
 */
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input)));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

/**
 * On Windows, a unicode BOM marker is always placed at the beginning of the input file. Very
 * annoying.
 */
fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> i64 {
    count_trees(input, 3, 1)
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    count_trees(input, 1, 1) * count_trees(input, 3, 1) * count_trees(input, 5, 1) * count_trees(input, 7, 1) * count_trees(input, 1, 2)
}

/**
 * Count the number of trees given a specific slope.
 */
fn count_trees(input: &str, dx:usize, dy:usize) -> i64 {
    let mut pos:usize = 0;
    let mut n_trees:i64 = 0;
    for line in input.lines().skip(dy).step_by(dy) {
        pos = (pos + dx) % line.len();
        if line.chars().nth(pos).unwrap() == '#' {
            n_trees += 1;
        }
    }
    n_trees
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("..##.......\n#...#...#..\n.#....#..#.\n..#.#...#.#\n.#...##..#.\n..#.##.....\n.#.#.#....#\n.#........#\n#.##...#...\n#...##....#\n.#..#...#.#\n"), 7);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("..##.......\n#...#...#..\n.#....#..#.\n..#.#...#.#\n.#...##..#.\n..#.##.....\n.#.#.#....#\n.#........#\n#.##...#...\n#...##....#\n.#..#...#.#\n"), 336);
    }
}
