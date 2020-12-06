use std::io::{self, Read};
use std::collections::{HashSet};

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

/*
 * Windows uses different line endings. Sigh.
 */
#[cfg(windows)]
const EMPTY_LINE: &'static str = "\r\n\r\n";
#[cfg(not(windows))]
const EMPTY_LINE: &'static str = "\n\n";

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> usize {
    input
        .split(EMPTY_LINE)
        .map(|group| {
            group
                .lines()
                .flat_map(|person| person.trim().chars())
                .collect::<HashSet<_>>()
                .len()
        }).sum()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> usize {
    input
        .split(EMPTY_LINE)
        .map(|group| {
            let mut group_iter = group
                .lines()
                .map(|person| {
                    person
                        .trim()
                        .chars()
                        .collect::<HashSet<char>>()
                });

            // No "reduce" yet in Rust :(
            // So we take out the first value manually and then fold
            let first_group = group_iter.next().unwrap();
            group_iter
                .fold(first_group, |x, y| {
                    x.intersection(&y).cloned().collect()
                })
                .len()
        }).sum()
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("abc\n\na\nb\nc\n\nab\nac\n\na\na\na\na\n\nb"), 5);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("abc\n\na\nb\nc\n\nab\nac\n\na\na\na\na\n\nb"), 6);
    }
}
