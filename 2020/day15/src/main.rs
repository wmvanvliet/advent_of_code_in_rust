use std::io::{self, Read};
use std::collections::HashMap;

/**
 * This reads in the puzzle input from stdin. So you would call this program like:
 *     cat input | cargo run
 * It then feeds the input as a string to the functions that solve both parts of the puzzle.
 */
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", solve(strip_bom(&input), 2020));
    println!("Part 2 answer: {}", solve(strip_bom(&input), 30_000_000));
}

/**
 * On Windows, a unicode BOM marker is always placed at the beginning of the input file. Very
 * annoying.
 */
fn strip_bom(input: &str) -> &str {
    match input.strip_prefix("\u{feff}") {
        Some(x) => x,
        _ => input
    }
}

/**
 * Solves parts 1 and 2 of the puzzle.
 */
fn solve(input: &str, n_turns: usize) -> usize {
    let mut turn:usize = 0;
    let mut last_spoken:usize = 0;
    let mut mem: HashMap<usize, usize> = input
        .trim()
        .split(',')
        .map(|x| { last_spoken = x.parse().unwrap(); turn += 1; (last_spoken, turn) })
        .collect();

    for turn in (mem.len()+1)..=n_turns {
        let num = match mem.get(&last_spoken) {
            Some(prev_seen) => turn - 1 - prev_seen,
            _ => 0,
        };
        mem.insert(last_spoken, turn - 1);
        last_spoken = num;
    }

    last_spoken
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve("0,3,6", 10), 0);
        assert_eq!(solve("1,3,2", 10), 1);
        assert_eq!(solve("2,1,3", 10), 10);
        assert_eq!(solve("1,2,3", 10), 27);
        assert_eq!(solve("2,3,1", 10), 78);
        assert_eq!(solve("3,2,1", 10), 438);
        assert_eq!(solve("3,1,2", 10), 1836);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
