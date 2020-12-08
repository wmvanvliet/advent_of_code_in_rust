use std::io::{self, Read};
use regex::Regex;

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
    match input.strip_prefix("\u{feff}") {
        Some(x) => x,
        _ => input
    }
}

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> i64 {
    let mut n_correct_passwords:i64 = 0;
    let re = Regex::new(r"^(\d+)-(\d+) ([a-z]): ([a-z]*)$").unwrap();
    for line in input.lines() {
        // Parse input line
        let caps = re.captures(line).unwrap();
        let min_count = caps.get(1).unwrap().as_str().parse::<usize>().unwrap();
        let max_count = caps.get(2).unwrap().as_str().parse::<usize>().unwrap();
        let letter = caps.get(3).unwrap().as_str().parse::<char>().unwrap();
        let password = caps.get(4).unwrap().as_str();

        // Count the number of times the given letter occurs in the password
        let n_matches = password.matches(letter).count();

        // Check the password policy
        if min_count <= n_matches && n_matches <= max_count {
            n_correct_passwords += 1;
        }
    }
    n_correct_passwords
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut n_correct_passwords:i64 = 0;
    let re = Regex::new(r"^(\d+)-(\d+) ([a-z]): ([a-z]*)$").unwrap();
    for line in input.lines() {
        // Parse the input line
        let caps = re.captures(line).unwrap();
        let pos1 = caps.get(1).unwrap().as_str().parse::<usize>().unwrap();
        let pos2 = caps.get(2).unwrap().as_str().parse::<usize>().unwrap();
        let letter = caps.get(3).unwrap().as_str().parse::<char>().unwrap();
        let password:Vec<char> = caps.get(4).unwrap().as_str().chars().collect();

        // Check the password policy
        if (password[pos1 - 1] == letter) ^ (password[pos2 - 1] == letter) {
            n_correct_passwords += 1;
        }
    }
    n_correct_passwords
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("1-3 a: abcde\n1-3 b: cdefg\n2-9 c: ccccccccc"), 2);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("1-3 a: abcde"), 1);
        assert_eq!(part2("1-3 b: cdefg"), 0);
        assert_eq!(part2("2-9 c: ccccccccc"), 0);
    }
}
