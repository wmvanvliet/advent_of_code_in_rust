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

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> i64 {
    let mut sum_of_counts:i64 = 0;
    for group in input.split("\r\n\r\n") {
        let mut unique_answers:HashSet<char> = HashSet::new();
        for person in group.lines() {
            let unique_answers_for_person:HashSet<char> = person.trim().chars().collect();
            unique_answers.extend(&unique_answers_for_person);
        }
        sum_of_counts += unique_answers.len() as i64;
    }
    sum_of_counts
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut sum_of_counts:i64 = 0;
    for group in input.split("\r\n\r\n") {
        let mut unique_answers:HashSet<_> = HashSet::new();
        for (i, person) in group.lines().enumerate() {
            let unique_answers_for_person:HashSet<char> = person.trim().chars().collect();
            if i == 0 {
                unique_answers = unique_answers_for_person.clone();
            } else {
                unique_answers = unique_answers.intersection(&unique_answers_for_person).map(|x| *x).collect();
            }
        }
        sum_of_counts += unique_answers.len() as i64;
    }
    sum_of_counts
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1(""), 1);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("abc\r\n\r\na\r\nb\r\nc\r\n\r\nab\r\nac\r\n\r\na\r\na\r\na\r\na\r\n\r\nb"), 6);
    }
}
