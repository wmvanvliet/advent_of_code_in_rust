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
    match input.strip_prefix("\u{feff}") {
        Some(x) => x,
        _ => input
    }
}

/**
 * Solves part 1 of the puzzle. We iterate over all the numbers by using the iterator produced by
 * input.lines() directly. No need to load them into a Vec first. At every iteration, we compare
 * the current number against the previous number.
 */
fn part1(input: &str) -> i64 {
    let numbers = input.lines()
        .map(|line| line.parse::<i64>().unwrap());
    let mut numbers_iter = numbers.into_iter();
    let mut answer:i64 = 0;
    let mut prev_depth:i64 = numbers_iter.next().unwrap();
    for current_depth in numbers_iter {
        if current_depth > prev_depth {
            answer += 1;
        }
        prev_depth = current_depth;
    }
    return answer;
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let numbers:Vec<i64> = input.lines()
        .map(|line| line.parse::<i64>().unwrap())
        .collect();

    let mut answer:i64 = 0;

    for i in 3..numbers.len() {
        if numbers[i] > numbers[i - 3] {
            answer += 1;
        }
    }
    return answer;
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("199\n200\n208\n210\n200\n207\n240\n269\n260\n263"), 7);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("199\n200\n208\n210\n200\n207\n240\n269\n260\n263"), 5);
    }
}
