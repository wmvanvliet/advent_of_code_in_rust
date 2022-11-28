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
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> i64 {
    let mut horizontal_position:i64 = 0;
    let mut depth:i64 = 0;
    for line in input.lines() {
        let val:u8 = line.chars().last().unwrap() as u8 - b'0' as u8;
        match line.chars().nth(0) {
            Some('f') => { horizontal_position += val as i64 },
            Some('u') => { depth -= val as i64 },
            Some('d') => { depth += val as i64 },
            _ => panic!("Invalid line: {}", line),
        }
    }
    return horizontal_position * depth;
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut aim:i64 = 0;
    let mut horizontal_position:i64 = 0;
    let mut depth:i64 = 0;
    for line in input.lines() {
        let val:u8 = line.chars().last().unwrap() as u8 - b'0' as u8;
        match line.chars().nth(0) {
            Some('f') => { horizontal_position += val as i64; depth += aim * val as i64 },
            Some('u') => { aim -= val as i64 },
            Some('d') => { aim += val as i64 },
            _ => panic!("Invalid line: {}", line),
        }
    }
    return horizontal_position * depth;
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("forward 5\ndown 5\nforward 8\nup 3\ndown 8\nforward 2"), 150);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("forward 5\ndown 5\nforward 8\nup 3\ndown 8\nforward 2"), 900);
    }
}
