use std::io::{self, Read};

// This reads in the puzzle input from stdin. So you would call this program like:
//     cat input | cargo run
// It then feeds the input as a string to the functions that solve both parts of the puzzle.
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input)));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

// On Windows, a unicode BOM marker is always placed at the beginning of the input file. Very
// annoying. This function strips it out.
fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

// Solves part 1 of the puzzle.
fn part1(input: &str) -> i64 {
    1
}

// Solves part 2 of the puzzle.
fn part2(input: &str) -> i64 {
    2
}

// Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1(""), 1);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
