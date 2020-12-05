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
    input.lines().map(|ticket| parse_ticket(ticket)).max().unwrap()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let seats:Vec<i64> = input.lines().map(|ticket| parse_ticket(ticket)).collect();
    let seats_sum:i64 = seats.iter().sum();

    // Compute the sum of the seats if all of them were present */
    let seats_min = seats.iter().min().unwrap();
    let seats_max = seats.iter().max().unwrap();
    let should_be = euler_sum(*seats_max) - euler_sum(seats_min - 1);

    // Our seat!
    should_be - seats_sum
}

/**
 * Binary search? Nah, binary number!
 */
fn parse_ticket(ticket: &str) -> i64 {
    let row_str = ticket[..7].replace('F', "0").replace('B', "1");
    let row = u64::from_str_radix(&row_str, 2).unwrap();
    let col_str = ticket[7..].replace('L', "0").replace('R', "1");
    let col = u64::from_str_radix(&col_str, 2).unwrap();
    (row * 8 + col) as i64
}

/**
 * Quicky compute the sum of 1..n
 */
fn euler_sum(n:i64) -> i64 {
    n * (n + 1) / 2
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("FBFBBFFRLR"), 357);
        assert_eq!(part1("BFFFBBFRRR"), 567);
        assert_eq!(part1("FFFBBBFRRR"), 119);
        assert_eq!(part1("BBFFBBFRLL"), 820);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
