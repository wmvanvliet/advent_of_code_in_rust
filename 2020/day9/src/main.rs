use std::io::{self, Read};
use std::collections::{HashSet, VecDeque};

/**
 * This reads in the puzzle input from stdin. So you would call this program like:
 *     cat input | cargo run
 * It then feeds the input as a string to the functions that solve both parts of the puzzle.
 */
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input), 25));
    println!("Part 2 answer: {}", part2(strip_bom(&input), 25));
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
fn part1(input: &str, preamble_len:usize) -> i64 {
    let numbers:Vec<i64> = input.lines()
                                .map(|x| x.trim().parse().unwrap())
                                .collect();

    let mut preamble_sums:VecDeque<HashSet<i64>> = VecDeque::new();
    for i in 0..preamble_len {
        let sums:HashSet<i64> = ((i+1)..preamble_len).map(|j| numbers[i] + numbers[j])
                                                     .collect();
        preamble_sums.push_back(sums);
    }

    for i in preamble_len..numbers.len() {
        if !preamble_sums.iter().any(|sums| sums.contains(&numbers[i])) {
            return numbers[i];
        }
        preamble_sums.pop_front();
        let preamble_numbers = &numbers[(i-preamble_len+1)..i];
        for (sums, old_num) in preamble_sums.iter_mut().zip(preamble_numbers) {
            sums.insert(numbers[i] + old_num);
        }
        preamble_sums.push_back(HashSet::new());
    }

    panic!("No solution found.");
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str, preamble_len:usize) -> i64 {
    let invalid_number = part1(input, preamble_len);
    let numbers:Vec<i64> = input.lines()
                                .map(|x| x.trim().parse().unwrap())
                                .collect();

    for i in 0..numbers.len() {
        let mut sum:i64 = numbers[i];
        for j in (i+1)..numbers.len() {
            sum += numbers[j];
            if sum == invalid_number {
                // Found the correct range of numbers. Now compute the encryption weakness.
                let min = numbers[i..=j].iter().min().unwrap();
                let max = numbers[i..=j].iter().max().unwrap();
                return min + max;
            } else if sum > invalid_number {
                // We overshot. Don't bother adding on even more numbers.
                break;
            }
        }
    }
    
    panic!("No solution found.");
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("35
                          20
                          15
                          25
                          47
                          40
                          62
                          55
                          65
                          95
                          102
                          117
                          150
                          182
                          127
                          219
                          299
                          277
                          309
                          576", 5), 127);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("35
                          20
                          15
                          25
                          47
                          40
                          62
                          55
                          65
                          95
                          102
                          117
                          150
                          182
                          127
                          219
                          299
                          277
                          309
                          576", 5), 62);
    }
}
