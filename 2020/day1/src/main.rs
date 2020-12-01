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
 *
 * We loop over all the numbers in the input. For each number, we know the other number that should
 * be present if their sum is to be 2020, namely `2020 - number`. So, the key to this puzzle is to
 * quickly check whether a number is present in the input. A HashSet seems like a great
 * datastructure for this.
 */
fn part1(input: &str) -> i64 {
    let numbers:HashSet<i64> = input.lines()
        .map(|line| line.parse::<i64>().unwrap())
        .collect();
    for number1 in numbers.iter() {
        let number2 = 2020 - number1;
        if number2 == *number1 {
            // Numbers must be unique
            continue;
        }
        if numbers.contains(&number2) {
            // Found solution!
            return number1 * number2;
        }
    }
    panic!("No answer found!");
}

/**
 * Solves part 2 of the puzzle.
 *
 * To find three numbers that add to 2020, we could just loop more. However, lets be a bit more
 * clever. If we sort all the numbers, we can ensure that number1 < number2 < number3. That allows
 * us to stop the loop early when 2020 - (number1 + number2) < number2.
 *
 * Turns out, this early stopping strategy is wholly unnecessary, as the solution is already found
 * on iteration 3, but oh well.
 */
fn part2(input: &str) -> i64 {
    // We store all the numbers in increasing order
    let mut numbers_ordered:Vec<i64> = input.lines()
        .map(|line| line.parse::<i64>().unwrap())
        .collect();
    numbers_ordered.sort();
    println!("{:?}", numbers_ordered);

    // For quick lookups for number3
    let numbers_set:HashSet<i64> = input.lines()
        .map(|line| line.parse::<i64>().unwrap())
        .collect();

    // The big loop over candidates number1, number2, number3
    for (i, number1) in numbers_ordered.iter().enumerate() {
        if 2020 - number1 < 2 * number1 {
            // We know that number2 and number3 will be larger than number1. They cannot fit
            // anymore between number1 and 2020. We are doomed!
            panic!("No answer found!");
        }
        for number2 in numbers_ordered[i+1..].iter() {
            let number3 = 2020 - (number1 + number2);
            if &number3 < number2 {
                // No solution possible for this combination of number1 and number2. We already
                // tried all the numbers smaller than number2.
                break;
            }
            if &number3 == number1 || &number3 == number2 {
                // Numbers must be unique
                continue;
            }
            if numbers_set.contains(&number3) {
                return number1 * number2 * number3;
            }

        }
    }
    // Dooooomed!
    panic!("No answer found!");
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("1721\n979\n366\n299\n675\n1456\n"), 514579);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("1721\n979\n366\n299\n675\n1456\n"), 241861950);
    }
}
