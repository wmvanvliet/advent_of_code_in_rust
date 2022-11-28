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
    let mut bit_count:Vec<i64> = vec![0; input.lines().nth(0).unwrap().len()];
    for line in input.lines() {
        for (val, count) in line.chars().zip(bit_count.iter_mut()) {
            if val == '1' {
                *count += 1;
            } else {
                *count -= 1;
            }
        }
    }
    let gamma_str:String = bit_count.iter().map(|&c| if c > 0 { '1' } else { '0' }).collect();
    let gamma = i64::from_str_radix(&*gamma_str, 2).unwrap();
    let epsilon_str:String = bit_count.iter().map(|&c| if c < 0 { '1' } else { '0' }).collect();
    let epsilon = i64::from_str_radix(&*epsilon_str, 2).unwrap();
    return gamma * epsilon;
}

fn seek(values: &Vec<u16>, most_frequent:bool) -> u16 {
    let mut from:usize = 0;
    let mut to:usize = values.len();
    let mut split:usize;
    for bit_pos in (0..12).rev() {
        split = from;
        for i in from..to {
            if values[i] & 1 << bit_pos == 0 {
                split += 1;
            } else {
                break;
            }
        }
        if split == to {
            continue;
        } else if split - from == (to - from) / 2 {
            if most_frequent {
                from = split;
            } else {
                to = split - 1;
            }
        } else if split - from > (to - from) / 2 {
            if most_frequent {
                to = split;
            } else {
                from = split;
            }
        } else {
            if most_frequent {
                from = split;
            } else {
                to = split;
            }
        }
        if to - from < 2 {
            break;
        }
    }
    return values[from];
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut values:Vec<u16> = Vec::new();
    for line in input.lines() {
        values.push(u16::from_str_radix(line, 2).unwrap());
    }
    values.sort();
    return seek(&values, true) as i64 * seek(&values, false) as i64;
}
 
/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("00100\n11110\n10110\n10111\n10101\n01111\n00111\n11100\n10000\n11001\n00010\n01010"), 198);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("00100\n11110\n10110\n10111\n10101\n01111\n00111\n11100\n10000\n11001\n00010\n01010"), 230);
    }
}
