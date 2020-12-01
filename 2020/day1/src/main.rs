use std::io::{self, Read};
use std::collections::{HashSet};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input)));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

fn part1(input: &str) -> i64 {
    let numbers:HashSet<i64> = input.lines()
        .map(|line| line.parse::<i64>().unwrap())
        .collect();
    for number in numbers.iter() {
        let other_number = 2020 - number;
        if numbers.contains(&other_number) {
            return number * other_number;
        }
    }
    panic!("No answer found!");
}

fn part2(input: &str) -> i64 {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("1721\n979\n366\n299\n675\n1456\n"), 514579);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
