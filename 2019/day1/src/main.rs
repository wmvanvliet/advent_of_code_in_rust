use std::io::{self, Read};
use std::iter;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    // Input is a string of numbers. Each number represents
    // the weight of a single module.
    let module_weights:Vec<i64> = input.lines()
        .map(|s| s.parse::<i64>().unwrap())
        .collect();

    // Compute total fuel requirements
    module_weights.iter()
        .map(|&weight| required_fuel(weight))
        .sum()
}

fn part2(input: &str) -> i64 {
    // Input is a string of numbers. Each number represents
    // the weight of a single module.
    let module_weights:Vec<i64> = input.lines()
        .map(|s| s.parse::<i64>().unwrap())
        .collect();

    // Compute total fuel requirements
    module_weights.iter()
        .map(|&weight| rocket_eq(weight))
        .sum()
}

/// Compute fuel to transport a module with a given weight
fn required_fuel(weight:i64) -> i64 {
    0.max(weight / 3 - 2)
}

/// Compute fuel to transport both a module and its required fuel
fn rocket_eq(weight:i64) -> i64 {
    iter::successors(Some(required_fuel(weight)), |&weight| {
        if weight <= 0 {
            None
        } else {
            Some(required_fuel(weight))
        }
    }).sum()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("12"), 2);
        assert_eq!(part1("14"), 2);
        assert_eq!(part1("1969"), 654);
        assert_eq!(part1("100756"), 33583);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("14"), 2);
        assert_eq!(part2("1969"), 966);
        assert_eq!(part2("100756"), 50346);
    }
}
