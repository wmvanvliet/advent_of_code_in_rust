use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let mut total:i64 = 0;
    for line in input.lines() {
        let mass = line.parse::<i64>().unwrap();
        total += (mass / 3) - 2;
    }

    return total;
}

fn part2(input: &str) -> i64 {
    fn rocket_eq(mass:i64) -> i64 {
        if mass <= 0 { return 0; }
        let mut fuel = (mass / 3) - 2;
        if fuel <= 0 { return 0; }
        fuel += rocket_eq(fuel);
        return fuel;
    }

    let mut total:i64 = 0;
    for line in input.lines() {
        let mass = line.parse::<i64>().unwrap();
        total += rocket_eq(mass);
    }

    return total;
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
