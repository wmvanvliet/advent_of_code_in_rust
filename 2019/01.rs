use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    part1(&input);
    part2(&input);
}

fn part1(input: &str) {
    let mut total:i64 = 0;
    for line in input.lines() {
        let mass = line.parse::<i64>().unwrap();
        total += (mass / 3) - 2;
    }

    println!("Part 1 answer: {}", total);
}

fn part2(input: &str) {
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

    println!("Part 2 answer: {}", total);
}
