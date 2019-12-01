use std::io::{self, Read};
use std::collections::HashSet;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    part1(&input);
    part2(&input);
}

fn part1(input: &str) {
    let mut total:i32 = 0;
    for line in input.lines() {
        let number = line.parse::<i32>().unwrap();
        total += number;
    }

    println!("Part 1 answer: {}", total);
}

fn part2(input: &str) {
    let mut total:i32 = 0;
    let mut seen = HashSet::new();
    seen.insert(0);

    loop {
        for line in input.lines() {
            let number = line.parse::<i32>().unwrap();
            total += number;
            if seen.contains(&total) {
                println!("Part 2 answer: {}", total);
                return;
            } else {
                seen.insert(total);
            }
        }
    }
}
