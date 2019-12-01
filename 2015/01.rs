use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    part1(&input);
    part2(&input);
}

fn part1(input: &str) {
    let mut floor:i64 = 0;
    for line in input.lines() {
        for direction in line.chars() {
            if direction == '(' {
                floor += 1;
            } else {
                floor -= 1;
            }
        }
    }

    println!("Part 1 answer: {}", floor);
}

fn part2(input: &str) {
    let mut floor:i64 = 0;
    let mut pos:i64 = 1;
    for line in input.lines() {
        for direction in line.chars() {
            if direction == '(' {
                floor += 1;
            } else {
                floor -= 1;
            }

            if floor == -1 {
                println!("Part 2 answer: {}", pos);
                return;
            } else {
                pos += 1;
            }
        }
    }
}
