use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    part1(&input);
    //part2(&input);
}

fn part1(input: &str) {
    let mut twos = 0;
    let mut threes = 0;

    for line in input.lines() {
        let mut chars: Vec<char> = line.chars().collect();
        chars.sort(); //_by(|a, b| b.cmp(a));

        let mut prev_char = ' ';
        let mut count = 1;
        let mut two_present = false;
        let mut three_present = false;

        for (i, &c) in chars.iter().enumerate() {
            if c == prev_char {
                count += 1;
            }
            if c != prev_char || i == chars.len() - 1 {
                prev_char = c;
                if count >= 3 {
                    three_present = true;
                } else if count == 2 {
                    two_present = true;
                }
                count = 1;
            }
        }
        if two_present {
            twos += 1;
        }
        if three_present {
            threes += 1;
        }
    }

    println!("Part 1 answer: {}", twos * threes);
}
