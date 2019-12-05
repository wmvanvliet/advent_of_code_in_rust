use std::io::{self, Read};
use std::collections::HashSet;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let mut seen:HashSet<(i64, i64)> = HashSet::new();
    let mut pos = (0, 0);

    seen.insert((0, 0));
    for line in input.lines() {
        for ch in line.trim_end().chars() {
            match ch {
                '>' => pos.0 += 1,
                '<' => pos.0 -= 1,
                '^' => pos.1 += 1,
                'v' => pos.1 -= 1,
                _ => panic!(),
            };
            seen.insert(pos);
        }
    }

    return seen.len() as i64;
}

fn part2(input: &str) -> i64 {
    let mut seen:HashSet<(i64, i64)> = HashSet::new();
    let mut pos = [(0, 0), (0, 0)];

    seen.insert((0, 0));
    for line in input.lines() {
        for (i, ch) in line.trim_end().chars().enumerate() {
            match ch {
                '>' => pos[i % 2].0 += 1,
                '<' => pos[i % 2].0 -= 1,
                '^' => pos[i % 2].1 += 1,
                'v' => pos[i % 2].1 -= 1,
                _ => panic!(),
            };
            seen.insert(pos[i % 2]);
        }
    }

    return seen.len() as i64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1(">"), 2);
        assert_eq!(part1("^>v<"), 4);
        assert_eq!(part1("^v^v^v^v^v"), 2);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("^v"), 3);
        assert_eq!(part2("^>v<"), 3);
        assert_eq!(part2("^v^v^v^v^v"), 11);
    }
}
