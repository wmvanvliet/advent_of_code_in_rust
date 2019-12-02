use std::io::{self, Read};
use std::collections::HashSet;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let mut seen:HashSet<[i64; 2]> = HashSet::new();
    let mut x = 0;
    let mut y = 0;

    seen.insert([0, 0]);
    for line in input.lines() {
        for ch in line.trim_end().chars() {
            match ch {
                '>' => x += 1,
                '<' => x -= 1,
                '^' => y += 1,
                'v' => y -= 1,
                _ => panic!(),
            };
            seen.insert([x, y]);
        }
    }

    return seen.len() as i64;
}

fn part2(input: &str) -> i64 {
    let mut seen:HashSet<[i64; 2]> = HashSet::new();
    let mut santa_x = 0;
    let mut santa_y = 0;
    let mut robot_x = 0;
    let mut robot_y = 0;
    let mut turn = 1;

    seen.insert([0, 0]);
    for line in input.lines() {
        for ch in line.trim_end().chars() {
            if turn == 1 {
                match ch {
                    '>' => santa_x += 1,
                    '<' => santa_x -= 1,
                    '^' => santa_y += 1,
                    'v' => santa_y -= 1,
                    _ => panic!(),
                };
                seen.insert([santa_x, santa_y]);
                turn = 2;
            } else {
                match ch {
                    '>' => robot_x += 1,
                    '<' => robot_x -= 1,
                    '^' => robot_y += 1,
                    'v' => robot_y -= 1,
                    _ => panic!(),
                };
                seen.insert([robot_x, robot_y]);
                turn = 1;
            }
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
