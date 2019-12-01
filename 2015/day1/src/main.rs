use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let mut floor:i64 = 0;
    for line in input.lines() {
        for ch in line.trim_end().chars() {
            match ch {
                '(' => floor += 1,
                ')' => floor -= 1,
                _ => panic!()
            }
        }
    }
    return floor;
}

fn part2(input: &str) -> usize {
    let mut floor:i64 = 0;
    for line in input.lines() {
        for (pos, ch) in line.trim_end().chars().enumerate() {
            match ch {
                '(' => floor += 1,
                ')' => floor -= 1,
                _ => panic!()
            }

            if floor == -1 { return pos + 1; }
        }
    }
    panic!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("(())"), 0);
        assert_eq!(part1("()()"), 0);
        assert_eq!(part1("((("), 3);
        assert_eq!(part1("(()(()("), 3);
        assert_eq!(part1("))((((("), 3);
        assert_eq!(part1("())"), -1);
        assert_eq!(part1("))("), -1);
        assert_eq!(part1(")))"), -3);
        assert_eq!(part1(")())())"), -3);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(")"), 1);
        assert_eq!(part2("()())"), 5);
    }
}
