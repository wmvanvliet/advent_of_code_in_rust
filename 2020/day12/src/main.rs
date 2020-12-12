use std::io::{self, Read};

/**
 * This reads in the puzzle input from stdin. So you would call this program like:
 *     cat input | cargo run
 * It then feeds the input as a string to the functions that solve both parts of the puzzle.
 */
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input)));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

/**
 * On Windows, a unicode BOM marker is always placed at the beginning of the input file. Very
 * annoying.
 */
fn strip_bom(input: &str) -> &str {
    match input.strip_prefix("\u{feff}") {
        Some(x) => x,
        _ => input
    }
}

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> i64 {
    let mut x: i64 = 0;
    let mut y: i64 = 0;
    let mut facing: i64 = 90;

    for line in input.lines() {
        let instr:char = line.trim().chars().nth(0).unwrap();
        let amount:i64 = line.trim()[1..].parse().unwrap();
        println!("{} {}", instr, amount);

        match instr {
            'N' => y += amount,
            'S' => y -= amount,
            'W' => x -= amount,
            'E' => x += amount,
            'L' => facing = (facing - amount) % 360,
            'R' => facing = (facing + amount) % 360,
            'F' => {
                match facing {
                    0 => y += amount,
                    90 => x += amount,
                    180 => y -= amount,
                    270 => x -= amount,
                    _ => panic!("Invalid direction"),
                }
            }
            _ => panic!("Invalid instruction"),
        }
        if facing < 0 {
            facing = 360 + facing;
        }
        println!("{}, {}, {}", x, y, facing);
    }

    x.abs() + y.abs()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut waypoint_x: i64 = 10;
    let mut waypoint_y: i64 = 1;
    let mut ship_x: i64 = 0;
    let mut ship_y: i64 = 0;

    for line in input.lines() {
        let instr:char = line.trim().chars().nth(0).unwrap();
        let amount:i64 = line.trim()[1..].parse().unwrap();
        println!("{} {}", instr, amount);

        match instr {
            'N' => waypoint_y += amount,
            'S' => waypoint_y -= amount,
            'W' => waypoint_x -= amount,
            'E' => waypoint_x += amount,
            'L' => {
                match amount {
                    90 => {let (new_waypoint_x, new_waypoint_y) = (-waypoint_y, waypoint_x); waypoint_x = new_waypoint_x; waypoint_y = new_waypoint_y;}
                    180 => {let (new_waypoint_x, new_waypoint_y) = (-waypoint_x, -waypoint_y); waypoint_x = new_waypoint_x; waypoint_y = new_waypoint_y;},
                    270 => {let (new_waypoint_x, new_waypoint_y) = (waypoint_y, -waypoint_x); waypoint_x = new_waypoint_x; waypoint_y = new_waypoint_y;},
                    _ => panic!("Invalid direction"),
                }
            }
            'R' => {
                match amount {
                    90 => {let (new_waypoint_x, new_waypoint_y) = (waypoint_y, -waypoint_x); waypoint_x = new_waypoint_x; waypoint_y = new_waypoint_y;},
                    180 => {let (new_waypoint_x, new_waypoint_y) = (-waypoint_x, -waypoint_y); waypoint_x = new_waypoint_x; waypoint_y = new_waypoint_y;},
                    270 => {let (new_waypoint_x, new_waypoint_y) = (-waypoint_y, waypoint_x); waypoint_x = new_waypoint_x; waypoint_y = new_waypoint_y;},
                    _ => panic!("Invalid direction"),
                }
            }
            'F' => {
                ship_x += waypoint_x * amount;
                ship_y += waypoint_y * amount;
            }
            _ => panic!("Invalid instruction"),
        }
        println!("{}, {}    {} {}", waypoint_x, waypoint_y, ship_x, ship_y);
    }

    ship_x.abs() + ship_y.abs()
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("F10
                          N3
                          F7
                          R90
                          F11"), 25);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("F10
                          N3
                          F7
                          R90
                          F11"), 286);
    }
}
