#[macro_use] extern crate itertools;
mod grid;

use std::io::{self, Read};
use grid::{Grid, Direction};

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
    let mut pos: (i64, i64) = (0, 0);
    let mut dir = Direction::E;

    for line in input.lines() {
        let instr:char = line.trim().chars().nth(0).unwrap();
        let amount:i64 = line.trim()[1..].parse().unwrap();
        println!("instr: {} {}", instr, amount);

        match instr {
            'N' => pos = Direction::N.stepn(pos, amount),
            'S' => pos = Direction::S.stepn(pos, amount),
            'W' => pos = Direction::W.stepn(pos, amount),
            'E' => pos = Direction::E.stepn(pos, amount),
            'L' => dir = dir.turn_left(),
            'R' => dir = dir.turn_right(),
            'F' => pos = dir.stepn(pos, amount),
            _ => panic!("Invalid instruction"),
        }
        println!("ship: {:?}, {:?}", pos, dir);
    }

    pos.0.abs() + pos.1.abs()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut waypoint: (i64, i64) = (10, 1);
    let mut ship: (i64, i64) = (0, 0);

    for line in input.lines() {
        let instr:char = line.trim().chars().nth(0).unwrap();
        let amount:i64 = line.trim()[1..].parse().unwrap();
        println!("instr: {} {}", instr, amount);

        match instr {
            'N' => waypoint = Direction::N.stepn(waypoint, amount),
            'S' => waypoint = Direction::S.stepn(waypoint, amount),
            'W' => waypoint = Direction::W.stepn(waypoint, amount),
            'E' => waypoint = Direction::E.stepn(waypoint, amount),
            'L' => {
                match amount {
                    90 => waypoint = (-waypoint.1, waypoint.0),
                    180 => waypoint = (-waypoint.0, -waypoint.1),
                    270 => waypoint = (waypoint.1, -waypoint.0),
                    _ => panic!("Invalid direction"),
                }
            }
            'R' => {
                match amount {
                    90 => waypoint = (waypoint.1, -waypoint.0),
                    180 => waypoint = (-waypoint.0, -waypoint.1),
                    270 => waypoint = (-waypoint.1, waypoint.0),
                    _ => panic!("Invalid direction"),
                }
            }
            'F' => ship = (ship.0 + amount * waypoint.0, ship.1 + amount * waypoint.1),
            _ => panic!("Invalid instruction"),
        }
        println!("ship: {:?}  waypoint: {:?}", ship, waypoint);
    }

    ship.0.abs() + ship.1.abs()
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
