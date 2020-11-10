use std::io::{self, Read};
use std::collections::HashSet;
use regex::Regex;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input)));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

fn part1(input: &str) -> i64 {
    // Parse the (x, y, z) positions of the moons
    let mut pos_x:[i64; 4] = [0; 4];
    let mut pos_y:[i64; 4] = [0; 4];
    let mut pos_z:[i64; 4] = [0; 4];
    for (i, line) in input.lines().enumerate() {
        let re = Regex::new(r"<x=(-?\d+), y=(-?\d+), z=(-?\d+)>").unwrap();
        for cap in re.captures_iter(line) {
            pos_x[i] = cap[1].parse::<i64>().unwrap();
            pos_y[i] = cap[2].parse::<i64>().unwrap();
            pos_z[i] = cap[3].parse::<i64>().unwrap();
        }
    }

    // Do the simulation
    let x = System::new(pos_x).take(1000).last().unwrap();
    let y = System::new(pos_y).take(1000).last().unwrap();
    let z = System::new(pos_z).take(1000).last().unwrap();
    compute_energy(&x, &y, &z)
}

fn part2(input: &str) -> i64 {
    // Parse the (x, y, z) positions of the moons
    let mut pos_x:[i64; 4] = [0; 4];
    let mut pos_y:[i64; 4] = [0; 4];
    let mut pos_z:[i64; 4] = [0; 4];
    for (i, line) in input.lines().enumerate() {
        let re = Regex::new(r"<x=(-?\d+), y=(-?\d+), z=(-?\d+)>").unwrap();
        for cap in re.captures_iter(line) {
            pos_x[i] = cap[1].parse::<i64>().unwrap();
            pos_y[i] = cap[2].parse::<i64>().unwrap();
            pos_z[i] = cap[3].parse::<i64>().unwrap();
        }
    }

    let loop_x = detect_loop(System::new(pos_x));
    let loop_y = detect_loop(System::new(pos_y));
    let loop_z = detect_loop(System::new(pos_z));
    lcd(lcd(loop_x, loop_y), loop_z)
}

fn sign(x:i64) -> i64 {
    if x.is_positive() {
        1
    } else if x.is_negative() {
        -1
    } else {
        0
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct System {
    pos:[i64; 4],
    vel:[i64; 4],
}

impl System {
    fn new(pos:[i64; 4]) -> System {
        System {
            pos: pos,
            vel: [0; 4],
        }
    }
}

/// Generate steps in the simulation
impl Iterator for System {
    type Item = System;
    fn next(&mut self) -> Option<System> {
        let mut acc:[i64; 4] = [0, 0, 0, 0];
        for (i, p1) in self.pos.iter().enumerate() {
            for p2 in self.pos.iter() {
                acc[i] += sign(p2 - p1);
            }
        }
        for (v, a) in self.vel.iter_mut().zip(&acc) { *v += a };
        for (p, v) in self.pos.iter_mut().zip(&self.vel) { *p += v };
        Some(self.clone())
    }
}

/// Compute total energy in the system
fn compute_energy(x:&System, y:&System, z:&System) -> i64 {
    let pot = x.pos.iter().zip(&y.pos).zip(&z.pos)
        .map(|((pos_x, pos_y), pos_z)| pos_x.abs() + pos_y.abs() + pos_z.abs());
    let kin = x.vel.iter().zip(&y.vel).zip(&z.vel)
        .map(|((vel_x, vel_y), vel_z)| vel_x.abs() + vel_y.abs() + vel_z.abs());
    pot.zip(kin).map(|(a, b)| a * b).sum()
}

/// Run the simulation until we detect that we've been here before
fn detect_loop(sys:System) -> i64 {
    let mut visited_states:HashSet<System> = HashSet::new();
    for (i, state) in sys.enumerate() {
        if visited_states.contains(&state) {
            return i as i64;
        } else {
            visited_states.insert(state);
        }
    }
    unreachable!();
}

/// Find the least common denominator (LCD) using Euclids method
fn lcd(a:i64, b:i64) -> i64 {
    fn gcd(a:i64, b:i64) -> i64 {
        if a == 0 {
            b
        } else {
            gcd(b % a, a)
        }
    }
    (a * b) / gcd(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let x = System::new([-1, 2, 4, 3]).take(10).last().unwrap();
        let y = System::new([0, -10, -8, 5]).take(10).last().unwrap();
        let z = System::new([2, -7, 8, -1]).take(10).last().unwrap();
        assert_eq!(x.pos, [2, 1, 3, 2]);
        assert_eq!(y.pos, [1, -8, -6, 0]);
        assert_eq!(z.pos, [-3, 0, 1, 4]);
        assert_eq!(x.vel, [-3, -1, 3, 1]);
        assert_eq!(y.vel, [-2, 1, 2, -1]);
        assert_eq!(z.vel, [1, 3, -3, -1]);
        assert_eq!(compute_energy(&x, &y, &z), 179);

        let x = System::new([-8, 5, 2, 9]).take(100).last().unwrap();
        let y = System::new([-10, 5, -7, -8]).take(100).last().unwrap();
        let z = System::new([0, 10, 3, -3]).take(100).last().unwrap();
        assert_eq!(x.pos, [8, 13, -29, 16]);
        assert_eq!(y.pos, [-12, 16, -11, -13]);
        assert_eq!(z.pos, [-9, -3, -1, 23]);
        assert_eq!(x.vel, [-7, 3, -3, 7]);
        assert_eq!(y.vel, [3, -11, 7, 1]);
        assert_eq!(z.vel, [0, -5, 4, 1]);
        assert_eq!(compute_energy(&x, &y, &z), 1940);
    }

    #[test]
    fn test_part2() {
        let loop_x = detect_loop(System::new([-1, 2, 4, 3]));
        let loop_y = detect_loop(System::new([0, -10, -8, 5]));
        let loop_z = detect_loop(System::new([2, -7, 8, -1]));
        assert_eq!(lcd(lcd(loop_x, loop_y), loop_z), 2772);
    }

    #[test]
    fn test_lcd() {
        assert_eq!(lcd(6, 8), 24);
        assert_eq!(lcd(lcd(6, 8), 15), 120);
    }
}
