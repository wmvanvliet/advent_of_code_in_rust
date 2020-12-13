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
fn part1(input: &str) -> usize {
    let mut map:Grid = input.parse().unwrap();

    loop {
        let new_map = step1(&map);
        let n_changes = count_changes(&map, &new_map);
        if n_changes == 0 {
            break;
        }
        map = new_map;
    }

    count_n_occupied(&map)
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> usize {
    let mut map: Grid = input.parse().unwrap();

    loop {
        let new_map = step2(&map);
        let n_changes = count_changes(&map, &new_map);
        if n_changes == 0 {
            break;
        }
        map = new_map;
    }

    count_n_occupied(&map)
}

fn step1(map: &Grid) -> Grid {
    let mut new_map = map.clone();

    for pos in map.iter_pos() {
        let n_occupied = map.neighbours_diag(pos)
            .filter(|&chr| chr == '#')
            .count();

        // If a seat is empty (L) and there are no occupied seats adjacent to it, the seat
        // becomes occupied.
        if map.get(pos).unwrap() == 'L' && n_occupied == 0 {
            new_map.set(pos, '#').unwrap();
        }

        // If a seat is occupied (#) and four or more seats adjacent to it are also occupied,
        // the seat becomes empty.
        if map.get(pos).unwrap() == '#' && n_occupied >= 4 {
            new_map.set(pos, 'L').unwrap();
        }
    }
    new_map
}

fn step2(map: &Grid) -> Grid {
    let mut new_map = map.clone();

    for pos in map.iter_pos() {
        let n_occupied = Direction::DIRECTIONS_DIAG.iter()
            .filter(|dir| {
                for chr in map.look_at(pos, **dir) {
                    if chr == '#' {
                        return true;
                    } else if chr == 'L' {
                        return false;
                    }
                }
                false
            })
           .count();

        // If a seat is empty (L) and there are no occupied seats in sight of it, the seat
        // becomes occupied.
        if map.get(pos).unwrap() == 'L' && n_occupied == 0 {
            new_map.set(pos, '#').unwrap();
        }

        // If a seat is occupied (#) and five or more seats in sight of it are also occupied,
        // the seat becomes empty.
        if map.get(pos).unwrap() == '#' && n_occupied >= 5 {
            new_map.set(pos, 'L').unwrap();
        }
    }
    new_map
}

fn count_changes(map1: &Grid, map2: &Grid) -> usize {
    map1.iter_pos().filter(|pos| map1.get(*pos).unwrap() != map2.get(*pos).unwrap()).count()
}

fn count_n_occupied(map: &Grid) -> usize {
    map.cells.iter()
       .map(|line| line.iter().filter(|x| x == &&'#').count())
       .sum()
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("L.LL.LL.LL
                          LLLLLLL.LL
                          L.L.L..L..
                          LLLL.LL.LL
                          L.LL.LL.LL
                          L.LLLLL.LL
                          ..L.L.....
                          LLLLLLLLLL
                          L.LLLLLL.L
                          L.LLLLL.LL"), 37);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("L.LL.LL.LL
                          LLLLLLL.LL
                          L.L.L..L..
                          LLLL.LL.LL
                          L.LL.LL.LL
                          L.LLLLL.LL
                          ..L.L.....
                          LLLLLLLLLL
                          L.LLLLLL.L
                          L.LLLLL.LL"), 26);
    }
}
