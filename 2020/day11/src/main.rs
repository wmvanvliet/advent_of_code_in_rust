use std::io::{self, Read};
use std::collections::HashMap;
use std::str::FromStr;
use std::string::ToString;
use std::fmt;
use std::ops::Range;
use std::clone::Clone;
use itertools::Itertools;

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
        let new_map = step(&map);
        let n_changes = count_changes(&map, &new_map);
        println!("n_changes: {}", n_changes);
        if n_changes == 0 {
            break;
        }
        map = new_map;
    }

    count_n_occupied(&map)
}

fn step(map: &Grid) -> Grid {
    let mut new_map = map.clone();

    for pos in map.iter_pos() {
        let n_occupied = Direction::DirectionsDiag.iter()
            .filter(|dir| map.get(dir.step(pos)).unwrap_or('~') == '#')
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

fn count_changes(map1: &Grid, map2: &Grid) -> usize {
    map1.iter_pos().filter(|pos| map1.get(*pos).unwrap() != map2.get(*pos).unwrap()).count()
}

fn count_n_occupied(map: &Grid) -> usize {
    map.cells.iter()
       .map(|line| line.iter().filter(|x| x == &&'#').count())
       .sum()
}

#[derive(Debug, Clone)]
struct Grid {
    cells: Vec<Vec<char>>,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Direction {
    dx: i64,
    dy: i64,
}

impl Direction {
    const North:Direction = Direction { dx: 0, dy: -1 };
    const NorthEast:Direction = Direction { dx: 1, dy: -1 };
    const East:Direction = Direction { dx: 1, dy: 0 };
    const SouthEast:Direction = Direction { dx: 1, dy: 1 };
    const South:Direction = Direction { dx: 0, dy: 1 };
    const SouthWest:Direction = Direction { dx: -1, dy: 1 };
    const West:Direction = Direction { dx: -1, dy: 0 };
    const NorthWest:Direction = Direction { dx: -1, dy: -1 };
    const Directions:[Direction; 4] = [Self::North, Self::East, Self::South, Self::West];
    const DirectionsDiag:[Direction; 8] = [Self::North, Self::NorthEast, Self::East, Self::SouthEast, Self::South, Self::SouthWest, Self::West, Self::NorthWest];

    // Move a position one step into the direction indicated by this object.
    fn step(&self, pos: (i64, i64)) -> (i64, i64) {
        (pos.0 + self.dx, pos.1 + self.dy)
    }

    // Move a position n steps into the direction indicated by this object.
    fn stepn(&self, pos: (i64, i64), n: usize) -> (i64, i64) {
        (pos.0 + n as i64 * self.dx, pos.1 + n as i64 * self.dy)
    }

    // Turn this direction 90 degrees to the left
    fn turn_left(&self) -> Direction {
        match *self {
            Self::North => Self::West,
            Self::East => Self::North,
            Self::South => Self::East,
            Self::West => Self::South,
            Self::NorthEast => Self::NorthWest,
            Self::SouthEast => Self::NorthEast,
            Self::SouthWest => Self::SouthEast,
            Self::NorthWest => Self::SouthWest,
            _ => panic!("Cannot turn an invalid direction")
        }
    }

    // Turn this direction 90 degrees to the right
    fn turn_right(&self) -> Direction {
        match *self {
            Self::North => Self::East,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
            Self::NorthEast => Self::SouthEast,
            Self::SouthEast => Self::SouthWest,
            Self::SouthWest => Self::NorthWest,
            Self::NorthWest => Self::NorthEast,
            _ => panic!("Cannot turn an invalid direction")
        }
    }
}

// Error indicating a Cell couldn't be parsed
#[derive(Debug, Clone)]
struct ParseError;

// Error indicating a position is out of bounds
#[derive(Debug, Clone)]
struct InvalidPos;

#[macro_use] extern crate itertools;
impl Grid {
    fn get(&self, pos: (i64, i64)) -> Result<char, InvalidPos> {
        if pos.0 < 0 || pos.1 < 0 {
            Err(InvalidPos)
        } else if pos.0 as usize >= self.width || pos.1  as usize>= self.height {
            Err(InvalidPos)
        } else {
            Ok(self.cells[pos.1 as usize][pos.0 as usize])
        }
    }

    fn set(&mut self, pos: (i64, i64), value: char) -> Result<(), InvalidPos> {
        if pos.0 < 0 || pos.1 < 0 {
            Err(InvalidPos)
        } else if pos.0 as usize >= self.width || pos.1 as usize >= self.height {
            Err(InvalidPos)
        } else {
            self.cells[pos.1 as usize ][pos.0 as usize ] = value;
            Ok(())
        }
    }

    fn iter_pos(&self) -> itertools::Product<Range<i64>, Range<i64>> {
        iproduct!(0..self.width as i64, (0..self.height as i64))
    }
}


// Parse a string into a Cell
impl FromStr for Grid {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cells: Vec<Vec<char>> = s.lines()
                                     .map(|line| line.trim().chars().collect())
                                     .collect();

        if cells.len() == 0 {
            return Err(ParseError);
        }

        let width = cells[0].len();
        let height = cells.len();

        Ok(Grid {
            cells: cells,
            width: width,
            height: height,
        })
    }
}

// Standard display function for printing the grid
impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for line in self.cells.iter() {
            for cell in line {
                s.push_str(&cell.to_string());
            }
            s.push_str("\n");
        }
        write!(f, "{}", s)
    }
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let grid: Grid = input.parse().unwrap();
    println!("{}", grid);
    2
}

/*
fn step(map: &Grid) -> Grid {
    let directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];

    let mut new_map = map.clone();

    for y in 0..map.height {
        let mut new_line: Vec<char> = Vec::new();
        for x in 0..map.width {
            let mut n_occupied = 0;
            for (dy, dx) in &directions {
                let new_y = y as i64 + dy;
                let new_x = x as i64 + dx;
                if new_y < 0 || new_y >= map_height {
                    continue;
                }
                if new_x < 0 || new_x >= map_width {
                    continue;
                }
                if map[new_y as usize][new_x as usize] == '#' {
                    n_occupied += 1;
                }
            }

            // If a seat is empty (L) and there are no occupied seats adjacent to it, the seat
            // becomes occupied.
            if map[y as usize][x as usize] == 'L' && n_occupied == 0 {
                new_line.push('#');
            }
            // If a seat is occupied (#) and four or more seats adjacent to it are also occupied,
            // the seat becomes empty.
            else if map[y as usize][x as usize] == '#' && n_occupied >= 4 {
                new_line.push('L');
            }
            else {
                new_line.push(map[y as usize][x as usize]);
            }

        }
        new_map.push(new_line);
    }
    new_map
}
*/

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
