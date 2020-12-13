use std::str::FromStr;
use std::string::ToString;
use std::fmt;
use std::ops::Range;
use std::clone::Clone;

#[derive(Debug, Clone)]
pub struct Grid {
    pub cells: Vec<Vec<char>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction { N, NE, E, SE, S, SW, W, NW }

// Error indicating number of degrees cannot be parsed to a direction
#[derive(Debug, Clone)]
pub struct InvalidDegrees;

impl Direction {
    pub const DIRECTIONS:[Direction; 4] = [Self::N, Self::E, Self::S, Self::W];
    pub const DIRECTIONS_DIAG:[Direction; 8] = [Self::N, Self::NE, Self::E, Self::SE, Self::S, Self::SW, Self::W, Self::NW];

    // Move a position one step into the direction indicated by this object.
    pub fn step(&self, pos: (i64, i64)) -> (i64, i64) {
        self.stepn(pos, 1)
    }

    // Move a position n steps into the direction indicated by this object.
    pub fn stepn(&self, pos: (i64, i64), n: i64) -> (i64, i64) {
        match self {
            Direction::N => (pos.0, pos.1 + n),
            Direction::NE => (pos.0 + n, pos.1 + n),
            Direction::E => (pos.0 + n, pos.1),
            Direction::SE => (pos.0 + n, pos.1 - n),
            Direction::S => (pos.0, pos.1 - n),
            Direction::SW => (pos.0 - n, pos.1 - n),
            Direction::W => (pos.0 - n, pos.1),
            Direction::NW => (pos.0 - n, pos.1 + n),
        }
    }

    // Make a Direction from a number of degrees. North is 0 degrees.
    pub fn from_deg(deg: i64) -> Result<Direction, InvalidDegrees> {
        match deg {
            0 => Ok(Direction::N),
            45 => Ok(Direction::NE),
            90 => Ok(Direction::E),
            135 => Ok(Direction::SE),
            180 => Ok(Direction::S),
            225 => Ok(Direction::SW),
            270 => Ok(Direction::W),
            315 => Ok(Direction::NW),
            _ => Err(InvalidDegrees),
        }
    }

    // Convert a Direction to a number of degrees. North is 0 degrees.
    pub fn to_deg(&self) -> i64 {
        match self {
            Self::N => 0,
            Self::NE => 45,
            Self::E => 90,
            Self::SE => 135,
            Self::S => 180,
            Self::SW => 225,
            Self::W => 270,
            Self::NW => 315,
        }
    }

    // Turn this direction 90 degrees to the left
    pub fn turn_left(&self) -> Direction {
        match *self {
            Self::N => Self::W,
            Self::E => Self::N,
            Self::S => Self::E,
            Self::W => Self::S,
            Self::NE => Self::NW,
            Self::SE => Self::NE,
            Self::SW => Self::SE,
            Self::NW => Self::SW,
        }
    }

    // Turn this direction 90 degrees to the right
    pub fn turn_right(&self) -> Direction {
        match *self {
            Self::N => Self::E,
            Self::E => Self::S,
            Self::S => Self::W,
            Self::W => Self::N,
            Self::NE => Self::SE,
            Self::SE => Self::SW,
            Self::SW => Self::NW,
            Self::NW => Self::NE,
        }
    }

    // Turn this direction a given amount of degrees to the right
    pub fn turn_right_deg(&self, deg: i64) -> Result<Direction, InvalidDegrees> {
        // 0 degrees is north
        let mut facing = self.to_deg();
        facing = (facing + deg) % 360;
        while facing < 0 {
            facing += 360;
        }
        Direction::from_deg(facing)
    }

    // Turn this direction a given amount of degrees to the left
    pub fn turn_left_deg(&self, deg: i64) -> Result<Direction, InvalidDegrees> {
        self.turn_right_deg(-deg)
    }
}

// Error indicating a Cell couldn't be parsed
#[derive(Debug, Clone)]
pub struct ParseError;

// Error indicating a position is out of bounds
#[derive(Debug, Clone)]
pub struct InvalidPos;

impl Grid {
    pub fn get(&self, pos: (i64, i64)) -> Result<char, InvalidPos> {
        if pos.0 < 0 || pos.1 < 0 {
            Err(InvalidPos)
        } else if pos.0 as usize >= self.width || pos.1  as usize>= self.height {
            Err(InvalidPos)
        } else {
            Ok(self.cells[pos.1 as usize][pos.0 as usize])
        }
    }

    pub fn set(&mut self, pos: (i64, i64), value: char) -> Result<(), InvalidPos> {
        if pos.0 < 0 || pos.1 < 0 {
            Err(InvalidPos)
        } else if pos.0 as usize >= self.width || pos.1 as usize >= self.height {
            Err(InvalidPos)
        } else {
            self.cells[pos.1 as usize ][pos.0 as usize ] = value;
            Ok(())
        }
    }

    pub fn iter_pos(&self) -> itertools::Product<Range<i64>, Range<i64>> {
        iproduct!(0..self.width as i64, (0..self.height as i64))
    }

    pub fn look_at(&self, pos: (i64, i64), dir: Direction) -> LookAtIterator {
        LookAtIterator { grid: &self, dir: dir, pos: pos }
    }

    pub fn neighbours(&self, pos: (i64, i64)) -> NeighboursIterator {
        NeighboursIterator { grid: &self, pos: pos, dirs: &Direction::DIRECTIONS, cur_dir: 0 }
    }

    pub fn neighbours_diag(&self, pos: (i64, i64)) -> NeighboursIterator {
        NeighboursIterator { grid: &self, pos: pos, dirs: &Direction::DIRECTIONS_DIAG, cur_dir: 0 }
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

// Iterator returned from the look_at function
pub struct LookAtIterator<'a> {
    grid: &'a Grid,
    dir: Direction,
    pos: (i64, i64),
}

// Look at all the cells in a certain direction (X-ray vision)
impl<'a> Iterator for LookAtIterator<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        self.pos = self.dir.step(self.pos);
        match self.grid.get(self.pos) {
            Ok(chr) => Some(chr),
            Err(_) => None,
        }
    }
}

// Iterator returned from the neighbours4 function
pub struct NeighboursIterator<'a> {
    grid: &'a Grid,
    pos: (i64, i64),
    dirs: &'a [Direction],
    cur_dir: usize,
}

// Iterate over all the neighbours of a cell
impl<'a> Iterator for NeighboursIterator<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        while self.cur_dir < self.dirs.len() {
            match self.grid.get(self.dirs[self.cur_dir].step(self.pos)) {
                Ok(chr) => {self.cur_dir += 1; return Some(chr)},
                Err(_) => {self.cur_dir += 1},
            }
        }
        None
    }
}
