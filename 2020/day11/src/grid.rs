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
pub struct Direction {
    dx: i64,
    dy: i64,
}

impl Direction {
    pub const NORTH:Direction = Direction { dx: 0, dy: -1 };
    pub const NORTHEAST:Direction = Direction { dx: 1, dy: -1 };
    pub const EAST:Direction = Direction { dx: 1, dy: 0 };
    pub const SOUTHEAST:Direction = Direction { dx: 1, dy: 1 };
    pub const SOUTH:Direction = Direction { dx: 0, dy: 1 };
    pub const SOUTHWEST:Direction = Direction { dx: -1, dy: 1 };
    pub const WEST:Direction = Direction { dx: -1, dy: 0 };
    pub const NORTHWEST:Direction = Direction { dx: -1, dy: -1 };
    pub const DIRECTIONS:[Direction; 4] = [Self::NORTH, Self::EAST, Self::SOUTH, Self::WEST];
    pub const DIRECTIONSDIAG:[Direction; 8] = [Self::NORTH, Self::NORTHEAST, Self::EAST, Self::SOUTHEAST, Self::SOUTH, Self::SOUTHWEST, Self::WEST, Self::NORTHWEST];

    // Move a position one step into the direction indicated by this object.
    pub fn step(&self, pos: (i64, i64)) -> (i64, i64) {
        (pos.0 + self.dx, pos.1 + self.dy)
    }

    // Move a position n steps into the direction indicated by this object.
    pub fn stepn(&self, pos: (i64, i64), n: usize) -> (i64, i64) {
        (pos.0 + n as i64 * self.dx, pos.1 + n as i64 * self.dy)
    }

    // Turn this direction 90 degrees to the left
    pub fn turn_left(&self) -> Direction {
        match *self {
            Self::NORTH => Self::WEST,
            Self::EAST => Self::NORTH,
            Self::SOUTH => Self::EAST,
            Self::WEST => Self::SOUTH,
            Self::NORTHEAST => Self::NORTHWEST,
            Self::SOUTHEAST => Self::NORTHEAST,
            Self::SOUTHWEST => Self::SOUTHEAST,
            Self::NORTHWEST => Self::SOUTHWEST,
            _ => panic!("Cannot turn an invalid direction")
        }
    }

    // Turn this direction 90 degrees to the right
    pub fn turn_right(&self) -> Direction {
        match *self {
            Self::NORTH => Self::EAST,
            Self::EAST => Self::SOUTH,
            Self::SOUTH => Self::WEST,
            Self::WEST => Self::NORTH,
            Self::NORTHEAST => Self::SOUTHEAST,
            Self::SOUTHEAST => Self::SOUTHWEST,
            Self::SOUTHWEST => Self::NORTHWEST,
            Self::NORTHWEST => Self::NORTHEAST,
            _ => panic!("Cannot turn an invalid direction")
        }
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

    pub fn neighbours4(&self, pos: (i64, i64)) -> NeighboursIterator {
        NeighboursIterator { grid: &self, pos: pos, dirs: &Direction::DIRECTIONS, cur_dir: 0 }
    }

    pub fn neighbours8(&self, pos: (i64, i64)) -> NeighboursIterator {
        NeighboursIterator { grid: &self, pos: pos, dirs: &Direction::DIRECTIONSDIAG, cur_dir: 0 }
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
