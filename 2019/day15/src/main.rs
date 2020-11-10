use std::io::{self, Read};
use std::collections::VecDeque;
use std::collections::HashMap;
use std::collections::HashSet;

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
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse::<i64>().expect("Could not parse IntCode program.") })
        .collect();

    let mut robot = Robot::new(&program);
    explore_map(&mut robot);
    let oxygen_loc = robot.oxygen_loc.expect("No oxygen found??");
    print_map(&robot.map);
    return find_shortest_path(&robot.map, oxygen_loc)
}

fn part2(input: &str) -> i64 {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    let mut robot = Robot::new(&program);
    explore_map(&mut robot);
    let oxygen_loc = robot.oxygen_loc.expect("No oxygen found??");
    return oxygen_spread(&robot.map, oxygen_loc);
}

fn explore_map(robot: &mut Robot) {
    for dir in [Direction::North, Direction::South, Direction::West, Direction::East].iter() {
        if robot.map.contains_key(&dir.apply(robot.position)) {
            continue;
        }
        let result = robot.try_move(*dir);
        if result == 2 {
            println!("Oxygen found at {:?}!", robot.position);
            robot.try_move(opposite(*dir));
        } else if result == 1 {
            explore_map(robot);
            robot.try_move(opposite(*dir));
        }
    }
}

fn find_shortest_path(map: &HashMap<(i64, i64), i64>, target: (i64, i64)) -> i64 {
    println!("Finding shortest path to {:?}", target);
    let mut visited:HashSet<(i64, i64)> = HashSet::new();
    let mut frontier:VecDeque<(i64, i64, i64)> = VecDeque::new();
    frontier.push_back((0, 0, 0));
    while frontier.len() > 0 {
        let entry = frontier.pop_front().unwrap();
        let loc = (entry.0, entry.1);
        let n_steps = entry.2;
        visited.insert(loc);
        for dir in [Direction::North, Direction::South, Direction::West, Direction::East].iter() {
            let next_loc = dir.apply(loc);
            if visited.contains(&next_loc) {
                continue;
            }
            if map[&next_loc] == 2 {
                return n_steps + 1;
            } else if map[&next_loc] == 1 {
                frontier.push_back((next_loc.0, next_loc.1, n_steps + 1));
            }
        }
    }

    0
}

fn oxygen_spread(map: &HashMap<(i64, i64), i64>, oxygen_loc: (i64, i64)) -> i64 {
    println!("Computing oxygen spread, with oxygen at location {:?}", oxygen_loc);
    let mut visited:HashSet<(i64, i64)> = HashSet::new();
    let mut frontier:VecDeque<(i64, i64, i64)> = VecDeque::new();
    let mut n_steps = 0;
    frontier.push_back((0, 0, 0));
    while frontier.len() > 0 {
        let entry = frontier.pop_front().unwrap();
        let loc = (entry.0, entry.1);
        n_steps = entry.2;
        visited.insert(loc);
        for dir in [Direction::North, Direction::South, Direction::West, Direction::East].iter() {
            let next_loc = dir.apply(loc);
            if visited.contains(&next_loc) {
                continue;
            } else if map[&next_loc] == 1 {
                frontier.push_back((next_loc.0, next_loc.1, n_steps + 1));
            }
        }
    }

    n_steps
}

fn print_map(map: &HashMap<(i64, i64), i64>) {
    for y in -30..30 {
        for x in -30..30 {
            if y == 0 && x == 0 {
                print!("0");
            } else if map.contains_key(&(x, y)) {
                match map.get(&(x, y)) {
                    None => print!(" "),
                    Some(0) => print!("#"),
                    Some(1) => print!("."),
                    Some(2) => print!("X"),
                    _ => print!(" "),
                }
            } else {
                print!(" ");
            }
        }
        print!("\n");
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
enum Direction {
    North = 1,
    South,
    West,
    East,
}

impl Direction {
    fn apply(&self, position: (i64, i64)) -> (i64, i64) {
        match self {
            Direction::North => (position.0, position.1 + 1),
            Direction::South => (position.0, position.1 - 1),
            Direction::West => (position.0 - 1, position.1),
            Direction::East => (position.0 + 1, position.1),
        }
    }
}

fn opposite(dir: Direction) -> Direction {
    match dir {
        Direction::North => Direction::South,
        Direction::South => Direction::North,
        Direction::West => Direction::East,
        Direction::East => Direction::West,
    }
}

struct Robot {
    computer: IntCode,
    position: (i64, i64),
    map: HashMap<(i64, i64), i64>,
    oxygen_loc: Option<(i64, i64)>,
}

impl Robot {
    fn new(program: &Vec<i64>) -> Self {
        Robot {
            computer: IntCode::new(program),
            position: (0, 0),
            map: HashMap::new(),
            oxygen_loc: None,
        }
    }

    fn try_move(&mut self, dir:Direction) -> i64 {
        self.computer.feed_input(dir as i64);
        let result = match self.computer.next() {
            Some(v) => v,
            None => match self.computer.status {
                StatusCode::Ready => panic!("Received None for no reason."),
                StatusCode::InputRequested => panic!("More input requested."),
                StatusCode::Halted => panic!("End of program."),
            }
        };

        self.map.insert(dir.apply(self.position), result);
        if result == 1 || result == 2 {
            self.position = dir.apply(self.position);
        }
        if result == 2 {
            self.oxygen_loc = Some(self.position);
        }

        return result;
    }
}


/*
fn find_oxygen(computer:&mut IntCode) -> i64 {
    match move_robot(computer, Direction::North) {
        0 =>,
        1 => {find_oxygen(computer); move_robot(computer, Direction::South);},
        2 => {println!("Oxygen found!"); move_robot(computer
        _ => {panic!("Unexpacted response from computer:
    }
    return result;
    1
}
*/

/// Intcode computer simulator
struct IntCode {
    memory: Vec<i64>,
    instr_ptr: usize,
    relative_base: i64,
    inputs: VecDeque<i64>,
    status: StatusCode,
}

#[derive(PartialEq)]
enum StatusCode {
    Ready,
    InputRequested,
    Halted,
}

impl IntCode {
    fn new(program:&Vec<i64>) -> Self {
        // Make a copy of the program to work with
        let mut memory = program.to_vec();
        memory.extend_from_slice(&[0; 1000]);
        IntCode {
            memory: memory,
            instr_ptr: 0,
            relative_base: 0,
            inputs: VecDeque::new(),
            status: StatusCode::Ready,
        }
    }

    fn feed_input(&mut self, val:i64) {
        self.inputs.push_back(val);
    }
}

impl Iterator for IntCode {
    type Item = i64;

    fn next(&mut self, ) -> Option<i64> {
        // Execute instructions until we can output a value.
        // Return None when no value can be outputted (either we are waiting for more input, or we
        // have halted)
        loop {
            if self.instr_ptr >= self.memory.len() {
                self.status = StatusCode::Halted;
                return None;  // End of memory
            }

            let instruction = Opcode::from(self.memory[self.instr_ptr]);

            let get_param = |offset, addressing_mode| {
                match addressing_mode {
                    0 => self.memory[self.memory[self.instr_ptr + offset] as usize], // Memory addressing mode
                    1 => self.memory[self.instr_ptr + offset], // Immediate addressing mode
                    2 => self.memory[(self.relative_base + self.memory[self.instr_ptr + offset]) as usize], // Relative addressing mode
                    _ => panic!("Invalid adressing mode")
                }
            };

            let get_target_ptr = |offset, addressing_mode| {
                match addressing_mode {
                    0 | 1 => self.memory[self.instr_ptr + offset] as usize, // Position addressing mode
                    2 => (self.relative_base + self.memory[self.instr_ptr + offset]) as usize, // Relative addressing mode
                    _ => panic!("Invalid adressing mode")
                }
            };

            match instruction.code {
                1 => { // Add
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = get_target_ptr(3, instruction.param3_mode);
                    self.memory[target_ptr] = left + right;
                    self.instr_ptr += 4;
                }
                2 => { // Multiply
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = get_target_ptr(3, instruction.param3_mode);
                    self.memory[target_ptr] = left * right;
                    self.instr_ptr += 4;
                }
                3 => { // Input
                    let target_ptr = get_target_ptr(1, instruction.param1_mode);
                    if self.inputs.len() == 0 {
                        self.status = StatusCode::InputRequested;
                        return None; // Wait for more input
                    } else {
                        self.memory[target_ptr] = self.inputs.pop_front().unwrap();
                        self.status = StatusCode::Ready;
                    }
                    self.instr_ptr += 2;
                }
                4 => { // Output
                    let out = get_param(1, instruction.param1_mode);
                    self.instr_ptr += 2;
                    return Some(out);
                }
                5 => { // Jump-if-true
                    let cmp = get_param(1, instruction.param1_mode);
                    let jmp = get_param(2, instruction.param2_mode);
                    if cmp != 0 {
                        self.instr_ptr = jmp as usize;
                    } else {
                        self.instr_ptr += 3;
                    }
                }
                6 => { // Jump-if-false
                    let cmp = get_param(1, instruction.param1_mode);
                    let jmp = get_param(2, instruction.param2_mode);
                    if cmp == 0 {
                        self.instr_ptr = jmp as usize;
                    } else {
                        self.instr_ptr += 3;
                    }
                }
                7 => { // Less then
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = get_target_ptr(3, instruction.param3_mode);
                    if left < right {
                        self.memory[target_ptr] = 1;
                    } else {
                        self.memory[target_ptr] = 0;
                    }
                    self.instr_ptr += 4;
                }
                8 => { // Equals
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = get_target_ptr(3, instruction.param3_mode);
                    if left == right {
                        self.memory[target_ptr] = 1;
                    } else {
                        self.memory[target_ptr] = 0;
                    }
                    self.instr_ptr += 4;
                }
                9 => { // Relative base offset
                    let offset = get_param(1, instruction.param1_mode);
                    self.relative_base += offset;
                    self.instr_ptr += 2;
                }
                99 => { // Halt
                    self.status = StatusCode::Halted;
                    return None;
                }
                _ => panic!("Unknown opcode: {}", instruction.code)
            };
        }
    }
}

struct Opcode {
    code:i64,
    param1_mode:i64,
    param2_mode:i64,
    param3_mode:i64,
}

// Parse an integer value into an opcode
impl From<i64> for Opcode {
    fn from(n:i64) -> Self {
        fn parse_digits_into(n:i64, digits:&mut Vec<i64>) {
            if n >= 10 {
                parse_digits_into(n / 10, digits);
            }
            digits.push(n % 10);
        }
        let mut digits = Vec::new();
        parse_digits_into(n, &mut digits);
        
        // Pad with zeros
        while digits.len() < 5 {
            digits.insert(0, 0);
        }

        Opcode {
            code: 10 * digits[3] + digits[4],
            param1_mode: digits[2],
            param2_mode: digits[1],
            param3_mode: digits[0],
        }
    }
}
