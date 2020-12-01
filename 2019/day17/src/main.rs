use std::io::{self, Read};
use std::collections::VecDeque;
use std::collections::HashMap;
use fancy_regex::Regex;

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

    // Spin up an IntCode computer and read the map
    let computer = IntCode::new(&program);
    let mut map:HashMap<(i64, i64), char> = HashMap::new();
    let mut x:i64 = 0;
    let mut y:i64 = 0;
    let mut map_width:i64 = 0;
    let mut map_height:i64 = 0;
    for output in computer {
        let output_char = output as u8 as char;
        match output_char {
            '\n' => { y += 1; x = 0; },
            _ => {map.insert((x, y), output_char); x += 1; }
        }
        if x > map_width {
            map_width = x;
        }
        if y > map_height {
            map_height = y;
        }
    }

    // Find intersections and sum up their coordinates
    let mut answer:i64 = 0;
    for y in 1..(map_height - 2) {
        for x in 1..(map_width - 1) {
            if *map.get(&(x, y)).unwrap() != '.' &&
               *map.get(&(x - 1, y)).unwrap() != '.' &&
               *map.get(&(x + 1, y)).unwrap() != '.' &&
               *map.get(&(x, y + 1)).unwrap() != '.' &&
               *map.get(&(x, y - 1)).unwrap() != '.' {
                map.insert((x, y), 'O');
                println!("Found intersection at ({}, {}) with value {}", x, y, x * y);
                answer += x * y;
            }
        }
    }

    print_map(&map, map_width, map_height);
    answer
}

fn part2(input: &str) -> i64 {
    let mut program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse::<i64>().expect("Could not parse IntCode program.") })
        .collect();

    // Spin up an IntCode computer and read the map.
    // This time, also parse out the robot location and direction.
    let computer = IntCode::new(&program);
    let mut map:HashMap<(i64, i64), char> = HashMap::new();
    let mut x:i64 = 0;
    let mut y:i64 = 0;
    let mut map_width:i64 = 0;
    let mut map_height:i64 = 0;
    let mut robot_pos:(i64, i64) = (0, 0);
    let mut robot_dir:char = '^';
    for output in computer {
        let output_char = output as u8 as char;
        match output_char {
            '\n' => { y += 1; x = 0; },
            '^' => {map.insert((x, y), output_char); robot_pos = (x, y); robot_dir = output_char; x += 1; },
            'v' => {map.insert((x, y), output_char); robot_pos = (x, y); robot_dir = output_char; x += 1; },
            '>' => {map.insert((x, y), output_char); robot_pos = (x, y); robot_dir = output_char; x += 1; },
            '<' => {map.insert((x, y), output_char); robot_pos = (x, y); robot_dir = output_char; x += 1; },
            _ => {map.insert((x, y), output_char); x += 1;  }
        }
        if x > map_width {
            map_width = x;
        }
        if y > map_height {
            map_height = y;
        }
    }

    // Find a path that will pass over all the scaffolding.
    // Looking at the map, we can just go straight until we run into a wall.
    // Then there is always only a single option available to us: either go right or left.
    // If we hit a dead end, we've reached the end of the path.
    let mut robot = Robot::new(robot_pos, robot_dir);
    loop {
        let next_turn = match robot.ori {
            '^' => {
                match map.get(&(robot.pos.0 + 1, robot.pos.1)) {
                    Some('#') => 'R',
                    _ => match map.get(&(robot.pos.0 - 1, robot.pos.1)) {
                        Some('#') => 'L',
                        _ => break,
                    }
                }
            },
            'v' => {
                match map.get(&(robot.pos.0 - 1, robot.pos.1)) {
                    Some('#') => 'R',
                    _ => match map.get(&(robot.pos.0 + 1, robot.pos.1)) {
                        Some('#') => 'L',
                        _ => break,
                    }
                }
            },
            '>' => {
                match map.get(&(robot.pos.0, robot.pos.1 + 1)) {
                    Some('#') => 'R',
                    _ => match map.get(&(robot.pos.0, robot.pos.1 - 1)) {
                        Some('#') => 'L',
                        _ => break,
                    }
                }
            },
            '<' => {
                match map.get(&(robot.pos.0, robot.pos.1 - 1)) {
                    Some('#') => 'R',
                    _ => match map.get(&(robot.pos.0, robot.pos.1 + 1)) {
                        Some('#') => 'L',
                        _ => break,
                    }
                }
            },
            _ => unreachable!()
        };
        robot.turn(next_turn);
        robot.move_forward(&map);
        map.insert(robot.pos, robot.ori);
    }
    print_map(&map, map_width, map_height);
    println!("Route: {:?}", robot.route);

    // Now that we know the route to take, we must find the optimal way to split it into 3
    // movement functions. Actually, this is a good job for an extremely complex regex :)
    let re = Regex::new(r"(?x)
        ^ 
        (((L|R),\d+,){1,5})  # Movement function A (capture group #1)
        \1*                  # function A may repeat before we get to function B
        (((L|R),\d+,){1,5})  # Movement function B (capture group #4)
        (\1|\4)*             # functions A and B may repeat before we get to function C
        (((L|R),\d+,){1,5})  # Movement function C (capture group #8)
        (\1|\4|\8)*          # all movement functions may repeat at will
        $
    ").unwrap();
   
    let captures = re.captures(&robot.route).expect("invalid regex").expect("no matches");
    let mut function_A = String::from(captures.get(1).unwrap().as_str());
    let mut function_B = String::from(captures.get(4).unwrap().as_str());
    let mut function_C = String::from(captures.get(8).unwrap().as_str());

    // Remove trailing comma's
    function_A.pop();
    function_B.pop();
    function_C.pop();

    // Compile the main movement routine
    let mut routine = robot.route.replace(&function_A, "A");
    routine = routine.replace(&function_B, "B");
    routine = routine.replace(&function_C, "C");
    routine.pop();
    println!("\nMain: {}\nA: {}\nB: {}\nC: {}\n", routine, function_A, function_B, function_C);

    // Feed the routine to the CPU
    program[0] = 2;
    let mut computer = IntCode::new(&program);
    for byte in routine.bytes() { computer.feed_input(byte as i64); }
    computer.feed_input('\n' as i64);
    for byte in function_A.bytes() { computer.feed_input(byte as i64); }
    computer.feed_input('\n' as i64);
    for byte in function_B.bytes() { computer.feed_input(byte as i64); }
    computer.feed_input('\n' as i64);
    for byte in function_C.bytes() { computer.feed_input(byte as i64); }
    computer.feed_input('\n' as i64);
    computer.feed_input('N' as i64);  // No continous feed
    computer.feed_input('\n' as i64);

    // Read out the answer from the CPU
    let mut answer:i64 = 0;
    let mut out = String::from("");
    for output in computer {
        if output < 127 {
            let output_char = output as u8 as char;
            out.push_str(&output_char.to_string());
        } else {
            answer = output;
        }
    }
    answer
}


struct Robot {
    pos: (i64, i64),
    ori: char,
    route: String,
}

impl Robot {
    fn new(pos:(i64, i64), ori:char) -> Self {
        Robot {
            pos: pos,
            ori: ori,
            route: String::new(),
        }
    }

    fn turn(&mut self, dir:char) {
        self.ori = match dir {
            'L' => match self.ori {
                '^' => '<',
                '>' => '^',
                'v' => '>',
                '<' => 'v',
                _ => unreachable!()
            },
            'R' => match self.ori {
                '^' => '>',
                '>' => 'v',
                'v' => '<',
                '<' => '^',
                _ => unreachable!()
            },
            _ => panic!("invalid dir.")
        };
        self.route.push_str(&dir.to_string());
        self.route.push_str(",");
    }

    fn move_forward(&mut self, map:&HashMap<(i64, i64), char>) {
        let mut steps_taken:usize = 0;
        let (dx, dy) = match self.ori {
            '^' => (0, -1),
            '>' => (1, 0),
            'v' => (0, 1),
            '<' => (-1, 0),
            _ => unreachable!()
        };

        loop {
            match map.get(&(self.pos.0 + dx, self.pos.1 + dy)) {
                Some('#') => { self.pos.0 += dx; self.pos.1 += dy; steps_taken += 1; },
                _ => { break; },
            }
        }

        self.route.push_str(&steps_taken.to_string());
        self.route.push_str(",");
    }
}

fn print_map(map:&HashMap<(i64, i64), char>, map_width:i64, map_height:i64) {
    let mut s = String::with_capacity((map_height * map_width) as usize);
    for y in 0..map_height {
        for x in 0..map_width {
            match map.get(&(x, y)) {
                Some(c) => s.push_str(&c.to_string()),
                None => s.push_str(" "),
            }
        }
        s.push_str("\n");
    }
    println!("{}", s);
}

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
        const MAX_MEM:usize = 5000;
        memory.extend_from_slice(&[0; MAX_MEM]);
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

/// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1(""), 1);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
