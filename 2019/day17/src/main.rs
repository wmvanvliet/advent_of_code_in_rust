use std::io::{self, Read};
use std::collections::VecDeque;
use std::collections::HashMap;
use std::collections::HashSet;
use std::{thread, time};
use termion::clear;

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
    let mut robot_pos:(i64, i64) = (0, 0);
    for output in computer {
        let output_char = output as u8 as char;
        match output_char {
            '\n' => { y += 1; x = 0; },
            '^' => {map.insert((x, y), output_char); robot_pos = (x, y); x += 1; },
            _ => {map.insert((x, y), output_char); x += 1;  }
        }
        if x > map_width {
            map_width = x;
        }
        if y > map_height {
            map_height = y;
        }
    }

    // Find Eularian path
    let mut unfinished_verts:VecDeque<(i64, i64)> = VecDeque::new();
    let mut route:Vec<(i64, i64)> = Vec::new();
    println!("Robot pos: {:?}", robot_pos);
    unfinished_verts.push_back(robot_pos);
    map.insert(robot_pos, 'V');
    while !unfinished_verts.is_empty() {
        let vert = unfinished_verts[unfinished_verts.len() - 1];
        let mut edges:Vec<(i64, i64)> = Vec::new();
        if vert.0 > 0 && map[&(vert.0 - 1, vert.1)] == '#' {
            edges.push((vert.0 - 1, vert.1));
        }
        if vert.0 < map_width - 1 && map[&(vert.0 + 1, vert.1)] == '#' {
            edges.push((vert.0 + 1, vert.1));
        }
        if vert.1 > 0 && map[&(vert.0, vert.1 - 1)] == '#' {
            edges.push((vert.0, vert.1 - 1));
        }
        if vert.1 < map_height - 2 && map[&(vert.0, vert.1 + 1)] == '#' {
            edges.push((vert.0, vert.1 + 1));
        }
        if edges.is_empty() {
            route.push(vert);
            unfinished_verts.pop_back();
        } else {
            for e in edges {
                unfinished_verts.push_back(e);
                map.insert(e, 'V');
            }
        }
    }

    // Convert path into directions for the robot
    let directions:Vec<char> = route[..].windows(2).map(|step| {
        let from = step[0];
        let to = step[1];
        if from.0 > to.0 {
            'L'
        } else if from.0 < to.0 {
            'R'
        } else if from.1 > to.1 {
            'D'
        } else if from.1 < to.1 {
            'U'
        } else {
            '?'
        }
    }).collect();
    println!("{:?}", directions);

    let mut optim_directions:Vec<(usize, char)> = Vec::new();
    let mut last_d = directions[0];
    let mut rep:usize = 1;
    for d in directions[1..].into_iter() {
        if d == &last_d {
            rep += 1;
        } else {
            optim_directions.push((rep, last_d));
            last_d = *d;
            rep = 1;
        }
    }
    println!("{:?} {:?}", optim_directions, optim_directions.len());

    let mut LCSRe:HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..(optim_directions.len() - 2) {
        for j in (i + 1)..(optim_directions.len() - 1) {
            if optim_directions[i] == optim_directions[j] && (j - i) > *LCSRe.get(&(i, j)).unwrap_or(&0) {
                LCSRe.insert((i + 1, j + 1), *LCSRe.get(&(i, j)).unwrap_or(&0) + 1);
            } else {
                LCSRe.insert((i + 1, j + 1), 0);
            }
        }
    }

    let mut max_substring:(usize, usize) = (0, 1);
    let mut max_substring_length:usize = 0;
    for (k, v) in LCSRe.into_iter() {
        if v > max_substring_length {
            max_substring = k;
            max_substring_length = v;
        }
    }
    let ans = &optim_directions[max_substring.0..max_substring.1];
    println!("{:?}, {:?}, {:?}, {:?}", ans, max_substring.0, max_substring.1, max_substring_length);

    2
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
