extern crate termion;

use std::io::{self, Read};
use std::collections::VecDeque;
use std::{thread, time};
use termion::clear;

const SCREEN_WIDTH:usize = 43;
const SCREEN_HEIGHT:usize = 26;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let mut program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    let mut screen:[[char; SCREEN_WIDTH]; SCREEN_HEIGHT] = [[' '; SCREEN_WIDTH]; SCREEN_HEIGHT];
    play_game(&mut program, &mut screen, 0);

    // Count number of blocks
    let mut n_blocks = 0;
    for line in screen.iter() {
        for tile in line.iter() {
            if tile == &'□' {
                n_blocks += 1;
            }
        }
    }
    n_blocks
}

fn part2(input: &str) -> i64 {
    let mut program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    let mut screen:[[char; SCREEN_WIDTH]; SCREEN_HEIGHT] = [[' '; SCREEN_WIDTH]; SCREEN_HEIGHT];
    play_game(&mut program, &mut screen, 2)
}

fn play_game(program:&mut Vec<i64>, screen:&mut [[char; SCREEN_WIDTH]; SCREEN_HEIGHT], n_coins:i64) -> i64 {
    if n_coins > 0 {
        program[0] = n_coins;
    }

    let mut computer = IntCode::new(program);
    let mut score:i64 = 0;
    let mut ball_x_pos:i64 = SCREEN_WIDTH as i64 / 2;
    let mut paddle_x_pos:i64 = SCREEN_WIDTH as i64 / 2;

    loop {
        let x = match computer.next() {
            Some(v) => v,
            None => match computer.status {
                StatusCode::Ready => panic!("Received None for no reason."),
                StatusCode::InputRequested => {
                    print_screen(&screen);
                    println!("Score: {}", score);
                    thread::sleep(time::Duration::new(0, 10_000_000));
                    if ball_x_pos < paddle_x_pos {
                        computer.feed_input(-1);
                    } else if ball_x_pos > paddle_x_pos {
                        computer.feed_input(1);
                    } else {
                        computer.feed_input(0);
                    }
                    continue;
                }
                StatusCode::Halted => {
                    print_screen(&screen);
                    println!("Score: {}", score);
                    println!("Game ended.");
                    break;
                }
            }
        };
        let y = computer.next().unwrap_or(0);
        let id = computer.next().unwrap_or(0);

        if x < 0 {
            score = id;
        } else {
            screen[y as usize][x as usize] = match id {
                0 => ' ',
                1 => '█',
                2 => '□',
                3 => {
                    paddle_x_pos = x;
                    '―' 
                }
                4 => {
                    ball_x_pos = x;
                    '·'
                }
                _ => panic!("Unknown tile type {}", id),
            };
        }
    }

    score
}

fn print_screen(screen:&[[char; SCREEN_WIDTH]; SCREEN_HEIGHT]) {
    let mut s = String::with_capacity(100 * 20);
    for line in screen.iter() {
        s.push_str(&line.iter().collect::<String>());
        s.push('\n');
    }
    println!("{clear}{content}",
        clear=clear::All,
        content=s,
    );
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
        let mut memory = program.to_vec();
        memory.extend_from_slice(&[0; 1000]);
        IntCode {
            // Make a copy of the program to work with
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
