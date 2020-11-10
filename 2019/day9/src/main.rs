use std::io::{self, Read};
use std::collections::VecDeque;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {:?}", part1(strip_bom(&input)));
    println!("Part 2 answer: {:?}", part2(strip_bom(&input)));
}

fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

fn part1(input: &str) -> Vec<i64> {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();
    //disassemble(&program);
    let mut computer = IntCode::new(&program);
    computer.feed_input(1);
    computer.collect()
}

fn part2(input: &str) -> Vec<i64> {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();
    let mut computer = IntCode::new(&program);
    computer.feed_input(2);
    computer.collect()
}

/// Intcode computer simulator
struct IntCode {
    memory: Vec<i64>,
    instr_ptr: usize,
    relative_base: i64,
    inputs: VecDeque<i64>,
    halted: bool,
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
            halted: false,
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
                self.halted = true;
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
                        return None; // Wait for more input
                    } else {
                        self.memory[target_ptr] = self.inputs.pop_front().unwrap();
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
                    self.halted = true;
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

fn disassemble(program:&Vec<i64>) {
    for (i, x) in program.iter().enumerate() {
        println!("{:4}: {}", i, x);
    }
    println!("--------------------------");
    fn param_mode_repr(mode:i64) -> &'static str {
        match mode {
            0 => "",
            1 => "$",
            2 => "+",
            _ => panic!("Invalid mode")
        }
    }

    let mut instr_ptr:usize = 0;
    loop {
        if instr_ptr >= program.len() {
            break;
        }

        let instruction = Opcode::from(program[instr_ptr]);
        match instruction.code {
            1 => {
                println!("{:4}: add {}{} {}{} {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                    param_mode_repr(instruction.param2_mode),
                    program[instr_ptr + 2],
                    param_mode_repr(instruction.param3_mode),
                    program[instr_ptr + 3],
                );
                instr_ptr += 4;
            }
            2 => {
                println!("{:4}: mul {}{} {}{} {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                    param_mode_repr(instruction.param2_mode),
                    program[instr_ptr + 2],
                    param_mode_repr(instruction.param3_mode),
                    program[instr_ptr + 3],
                );
                instr_ptr += 4;
            }
            3 => {
                println!("{:4}: inp {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1]
                );
                instr_ptr += 2;
            }
            4 => {
                println!("{:4}: out {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                );
                instr_ptr += 2;
            }
            5 => {
                println!("{:4}: jpt {}{} {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                    param_mode_repr(instruction.param2_mode),
                    program[instr_ptr + 2],
                );
                instr_ptr += 3;
            }
            6 => {
                println!("{:4}: jpf {}{} {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                    param_mode_repr(instruction.param2_mode),
                    program[instr_ptr + 2],
                );
                instr_ptr += 3;
            }
            7 => {
                println!("{:4}: lt {}{} {}{} {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                    param_mode_repr(instruction.param2_mode),
                    program[instr_ptr + 2],
                    param_mode_repr(instruction.param3_mode),
                    program[instr_ptr + 3],
                );
                instr_ptr += 4;
            }
            8 => {
                println!("{:4}: eq {}{} {}{} {}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                    param_mode_repr(instruction.param2_mode),
                    program[instr_ptr + 2],
                    program[instr_ptr + 3],
                );
                instr_ptr += 4;
            }
            9 => {
                println!("{:4}: rbo {}{}",
                    instr_ptr,
                    param_mode_repr(instruction.param1_mode),
                    program[instr_ptr + 1],
                );
                instr_ptr += 2;
            }
            99 => {
                println!("{:4}: hlt", instr_ptr);
                instr_ptr += 1;
                //break;
            }
            _ => {
                println!("{:4}: [{}]",
                     instr_ptr,
                     instruction.code
                 );
                instr_ptr += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99"), vec![109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99]);
        assert_eq!(part1("1102,34915192,34915192,7,4,7,99,0"), vec![1219070632396864]);
        assert_eq!(part1("104,1125899906842624,99"), vec![1125899906842624]);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("0,1"), 2);
    }
}
