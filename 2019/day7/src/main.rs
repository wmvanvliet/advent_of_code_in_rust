use std::io::{self, Read};
use std::collections::VecDeque;
use itertools::Itertools;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    (0..5).permutations(5)
        .map(|phase_code| amplifier_series(&program, phase_code))
        .max().unwrap()
}

fn part2(input: &str) -> i64 {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    (5..10).permutations(5)
        .map(|phase_code| amplifier_loop(&program, phase_code))
        .max().unwrap()
}

/// Compute the thruster signal output of the complete array of amplifiers
fn amplifier_series(program:&Vec<i64>, phase_code:Vec<i64>) -> i64 {
    let mut amplifiers = [
        IntCode::new(&program),
        IntCode::new(&program),
        IntCode::new(&program),
        IntCode::new(&program),
        IntCode::new(&program),
    ];

    // Lock in phase value
    for (amp, val) in amplifiers.iter_mut().zip(phase_code.iter()) {
        amp.feed_input(*val);
    }

    amplifiers[0].feed_input(0);

    // Run the programs in the amplifiers
    for i in 0..4 {
        if let Some(output) = amplifiers[i].next() {
            amplifiers[i + 1].feed_input(output);
        }
    }

    amplifiers[4].next().unwrap()
}

/// Run the amplifiers with a feedback loop
fn amplifier_loop(program:&Vec<i64>, phase_code:Vec<i64>) -> i64 {
    let mut amplifiers = [
        IntCode::new(&program),
        IntCode::new(&program),
        IntCode::new(&program),
        IntCode::new(&program),
        IntCode::new(&program),
    ];

    // Lock in phase value
    for (amp, val) in amplifiers.iter_mut().zip(phase_code.iter()) {
        amp.feed_input(*val);
    }

    amplifiers[0].feed_input(0);

    let mut thruster_output:i64 = 0; // Keeps track of the output of the last amplifier
    while !amplifiers.iter().fold(true, |halted, amp| (halted && amp.halted)) {
        // Run the programs in the amplifiers
        for i in 0..5 {
            if let Some(output) = amplifiers[i].next() {
                if i == 4 {
                    thruster_output = output;
                }
                amplifiers[(i + 1) % 5].feed_input(output);
            }
        }
    }

    thruster_output
}

/// Intcode computer simulator
struct IntCode {
    program: Vec<i64>,
    instr_ptr: usize,
    inputs: VecDeque<i64>,
    halted: bool,
}

impl IntCode {
    fn new(program:&Vec<i64>) -> Self {
        IntCode {
            // Make a copy of the program to work with
            program: program.to_vec(),
            instr_ptr: 0,
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
            if self.instr_ptr >= self.program.len() {
                self.halted = true;
                return None;  // End of program
            }

            let instruction = Opcode::from(self.program[self.instr_ptr]);

            let get_param = |offset, addressing_mode| {
                match addressing_mode {
                    0 => self.program[self.program[self.instr_ptr + offset] as usize], // Memory addressing mode
                    1 => self.program[self.instr_ptr + offset], // Immediate addressing mode
                    _ => panic!("Invalid adressing mode")
                }
            };

            match instruction.code {
                1 => { // Add
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = self.program[self.instr_ptr + 3] as usize;
                    self.program[target_ptr] = left + right;
                    self.instr_ptr += 4;
                }
                2 => { // Multiply
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = self.program[self.instr_ptr + 3] as usize;
                    self.program[target_ptr] = left * right;
                    self.instr_ptr += 4;
                }
                3 => { // Input
                    let target_ptr = self.program[self.instr_ptr + 1] as usize;
                    if self.inputs.len() == 0 {
                        return None; // Wait for more input
                    } else {
                        self.program[target_ptr] = self.inputs.pop_front().unwrap();
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
                    let target_ptr = self.program[self.instr_ptr + 3] as usize;
                    if left < right {
                        self.program[target_ptr] = 1;
                    } else {
                        self.program[target_ptr] = 0;
                    }
                    self.instr_ptr += 4;
                }
                8 => { // Equals
                    let left = get_param(1, instruction.param1_mode);
                    let right = get_param(2, instruction.param2_mode);
                    let target_ptr = self.program[self.instr_ptr + 3] as usize;
                    if left == right {
                        self.program[target_ptr] = 1;
                    } else {
                        self.program[target_ptr] = 0;
                    }
                    self.instr_ptr += 4;
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
    // param3_mode:i64,
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
            // param3_mode: digits[0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplifier_series() {
        assert_eq!(amplifier_series(&vec![3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0], vec![4,3,2,1,0]), 43210);
        assert_eq!(amplifier_series(&vec![3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0], vec![0,1,2,3,4]), 54321);
        assert_eq!(amplifier_series(&vec![3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0], vec![1,0,4,3,2]), 65210);
    }

    #[test]
    fn test_amplifier_loop() {
        assert_eq!(amplifier_loop(&vec![3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5], vec![9,8,7,6,5]), 139629729);
        assert_eq!(amplifier_loop(&vec![3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10], vec![9,7,8,5,6]), 18216);
    }

    #[test]
    fn test_part1() {
        assert_eq!(part1("3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0"), 43210);
        assert_eq!(part1("3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0"), 54321);
        assert_eq!(part1("3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0"), 65210);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5"), 139629729);
        assert_eq!(part2("3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10"), 18216);
    }
}
