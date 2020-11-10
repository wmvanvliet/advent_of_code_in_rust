use std::io::{self, Read};

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
        .map(|val| { val.parse().unwrap() })
        .collect();

    let outputs = execute_program(&program, &vec![1]);
    println!("Outputs: {:?}", outputs);
    return outputs.last().unwrap().clone();
}

fn part2(input: &str) -> i64 {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    let outputs = execute_program(&program, &vec![5]);
    println!("Outputs: {:?}", outputs);
    return outputs.last().unwrap().clone();
}

/// Execute an Intcode program. A vector of inputs needs to be supplied. Everytime the program
/// encountes an "input" instruction, the next input in the vector is used. This function produces
/// a vector of outputs, populated whenever the program encounters an "output" instruction.
fn execute_program(program:&Vec<i64>, inputs:&Vec<i64>) -> Vec<i64> {
    // Make a copy of the program to work with
    let mut program = program.to_vec();

    let mut outputs:Vec<i64> = Vec::new();

    // Execute all instructions
    let mut instr_ptr:usize = 0;
    let mut input_ptr:usize = 0;
    loop {
        if instr_ptr >= program.len() {
            break;  // End of program
        }

        let instruction = Opcode::from(program[instr_ptr]);

        let get_param = |offset, addressing_mode| {
            match addressing_mode {
                0 => program[program[instr_ptr + offset] as usize], // Memory addressing mode
                1 => program[instr_ptr + offset], // Immediate addressing mode
                _ => panic!("Invalid adressing mode")
            }
        };

        match instruction.code {
            1 => { // Add
                let left = get_param(1, instruction.param1_mode);
                let right = get_param(2, instruction.param2_mode);
                let target_ptr = program[instr_ptr + 3] as usize;
                program[target_ptr] = left + right;
                instr_ptr += 4;
            }
            2 => { // Multiply
                let left = get_param(1, instruction.param1_mode);
                let right = get_param(2, instruction.param2_mode);
                let target_ptr = program[instr_ptr + 3] as usize;
                program[target_ptr] = left * right;
                instr_ptr += 4;
            }
            3 => { // Input
                let target_ptr = program[instr_ptr + 1] as usize;
                program[target_ptr] = inputs[input_ptr];
                input_ptr += 1;
                instr_ptr += 2;
            }
            4 => { // Output
                let out = get_param(1, instruction.param1_mode);
                outputs.push(out);
                instr_ptr += 2;
            }
            5 => { // Jump-if-true
                let cmp = get_param(1, instruction.param1_mode);
                let jmp = get_param(2, instruction.param2_mode);
                if cmp != 0 {
                    instr_ptr = jmp as usize;
                } else {
                    instr_ptr += 3;
                }
            }
            6 => { // Jump-if-false
                let cmp = get_param(1, instruction.param1_mode);
                let jmp = get_param(2, instruction.param2_mode);
                if cmp == 0 {
                    instr_ptr = jmp as usize;
                } else {
                    instr_ptr += 3;
                }
            }
            7 => { // Less then
                let left = get_param(1, instruction.param1_mode);
                let right = get_param(2, instruction.param2_mode);
                let target_ptr = program[instr_ptr + 3] as usize;
                if left < right {
                    program[target_ptr] = 1;
                } else {
                    program[target_ptr] = 0;
                }
                instr_ptr += 4;
            }
            8 => { // Equals
                let left = get_param(1, instruction.param1_mode);
                let right = get_param(2, instruction.param2_mode);
                let target_ptr = program[instr_ptr + 3] as usize;
                if left == right {
                    program[target_ptr] = 1;
                } else {
                    program[target_ptr] = 0;
                }
                instr_ptr += 4;
            }
            99 => { // Halt
                break;
            }
            _ => panic!("Unknown opcode: {}", instruction.code)
        };
    }

    outputs
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
    fn test_part1() {
        assert_eq!(execute_program(&vec![3,0,4,0,99], &vec![42])[0], 42);
        assert_eq!(execute_program(&vec![1002,7,3,7,4,7,99,33], &vec![])[0], 99);
        assert_eq!(execute_program(&vec![1101,100,-1,7,4,7,99,0], &vec![])[0], 99);
    }
}
