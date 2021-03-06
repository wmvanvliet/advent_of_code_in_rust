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

    return execute_program(&program, 12, 2);
}

fn part2(input: &str) -> i64 {
    let program:Vec<i64> = input
        .trim_end()
        .split(',')
        .map(|val| { val.parse().unwrap() })
        .collect();

    for noun in {0..99} {
        for verb in {0..99} {
            if execute_program(&program, noun, verb) == 19690720 {
                return 100 * noun + verb;
            }
        }
    }

    panic!();
}

/// Execute an Intcode program
fn execute_program(initial_program:&Vec<i64>, noun:i64, verb:i64) -> i64 {
    // Initial setup of the program
    let mut program = initial_program.to_vec();
    program[1] = noun;
    program[2] = verb;

    // Execute all instructions
    let mut instr_ptr:usize = 0;
    loop {
        if instr_ptr >= program.len() {
            break;
        }
        match program[instr_ptr] {
            1 => { // Add
                let left_ptr = program[instr_ptr + 1] as usize;
                let right_ptr = program[instr_ptr + 2] as usize;
                let target_ptr = program[instr_ptr + 3] as usize;
                program[target_ptr] = program[left_ptr] + program[right_ptr];
                instr_ptr += 4;
            }
            2 => { // Multiply
                let left_ptr = program[instr_ptr + 1] as usize;
                let right_ptr = program[instr_ptr + 2] as usize;
                let target_ptr = program[instr_ptr + 3] as usize;
                program[target_ptr] = program[left_ptr] * program[right_ptr];
                instr_ptr += 4;
            }
            99 => { // Halt
                break;
            }
            _ => panic!()
        };
    }

    program[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(execute_program(&vec![1,9,10,3,2,3,11,0,99,30,40,50], 9, 10), 3500);
        assert_eq!(execute_program(&vec![1,0,0,0,99], 0, 0), 2);
        assert_eq!(execute_program(&vec![2,3,0,3,99], 3, 0), 2);
        assert_eq!(execute_program(&vec![2,4,4,5,99,0], 4, 4), 2);
        assert_eq!(execute_program(&vec![1,1,1,4,99,5,6,0,99], 1, 1), 30);
    }
}
