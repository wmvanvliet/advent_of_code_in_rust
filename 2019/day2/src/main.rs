use std::io::{self, Read};

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
        instr_ptr += match program[instr_ptr] {
            1 => add_instr(program[instr_ptr + 1], program[instr_ptr + 2], program[instr_ptr + 3], &mut program),
            2 => mult_instr(program[instr_ptr + 1], program[instr_ptr + 2], program[instr_ptr + 3], &mut program),
            99 => return program[0],
            _ => panic!()
        };

        if instr_ptr >= program.len() {
            return program[0];
        }
    }
}

/// Opcode 1: add
fn add_instr(pos1:i64, pos2:i64, target:i64, program:&mut Vec<i64>) -> usize {
    program[target as usize] = program[pos1 as usize] + program[pos2 as usize];
    return 4;
}

/// Opcode 2: multiply
fn mult_instr(pos1:i64, pos2:i64, target:i64, program:&mut Vec<i64>) -> usize {
    program[target as usize] = program[pos1 as usize] * program[pos2 as usize];
    return 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("1,9,10,3,2,3,11,0,99,30,40,50"), 3500);
        assert_eq!(part1("1,0,0,0,99"), 2);
        assert_eq!(part1("2,3,0,3,99"), 2);
        assert_eq!(part1("2,4,4,5,99,0"), 2);
        assert_eq!(part1("1,1,1,4,99,5,6,0,99"), 30);
    }
}
