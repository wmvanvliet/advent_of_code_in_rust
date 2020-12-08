use std::io::{self, Read};
use std::str::FromStr;
use std::collections::{HashSet};
use itertools::Itertools;

/**
 * This reads in the puzzle input from stdin. So you would call this program like:
 *     cat input | cargo run
 * It then feeds the input as a string to the functions that solve both parts of the puzzle.
 */
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input)));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

/**
 * On Windows, a unicode BOM marker is always placed at the beginning of the input file. Very
 * annoying.
 */
fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> i64 {
    let mut computer = Computer::new(input);
    computer.run();
    computer.acc
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut computer = Computer::new(input);
    // We don't want to create an iterator, because we'll be mutating the computers memory as we go
    for i in 0..computer.memory.len() {
        let instr = computer.memory.get(i).unwrap();
        // Change NOP <-> JMP
        computer.memory[i] = match instr.op {
            Operation::NOP => Instruction{op: Operation::JMP, arg: instr.arg},
            Operation::JMP => Instruction{op: Operation::NOP, arg: instr.arg},
            Operation::ACC => Instruction{op: Operation::ACC, arg: instr.arg},
        };
        // Does it halt now?
        if computer.run() {
            return computer.acc;
        }
        // Reset the computer and try again
        computer = Computer::new(input);
    }
    panic!("No solution found.");
}

#[derive(Debug)]
enum Operation {
    NOP,
    ACC,
    JMP,
}

#[derive(Debug)]
struct Instruction {
    op: Operation,
    arg: i64,
}

// Error indicating an instruction couldn't be parsed
#[derive(Debug, Clone)]
struct ParseError;

// Parse a string into an Instruction
impl FromStr for Instruction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (op_str, arg_str) = s.trim().split_whitespace().next_tuple().unwrap();
        let op = match op_str {
            "nop" => Operation::NOP,
            "acc" => Operation::ACC,
            "jmp" => Operation::JMP,
            _ => return Err(ParseError)
        };
        Ok(Instruction {
            op: op,
            arg: arg_str.parse().unwrap(),
        })
    }
}

#[derive(Debug)]
struct Computer {
    memory: Vec<Instruction>,
    instr_ptr: usize,
    acc: i64,
}

impl Computer {
    fn new(program:&str) -> Self {
        // Parse the program
        let memory:Vec<Instruction> = program
            .lines()
            .map(|i| i.parse().unwrap())
            .collect();
        Computer {
            memory: memory,
            instr_ptr: 0,
            acc: 0,
        }
    }

    /**
     * Run a single step of the program.
     */
    fn step(&mut self) {
        let instr = self.memory.get(self.instr_ptr).unwrap();
        match instr.op {
            Operation::NOP => { self.instr_ptr += 1 },
            Operation::ACC => { self.acc += instr.arg; self.instr_ptr += 1},
            Operation::JMP => self.instr_ptr = (self.instr_ptr as i64 + instr.arg) as usize,
        }
    }

    /**
     * Run the computer until it either finishes or detects it's in an infinite loop.
     */
    fn run(&mut self) -> bool {
        let mut visited_addresses:HashSet<usize> = HashSet::new();
        visited_addresses.insert(0);
        loop {
            self.step();
            if visited_addresses.contains(&self.instr_ptr) {
               return false
            }
            visited_addresses.insert(self.instr_ptr);
            if self.instr_ptr >= self.memory.len() {
                return true;
            }
        }
    }
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("nop +0
                          acc +1
                          jmp +4
                          acc +3
                          jmp -3
                          acc -99
                          acc +1
                          jmp -4
                          acc +6"), 5);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("nop +0
                          acc +1
                          jmp +4
                          acc +3
                          jmp -3
                          acc -99
                          acc +1
                          jmp -4
                          acc +6"), 8);
    }
}
