use std::io::{self, Read};
use std::collections::HashMap;

 #[macro_use] extern crate scan_fmt;
 #[macro_use] extern crate itertools;

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
    match input.strip_prefix("\u{feff}") {
        Some(x) => x,
        _ => input
    }
}

/**
 * Solves part 1 of the puzzle.
 */
fn part1(input: &str) -> u64 {
    let mut or_mask:u64 = 0;
    let mut and_mask:u64 = 0;
    let mut mem:HashMap<u64, u64> = HashMap::new();

    for line in input.lines() {
        if let Some(mask_str) = line.strip_prefix("mask =") {
            or_mask = u64::from_str_radix(&mask_str.trim().replace('X', "0"), 2).unwrap();
            and_mask = u64::from_str_radix(&mask_str.trim().replace('X', "1"), 2).unwrap();
        } else {
            let (addr, val) = scan_fmt!(line.trim(), "mem[{d}] = {d}", u64, u64).unwrap();
            mem.insert(addr, (val | or_mask) & and_mask);
        }
    }
   
    mem.values().sum()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> u64 {
    let mut mem:HashMap<u64, u64> = HashMap::new();
    let mut floating_bits:Vec<usize> = Vec::new();
    let mut mask:u64 = 0;

    for line in input.lines().map(|l| l.trim()) {
        if let Some(mask_str) = line.strip_prefix("mask =") {
            mask = u64::from_str_radix(&mask_str.trim().replace('X', "0"), 2).unwrap();
            floating_bits = mask_str.match_indices('X').map(|(x, _)| 36 - x).collect();
        } else {
            let (mut addr, val) = scan_fmt!(line, "mem[{d}] = {d}", u64, u64).unwrap();
            addr |= mask;

            // Generate all possible combinations of 0's and 1's
            for bit_val in 0..2_u64.pow(floating_bits.len() as u32) {
                // Assign 0's and 1's to the floating bits
                for (i, bit_pos) in floating_bits.iter().enumerate() {
                    // Check the i'th position for a 0 or 1
                    if (bit_val >> i) % 2 == 1 {
                        addr |= 1 << bit_pos;  // Set bit to 1
                    } else {
                        addr &= !(1 << bit_pos); // Set bit to 0
                    }
                }
                mem.insert(addr, val);
            }
        }
    }
   
    mem.values().sum()
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X
                          mem[8] = 11
                          mem[7] = 101
                          mem[8] = 0"), 165);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("mask = 000000000000000000000000000000X1001X
                          mem[42] = 100
                          mask = 00000000000000000000000000000000X0XX
                          mem[26] = 1"), 208);
    }
}
