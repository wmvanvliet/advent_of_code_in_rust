use std::io::{self, Read};
use std::collections::{HashMap, HashSet};
use itertools::Itertools;
use regex::Regex;

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
    let mut n_correct_passports:i64 = 0;
    for entry in input.split("\n\n") {  // Password entries are separated by a blank line
        match check_passport(&parse_passport(entry)) {
            Evaluation::VALID => n_correct_passports += 1,
            Evaluation::PRESENT => n_correct_passports += 1,
            Evaluation::INVALID => {},
        }
    }
    n_correct_passports
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut n_correct_passports:i64 = 0;
    for entry in input.split("\n\n") {  // Password entries are separated by a blank line
        match check_passport(&parse_passport(entry)) {
            Evaluation::VALID => n_correct_passports += 1,
            Evaluation::PRESENT => {},
            Evaluation::INVALID => {},
        }
    }
    n_correct_passports
}



#[derive(PartialEq, Debug)]
enum Evaluation { VALID, PRESENT, INVALID }

fn parse_passport(entry:&str) -> HashMap<&str, &str> {
    entry.split_whitespace()
        .flat_map(|pair| pair.split(':')).tuples()
        .collect::<HashMap<_, _>>()
}

fn check_passport(passport:&HashMap<&str, &str>) -> Evaluation {
    // let required_keys = ["byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid", "cid"]
    let required_keys = ["byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"];
    for key in required_keys.iter() {
        if !passport.contains_key(key) {
            return Evaluation::INVALID;
        }
    }

    // At this point, all required keys are present. But are they valid?
    let hcl_re = Regex::new(r"^#[0-9a-f]{6}$").unwrap();
    let valid = passport.iter().all(|(&k, v)| match k {
        "byr" => {
            match v.parse::<u64>() {
                Ok(byr) => byr >= 1920 && byr <= 2002,
                Err(_) => false,
            }
        },
        "iyr" => {
            match v.parse::<u64>() {
                Ok(iyr) => iyr >= 2010 && iyr <= 2020,
                Err(_) => false,
            }
        },
        "eyr" => {
            match v.parse::<u64>() {
                Ok(eyr) => eyr >= 2020 && eyr <= 2030,
                Err(_) => false,
            }
        },
        "hgt" => {
            if let Some(height_cm) = v.strip_suffix("cm") {
                match height_cm.parse::<u64>() {
                    Ok(n) => n >= 150 && n <= 193,
                    Err(_) => false,
                }
            } else if let Some(height_in) = v.strip_suffix("in") {
                match height_in.parse::<u64>() {
                    Ok(n) => n >= 59 && n <= 76,
                    Err(_) => false,
                }
            } else {
                false
            }
        },
        "hcl" => hcl_re.is_match(v),
        "ecl" => ["amb", "blu", "brn", "gry", "grn", "hzl", "oth"].contains(v),
        "pid" => v.len() == 9 && v.chars().all(|c| c.is_ascii_digit()),
        "cid" => true,
        _ => unreachable!()
    });

    if valid {
        Evaluation::VALID
    } else {
        Evaluation::PRESENT
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
        assert_eq!(part1("ecl:gry pid:860033327 eyr:2020 hcl:#fffffd\n\
                          byr:1937 iyr:2017 cid:147 hgt:183cm\n\
                          \n\
                          iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884\n\
                          hcl:#cfa07d byr:1929\n\
                          \n\
                          hcl:#ae17e1 iyr:2013\n\
                          eyr:2024\n\
                          ecl:brn pid:760753108 byr:1931\n\
                          hgt:179cm\n\
                          \n\
                          hcl:#cfa07d eyr:2025 pid:166559648\n\
                          iyr:2011 ecl:brn hgt:59in\n\
                          "), 2);
    }

    #[test]
    fn test_part2() {
        assert_eq!(
            check_passport(&parse_passport(
                "eyr:1972 cid:100 hcl:#18171d ecl:amb hgt:170 pid:186cm iyr:2018 byr:1926"
            )), Evaluation::PRESENT
        );
        assert_eq!(
            check_passport(&parse_passport(
                "pid:087499704 hgt:74in ecl:grn iyr:2012 eyr:2030 byr:1980 hcl:#623a2f"
            )), Evaluation::VALID
        );
    }
}
