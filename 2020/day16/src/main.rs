use std::io::{self, Read};
use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
use regex::Regex;
use std::str::FromStr;

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
    let puzzle: Puzzle = input.parse().unwrap();
    puzzle.nearby_tickets.iter()
        .flat_map(|ticket| find_invalid_numbers(ticket, &puzzle.rules))
        .sum()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> u64 {
    let puzzle: Puzzle = input.parse().unwrap();

    // Drop invalid tickets
    let valid_tickets: Vec<_> = puzzle.nearby_tickets
        .iter()
        .filter(|ticket| find_invalid_numbers(ticket, &puzzle.rules).len() == 0)
        .collect();

    // For each field, find all possible names for which the corresponding rule is satisfied
    let n_fields = puzzle.your_ticket.len();
    let mut possible_field_names: Vec<HashSet<&String>> = (0..n_fields)
        .map(|field_num|
            puzzle.rules.iter().filter(|(_, (range1, range2))|
                valid_tickets.iter().all(|ticket|
                    range1.contains(&ticket[field_num]) || range2.contains(&ticket[field_num])
                )
            )
            .map(|(name, _)| name)
            .collect()
        )
        .collect();


    // Start decyphering field names one at the time.
    // A field name can be decyphered when there is only one possible name for it.
    let mut decyphered_ticket: HashMap<&str, u64> = HashMap::new();
    let mut unassigned_names: HashSet<_> = puzzle.rules.keys().collect();
    while !unassigned_names.is_empty() {
        let field_num = possible_field_names.iter().position(|names| names.len() == 1).unwrap();
        let name = possible_field_names[field_num].iter().copied().next().unwrap();
        decyphered_ticket.insert(name, puzzle.your_ticket[field_num]);
        unassigned_names.remove(name);
        for names in possible_field_names.iter_mut() {
            names.remove(name);
        }
    }

    // Multiple all fields that start with "departure"
    decyphered_ticket.iter()
        .filter(|(name, _)| name.starts_with("departure"))
        .map(|(_, value)| value)
        .product()
}

type Rules = HashMap<String, (RangeInclusive<u64>, RangeInclusive<u64>)>;

#[derive(Debug)]
struct Puzzle {
    rules: Rules,
    your_ticket: Vec<u64>,
    nearby_tickets: Vec<Vec<u64>>,
}

#[derive(Debug, PartialEq)]
struct ParseError;

impl FromStr for Puzzle {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts:Vec<_> = s.split("\r\n\r\n").collect();
        let mut rules:HashMap<_, _> = HashMap::new();
        let parser = Regex::new(r"\W*([\w ]+): (\d+)-(\d+) or (\d+)-(\d+)\W*").unwrap();
        for cap in parser.captures_iter(parts[0]) {
            rules.insert(String::from(&cap[1]),
                (RangeInclusive::new(cap[2].parse::<u64>().unwrap(), cap[3].parse::<u64>().unwrap()),
                 RangeInclusive::new(cap[4].parse::<u64>().unwrap(), cap[5].parse::<u64>().unwrap())));
        }

        let your_ticket:Vec<u64> = parts[1].lines()
                                           .skip(1)
                                           .next()
                                           .unwrap()
                                           .split(',')
                                           .map(|x| x.trim().parse().unwrap())
                                           .collect();

        let mut nearby_tickets:Vec<Vec<u64>> = Vec::new();
        for line in parts[2].lines().skip(1) {
            nearby_tickets.push(line.trim()
                                   .split(',')
                                   .map(|x| x.parse().unwrap()).collect());
        }

        Ok(Puzzle {
            rules: rules,
            your_ticket: your_ticket,
            nearby_tickets: nearby_tickets,
        })
    }
}

fn find_invalid_numbers(ticket: &Vec<u64>, rules: &Rules) -> Vec<u64> {
    ticket.iter()
        .copied()
        .filter(|num|
            rules.values()
                .filter(|(range1, range2)| range1.contains(num) || range2.contains(num)).count() == 0
        )
        .collect()
}


/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("class: 1-3 or 5-7\r
                          row: 6-11 or 33-44\r
                          seat: 13-40 or 45-50\r
\r
                          your ticket:\r
                          7,1,14\r
\r
                          nearby tickets:\r
                          7,3,47\r
                          40,4,50\r
                          55,2,20\r
                          38,6,12"), 71);
    }
}
