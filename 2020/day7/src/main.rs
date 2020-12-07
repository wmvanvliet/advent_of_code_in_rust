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

// I'm not typing this type out every time, so let's make it a new type
type RuleBook = HashMap<String, HashMap<String, i64>>;

fn part1(input: &str) -> usize {
    let rules:RuleBook = parse_rules(input);

    // Count the number of colors that (eventually) must contain "shiny gold"
    rules.keys().filter(|x| must_contain_shiny_gold(x, &rules)).count()
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let rules:RuleBook = parse_rules(input);
    count_bags(&String::from("shiny gold"), &rules) - 1
}

/**
 * Parses the rules in engligh into a machine readable RuleBook.
 */
fn parse_rules(input: &str) -> RuleBook {
    let mut rules:RuleBook = HashMap::new();

    // Parse the rules
    let re = Regex::new(r"(\d+) ([a-z]+ [a-z]+)").unwrap();
    for rule in input.lines() {
        let (color, must_contain_str) = rule.trim().split(" bags contain ").next_tuple().unwrap();
        let must_contain:HashMap<String, i64> = re
            .captures_iter(must_contain_str)
            .map(|cap| {
                let color = String::from(&cap[2]);
                let n:i64 = cap[1].parse().unwrap();
                (color, n)
            })
            .collect();
        rules.insert(String::from(color), must_contain);
    }

    rules
}

/**
 * Determine whether a bag with the given color must (eventually) contain a "shiny gold" bag.
 */
fn must_contain_shiny_gold(color: &String, rules: &RuleBook) -> bool {
    let must_contain = &rules[color];
    if must_contain.contains_key("shiny gold") {
        return true
    } else {
        for must_contain_color in must_contain.keys() {
            if must_contain_shiny_gold(must_contain_color, rules) {
                return true;
            }
        }
    }
    false
}

/**
 * Count the total number of bags you have, if you have one bag of the given color.
 */
fn count_bags(color: &String, rules: &RuleBook) -> i64 {
    let mut n_bags:i64 = 1;  // The bag itself

    // Add any bags that must be inside the bag
    for (inside_bag_color, inside_bag_number) in rules[color].iter() {
        n_bags += inside_bag_number * count_bags(inside_bag_color, rules);
    }

    n_bags
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("light red bags contain 1 bright white bag, 2 muted yellow bags.
                          dark orange bags contain 3 bright white bags, 4 muted yellow bags.
                          bright white bags contain 1 shiny gold bag.
                          muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
                          shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
                          dark olive bags contain 3 faded blue bags, 4 dotted black bags.
                          vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
                          faded blue bags contain no other bags.
                          dotted black bags contain no other bags."), 4);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("light red bags contain 1 bright white bag, 2 muted yellow bags.
                          dark orange bags contain 3 bright white bags, 4 muted yellow bags.
                          bright white bags contain 1 shiny gold bag.
                          muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
                          shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
                          dark olive bags contain 3 faded blue bags, 4 dotted black bags.
                          vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
                          faded blue bags contain no other bags.
                          dotted black bags contain no other bags."), 32);

        assert_eq!(part2("shiny gold bags contain 2 dark red bags.
                          dark red bags contain 2 dark orange bags.
                          dark orange bags contain 2 dark yellow bags.
                          dark yellow bags contain 2 dark green bags.
                          dark green bags contain 2 dark blue bags.
                          dark blue bags contain 2 dark violet bags.
                          dark violet bags contain no other bags."), 126);

    }
}
