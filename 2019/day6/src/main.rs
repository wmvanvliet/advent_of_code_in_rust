use std::io::{self, Read};
use std::collections::HashMap;
use std::iter;

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
    let mut bodies:HashMap<String, Vec<String>> = HashMap::new();
    for line in input.lines() {
        let mut parts = line.split(')');
        let parent_name = parts.next().unwrap();
        let child_name = parts.next().unwrap();
        bodies.entry(child_name.to_string()).or_insert(Vec::new());
        let children = bodies.entry(parent_name.to_string()).or_insert(Vec::new());
        children.push(child_name.to_string());
    }

    fn count_orbits(bodies:&HashMap<String, Vec<String>>, name:&String, depth:i64) -> i64{
        let mut total_orbits:i64 = depth;
        for child in &bodies[name] {
            total_orbits += count_orbits(bodies, child, depth + 1);
        }
        total_orbits
    }

    count_orbits(&bodies, &String::from("COM"), 0)
}

fn part2(input: &str) -> usize {
    let mut orbits:HashMap<String, String> = HashMap::new();
    for line in input.lines() {
        let mut parts = line.split(')');
        let parent_name = parts.next().unwrap();
        let child_name = parts.next().unwrap();
        orbits.insert(child_name.to_string(), parent_name.to_string());
    }

    let you = String::from("YOU");
    let san = String::from("SAN");
    let mut chain_you:Vec<&String> = iter::successors(Some(&you), |&body| orbits.get(body)).collect();
    let mut chain_san:Vec<&String> = iter::successors(Some(&san), |&body| orbits.get(body)).collect();
    chain_you.reverse();
    chain_san.reverse();
    let mut i:usize = 0;
    while chain_you[i] == chain_san[i] {
        i += 1;
    }
    (chain_you.len() - i - 1) + (chain_san.len() - i - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("COM)B\nB)C\nC)D\nD)E\nE)F\nB)G\nG)H\nD)I\nE)J\nJ)K\nK)L"), 42);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("COM)B\nB)C\nC)D\nD)E\nE)F\nB)G\nG)H\nD)I\nE)J\nJ)K\nK)L\nK)YOU\nI)SAN"), 4);
    }
}
