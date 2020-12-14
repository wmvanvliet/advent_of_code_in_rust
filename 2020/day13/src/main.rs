use std::io::{self, Read};

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
    let earliest_time:u64 = input.lines().nth(0).unwrap().trim().parse().unwrap();
    let mut ids: Vec<u64> = Vec::new();
    for val in input.lines().nth(1).unwrap().trim().split(',') {
        match val {
            "x" => (),
            i => ids.push(i.parse().unwrap()),
        }
    }
    let mut min_id: u64 = 0;
    let mut min_wait_time: u64 = 1000000;
    for i in ids {
        let wait_time = wait_time(earliest_time, i);
        if wait_time < min_wait_time {
            min_wait_time = wait_time;
            min_id = i;
        }
    }
    min_wait_time * min_id
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> u64 {
    let mut buses: Vec<(u64, u64)> = Vec::new();
    for (i, val) in input.lines().nth(1).unwrap().trim().split(',').enumerate() {
        match val {
            "x" => (),
            id => buses.push((id.parse().unwrap(), i as u64)),
        }
    }

    let mut cycle1:u64 = buses[0].0;
    buses.iter()
         .skip(1)
         .fold(cycle1, |time, &(cycle2, offset)| {
             let next_time = cycle_time(time, cycle1, cycle2, offset);
             cycle1 *= cycle2;
             next_time
         })
}

fn wait_time(time: u64, cycle: u64) -> u64 {
    cycle - (time % cycle)
}

fn cycle_time(start_time: u64, cycle1: u64, cycle2: u64, offset: u64) -> u64 {
    let mut time = start_time;
    loop {
        if (time + offset) % cycle2 == 0 {
            return time;
        }
        time += cycle1;
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
        assert_eq!(part1("939
                          7,13,x,x,59,x,31,19"), 295);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("\n17,x,13,19"), 3417);
        assert_eq!(part2("\n67,7,59,61"), 754018);
    }
}
