use std::io::{self, Read};
use std::collections::HashMap;

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
    let mut lines = input.lines();
    let directions1 = lines.next().unwrap();
    let directions2 = lines.next().unwrap();
    let wire1_path = trace_path(directions1);
    let wire2_path = trace_path(directions2);
    return wire1_path.keys() // Iterate over all visited locations
        .filter(|loc| wire2_path.contains_key(loc)) // Get intersection points
        .map(|i| { i.0.abs() + i.1.abs() }) // Compute Manhattan distance
        .min().unwrap_or(0);
}

fn part2(input: &str) -> i64 {
    let mut lines = input.lines();
    let wire1_directions = lines.next().unwrap();
    let wire2_directions = lines.next().unwrap();
    let wire1_path = trace_path(wire1_directions);
    let wire2_path = trace_path(wire2_directions);
    return wire1_path.keys() // Iterate over all visited locations
        .filter(|loc| wire2_path.contains_key(loc)) // Get intersection points
        .map(|loc| { wire1_path[loc] + wire2_path[loc] }) // Add number of steps for both wires
        .min().unwrap_or(0);
}

/// Construct a HashMap containing as keys all the (x, y) locations visited by a wire. The values
/// are the total number of steps taken to reach the location. If a wire has visited a location
/// more than once, only the visit with the least amount of steps taken is stored.
fn trace_path(directions:&str) -> HashMap<(i64, i64), i64> {
    let mut path:HashMap<(i64, i64), i64> = HashMap::new();
    let mut x:i64 = 0;  // Current location of the wire
    let mut y:i64 = 0;
    let mut total_steps:i64 = 0;
    for instruction in directions.split(',') {
        let (direction, n_steps) = instruction.split_at(1);
        let mut dx:i64 = 0;
        let mut dy:i64 = 0;
        match direction {
            "R" => dx = 1,
            "L" => dx = -1,
            "U" => dy = 1,
            "D" => dy = -1,
            _ => panic!(),
        }
        for _ in 0..n_steps.parse::<i64>().unwrap() {
            x += dx;
            y += dy;
            total_steps += 1;
            path.entry((x, y)).or_insert(total_steps);
        }
    }
    path
}
   

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("R75,D30,R83,U83,L12,D49,R71,U7,L72\nU62,R66,U55,R34,D71,R55,D58,R83"), 159);
        assert_eq!(part1("R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51\nU98,R91,D20,R16,D67,R40,U7,R15,U6,R7"), 135);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("R75,D30,R83,U83,L12,D49,R71,U7,L72\nU62,R66,U55,R34,D71,R55,D58,R83"), 610);
        assert_eq!(part2("R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51\nU98,R91,D20,R16,D67,R40,U7,R15,U6,R7"), 410);
    }
}
