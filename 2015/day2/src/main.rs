use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let mut paper_needed:i64 = 0;
    for line in input.lines() {
        let dims = parse_dims(line);
        paper_needed += areas(&get_sides(&dims)).iter().sum::<i64>();

        // Add the smallest area again for slack
        paper_needed += areas(&get_sides(&dims)).iter().min().unwrap();
    }

    return paper_needed;
}

fn part2(input: &str) -> i64 {
    let mut ribbon_needed:i64 = 0;
    for line in input.lines() {
        let mut dims = parse_dims(line);

        // By storing the dimensions sorted, we ensure that the sides are also 
        // computed shortest to longest.
        dims.sort();
        let sides = get_sides(&dims);
        ribbon_needed += 2 * sides[0][0] + 2 * sides[0][1];

        // Add bow
        ribbon_needed += volume(&dims);
    }

    return ribbon_needed;
}

/// Parse dimension strings of the form "a x b x c"
fn parse_dims(line:&str) -> Vec<i64> {
    let dims = line.trim_end()
        .split('x')
        .map(|d| {d.parse().unwrap()})
        .collect::<Vec<i64>>();

    assert_eq!(dims.len(), 3);
    return dims;
}

/// Get the lengths of the sides of a box, given its dimensions
fn get_sides(dims:&Vec<i64>) -> [[i64; 2]; 6] {
    return [
        [dims[0], dims[1]],
        [dims[0], dims[2]],
        [dims[1], dims[2]],
        [dims[0], dims[1]],
        [dims[0], dims[2]],
        [dims[1], dims[2]],
    ];
}

/// Compute the area of each sides of a box, given the length of the sides
fn areas(sides:&[[i64; 2]; 6]) -> Vec<i64> {
    sides.iter().map(|side| {side[0] * side[1]}).collect()
}

/// Compute the volume of a box, given its dimensions
fn volume(dims:&Vec<i64>) -> i64  {
    dims.iter().product::<i64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("2x3x4"), 58);
        assert_eq!(part1("1x1x10"), 43);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("2x3x4"), 34);
        assert_eq!(part2("1x1x10"), 14);
    }
}
