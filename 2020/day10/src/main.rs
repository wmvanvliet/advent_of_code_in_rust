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
fn part1(input: &str) -> i64 {
    let mut jolts:Vec<i64> = input.lines().map(|x| x.trim().parse().unwrap()).collect();
    jolts.sort();
    jolts.insert(0, 0);
    jolts.push(jolts[jolts.len()-1] + 3);
    let mut n_diff_1 = 0;
    let mut n_diff_3 = 0;
    for (x, y) in jolts[..(jolts.len()-1)].iter().zip(jolts[1..].iter()) {
        match y - x {
            1 => n_diff_1 += 1,
            3 => n_diff_3 += 1,
            _ => (),
        }
    }

    n_diff_1 * n_diff_3
}

/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut jolts:Vec<i64> = input.lines().map(|x| x.trim().parse().unwrap()).collect();
    jolts.sort();
    jolts.insert(0, 0);
    jolts.push(jolts[jolts.len()-1] + 3);
    println!("{:?}", jolts);
    let mut n_combs:Vec<i64> = vec![1, 1];
    for i in 2..jolts.len() {
        let mut n_comb:i64 = 0;

        if (i > 2) && ((jolts[i] - jolts[i - 3]) < 4) {
            n_comb += n_combs[i - 3];
        }
        if (jolts[i] - jolts[i - 2]) < 4 {
            n_comb += n_combs[i - 2];
        }
        if (jolts[i] - jolts[i - 1]) < 4 {
            n_comb += n_combs[i - 1];
        }
        println!("{} ({}, {}) {}", jolts[i], jolts[i-2], jolts[i-1], n_comb);
        n_combs.push(n_comb)
    }
    n_combs[n_combs.len()-1]
}

/**
 * Unit tests! All the examples given in the puzzle descriptions are added here as unit tests.
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("16
                          10
                          15
                          5
                          1
                          11
                          7
                          19
                          6
                          12
                          4"), 35);

        assert_eq!(part1("28
                          33
                          18
                          42
                          31
                          14
                          46
                          20
                          48
                          47
                          24
                          23
                          49
                          45
                          19
                          38
                          39
                          11
                          1
                          32
                          25
                          35
                          8
                          17
                          7
                          9
                          4
                          2
                          34
                          10
                          3"), 220);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("16
                          10
                          15
                          5
                          1
                          11
                          7
                          19
                          6
                          12
                          4"), 8);

        assert_eq!(part2("28
                          33
                          18
                          42
                          31
                          14
                          46
                          20
                          48
                          47
                          24
                          23
                          49
                          45
                          19
                          38
                          39
                          11
                          1
                          32
                          25
                          35
                          8
                          17
                          7
                          9
                          4
                          2
                          34
                          10
                          3"), 19208);
    }
}
