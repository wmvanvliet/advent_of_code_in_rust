use std::io::{self, Read};

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

fn part1(input: &str) -> &str {
    let digits:Vec<i64> = input.trim().chars()
        .map(|x| x.to_digit(10).unwrap() as i64)
        .collect();

    let pattern = vec![0, 1, 0, -1];

    for output in 0..digits.len() {
        let pattern_iter = pattern.iter()
            .flat_map(|x| std::iter::repeat(x).take(output + 1))  // Repeat each element based on the output we are computing
            .cycle()  // Repeat the whole thing endlessly
            .skip(1);  // But skip the first element

        let result:i64 = digits.iter().zip(pattern_iter)
            .map(|(x, y)| x * y)
            .sum::<i64>()
            .abs()
            % 10;
        println!("{:?}", result);
    }

    "1"
}

fn part2(input: &str) -> i64 {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("12345678"), "48226158");
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
