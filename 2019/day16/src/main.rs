use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(strip_bom(&input), 100));
    println!("Part 2 answer: {}", part2(strip_bom(&input)));
}

fn strip_bom(input: &str) -> &str {
    return if input.starts_with("\u{feff}") {
        &input[3..]
    } else {
        input
    }
}

fn part1(input: &str, n_phases: i64) -> String {
    let mut digits:Vec<i64> = input.trim().chars()
        .map(|x| x.to_digit(10).unwrap() as i64)
        .collect();

    let pattern = vec![0, 1, 0, -1];

    for _ in 0..n_phases {
        let mut phase_result:Vec<i64> = Vec::new();

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
            phase_result.push(result);
        }

        digits = phase_result;
    }

    let mut answer = digits.into_iter().map(|x| x.to_string()).collect::<String>();
    answer.truncate(8);
    answer
}

fn part2(input: &str) -> i64 {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        assert_eq!(part1("12345678", 4), "01029498");
        assert_eq!(part1("80871224585914546619083218645595", 100), "24176176");
        assert_eq!(part1("19617804207202209144916044189917", 100), "73745418");
        assert_eq!(part1("69317163492948606335995924319873", 100), "52432133");
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2(""), 2);
    }
}
