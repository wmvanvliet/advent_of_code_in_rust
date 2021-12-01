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
    let mut sum = 0;
    for line in input.lines() {
        //println!("Parsing equation: {}", line);
        let chars:Vec<char> = line.trim().chars().filter(|&x| x != ' ').collect();
        let (val, _) = parse_equation(&chars[..], true);
        sum += val;
    }
    sum
}


/**
 * Solves part 2 of the puzzle.
 */
fn part2(input: &str) -> i64 {
    let mut sum = 0;
    for line in input.lines() {
        //println!("Parsing equation: {}", line);
        let chars:Vec<char> = line.trim().chars().filter(|&x| x != ' ').collect();
        let (val, _) = parse_equation(&chars[..], false);
        sum += val;
    }
    sum
}


/**
 * Parse an equation, such as 1 + 2 * 3, and return the result
 */
fn parse_equation(chars: &[char], part1: bool) -> (i64, &[char]) {
    let mut op_stack:Vec<char> = Vec::new();
    let mut exp_stack:Vec<i64> = Vec::new();

    let (exp, mut chars) = parse_expression(&chars, part1);
    exp_stack.push(exp);
    loop {
        let op = chars[0];
        chars = &chars[1..];

        // Handle operators with higher precedence
        while let Some(&prev_op) = op_stack.last() {
            if part1 || ((op == '*' || op == '/') && (prev_op == '+' || prev_op == '-')) {
                do_operator(&mut op_stack, &mut exp_stack);
            } else {
                break;
            }
        }
        op_stack.push(op);

        let (exp, c) = parse_expression(chars, part1);
        chars = c;
        exp_stack.push(exp);

        if chars.len() == 0 || chars[0] == ')' {
            while op_stack.len() > 0 {
                do_operator(&mut op_stack, &mut exp_stack);
            }
            return (exp_stack.pop().unwrap(), chars)
        }
    }
}

fn do_operator(op_stack: &mut Vec<char>, exp_stack: &mut Vec<i64>) {
    let op = op_stack.pop().unwrap();
    let right = exp_stack.pop().unwrap();
    let left = exp_stack.pop().unwrap();
    let val = match op {
        '+' => left + right,
        '-' => left - right,
        '*' => left * right,
        '/' => left / right,
        _ => panic!("Unknown operator")
    };
    exp_stack.push(val);
}

/**
 * Parse an expression, such as 1, 2, or (1 + 1), and return the result
 */
fn parse_expression(chars: &[char], part1: bool) -> (i64, &[char]) {
    match chars[0] {
        '(' => {
            let (val, chars) = parse_equation(&chars[1..], part1);
            // Eat the )
            assert_eq!(chars[0], ')'); 
            (val, &chars[1..])
        },
        '0'..='9' => (chars[0].to_digit(10).unwrap() as i64, &chars[1..]),
        _ => panic!("Expecing a number or ()"),
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
        assert_eq!(part1("1 + 1"), 2);
        assert_eq!(part1("(1 * 2) + (1 - 2)"), 1);
        assert_eq!(part1("1 + 2 * 3 + 4 * 5 + 6"), 71);
        assert_eq!(part1("1 + (2 * 3) + (4 * (5 + 6))"), 51);
        assert_eq!(part1("2 * 3 + (4 * 5)"), 26);
        assert_eq!(part1("5 + (8 * 3 + 9 + 3 * 4 * 3)"), 437);
        assert_eq!(part1("5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))"), 12240);
        assert_eq!(part1("((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2"), 13632);
    }

    #[test]
    fn test_part2() {
        assert_eq!(part2("1 + 2 * 3 + 4 * 5 + 6"), 231);
        assert_eq!(part2("1 + (2 * 3) + (4 * (5 + 6))"), 51);
        assert_eq!(part2("2 * 3 + (4 * 5)"), 46);
        assert_eq!(part2("5 + (8 * 3 + 9 + 3 * 4 * 3)"), 1445);
        assert_eq!(part2("5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))"), 669060);
        assert_eq!(part2("((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2"), 23340);
    }
}
