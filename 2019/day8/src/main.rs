use std::io::{self, Read};

const IMG_WIDTH:usize = 25;
const IMG_HEIGHT:usize = 6;
const LAYER_SIZE:usize = IMG_WIDTH * IMG_HEIGHT;

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

fn part1(input_str: &str) -> u32 {
    const IMG_WIDTH:usize = 25;
    const IMG_HEIGHT:usize = 6;
    const LAYER_SIZE:usize = IMG_WIDTH * IMG_HEIGHT;

    let input:Vec<u32> = input_str.trim().chars()
        .map(|c| c.to_digit(10).unwrap()).collect();
    let n_layers = input.len() / (IMG_WIDTH * IMG_HEIGHT);

    // Extract a slice of the input vector that corresponds to a single layer
    let get_layer = |n| (&input[(n * LAYER_SIZE)..((n + 1) * LAYER_SIZE)]);

    // Count the number of times 'n' appears inside a layer
    let count_n = |layer, n| get_layer(layer).iter().map(|x| if *x == n {1} else {0}).sum::<u32>();

    let min_zeros_layer = (0..n_layers)
        .map(|layer| count_n(layer, 0))
        .enumerate().min_by(|(_, x), (_, y)| x.cmp(y)).unwrap().0; // argmin
    count_n(min_zeros_layer, 1) * count_n(min_zeros_layer, 2)
}

fn part2(input_str: &str) -> String {
    let input:Vec<u32> = input_str.trim().chars()
        .map(|c| c.to_digit(10).unwrap()).collect();
    let n_layers = input.len() / (IMG_WIDTH * IMG_HEIGHT);

    // Extract a slice of the input vector that corresponds to a single layer
    let get_layer = |n| (&input[(n * LAYER_SIZE)..((n + 1) * LAYER_SIZE)]);

    // Add the layers together
    let mut output:Vec<u32> = vec![2; LAYER_SIZE];
    for layer in 0..n_layers {
        for (i, num) in get_layer(layer).iter().enumerate() {
            if output[i] == 2 {
                output[i] = *num;
            }
        }
    }

    // Format the result
    let to_char = |x:&u32| {
        match x {
            0 => String::from("█"), // Unicode "full block" character
            _ => String::from(" "),
        }
    };
    let mut rows:Vec<String> = Vec::new();
    rows.push(String::from(""));
    for y in 0..IMG_HEIGHT {
        let row:Vec<String> = output[(y * IMG_WIDTH)..((y + 1) * IMG_WIDTH)].iter().map(to_char).collect();
        rows.push(row.join(""));
    }
    rows.join("\n")
}
