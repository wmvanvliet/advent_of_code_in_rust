use std::io::{self, Read};
use std::collections::{HashSet, HashMap};
use regex::Regex;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {}", part1(&input));
    println!("Part 2 answer: {}", part2(&input));
}

fn part1(input: &str) -> i64 {
    let ore = String::from("ORE");

    let mut cookbook:HashMap<String, Recipe> = HashMap::new();
    let mut ingredients:HashSet<String> = HashSet::new();

    // ORE is free to produce
    cookbook.insert(ore.to_string(), Recipe {
        for_ingredient: ore.to_string(),
        num_produced: 1,
        required_ingredients: HashSet::new(),
        required_amounts: HashMap::new(),
    });

    // Read the other recipies from the input file
    for line in input.lines() {
        let mut parts:Vec<(i64, String)> = Vec::new();
        let re = Regex::new(r"(\d+) ([A-Z]+)").unwrap();
        for cap in re.captures_iter(line) {
            parts.push((cap[1].parse::<i64>().unwrap(), cap[2].to_string()));
            ingredients.insert(cap[2].to_string());
        }

        let (num_produced, for_ingredient) = parts.pop().unwrap();
        let mut required_ingredients:HashSet<String> = HashSet::new();
        let mut required_amounts:HashMap<String, i64> = HashMap::new();
        for (n, i) in parts {
            required_ingredients.insert(i.to_string());
            required_amounts.insert(i, n);
        }
        cookbook.insert(for_ingredient.to_string(), Recipe {
            for_ingredient: for_ingredient,
            num_produced: num_produced,
            required_ingredients: required_ingredients,
            required_amounts: required_amounts,
        });
    }

    let ingredient_order = get_ingredient_order(&cookbook);
    ore_needed(1, &cookbook, &ingredient_order)
}

fn part2(input: &str) -> i64 {
    let ore = String::from("ORE");

    let mut cookbook:HashMap<String, Recipe> = HashMap::new();
    let mut ingredients:HashSet<String> = HashSet::new();

    // ORE is free to produce
    cookbook.insert(ore.to_string(), Recipe {
        for_ingredient: ore.to_string(),
        num_produced: 1,
        required_ingredients: HashSet::new(),
        required_amounts: HashMap::new(),
    });

    // Read the other recipies from the input file
    for line in input.lines() {
        let mut parts:Vec<(i64, String)> = Vec::new();
        let re = Regex::new(r"(\d+) ([A-Z]+)").unwrap();
        for cap in re.captures_iter(line) {
            parts.push((cap[1].parse::<i64>().unwrap(), cap[2].to_string()));
            ingredients.insert(cap[2].to_string());
        }

        let (num_produced, for_ingredient) = parts.pop().unwrap();
        let mut required_ingredients:HashSet<String> = HashSet::new();
        let mut required_amounts:HashMap<String, i64> = HashMap::new();
        for (n, i) in parts {
            required_ingredients.insert(i.to_string());
            required_amounts.insert(i, n);
        }
        cookbook.insert(for_ingredient.to_string(), Recipe {
            for_ingredient: for_ingredient,
            num_produced: num_produced,
            required_ingredients: required_ingredients,
            required_amounts: required_amounts,
        });
    }

    let ingredient_order = get_ingredient_order(&cookbook);
    let mut max: i64 = 1_000_000_000_000;
    let mut min = max / ore_needed(1, &cookbook, &ingredient_order);
    while (max - min) > 1 {
        let t = min + (max - min) / 2;
        if ore_needed(t, &cookbook, &ingredient_order) <= 1_000_000_000_000 {
            min = t;
        } else {
            max = t;
        }
    }
    min
}

#[derive(Debug)]
struct Recipe {
    for_ingredient: String,
    num_produced: i64,
    required_ingredients: HashSet<String>,
    required_amounts: HashMap<String, i64>,
}

// Topological sorting algorithm
fn get_ingredient_order(cookbook:&HashMap<String, Recipe>) -> Vec<String> {
    let fuel = String::from("FUEL");
    let mut ingredient_order:Vec<String> = Vec::new();
    fn visit<'a>(ingredient:&'a String, cookbook:&'a HashMap<String, Recipe>, ingredient_order:&mut Vec<String>) {
        if ingredient_order.contains(&ingredient) {
            return;
        }
        for i in &cookbook.get(ingredient).unwrap().required_ingredients {
            visit(i, cookbook, ingredient_order);
        }
        ingredient_order.push(ingredient.to_string());
    }
    visit(&fuel, &cookbook, &mut ingredient_order);
    ingredient_order.reverse();
    ingredient_order
}

fn ore_needed(n:i64, cookbook:&HashMap<String, Recipe>, ingredient_order:&Vec<String>) -> i64 {
    let ore = String::from("ORE");
    let fuel = String::from("FUEL");
    let mut needed:HashMap<&String, i64> = HashMap::new();
    needed.insert(&fuel, n);
    for ingredient in ingredient_order {
        let recipe = cookbook.get(ingredient).unwrap();
        let times = ((*needed.get(ingredient).unwrap() as f64) / (recipe.num_produced as f64)).ceil() as i64;
        for req_ingredient in recipe.required_ingredients.iter() {
            let req_amount = recipe.required_amounts.get(req_ingredient).unwrap();
            *needed.entry(req_ingredient).or_insert(0) += times * req_amount;
        }
    }

    *needed.get(&ore).unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let input = "10 ORE => 10 A
                     1 ORE => 1 B
                     7 A, 1 B => 1 C
                     7 A, 1 C => 1 D
                     7 A, 1 D => 1 E
                     7 A, 1 E => 1 FUEL";
        assert_eq!(part1(&input), 31);

        let input = "9 ORE => 2 A
                     8 ORE => 3 B
                     7 ORE => 5 C
                     3 A, 4 B => 1 AB
                     5 B, 7 C => 1 BC
                     4 C, 1 A => 1 CA
                     2 AB, 3 BC, 4 CA => 1 FUEL";
        assert_eq!(part1(&input), 165);

        let input = "157 ORE => 5 NZVS
                     165 ORE => 6 DCFZ
                     44 XJWVT, 5 KHKGT, 1 QDVJ, 29 NZVS, 9 GPVTF, 48 HKGWZ => 1 FUEL
                     12 HKGWZ, 1 GPVTF, 8 PSHF => 9 QDVJ
                     179 ORE => 7 PSHF
                     177 ORE => 5 HKGWZ
                     7 DCFZ, 7 PSHF => 2 XJWVT
                     165 ORE => 2 GPVTF
                     3 DCFZ, 7 NZVS, 5 HKGWZ, 10 PSHF => 8 KHKGT";
        assert_eq!(part1(&input), 13312);

        let input = "2 VPVL, 7 FWMGM, 2 CXFTF, 11 MNCFX => 1 STKFG
                     17 NVRVD, 3 JNWZP => 8 VPVL
                     53 STKFG, 6 MNCFX, 46 VJHF, 81 HVMC, 68 CXFTF, 25 GNMV => 1 FUEL
                     22 VJHF, 37 MNCFX => 5 FWMGM
                     139 ORE => 4 NVRVD
                     144 ORE => 7 JNWZP
                     5 MNCFX, 7 RFSQX, 2 FWMGM, 2 VPVL, 19 CXFTF => 3 HVMC
                     5 VJHF, 7 MNCFX, 9 VPVL, 37 CXFTF => 6 GNMV
                     145 ORE => 6 MNCFX
                     1 NVRVD => 8 CXFTF
                     1 VJHF, 6 MNCFX => 4 RFSQX
                     176 ORE => 6 VJHF";
        assert_eq!(part1(&input), 180697);

        let input = "171 ORE => 8 CNZTR
                     7 ZLQW, 3 BMBT, 9 XCVML, 26 XMNCP, 1 WPTQ, 2 MZWV, 1 RJRHP => 4 PLWSL
                     114 ORE => 4 BHXH
                     14 VRPVC => 6 BMBT
                     6 BHXH, 18 KTJDG, 12 WPTQ, 7 PLWSL, 31 FHTLT, 37 ZDVW => 1 FUEL
                     6 WPTQ, 2 BMBT, 8 ZLQW, 18 KTJDG, 1 XMNCP, 6 MZWV, 1 RJRHP => 6 FHTLT
                     15 XDBXC, 2 LTCX, 1 VRPVC => 6 ZLQW
                     13 WPTQ, 10 LTCX, 3 RJRHP, 14 XMNCP, 2 MZWV, 1 ZLQW => 1 ZDVW
                     5 BMBT => 4 WPTQ
                     189 ORE => 9 KTJDG
                     1 MZWV, 17 XDBXC, 3 XCVML => 2 XMNCP
                     12 VRPVC, 27 CNZTR => 2 XDBXC
                     15 KTJDG, 12 BHXH => 5 XCVML
                     3 BHXH, 2 VRPVC => 7 MZWV
                     121 ORE => 7 VRPVC
                     7 XCVML => 6 RJRHP
                     5 BHXH, 4 VRPVC => 5 LTCX";
        assert_eq!(part1(&input), 2210736);
    }

    #[test]
    fn test_part2() {
        let input = "157 ORE => 5 NZVS
                     165 ORE => 6 DCFZ
                     44 XJWVT, 5 KHKGT, 1 QDVJ, 29 NZVS, 9 GPVTF, 48 HKGWZ => 1 FUEL
                     12 HKGWZ, 1 GPVTF, 8 PSHF => 9 QDVJ
                     179 ORE => 7 PSHF
                     177 ORE => 5 HKGWZ
                     7 DCFZ, 7 PSHF => 2 XJWVT
                     165 ORE => 2 GPVTF
                     3 DCFZ, 7 NZVS, 5 HKGWZ, 10 PSHF => 8 KHKGT";
        assert_eq!(part2(&input), 82892753);

        let input = "2 VPVL, 7 FWMGM, 2 CXFTF, 11 MNCFX => 1 STKFG
                     17 NVRVD, 3 JNWZP => 8 VPVL
                     53 STKFG, 6 MNCFX, 46 VJHF, 81 HVMC, 68 CXFTF, 25 GNMV => 1 FUEL
                     22 VJHF, 37 MNCFX => 5 FWMGM
                     139 ORE => 4 NVRVD
                     144 ORE => 7 JNWZP
                     5 MNCFX, 7 RFSQX, 2 FWMGM, 2 VPVL, 19 CXFTF => 3 HVMC
                     5 VJHF, 7 MNCFX, 9 VPVL, 37 CXFTF => 6 GNMV
                     145 ORE => 6 MNCFX
                     1 NVRVD => 8 CXFTF
                     1 VJHF, 6 MNCFX => 4 RFSQX
                     176 ORE => 6 VJHF";
        assert_eq!(part2(&input), 5586022);

        let input = "171 ORE => 8 CNZTR
                     7 ZLQW, 3 BMBT, 9 XCVML, 26 XMNCP, 1 WPTQ, 2 MZWV, 1 RJRHP => 4 PLWSL
                     114 ORE => 4 BHXH
                     14 VRPVC => 6 BMBT
                     6 BHXH, 18 KTJDG, 12 WPTQ, 7 PLWSL, 31 FHTLT, 37 ZDVW => 1 FUEL
                     6 WPTQ, 2 BMBT, 8 ZLQW, 18 KTJDG, 1 XMNCP, 6 MZWV, 1 RJRHP => 6 FHTLT
                     15 XDBXC, 2 LTCX, 1 VRPVC => 6 ZLQW
                     13 WPTQ, 10 LTCX, 3 RJRHP, 14 XMNCP, 2 MZWV, 1 ZLQW => 1 ZDVW
                     5 BMBT => 4 WPTQ
                     189 ORE => 9 KTJDG
                     1 MZWV, 17 XDBXC, 3 XCVML => 2 XMNCP
                     12 VRPVC, 27 CNZTR => 2 XDBXC
                     15 KTJDG, 12 BHXH => 5 XCVML
                     3 BHXH, 2 VRPVC => 7 MZWV
                     121 ORE => 7 VRPVC
                     7 XCVML => 6 RJRHP
                     5 BHXH, 4 VRPVC => 5 LTCX";
        assert_eq!(part2(&input), 460664);
    }
}
