use std::io::{self, Read};
use std::collections::HashSet;
use std::iter::FromIterator;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("Part 1 answer: {:?}", part1(strip_bom(&input)));
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
    // Parse the map to find the coordinates of all astroids.
    let mut astroids:HashSet<(i64, i64)> = HashSet::new();
    for (y, line) in input.lines().enumerate() {
        for (x, _) in line.trim().match_indices('#') {
            astroids.insert((x as i64, y as i64));
        }
    }
    
    // Determine the most other astroids ever seen
    astroids.iter()
        .map(|(x_origin, y_origin)| can_see(&astroids, *x_origin, *y_origin).len())
        .max().unwrap() as i64
}

fn part2(input: &str) -> i64 {
    // Parse the map to find the coordinates of all astroids.
    let mut astroids:HashSet<(i64, i64)> = HashSet::new();
    for (y, line) in input.lines().enumerate() {
        for (x, _) in line.trim().match_indices('#') {
            astroids.insert((x as i64, y as i64));
        }
    }

    // Determine astroid that sees the most other astroids
    let (x_origin, y_origin) = *astroids.iter()
        .max_by(|(x1, y1), (x2, y2)| {
            let n1 = can_see(&astroids, *x1, *y1).len();
            let n2 = can_see(&astroids, *x2, *y2).len();
            n1.cmp(&n2)
        }).unwrap();

    // Determine the 200th astroid that is shot 
    let mut astroids_shot:i64 = 0;
    loop {
        let shot = can_see(&astroids, x_origin, y_origin);
        for asteroid in &shot {
            astroids.remove(asteroid); // BOOM!
            astroids_shot += 1;
            if astroids_shot == 200 {
                return asteroid.0 * 100 + asteroid.1;
            }
        }
    }
}

/// Find greatest common divisor using Euclids method
fn gcd(a:i64, b:i64) -> i64 {
    if a == 0 {
        b
    } else {
        gcd(b % a, a)
    }
}

/// Find all points along the way between (0, 0) and (x, y),
/// excluding the two end points.
fn points_along_the_way(x:i64, y:i64) -> Vec<(i64, i64)> {
    let d = gcd(x, y).abs();
    let stride_x = x / d;
    let stride_y = y / d;
    (1..d).map(|i| (i * stride_x, i * stride_y)).collect()
}

/// Compute which astroids can be seen from a certain origin point.
/// Results are returned in clockwise order.
fn can_see(astroids:&HashSet<(i64, i64)>, x_origin:i64, y_origin:i64) -> Vec<(i64, i64)> {
    // For each astroid, determine the number of other astroids it can see
    let mut can_see:Vec<(i64, i64)> = Vec::new();
    for (x_target, y_target) in astroids.iter() {
        if x_origin == *x_target && y_origin == *y_target {
            continue
        }
        // Check all intermediate points between the origin astroid and the target astroid for
        // possible blocking astroids.
        let mut view_blocked = false;
        for (x_inter, y_inter) in points_along_the_way(x_target - x_origin, y_target - y_origin) {
            if astroids.contains(&(x_origin + x_inter, y_origin + y_inter)) {
                view_blocked = true;
                break;
            }
        }
        if !view_blocked {
            can_see.push((*x_target, *y_target));
        }
    }

    // Sort in clockwise order
    can_see.sort_by(|(x1, y1), (x2, y2)| {
        let angle1 = clockwise_angle(x_origin, y_origin, *x1, *y1);
        let angle2 = clockwise_angle(x_origin, y_origin, *x2, *y2);
        angle1.partial_cmp(&angle2).unwrap()
    });

    can_see
}

/// Compute the angle between two points in a clockwise manner.
/// 12h is 0 rad
/// 3h is 1/2 * π rad
/// 6h is π rad
/// 9h is 3/2 * π rad
fn clockwise_angle(x_origin:i64, y_origin:i64, x_target:i64, y_target:i64) -> f64 {
    let mut angle = ((x_target - x_origin) as f64).atan2((y_origin - y_target) as f64);
    // Make angle increase monotonically instead of becoming negative
    if angle < 0.0 {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(10, 15), 5);
        assert_eq!(gcd(35, 10), 5);
        assert_eq!(gcd(31, 2), 1);
        assert_eq!(gcd(0, -2), -2);
    }

    #[test]
    fn test_points_along_the_way() {
        assert_eq!(points_along_the_way(0, -2), vec![(0, -1)]);
        assert_eq!(points_along_the_way(3, 3), vec![(1, 1), (2, 2)]);
        assert_eq!(points_along_the_way(15, 10), vec![(3, 2), (6, 4), (9, 6), (12, 8)]);
        assert_eq!(points_along_the_way(9, 7), vec![]);
    }

    #[test]
    fn test_part1() {
        let input = ".#..#
                     .....
                     #####
                     ....#
                     ...##";
        assert_eq!(part1(&input), 8);

        let input = "......#.#.
                     #..#.#....
                     ..#######.
                     .#.#.###..
                     .#..#.....
                     ..#....#.#
                     #..#....#.
                     .##.#..###
                     ##...#..#.
                     .#....####";
        assert_eq!(part1(&input), 33);

        let input ="#.#...#.#.
                    .###....#.
                    .#....#...
                    ##.#.#.#.#
                    ....#.#.#.
                    .##..###.#
                    ..#...##..
                    ..##....##
                    ......#...
                    .####.###.";
        assert_eq!(part1(&input), 35);

        let input = ".#..#..###
                     ####.###.#
                     ....###.#.
                     ..###.##.#
                     ##.##.#.#.
                     ....###..#
                     ..#.#..#.#
                     #..#.#.###
                     .##...##.#
                     .....#.#..";
        assert_eq!(part1(&input), 41);

        let input =".#..##.###...#######
                    ##.############..##.
                    .#.######.########.#
                    .###.#######.####.#.
                    #####.##.#.##.###.##
                    ..#####..#.#########
                    ####################
                    #.####....###.#.#.##
                    ##.#################
                    #####.##.###..####..
                    ..######..##.#######
                    ####.##.####...##..#
                    .#####..#.######.###
                    ##...#.##########...
                    #.##########.#######
                    .####.#.###.###.#.##
                    ....##.##.###..#####
                    .#.#.###########.###
                    #.#.#.#####.####.###
                    ###.##.####.##.#..##";
        assert_eq!(part1(&input), 210);
    }

    #[test]
    fn test_angle() {
        assert_eq!(clockwise_angle(1, 1, 1, 0), 0.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 2, 0), 1.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 2, 1), 2.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 2, 2), 3.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 1, 2), 4.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 0, 2), 5.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 0, 1), 6.0 / 4.0 * std::f64::consts::PI);
        assert_eq!(clockwise_angle(1, 1, 0, 0), 7.0 / 4.0 * std::f64::consts::PI);
    }

    #[test]
    fn test_can_see() {
        // These astroids are defined in a random order
        let astroids:HashSet<(i64, i64)> = HashSet::from_iter([(1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2)].iter().cloned());
        // Result should be in clockwise order
        assert_eq!(can_see(&astroids, 1, 1), vec![(1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0)]);
    }
}
