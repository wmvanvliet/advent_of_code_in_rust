using DataStructures
using Base.Iterators

function day1(input::IO)
    pattern = "1|2|3|4|5|6|7|8|9|one|two|three|four|five|six|seven|eight|nine"
    convert = Dict(map(x -> (x[2], (x[1] - 1) % 9 + 1), enumerate(split(pattern, "|"))))

    answer_part1 = 0
    answer_part2 = 0
    for line in eachline(input)
        digits_part1 = map(m -> parse(Int64, m[1]), eachmatch(r"([1-9])", line))
        digits_part2 = map(m -> convert[m[1]], eachmatch(Regex("($pattern)"), line, overlap=true))
        answer_part1 += digits_part1[1] * 10 + digits_part1[end]
        answer_part2 += digits_part2[1] * 10 + digits_part2[end]
    end

    return answer_part1, answer_part2
end

function day2(input::IO)
    answer_part1 = 0
    answer_part2 = 0
    for line in eachline(input)
        game, draws = split(line, ": ")
        game = parse(Int64, split(game, " ")[2])
        max_red = 0
        max_green = 0
        max_blue = 0
        for draw in split(draws, "; ")
            for cube in split(draw, ", ")
                number, color = split(cube, " ")
                number = parse(Int64, number)
                if color == "red" && number > max_red
                    max_red = number
                elseif color == "green" && number > max_green
                    max_green = number
                elseif color == "blue" && number > max_blue
                    max_blue = number
                end
            end
        end
        if max_red <= 12 && max_green <= 13 && max_blue <= 14
            answer_part1 += game
        end
        answer_part2 += max_red * max_green * max_blue
    end
    return answer_part1, answer_part2
end

function day3(input::IO)
    answer_part1 = 0
    answer_part2 = 0
    numbers = []
    symbols = []
    for (y, line) in enumerate(eachline(input))
        for m in eachmatch(r"(\d+)", line)
            push!(numbers, (y, UnitRange(m.offset, m.offset + length(m.match) - 1), parse(Int64, m.match)))
        end
        for m in eachmatch(r"[^\d\.]", line)
            push!(symbols, (y, m.offset, m.match))
        end
    end

    # Part1
    for (num_y, num_xrange, num) in numbers
        for y in (num_y - 1):(num_y + 1)
            for x in (num_xrange.start - 1):(num_xrange.stop + 1)
                for (symb_y, symb_x, symb) in symbols
                    if symb_x == x && symb_y == y
                        answer_part1 += num
                    end
                end
            end
        end
    end

    # Part2
    for (symb_y, symb_x, symb) in symbols
        if symb != "*"
            continue
        end
        ratios = Set()
        for y in (symb_y - 1):(symb_y + 1)
            for x in (symb_x - 1):(symb_x + 1)
                for (num_y, num_xrange, num) in numbers
                    if num_y == y && x in num_xrange
                        push!(ratios, num)
                    end
                end
            end
        end
        if length(ratios) == 2
            ratio = 1
            for r in ratios
                ratio *= r
            end
            answer_part2 += ratio
        end
    end

    return answer_part1, answer_part2
end

function day4(input::IO)
    answer_part1 = 0
    answer_part2 = 0
    n_cards = DefaultDict(1)
    for line in eachline(input)
        card, numbers = split(line, ": ")
        card = parse(Int64, split(card)[2])
        winning_numbers, numbers_on_card = split(numbers, " | ")
        winning_numbers = Set([parse(Int64, n) for n in split(winning_numbers)])
        numbers_on_card = Set([parse(Int64, n) for n in split(numbers_on_card)])
        n_correct = length(intersect(winning_numbers, numbers_on_card))
        if n_correct > 0
            answer_part1 += 2 ^ (n_correct - 1)
        end
        for i in card + 1:card + n_correct
            n_cards[i] += n_cards[card]
        end
        if !haskey(n_cards, card)
            n_cards[card] = 1
        end
    end
    answer_part2 = sum(values(n_cards))
    return answer_part1, answer_part2
end


function read_map(input::IO)
    map::Dict{UnitRange, Int64} = Dict()
    readline(input)
    while (line = readline(input)) != ""
        destination_start, source_start, len = [parse(Int64, x) for x in split(line)]
        map[range(source_start, source_start + len - 1)] = destination_start - source_start
    end
    return map
end

function apply_map(map::Dict, seed_ranges)
    destinations::Vector{UnitRange} = []
    for (source_range, offset) in map
        next_seed_ranges = []
        for seed_range in seed_ranges
            modified_range = intersect(seed_range, source_range)
            if length(modified_range) == 0
                # No part of the seed range is modified
                push!(next_seed_ranges, seed_range)
                continue
            end
            if length(modified_range) == length(seed_range)
                # The entire seed range is modified
                push!(destinations, range(modified_range.start + offset, modified_range.stop + offset))
                continue
            end
            # Part of the seed range is modified, part of the range will be modified,
            # and we could be left with 1 or 2 parts of the original range.
            push!(destinations, range(modified_range.start + offset, modified_range.stop + offset))
            if modified_range.start > seed_range.start
                push!(next_seed_ranges, range(seed_range.start, modified_range.start - 1))
            end
            if modified_range.stop < seed_range.stop
                push!(next_seed_ranges, range(modified_range.stop + 1, seed_range.stop))
            end
        end
        seed_ranges = next_seed_ranges
    end
    # Any seed ranges left unmodified
    append!(destinations, seed_ranges)
    return destinations
end

function day5(input::IO)
    seeds = [parse(Int64, x) for x in
             split(chopprefix(readline(input), "seeds: "))]
    readline(input)

    # Read all the maps
    maps = [read_map(input) for i in 1:7]

    # Part 1, every seed is a range of length 1
    destinations = [range(seed, seed) for seed in seeds]
    for map in maps
         destinations = apply_map(map, destinations)
    end
    answer_part1 = minimum([location_range.start for location_range in destinations])

    # Part 2, proper seed ranges
    destinations = [range(start, start + len - 1)
                    for (start, len) in zip(seeds[1:2:end], seeds[2:2:end])]
    for map in maps
         destinations = apply_map(map, destinations)
    end
    answer_part2 = minimum([location_range.start for location_range in destinations])

    return answer_part1, answer_part2
end

function day6(input::IO)
    duration = (time, distance) -> floor(0.5 * (time - sqrt(time^2 - 4*distance))) + 1
    flexibility = (time, record) -> trunc(Int64, floor(time - duration(time, record)) - ceil(duration(time, record))) + 1

    # Part 1
    times = [parse(Int64, x) for x in split(readline(input))[2:end]]
    records = [parse(Int64, x) for x in split(readline(input))[2:end]]
    println([flexibility(t, d) for (t, d) in zip(times, records)])
    answer_part1 = prod([flexibility(t, d) for (t, d) in zip(times, records)])

    # Part 2
    time = parse(Int64, join(times))
    record = parse(Int64, join(records))
    answer_part2 = flexibility(time, record)

    return answer_part1, answer_part2
end

function day7(input::IO)
    hands = []
    scores_part1 = []
    scores_part2 = []
    bets = []
    for line in eachline(input)
        (hand, bet) = split(line)
        push!(hands, collect(hand))
        push!(bets, parse(Int64, bet))
        score_part1, score_part2 = hand_score(collect(hand))
        push!(scores_part1, score_part1)
        push!(scores_part2, score_part2)

    end

    # high-card, one-pair, three-of-a-kind, four-of-a-kind, yatzee
    pair_scores = [0, 1, 3, 5, 7]

    # Part 1
    scores_part1 = []
    for hand in hands
        pairs = diff([0; findall(diff(sort(hand)) .!= 0); length(hand)]);
        scores_part1.append(sum(pair_scores[pairs]))
    end
    card_value = Dict(zip("23456789TJQKA", 1:13))
    order = sortperm([(s, [card_value[c] for c in h]) for (s, h) in zip(scores_part1, hands)])
    answer_part1 = 0
    for (r, b) in enumerate(bets[order])
        answer_part1 += r * b
    end

    # Part 2
    scores_part2 = []
    for hand in hands
        # remove jokers
        n_jokers = count(==('J'), hand)
        hand = filter(!=('J'), hand)
        pairs = diff([0; findall(diff(sort(hand)) .!= 0); length(hand)]);

        # jokers add to the largest pair (this maximizes score)
        val, ind = findmax(pairs)
        pairs[ind] = val + n_jokers
        scores_part2.append(sum(values[pairs]))
    end
    card_value = Dict(zip("J23456789TQKA", 1:13))
    order = sortperm([(s, [card_value[c] for c in h]) for (s, h) in zip(scores_part2, hands)])
    answer_part2 = 0
    for (r, b) in enumerate(bets[order])
        answer_part2 += r * b
    end

    return answer_part1, answer_part2
end


function walk_to_z(node, connections, directions)
    n_steps = 0
    for (step, direction) in cycle(enumerate(directions))
        node = connections[node][direction]
        n_steps += 1
        if endswith(node, "Z")
            return n_steps, step
        end
    end
end

function day8(input::IO)
    directions = collect(readline(input))
    readline(input)

    connections = Dict()
    for line in eachline(input)
        (name, left, right) = match(r"([A-Z]+) = \(([A-Z]+), ([A-Z]+)\)", line)
        connections[name] = Dict('L' => left, 'R' => right)
    end

    start_nodes = [name for name in keys(connections) if endswith(name, 'A')]
    steps_until_end_reached = []
    for current_node in start_nodes
        steps = 0
        for direction in cycle(directions)
            current_node = connections[current_node][direction]
            steps += 1
            if endswith(current_node, 'Z')
                println(current_node, " ", steps)
                push!(steps_until_end_reached, steps)
                break
            end
        end
    end

    return steps_until_end_reached[1], lcm(steps_until_end_reached...)
end

function day9(input::IO)
    answer_part1 = 0
    answer_part2 = 0
    for line in eachline(input)
        next_val = 0
        first_vals = []
        readings = [parse(Int64, x) for x in split(line)]
        while !all(readings .== 0)
            push!(first_vals, first(readings))
            next_val += last(readings)
            readings = diff(readings)
        end
        answer_part1 += next_val

        prev_val = 0
        for v in reverse(first_vals)
            prev_val = v - prev_val
        end
        answer_part2 += prev_val
    end
    return answer_part1, answer_part2
end


function add_dir(a::Tuple{Int64, Int64}, b::Tuple{Int64, Int64})
    return (a[1] + b[1], a[2] + b[2])
end

function day10(input::IO)
    answer_part1 = 0
    answer_part2 = 0

    pieces = Dict(
        '|' => Dict((0, 1) => (0, 1), (0, -1) => (0, -1)), # a vertical pipe connecting north and south.
        '-' => Dict((1, 0) => (1, 0), (-1, 0) => (-1, 0)), # a horizontal pipe connecting east and west.
        'L' => Dict((0, 1) => (1, 0), (-1, 0) => (0, -1)), # a 90-degree bend connecting north and east.
        'J' => Dict((0, 1) => (-1, 0), (1, 0) => (0, -1)), # a 90-degree bend connecting north and west.
        '7' => Dict((1, 0) => (0, 1), (0, -1) => (-1, 0)), # a 90-degree bend connecting south and west.
        'F' => Dict((0, -1) => (1, 0), (-1, 0) => (0, 1)), # a 90-degree bend connecting south and east.
    )
    possible_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]


    # Parse map
    map = Dict()
    map_width = 0
    map_height = 0
    path = Set()
    start_pos = undef
    for (y, line) in enumerate(eachline(input))
        map_width = length(line)
        map_height += 1
        for (x, char) in enumerate(line)
            map[(x, y)] = char
            if char == 'S'
                start_pos = (x, y)
            end
        end
    end

    # Find possible ways to go
    animal_pos = start_pos
    animal_dir = undef

    for possible_dir in possible_dirs
        try_pos = add_dir(animal_pos, possible_dir)
        if (try_pos in keys(map))
            piece = map[try_pos]
            if piece in keys(pieces)
                piece_dirs = pieces[piece]
                if possible_dir in keys(piece_dirs)
                    animal_dir = possible_dir
                    break
                end
            end
        end
    end

    while true
        push!(path, animal_pos)
        animal_pos = add_dir(animal_pos, animal_dir)
        answer_part1 += 1
        if map[animal_pos] == 'S'
            break
        end
        animal_dir = pieces[map[animal_pos]][animal_dir]
    end
    answer_part1 /= 2

    for y in 1:map_height
        for x in 1:map_width
            if !((x, y) in path)
                n_crossings_x = 0
                for z in (x + 1):map_width
                    if ((z, y) in path)
                        piece = map[(z, y)]
                        if piece == '|' || piece == 'L' || piece == 'J'
                            n_crossings_x += 1
                        end
                    end
                end
                if (n_crossings_x % 2) != 0
                    answer_part2 += 1
                end
            end
        end
    end

    return answer_part1, answer_part2
end

function day11(input::IO)
    answer_part1 = 0
    answer_part2 = 0

    # parse input
    galaxy_xs = []
    galaxy_ys = []
    for (y, line) in enumerate(eachline(input))
        for (x, char) in enumerate(line)
            if char == '#'
                push!(galaxy_xs, x)
                push!(galaxy_ys, y)
            end
        end
    end

    # expand space
    empty_xs = setdiff(1:maximum(galaxy_xs), galaxy_xs)
    empty_ys = setdiff(1:maximum(galaxy_ys), galaxy_ys)
    galaxy_xs_part1 = [x + sum(empty_xs .< x) for x in galaxy_xs]
    galaxy_ys_part1 = [y + sum(empty_ys .< y) for y in galaxy_ys]
    galaxy_xs_part2 = [x + (999999 * sum(empty_xs .< x)) for x in galaxy_xs]
    galaxy_ys_part2 = [y + (999999 * sum(empty_ys .< y)) for y in galaxy_ys]

    # compute all-to-all distances
    for i in 1:length(galaxy_xs)
        for j in i:length(galaxy_xs)
            dist_part1 = abs(galaxy_xs_part1[i] - galaxy_xs_part1[j]) + abs(galaxy_ys_part1[i] - galaxy_ys_part1[j])
            dist_part2 = abs(galaxy_xs_part2[i] - galaxy_xs_part2[j]) + abs(galaxy_ys_part2[i] - galaxy_ys_part2[j])
            answer_part1 += dist_part1
            answer_part2 += dist_part2
        end
    end

    return answer_part1, answer_part2
end


function place_chunks(map, sequence, depth=0)
    # for i in 1:depth
    #     print(" ")
    # end

    # println(" ", map, " ", sequence)
    if length(sequence) == 0
        return 1
    end

    if length(map) == 0
        return 0
    end

    chunk_len = first(sequence)
    if length(map) < (chunk_len + 1)
        return 0
    end

    n_ways = []
    for start in 1:(length(map) - chunk_len)
        if map[start] != '.' && map[start] != '?'
            break
        end
        stop = start + chunk_len
        candidate_range = map[start+1:stop]
        if all((candidate_range .== '#') .| (candidate_range .== '?'))
            n = place_chunks(map[stop+1:end], sequence[2:end], depth+1)
            # println(n)
            n_ways += n
        end
    end
    return n_ways
end

function day12(input::IO)
    answer_part1 = 0
    for line in eachline(input)
        (map, sequence) = split(("." * line), ' ')
        map = split(map, '.')
        map = [length(x) for x in map if length(x) > 0]
        sequence = [parse(Int64, c) for c in split(sequence, ',')]
        println(line, " ", map, " ", sequence)
    end

    return answer_part1
end
