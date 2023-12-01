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
