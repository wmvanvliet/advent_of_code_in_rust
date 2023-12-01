answer = 0

for line in eachline("input_day1.txt")
    numbers = []
    for char in line
        try
            push!(numbers, parse(Integer, char))
        catch
            continue
        end
    end
    global answer += numbers[1] * 10 + numbers[end]
end

println(answer)
