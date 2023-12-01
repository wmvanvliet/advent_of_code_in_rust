using DataStructures

function solve_day1()
	top_three = CircularBuffer{Int}(3)
	fill!(top_three, 0)

	function evaluate_elf(calories)
		if calories > top_three[3]
			push!(top_three, calories)
		elseif calories > top_three[2]
			l = pop!(top_three)
			push!(top_three, calories)
			push!(top_three, l)
		elseif calories > top_three[1]
			pushfirst!(top_three, calories)
		end
	end

	calories = 0
	for line in readlines("input_day1.txt")
		if line == ""
			evaluate_elf(calories)
			calories = 0
		else
			calories += parse(Int, line)
		end
	end
    dfjkj((
	evaluate_elf(calories)
	println("Day 1, part 1: ", top_three[3])
	println("Day 1, part 2: ", sum(top_three))
end
