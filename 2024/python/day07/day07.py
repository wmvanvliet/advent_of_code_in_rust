def is_solvable(answer, terms, allow_concat=False):
    if len(terms) == 1:
        return terms[0] == answer
    elif terms[0] >= answer:
        return False
    else:  # len(terms) >= 2
        *first_terms, final_term = terms

        if allow_concat:
            # Check if the final digits of the answer match the final term
            final_term_str = str(final_term)
            answer_str = str(answer)
            if (
                len(answer_str) > len(final_term_str)
                and answer_str[-len(final_term_str) :] == final_term_str
            ):
                # Concatenation is an option for the solution. Check the rest.
                if is_solvable(
                    int(answer_str[: -len(final_term_str)]), first_terms, allow_concat
                ):
                    return True

        # Multiplication is only an option if the answer is dividable by the final term.
        if answer % final_term == 0:
            if is_solvable(answer // final_term, first_terms, allow_concat):
                return True

        # Addition is only an option if the answer is more than the final term.
        if answer > final_term:
            if is_solvable(answer - final_term, first_terms, allow_concat):
                return True

        # No options remain.
        return False


equations = list()
with open("day07.txt") as f:
    for line in f:
        answer, terms = line.split(": ", 1)
        answer = int(answer)
        terms = [int(t) for t in terms.split(" ")]
        equations.append((answer, terms))

part1 = sum(answer for answer, terms in equations if is_solvable(answer, terms, False))
part2 = sum(answer for answer, terms in equations if is_solvable(answer, terms, True))
print("part 1:", part1)
print("part 2:", part2)
