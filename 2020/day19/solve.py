from pprint import pprint

rules = dict()
messages = list()

with open('input.txt') as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            break
        num, rule = line.split(': ')
        num = int(num)
        branches = rule.split(' | ')
        branches = [[x[1:-1] if x.startswith('"') else int(x) for x in branch.split()]
                    for branch in branches]
        rules[num] = branches

    for line in f:
        messages.append(line.strip())

# 8: 42 | 42 8
# 11: 42 31 | 42 11 31
rules[8] = [[42], [42, 8]]
rules[11] = [[42, 31], [42, 11, 31]]


def apply_rule(num, msg):
    if len(msg) == 0:
        raise ValueError('Could not apply rule')
    # print(num, msg)
    for branch in rules[num]:
        pos_msg = str(msg)
        try:
            for part in branch:
                if type(part) == int:
                    pos_msg = apply_rule(part, pos_msg)
                else:
                    if pos_msg[0] != part:
                        raise ValueError('Coult not apply branch')
                    else:
                        pos_msg = pos_msg[1:]
        except ValueError:
            continue
        return pos_msg
    else:
        raise ValueError('Could not apply rule')

# pprint(rules)
# pprint(messages)

num_valid_msgs = 0
for msg in messages:
    try:
        msg = apply_rule(0, msg)
        if len(msg) == 0:
            num_valid_msgs += 1
    except ValueError:
        pass

print(num_valid_msgs)
