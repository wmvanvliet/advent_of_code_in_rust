def eval(line):
    hand, bid = line.split()
    hand = hand.translate(str.maketrans('TJQKA', face))
    best = max(type(hand.replace('0', r)) for r in hand)
    return best, hand, int(bid)

def type(hand):
    return sorted(map(hand.count, hand), reverse=True)

for face in 'ABCDE', 'A0CDE':
    ans = 0
    for rank, (*x, bid) in enumerate(sorted(map(eval, open('input_day7.txt'))), start=1):
        print(rank, x, bid)
        ans += rank * bid
