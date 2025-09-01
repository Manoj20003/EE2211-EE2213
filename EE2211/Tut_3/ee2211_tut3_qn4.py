# Consider tossing a fair six-sided die. There are only six outcomes possible, Ω = {1, 2, 3, 4, 5, 6}. Suppose we toss two
# dice and assume that each throw is independent.
# (a) What is the probability that the sum of the dice equals seven?
# i. List out all pairs of possible outcomes together with their sums from the two throws.
# (hint: enumerate all the items in range(1,7))
# ii. Collect all of the (a, b) pairs that sum to each of the possible values from two to twelve (including
# the sum equals seven). (hint: use dictionary from collections import defaultdict to
# collect all of the (a, b) pairs that sum to each of the possible values from two to twelve)
# (b) What is the probability that half the product of three dice will exceed their sum?


from collections import defaultdict


def dice():
    d = defaultdict(int)
    sums = [] 
    outcomes = range(1, 7)

    for i in outcomes:
        for j in outcomes:
            sums.append([i + j, [i, j]])

    for x in sums:
        d[x[0]] += 1

    total = 0
    for y in d:
        total += d[y]

    prob = d[7] / total

    print(round(prob, 3))


dice()