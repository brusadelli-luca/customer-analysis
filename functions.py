from math import sqrt

def dist_calc(P1, P2):

    x1, y1, x2, y2 = P1[0], P1[1], P2[0], P2[1]

    return round(sqrt( (x2 - x1)**2 + (y2 - y1)**2 ), 2)


def dist_calc_n(X, C):

    dists_c = [dist_calc(p, C) for p in X]

    return round(sum(dists_c), 2)
