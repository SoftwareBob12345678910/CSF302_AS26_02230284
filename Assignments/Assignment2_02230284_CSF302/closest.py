"""Closest pair of camera traps (Assignment 2, CSF302).

Two implementations of the same problem:

    closest_pair_brute(points)  -> (d, i, j)   Theta(n^2), checks every pair
    closest_pair(points)        -> (d, i, j)   Theta(n log n), divide and conquer

Both take a list of (x, y) points with n >= 2 and return the smallest distance
``d`` together with the indices ``i < j`` (positions in the ORIGINAL list) of a
pair of points that achieves it.  If several pairs tie for the minimum, either
function may return any one of them; the distance ``d`` is always the same.

Distances are compared as squared distances (no square root, and exact for
integer coordinates); the square root is taken once, at the very end.
"""

import math

__all__ = ["closest_pair_brute", "closest_pair"]

# Sub-problems of at most this many points are solved by brute force.
_BASE_CASE = 3


# --------------------------------------------------------------------------
# Part A.1  Brute force: Theta(n^2)
# --------------------------------------------------------------------------
def closest_pair_brute(points):
    """Return (d, i, j) by comparing every pair of points: n(n-1)/2 checks."""
    n = len(points)
    if n < 2:
        raise ValueError("at least 2 points are required")

    best = math.inf
    best_i, best_j = 0, 1
    for i in range(n - 1):
        xi, yi = points[i]
        for j, (xj, yj) in enumerate(points[i + 1:], i + 1):
            dx = xj - xi
            dy = yj - yi
            d2 = dx * dx + dy * dy
            if d2 < best:
                best, best_i, best_j = d2, i, j
    return math.sqrt(best), best_i, best_j


# --------------------------------------------------------------------------
# Part A.2  Divide and conquer: Theta(n log n)
# --------------------------------------------------------------------------
def closest_pair(points):
    """Return (d, i, j) using divide and conquer in Theta(n log n) time.

    1. Sort the points by x once (Theta(n log n)); each point carries its
       original index so the answer can be reported in the caller's order.
    2. Split the x-sorted list at the middle and solve both halves recursively.
       Each recursive call also hands back its points sorted by y, so the
       parent obtains a y-sorted list with a Theta(n) merge (no re-sorting).
    3. Let delta be the smaller of the two half answers.  Any pair closer than
       delta must straddle the dividing line and lie in the vertical strip of
       width 2*delta around it.  Scan the strip in y order; each point only has
       to be compared with the few points above it that are less than delta
       higher (at most 7 by the packing argument).
    """
    n = len(points)
    if n < 2:
        raise ValueError("at least 2 points are required")

    # (x, y, original_index); sorting the tuples sorts by x, then y, then index.
    by_x = sorted((p[0], p[1], i) for i, p in enumerate(points))
    (d2, i, j), _ = _solve(by_x)
    return math.sqrt(d2), i, j


def _solve(by_x):
    """Solve one sub-problem.

    ``by_x`` is a list of (x, y, index) sorted by x.  Returns
    ``((best_d2, i, j), by_y)`` where ``by_y`` is the same points sorted by y.
    """
    n = len(by_x)

    # ---- base case: Theta(1) work for at most 3 points ----
    if n <= _BASE_CASE:
        best = (math.inf, -1, -1)
        for a in range(n - 1):
            xa, ya, ia = by_x[a]
            for b in range(a + 1, n):
                xb, yb, ib = by_x[b]
                d2 = (xb - xa) ** 2 + (yb - ya) ** 2
                if d2 < best[0]:
                    best = (d2, min(ia, ib), max(ia, ib))
        return best, sorted(by_x, key=_y_of)

    # ---- divide: split by position, so duplicate x values cannot unbalance it ----
    mid = n // 2
    mid_x = by_x[mid][0]                    # x of the dividing line
    best_left, left_by_y = _solve(by_x[:mid])
    best_right, right_by_y = _solve(by_x[mid:])

    # ---- combine: merge by y in Theta(n), then scan the strip in Theta(n) ----
    best = min(best_left, best_right)        # tuples compare by d2 first
    by_y = _merge_by_y(left_by_y, right_by_y)

    # Keep only points strictly closer than delta to the dividing line.
    strip = [p for p in by_y if (p[0] - mid_x) ** 2 < best[0]]

    for a in range(len(strip)):
        xa, ya, ia = strip[a]
        for b in range(a + 1, len(strip)):
            xb, yb, ib = strip[b]
            dy = yb - ya
            if dy * dy >= best[0]:           # too far above: nothing further can help
                break
            d2 = (xb - xa) ** 2 + dy * dy
            if d2 < best[0]:
                best = (d2, min(ia, ib), max(ia, ib))

    return best, by_y


def _y_of(point):
    return point[1]


def _merge_by_y(a, b):
    """Merge two lists of (x, y, index) that are each sorted by y: Theta(len(a)+len(b))."""
    out = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i][1] <= b[j][1]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


if __name__ == "__main__":
    demo = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    print(closest_pair_brute(demo))   # (1.4142135623730951, 0, 5)
    print(closest_pair(demo))         # (1.4142135623730951, 0, 5)