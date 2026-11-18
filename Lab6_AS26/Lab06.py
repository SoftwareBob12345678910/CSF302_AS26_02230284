#!/usr/bin/env python3
"""
CSF302 - Lab 6: Dynamic Programming
===================================
Q1  Rod cutting         (naive / top-down memoized / bottom-up)
Q2  Matrix chain        (bottom-up DP, parenthesisation, brute-force check)
Q3  Longest common subsequence (table, reconstruction, two rows, brute force, text diff)
Q4  Optimal binary search tree (optional)
"""

import argparse
import csv
import math
import os
import random
import sys
import time

OUT_DIR = "results6"
SEED = 2026


def timed_avg(fn, min_total=0.05):
    """Average running time of fn() in seconds (repeats fast calls so the clock can measure them)."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first >= min_total:
        return first
    reps = max(2, int(min_total / max(first, 1e-7)))
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# =====================================================================
# Q1  ROD CUTTING
# =====================================================================
BASE_PRICES = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]      # p[0] = 0, p[1..10]


def price_table(n):
    """Prices for lengths 0..n; beyond 10 the lab uses p[i] = 3*i."""
    return BASE_PRICES[:n + 1] + [3 * i for i in range(len(BASE_PRICES), n + 1)]


def rod_naive(p, n, stats):
    """Plain recursion: T(n) = 1 + T(0) + ... + T(n-1) = 2^n calls."""
    stats["calls"] += 1
    if n == 0:
        return 0
    best = -1
    for i in range(1, n + 1):
        v = p[i] + rod_naive(p, n - i, stats)
        if v > best:
            best = v
    return best


def rod_memo(p, n, stats):
    """Top-down: same recursion, but every subproblem's answer is stored the first time."""
    memo = [None] * (n + 1)
    cut = [0] * (n + 1)

    def solve(m):
        stats["calls"] += 1
        if memo[m] is not None:                 # already solved -> constant-time return
            return memo[m]
        if m == 0:
            memo[0] = 0
            return 0
        best = -1
        for i in range(1, m + 1):
            v = p[i] + solve(m - i)
            if v > best:
                best, cut[m] = v, i
        memo[m] = best
        return best

    return solve(n), cut


def rod_bottom_up(p, n, stats):
    """Bottom-up: solve lengths 1, 2, ..., n in order; no recursion at all."""
    rev = [0] * (n + 1)
    cut = [0] * (n + 1)
    for j in range(1, n + 1):
        best = -1
        for i in range(1, j + 1):
            stats["iterations"] += 1
            v = p[i] + rev[j - i]
            if v > best:
                best, cut[j] = v, i
        rev[j] = best
    return rev[n], cut


def pieces_from_cuts(cut, n):
    out = []
    while n > 0:
        out.append(cut[n])
        n -= cut[n]
    return out


def run_q1(quick=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Q1 part 4 - maximum revenue and the pieces, n = 1..10")
    p10 = price_table(10)
    print(f"  prices p[1..10] = {p10[1:]}")
    print(f"{'n':>4}{'max revenue':>14}   pieces")
    for n in range(1, 11):
        r1 = rod_naive(p10, n, {"calls": 0})
        r2, _ = rod_memo(p10, n, {"calls": 0})
        r3, cut = rod_bottom_up(p10, n, {"iterations": 0})
        assert r1 == r2 == r3, "the three versions disagree!"
        pieces = pieces_from_cuts(cut, n)
        assert sum(pieces) == n and sum(p10[x] for x in pieces) == r3
        print(f"{n:>4}{r3:>14}   {' + '.join(map(str, pieces))}")
    print("  (all three versions give the same revenue for every n)")

    sizes = [5, 10, 15, 20] if quick else [5, 10, 15, 20, 25]
    rows = []
    print("\nQ1 part 5/6 - recursive calls and running time (p[i] = 3*i for i > 10)")
    print(f"{'n':>3}{'naive calls':>14}{'2^n':>12}{'memo calls':>12}{'1+n(n+1)/2':>12}"
          f"{'bottom-up iters':>17}{'t naive (s)':>13}{'t memo (s)':>12}{'t bottom-up (s)':>17}")
    for n in sizes:
        p = price_table(n)
        s1, s2, s3 = {"calls": 0}, {"calls": 0}, {"iterations": 0}
        t1 = timed_avg(lambda: rod_naive(p, n, {"calls": 0}))
        r1 = rod_naive(p, n, s1)
        t2 = timed_avg(lambda: rod_memo(p, n, {"calls": 0}))
        r2, _ = rod_memo(p, n, s2)
        t3 = timed_avg(lambda: rod_bottom_up(p, n, {"iterations": 0}))
        r3, _ = rod_bottom_up(p, n, s3)
        assert r1 == r2 == r3
        assert s1["calls"] == 2 ** n and s2["calls"] == 1 + n * (n + 1) // 2 and s3["iterations"] == n * (n + 1) // 2
        rows.append((n, r1, s1["calls"], s2["calls"], s3["iterations"], t1, t2, t3))
        print(f"{n:>3}{s1['calls']:>14,}{2 ** n:>12,}{s2['calls']:>12,}{1 + n * (n + 1) // 2:>12,}"
              f"{s3['iterations']:>17,}{t1:>13.6f}{t2:>12.6f}{t3:>17.6f}", flush=True)
    with open(f"{OUT_DIR}/q1_rod_cutting.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "max_revenue", "naive_calls", "memo_calls", "bottom_up_iterations",
                    "naive_s", "memo_s", "bottom_up_s"])
        w.writerows(rows)
    plot_q1(rows)


def plot_q1(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ns = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ns, [r[2] for r in rows], "o-", label="naive  (2^n)")
    axes[0].plot(ns, [r[3] for r in rows], "s-", label="memoized  (1 + n(n+1)/2)")
    axes[0].plot(ns, [r[4] for r in rows], "^--", label="bottom-up iterations  (n(n+1)/2)")
    axes[0].set_yscale("log")
    axes[0].set_title("Number of recursive calls / loop iterations")
    axes[1].plot(ns, [r[5] for r in rows], "o-", label="naive")
    axes[1].plot(ns, [r[6] for r in rows], "s-", label="memoized")
    axes[1].plot(ns, [r[7] for r in rows], "^-", label="bottom-up")
    axes[1].set_yscale("log")
    axes[1].set_title("Running time (s)")
    for ax in axes:
        ax.set_xlabel("rod length n")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q1_rod_cutting.png", dpi=130)
    plt.close(fig)


# =====================================================================
# Q2  MATRIX CHAIN MULTIPLICATION
# Matrix A_i has size p[i-1] x p[i], i = 1..n  (so len(p) = n + 1).
# =====================================================================

def matrix_chain(p):
    """Bottom-up DP.  m[i][j] = fewest scalar multiplications for A_i..A_j,
    s[i][j] = the split point k that achieves it.  Time O(n^3), space O(n^2)."""
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):                  # chain length
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = math.inf
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j], s[i][j] = q, k
    return m, s


def parenthesise(s, i, j):
    if i == j:
        return f"A{i}"
    k = s[i][j]
    return "(" + parenthesise(s, i, k) + parenthesise(s, k + 1, j) + ")"


def cost_of_split_tree(p, s, i, j):
    """Re-evaluate the cost by walking the chosen parenthesisation (independent check)."""
    if i == j:
        return 0
    k = s[i][j]
    return (cost_of_split_tree(p, s, i, k) + cost_of_split_tree(p, s, k + 1, j)
            + p[i - 1] * p[k] * p[j])


def left_to_right_cost(p):
    """Cost of ((A1 A2) A3) ... An."""
    return sum(p[0] * p[i] * p[i + 1] for i in range(1, len(p) - 1))


def matrix_chain_brute(p, i=None, j=None):
    """Try every parenthesisation (Catalan(n-1) of them) with plain recursion."""
    if i is None:
        i, j = 1, len(p) - 1
    if i == j:
        return 0
    return min(matrix_chain_brute(p, i, k) + matrix_chain_brute(p, k + 1, j) + p[i - 1] * p[k] * p[j]
               for k in range(i, j))


def run_q2(quick=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    tests = [("CLRS example", [30, 35, 15, 5, 10, 20, 25]), ("second example", [5, 4, 6, 2, 7])]
    for t in range(1, 6):
        n = rng.randint(4, 8)
        tests.append((f"random chain {t}", [rng.randint(5, 60) for _ in range(n + 1)]))

    rows = []
    for name, p in tests:
        n = len(p) - 1
        m, s = matrix_chain(p)
        best = m[1][n]
        par = parenthesise(s, 1, n)
        ltr = left_to_right_cost(p)
        brute = matrix_chain_brute(p) if n <= 8 else None
        assert cost_of_split_tree(p, s, 1, n) == best
        assert brute is None or brute == best
        rows.append((name, p, best, par, ltr, brute))
        print(f"\n{name}: dimensions <{', '.join(map(str, p))}>  ({n} matrices)")
        print(f"  optimal cost          : {best:,}")
        print(f"  optimal parenthesis.  : {par}")
        print(f"  left-to-right cost    : {ltr:,}  ({ltr / best:.2f}x the optimum)")
        print(f"  brute-force check     : {brute:,}  -> {'MATCH' if brute == best else 'MISMATCH'}")
    assert rows[0][2] == 15125, "expected 15,125 for the CLRS example"
    print("\n  CLRS example gives 15,125 as expected.")

    m, s = matrix_chain(tests[0][1])
    print("\n  m[i][j] table for the CLRS example (rows i, columns j):")
    print("       " + "".join(f"{j:>8}" for j in range(1, 7)))
    for i in range(1, 7):
        print(f"  {i:>3}  " + "".join(f"{(m[i][j] if j >= i else ''):>8}" for j in range(1, 7)))

    trials = 100 if quick else 500
    for _ in range(trials):
        n = rng.randint(1, 8)
        p = [rng.randint(1, 40) for _ in range(n + 1)]
        mm, ss = matrix_chain(p)
        assert mm[1][n] == matrix_chain_brute(p) == cost_of_split_tree(p, ss, 1, n)
    print(f"\n  extra stress test: DP == brute force on {trials} more random chains of 1..8 matrices  OK")

    with open(f"{OUT_DIR}/q2_matrix_chain.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "dimensions", "optimal_cost", "parenthesisation", "left_to_right_cost", "brute_force"])
        for name, p, best, par, ltr, brute in rows:
            w.writerow([name, " ".join(map(str, p)), best, par, ltr, brute])


# =====================================================================
# Q3  LONGEST COMMON SUBSEQUENCE
# =====================================================================

def lcs_table(x, y):
    """c[i][j] = length of an LCS of x[:i] and y[:j].  Time and space O(mn)."""
    m, n = len(x), len(y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i - 1][j], c[i][j - 1])
    return c


def print_lcs_table(c, x, y):
    print("        " + "  ".join(f"{ch:>2}" for ch in " " + y))
    for i, row in enumerate(c):
        label = "ε" if i == 0 else x[i - 1]
        print(f"   {label:>2}   " + "  ".join(f"{v:>2}" for v in row))


def lcs_reconstruct(c, x, y):
    """Walk back from c[m][n]; one LCS (ties are resolved by moving up)."""
    i, j = len(x), len(y)
    out = []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            out.append(x[i - 1])
            i, j = i - 1, j - 1
        elif c[i - 1][j] >= c[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def lcs_all(c, x, y):
    """Bonus: every distinct LCS string (follows both directions on ties)."""
    memo = {}

    def go(i, j):
        if i == 0 or j == 0:
            return {""}
        if (i, j) in memo:
            return memo[(i, j)]
        if x[i - 1] == y[j - 1]:
            res = {s + x[i - 1] for s in go(i - 1, j - 1)}
        else:
            res = set()
            if c[i - 1][j] >= c[i][j - 1]:
                res |= go(i - 1, j)
            if c[i][j - 1] >= c[i - 1][j]:
                res |= go(i, j - 1)
        memo[(i, j)] = res
        return res

    return sorted(go(len(x), len(y)))


def lcs_length_two_rows(x, y):
    """Length only, with O(min(m, n)) memory: row i only needs row i-1."""
    if len(y) > len(x):
        x, y = y, x                                    # keep rows as short as possible
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if xi == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] > prev[j] else prev[j]
        prev = cur
    return prev[-1]


def is_subsequence(sub, y):
    it = iter(y)
    return all(ch in it for ch in sub)               # `in` on an iterator consumes it: one left-to-right pass


def lcs_brute(x, y):
    """Try all 2^m subsequences of x; keep the longest that is also a subsequence of y."""
    m = len(x)
    best = ""
    for mask in range(1 << m):
        sub = "".join(x[i] for i in range(m) if (mask >> i) & 1)
        if len(sub) > len(best) and is_subsequence(sub, y):
            best = sub
    return best


def line_diff(a_lines, b_lines):
    """Unchanged / removed / added lines using the LCS of the two line lists."""
    c = lcs_table(a_lines, b_lines)
    i, j = len(a_lines), len(b_lines)
    out = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a_lines[i - 1] == b_lines[j - 1]:
            out.append(("  ", a_lines[i - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and (i == 0 or c[i][j - 1] >= c[i - 1][j]):
            out.append(("+ ", b_lines[j - 1]))
            j -= 1
        else:
            out.append(("- ", a_lines[i - 1]))
            i -= 1
    return list(reversed(out))


def diff_files(path_a, path_b):
    with open(path_a, encoding="utf-8") as f:
        a = f.read().splitlines()
    with open(path_b, encoding="utf-8") as f:
        b = f.read().splitlines()
    print(f"--- {path_a}\n+++ {path_b}")
    result = line_diff(a, b)
    for tag, line in result:
        print(tag + line)
    same = sum(1 for t, _ in result if t == "  ")
    print(f"\n{same} unchanged, {sum(1 for t, _ in result if t == '- ')} removed, "
          f"{sum(1 for t, _ in result if t == '+ ')} added")


DEMO_OLD = """def greet(name):
    print("Hello " + name)

def add(a, b):
    return a + b

print(greet("Ann"))
print(add(1, 2))
"""
DEMO_NEW = """import sys

def greet(name):
    print(f"Hello {name}")

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

print(add(1, 2))
"""


def run_q3(quick=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    X, Y = "ABCBDAB", "BDCABA"
    c = lcs_table(X, Y)
    print(f"Q3 part 1 - LCS table for X = {X}, Y = {Y}\n")
    print_lcs_table(c, X, Y)
    lcs = lcs_reconstruct(c, X, Y)
    print(f"\nQ3 part 2 - one LCS: {lcs}   (length {len(lcs)} = c[{len(X)}][{len(Y)}] = {c[-1][-1]})")
    print(f"           all LCSs: {', '.join(lcs_all(c, X, Y))}")
    assert is_subsequence(lcs, X) and is_subsequence(lcs, Y) and len(lcs) == c[-1][-1]
    print(f"Q3 part 3 - two-row version returns length {lcs_length_two_rows(X, Y)}")
    assert lcs_length_two_rows(X, Y) == c[-1][-1]

    rng = random.Random(SEED)
    for _ in range(300):                                 # randomised correctness test of the DP versions
        a = "".join(rng.choice("ABC") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("ABC") for _ in range(rng.randint(0, 12)))
        cc = lcs_table(a, b)
        assert cc[-1][-1] == lcs_length_two_rows(a, b) == len(lcs_brute(a, b))
        assert len(lcs_reconstruct(cc, a, b)) == cc[-1][-1]
    print("           (two-row, table and brute force agree on 300 random string pairs)")

    lengths = list(range(5, 15)) if quick else list(range(5, 19))
    rows = []
    print("\nQ3 part 4 - DP vs brute force (random strings over ABCD, |X| = |Y| = L)")
    print(f"{'L':>4}{'LCS len':>9}{'DP time (s)':>14}{'brute time (s)':>16}{'brute / DP':>13}")
    for L in lengths:
        a = "".join(rng.choice("ABCD") for _ in range(L))
        b = "".join(rng.choice("ABCD") for _ in range(L))
        t_dp = timed_avg(lambda: lcs_table(a, b))
        t0 = time.perf_counter()
        best = lcs_brute(a, b)
        t_bf = time.perf_counter() - t0
        assert len(best) == lcs_table(a, b)[-1][-1]
        rows.append((L, len(best), t_dp, t_bf))
        print(f"{L:>4}{len(best):>9}{t_dp:>14.6f}{t_bf:>16.4f}{t_bf / t_dp:>13,.0f}", flush=True)
    with open(f"{OUT_DIR}/q3_lcs_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "lcs_length", "dp_s", "brute_s"])
        w.writerows(rows)
    plot_q3(rows)

    old_p, new_p = f"{OUT_DIR}/demo_old.txt", f"{OUT_DIR}/demo_new.txt"
    with open(old_p, "w", encoding="utf-8") as f:
        f.write(DEMO_OLD)
    with open(new_p, "w", encoding="utf-8") as f:
        f.write(DEMO_NEW)
    print("\nQ3 part 5 - line-by-line diff of two small files (your own: python lab06.py diff a.txt b.txt)\n")
    diff_files(old_p, new_p)


def plot_q3(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="DP table  O(mn)")
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "s-", label="brute force  O(2^m · n)")
    ax.set_yscale("log")
    ax.set_xlabel("string length L")
    ax.set_ylabel("time (s)")
    ax.set_title("LCS: dynamic programming vs brute force")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q3_lcs_timing.png", dpi=130)
    plt.close(fig)


# =====================================================================
# Q4  OPTIMAL BINARY SEARCH TREE (optional)
# Probabilities are scaled by 100 and kept as integers, so the DP is
# exact (no floating-point rounding); results are divided by 100 to print.
# =====================================================================

def optimal_bst(p, q):
    """p[1..n] key probabilities (p[0] unused), q[0..n] dummy-key probabilities (integers).
    e[i][j] = expected cost of an optimal tree on keys k_i..k_j, w[i][j] = sum of their probabilities,
    root[i][j] = best root.  Time O(n^3)."""
    n = len(p) - 1
    e = [[0] * (n + 2) for _ in range(n + 2)]
    w = [[0] * (n + 2) for _ in range(n + 2)]
    root = [[0] * (n + 1) for _ in range(n + 2)]
    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]                         # empty tree = just the dummy key d_{i-1}
        w[i][i - 1] = q[i - 1]
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            e[i][j] = math.inf
            w[i][j] = w[i][j - 1] + p[j] + q[j]
            for r in range(i, j + 1):
                t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if t < e[i][j]:
                    e[i][j], root[i][j] = t, r
    return e, w, root


def bst_structure(root, i, j, parent, side, depth, lines, p, q):
    """Print the tree rooted at the best root of k_i..k_j; returns its expected-cost contribution (scaled)."""
    pad = "    " * depth
    if j < i:                                          # dummy leaf d_j
        lines.append(f"{pad}d{j} is the {side} child of {parent}    (prob {q[j] / 100:.2f}, depth {depth})")
        return q[j] * (depth + 1)
    r = root[i][j]
    if parent is None:
        lines.append(f"k{r} is the root    (prob {p[r] / 100:.2f}, depth 0)")
    else:
        lines.append(f"{pad}k{r} is the {side} child of {parent}    (prob {p[r] / 100:.2f}, depth {depth})")
    cost = p[r] * (depth + 1)
    cost += bst_structure(root, i, r - 1, f"k{r}", "left", depth + 1, lines, p, q)
    cost += bst_structure(root, r + 1, j, f"k{r}", "right", depth + 1, lines, p, q)
    return cost


def bst_brute(p, q, i, j):
    """Try every possible root recursively (no memo) - used only to double-check the DP on small inputs."""
    w = sum(p[i:j + 1]) + sum(q[i - 1:j + 1])
    if j < i:
        return q[i - 1]
    return min(bst_brute(p, q, i, r - 1) + bst_brute(p, q, r + 1, j) for r in range(i, j + 1)) + w


def run_q4(quick=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = [0] + [round(x * 100) for x in (0.15, 0.10, 0.05, 0.10, 0.20)]
    q = [round(x * 100) for x in (0.05, 0.10, 0.05, 0.05, 0.05, 0.10)]
    assert sum(p) + sum(q) == 100, "probabilities must sum to 1"
    e, w, root = optimal_bst(p, q)
    n = len(p) - 1
    print("Q4 - optimal binary search tree")
    print(f"  p = 0.15 0.10 0.05 0.10 0.20     q = 0.05 0.10 0.05 0.05 0.05 0.10")
    print(f"  expected search cost e[1][{n}] = {e[1][n] / 100:.2f}   (expected: 2.75)")
    assert e[1][n] == 275

    print("\n  root[i][j] table:")
    print("       " + "".join(f"{j:>5}" for j in range(1, n + 1)))
    for i in range(1, n + 1):
        print(f"  {i:>3}  " + "".join(f"{(root[i][j] if j >= i else ''):>5}" for j in range(1, n + 1)))
    print("\n  e[i][j] table (expected cost of the best subtree on k_i..k_j):")
    print("       " + "".join(f"{j:>7}" for j in range(0, n + 1)))
    for i in range(1, n + 2):
        print(f"  {i:>3}  " + "".join(f"{(e[i][j] / 100 if j >= i - 1 else ''):>7}" for j in range(0, n + 1)))

    lines = []
    cost = bst_structure(root, 1, n, None, "", 0, lines, p, q)
    print("\n  Structure of the optimal tree (each key's depth counts as depth + 1 comparisons):\n")
    for ln in lines:
        print("  " + ln)
    print(f"\n  cost recomputed from the printed tree = {cost / 100:.2f}")
    assert cost == e[1][n] == bst_brute(p, q, 1, n)
    print("  brute-force recursion over every possible tree agrees.")

    rng = random.Random(SEED)
    for _ in range(40 if quick else 150):              # stress test on random distributions
        n2 = rng.randint(1, 6)
        raw = [rng.randint(1, 9) for _ in range(2 * n2 + 1)]
        p2, q2 = [0] + raw[:n2], raw[n2:]
        e2, _, _ = optimal_bst(p2, q2)
        assert e2[1][n2] == bst_brute(p2, q2, 1, n2)
    print("  extra stress test: DP == brute force on random instances  OK")


# =====================================================================
def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="CSF302 Lab 6 - dynamic programming")
    ap.add_argument("part", choices=["q1", "q2", "q3", "q4", "all", "diff"])
    ap.add_argument("files", nargs="*", help="diff: old.txt new.txt")
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke test")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.out
    sys.setrecursionlimit(10000)
    if args.part == "diff":
        if len(args.files) != 2:
            ap.error("diff needs two files:  python lab06.py diff old.txt new.txt")
        diff_files(*args.files)
        return
    for name, fn in (("q1", run_q1), ("q2", run_q2), ("q3", run_q3), ("q4", run_q4)):
        if args.part in (name, "all"):
            if args.part == "all":
                print("\n" + "=" * 78 + f"\n{name.upper()}\n" + "=" * 78)
            fn(args.quick)


if __name__ == "__main__":
    main()