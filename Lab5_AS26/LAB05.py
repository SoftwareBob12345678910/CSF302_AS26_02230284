#!/usr/bin/env python3
"""
CSF302 - Lab 5: Sorting Algorithms
==================================
Q1  Quicksort variants (Lomuto, Hoare, randomised, three-way)
Q2  Counting sort and LSD radix sort (vs merge sort)
Q3  Counting inversions (brute force vs merge sort) + Kendall tau distance
"""

import argparse
import csv
import itertools
import os
import random
import sys
import time

OUT_DIR = "results"
SEED = 2026


# =====================================================================
# Q1  QUICKSORT VARIANTS
# ---------------------------------------------------------------------
# Every variant sorts the list IN PLACE and returns the number of key
# comparisons it made.  What counts as one comparison:
#   * Lomuto   : each "a[j] <= pivot" test
#   * Hoare    : each "a[i] < pivot" and "a[j] > pivot" test
#   * Random   : same as Lomuto (random pivot swapped to the end)
#   * 3-way    : each "x < pivot" and "x > pivot" test that is evaluated
# To keep the stack shallow every variant recurses on the SMALLER part
# and loops on the larger one, so the recursion depth is <= log2(n)
# even when the partitions are completely unbalanced (sorted input).
# =====================================================================

def quicksort_lomuto(a):
    """Lomuto partition, pivot = last element."""
    cmp = 0

    def partition(lo, hi):
        nonlocal cmp
        pivot = a[hi]
        i = lo - 1
        for j in range(lo, hi):
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
        cmp += hi - lo                       # one test per scanned element
        a[i + 1], a[hi] = a[hi], a[i + 1]
        return i + 1                         # pivot is now in its final place

    def sort(lo, hi):
        while lo < hi:
            p = partition(lo, hi)
            if p - lo < hi - p:              # left part is smaller
                sort(lo, p - 1)              # recurse on the smaller part
                lo = p + 1                   # loop on the larger part
            else:
                sort(p + 1, hi)
                hi = p - 1

    sort(0, len(a) - 1)
    return cmp


def quicksort_hoare(a, pivot_rule="middle"):
    """Hoare partition.  pivot_rule = 'middle' (default) or 'first'."""
    cmp = 0

    def partition(lo, hi):
        nonlocal cmp
        pivot = a[(lo + hi) // 2] if pivot_rule == "middle" else a[lo]
        i, j = lo - 1, hi + 1
        while True:
            i += 1
            cmp += 1
            while a[i] < pivot:
                i += 1
                cmp += 1
            j -= 1
            cmp += 1
            while a[j] > pivot:
                j -= 1
                cmp += 1
            if i >= j:
                return j                     # parts are [lo..j] and [j+1..hi]
            a[i], a[j] = a[j], a[i]

    def sort(lo, hi):
        while lo < hi:
            p = partition(lo, hi)
            if p - lo + 1 < hi - p:
                sort(lo, p)
                lo = p + 1
            else:
                sort(p + 1, hi)
                hi = p

    sort(0, len(a) - 1)
    return cmp


def quicksort_randomised(a):
    """Lomuto partition with a uniformly random pivot."""
    cmp = 0

    def partition(lo, hi):
        nonlocal cmp
        r = random.randint(lo, hi)
        a[r], a[hi] = a[hi], a[r]            # move random pivot to the end
        pivot = a[hi]
        i = lo - 1
        for j in range(lo, hi):
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
        cmp += hi - lo
        a[i + 1], a[hi] = a[hi], a[i + 1]
        return i + 1

    def sort(lo, hi):
        while lo < hi:
            p = partition(lo, hi)
            if p - lo < hi - p:
                sort(lo, p - 1)
                lo = p + 1
            else:
                sort(p + 1, hi)
                hi = p - 1

    sort(0, len(a) - 1)
    return cmp


def quicksort_3way(a):
    """Three-way (Dutch national flag) partition, random pivot.

    Invariant:  a[lo:lt] < pivot,  a[lt:i] == pivot,  a[i:gt+1] unknown,
                a[gt+1:hi+1] > pivot.   All keys equal to the pivot are
                finished after one pass and never touched again.
    """
    cmp = 0

    def sort(lo, hi):
        nonlocal cmp
        while lo < hi:
            pivot = a[random.randint(lo, hi)]
            lt, i, gt = lo, lo, hi
            while i <= gt:
                x = a[i]
                cmp += 1
                if x < pivot:
                    a[lt], a[i] = a[i], a[lt]
                    lt += 1
                    i += 1
                else:
                    cmp += 1
                    if x > pivot:
                        a[i], a[gt] = a[gt], a[i]
                        gt -= 1
                    else:
                        i += 1
            # a[lt..gt] == pivot is already in place; recurse on < and > only
            if lt - lo < hi - gt:
                sort(lo, lt - 1)
                lo = gt + 1
            else:
                sort(gt + 1, hi)
                hi = lt - 1

    sort(0, len(a) - 1)
    return cmp


Q1_VARIANTS = [
    ("Lomuto", quicksort_lomuto),
    ("Hoare", quicksort_hoare),
    ("Randomised", quicksort_randomised),
    ("3-way", quicksort_3way),
]
Q1_INPUTS = ["random", "sorted", "reverse", "duplicates"]


def make_input(kind, n, rng):
    if kind == "random":
        return rng.sample(range(10 * n), n)          # distinct random keys
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "duplicates":
        return [rng.randint(0, 9) for _ in range(n)]  # only 10 distinct values
    raise ValueError(kind)


def print_grid(title, row_labels, col_labels, cells, fmt):
    print("\n" + title)
    w0 = max(len(r) for r in row_labels) + 2
    print(" " * w0 + "".join(f"{c:>14}" for c in col_labels))
    for r, row in zip(row_labels, cells):
        print(f"{r:<{w0}}" + "".join(f"{fmt(v):>14}" for v in row))


def run_q1(quick=False):
    sizes = [1000, 2000, 4000] if quick else [1000, 2000, 4000, 8000, 16000]
    rng = random.Random(SEED)
    random.seed(SEED)
    rows = []                                   # (input, variant, n, comparisons, seconds)
    for kind in Q1_INPUTS:
        for n in sizes:
            base = make_input(kind, n, rng)
            expected = sorted(base)
            for name, fn in Q1_VARIANTS:
                a = base[:]                     # every variant gets the same data
                t0 = time.perf_counter()
                c = fn(a)
                t = time.perf_counter() - t0
                assert a == expected, f"{name} failed on {kind} n={n}"
                rows.append((kind, name, n, c, t))
                print(f"  done {kind:<10} n={n:<6} {name:<11} cmp={c:>12,} time={t:8.4f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/q1_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "variant", "n", "comparisons", "seconds"])
        w.writerows(rows)

    names = [v[0] for v in Q1_VARIANTS]
    for kind in Q1_INPUTS:
        for metric, idx, fmt in (("comparisons", 3, lambda v: f"{v:,}"),
                                 ("time (s)", 4, lambda v: f"{v:.4f}")):
            cells = []
            for name in names:
                cells.append([next(r[idx] for r in rows if r[0] == kind and r[1] == name and r[2] == n)
                              for n in sizes])
            print_grid(f"Q1 - {kind} input - {metric}", names, [f"n={n}" for n in sizes], cells, fmt)

    # growth factor when n doubles: ~2 => n log n, ~4 => quadratic
    print("\nQ1 - growth factor of comparisons when n doubles (2 ~ n log n, 4 ~ n^2, ~2 flat = linear)")
    print(f"{'input':<12}{'variant':<12}" + "".join(f"{f'{a}->{b}':>14}" for a, b in zip(sizes, sizes[1:])))
    for kind in Q1_INPUTS:
        for name in names:
            cs = [next(r[3] for r in rows if r[0] == kind and r[1] == name and r[2] == n) for n in sizes]
            print(f"{kind:<12}{name:<12}" + "".join(f"{b / a:>14.2f}" for a, b in zip(cs, cs[1:])))

    # bonus: why the choice of pivot position matters for Hoare
    print("\nBonus - Hoare with FIRST-element pivot (textbook version), n = 2000")
    for kind in ("sorted", "reverse"):
        a = make_input(kind, 2000, rng)
        print(f"  {kind:<8} comparisons = {quicksort_hoare(a, 'first'):>10,}   (middle pivot: "
              f"{quicksort_hoare(make_input(kind, 2000, rng), 'middle'):,})")

    plot_q1(rows, sizes)
    return rows


def plot_q1(rows, sizes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping Q1 plots)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, kind in zip(axes.flat, Q1_INPUTS):
        for name, _ in Q1_VARIANTS:
            ys = [next(r[3] for r in rows if r[0] == kind and r[1] == name and r[2] == n) for n in sizes]
            ax.plot(sizes, ys, marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{kind} input")
        ax.set_xlabel("n")
        ax.set_ylabel("comparisons")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend()
    fig.suptitle("Q1: comparisons vs n (log-log; slope 2 = quadratic, ~1 = n log n)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q1_comparisons.png", dpi=130)
    plt.close(fig)


# =====================================================================
# Q2  COUNTING SORT, RADIX SORT, MERGE SORT
# =====================================================================

def _counting_sort_by_keys(a, keys, k):
    """Stable counting sort of list a using keys[i] in 0..k as the key of a[i]."""
    n = len(a)
    count = [0] * (k + 1)
    for key in keys:                       # 1. histogram            O(n)
        count[key] += 1
    for v in range(1, k + 1):              # 2. prefix sums          O(k)
        count[v] += count[v - 1]           #    count[v] = #keys <= v
    out = [None] * n
    for i in range(n - 1, -1, -1):         # 3. place, right-to-left O(n)  (=> stable)
        key = keys[i]
        count[key] -= 1
        out[count[key]] = a[i]
    return out


def counting_sort(a, k, key=None):
    """Stable counting sort for integer keys in 0..k.  Time O(n + k), space O(n + k).

    key: optional function giving the integer key of an element, so records
    such as (key, original_index) can be sorted.  Without it the elements
    themselves are the keys.
    """
    keys = a if key is None else [key(x) for x in a]
    if keys and (min(keys) < 0 or max(keys) > k):
        raise ValueError(f"keys must lie in 0..{k}")
    return _counting_sort_by_keys(a, keys, k)


def radix_sort(a, base=10):
    """LSD radix sort for non-negative integers: one stable counting sort per digit.
    Time O(d (n + base)) with d = number of base-`base` digits of the maximum."""
    if not a:
        return []
    if min(a) < 0:
        raise ValueError("radix_sort handles non-negative integers only")
    biggest = max(a)
    out = list(a)
    exp = 1
    while biggest // exp > 0:              # least-significant digit first
        digits = [(x // exp) % base for x in out]
        out = _counting_sort_by_keys(out, digits, base - 1)
        exp *= base
    return out


def merge_sort(a):
    """Standard top-down merge sort (stand-in for the Lab 1 merge sort).  Returns a new list."""
    n = len(a)
    if n <= 1:
        return a[:]
    mid = n // 2
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])
    out = []
    i = j = 0
    nl, nr = len(left), len(right)
    while i < nl and j < nr:
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def timed(fn, *args, repeats=1):
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, result


def demo_stability():
    print("\nQ2 part 7 - stability of counting sort on (key, original index) pairs")
    rng = random.Random(SEED)
    pairs = [(rng.randint(0, 4), i) for i in range(15)]
    print("  input  :", pairs)
    out = counting_sort(pairs, 4, key=lambda p: p[0])
    print("  output :", out)
    ok = all(out[i][0] < out[i + 1][0] or (out[i][0] == out[i + 1][0] and out[i][1] < out[i + 1][1])
             for i in range(len(out) - 1))
    print("  equal keys keep increasing original indices ->", "STABLE" if ok else "NOT STABLE")
    assert ok
    # larger automatic check
    big = [(rng.randint(0, 99), i) for i in range(50000)]
    res = counting_sort(big, 99, key=lambda p: p[0])
    assert res == sorted(big, key=lambda p: p[0])      # Python's sort is stable: identical output
    print("  50,000 random pairs: identical to Python's stable sorted(key=...) -> STABLE")


def run_q2(quick=False):
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # correctness first
    for _ in range(200):
        arr = [rng.randint(0, 1000) for _ in range(rng.randint(0, 300))]
        assert counting_sort(arr, 1000) == sorted(arr)
        assert radix_sort(arr, 10) == sorted(arr)
        assert radix_sort(arr, 256) == sorted(arr)
    print("Q2 correctness: counting sort and radix sort (base 10, base 256) match sorted() on 200 random tests")
    demo_stability()

    # ---- part 4: n fixed, k varies (counting sort vs merge sort) ----
    n = 20000 if quick else 100000
    ks = [10, 10**3, 10**5, 10**6] if quick else [10, 10**3, 10**5, 10**7]
    rows_k = []
    print(f"\nQ2 part 4 - counting sort, n = {n:,}, varying k (best of 3)")
    print(f"{'k':>12}{'counting':>12}{'merge sort':>12}{'sorted() (C)':>14}")
    for k in ks:
        data = [rng.randint(0, k) for _ in range(n)]
        t_cs, r1 = timed(counting_sort, data, k, repeats=3)
        t_ms, r2 = timed(merge_sort, data, repeats=3)
        t_py, r3 = timed(sorted, data, repeats=3)
        assert r1 == r2 == r3
        rows_k.append((k, t_cs, t_ms, t_py))
        print(f"{k:>12,}{t_cs:>12.4f}{t_ms:>12.4f}{t_py:>14.4f}", flush=True)
    with open(f"{OUT_DIR}/q2_counting_vs_k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "k", "counting_s", "merge_s", "python_sorted_s"])
        for k, a, b, c in rows_k:
            w.writerow([n, k, a, b, c])

    # ---- part 5: k fixed = 1e9, n varies (radix vs merge) ----
    K = 10**9
    ns = [5000, 10000, 20000, 50000] if quick else [10000, 50000, 100000, 250000, 500000, 1000000]
    rows_n = []
    print(f"\nQ2 part 5 - k = 10^9 fixed, varying n")
    print(f"{'n':>10}{'radix b=10':>12}{'radix b=256':>13}{'merge sort':>12}{'sorted() (C)':>14}")
    for n in ns:
        data = [rng.randint(0, K) for _ in range(n)]
        t10, r1 = timed(radix_sort, data, 10)
        t256, r2 = timed(radix_sort, data, 256)
        tms, r3 = timed(merge_sort, data)
        tpy, r4 = timed(sorted, data)
        assert r1 == r2 == r3 == r4
        rows_n.append((n, t10, t256, tms, tpy))
        print(f"{n:>10,}{t10:>12.3f}{t256:>13.3f}{tms:>12.3f}{tpy:>14.3f}", flush=True)
    with open(f"{OUT_DIR}/q2_radix_vs_n.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "radix10_s", "radix256_s", "merge_s", "python_sorted_s"])
        w.writerows(rows_n)

    plot_q2(rows_k, rows_n, n_fixed=(20000 if quick else 100000))


def plot_q2(rows_k, rows_n, n_fixed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping Q2 plots)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r[0] for r in rows_k], [r[1] for r in rows_k], "o-", label="counting sort  O(n + k)")
    ax.plot([r[0] for r in rows_k], [r[2] for r in rows_k], "s--", label="merge sort  O(n log n)")
    ax.set_xscale("log")
    ax.set_xlabel("k (range of keys)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Counting sort vs merge sort, n = {n_fixed:,}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q2_counting_vs_k.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, log in zip(axes, (False, True)):
        ax.plot([r[0] for r in rows_n], [r[1] for r in rows_n], "o-", label="radix sort, base 10")
        ax.plot([r[0] for r in rows_n], [r[2] for r in rows_n], "^-", label="radix sort, base 256")
        ax.plot([r[0] for r in rows_n], [r[3] for r in rows_n], "s--", label="merge sort")
        ax.set_xlabel("n")
        ax.set_ylabel("time (s)")
        ax.grid(True, which="both", alpha=0.3)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("log-log view")
        else:
            ax.set_title("k = 10^9, linear axes")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q2_radix_vs_n.png", dpi=130)
    plt.close(fig)


# =====================================================================
# Q3  COUNTING INVERSIONS
# =====================================================================

def inversions_brute(a, deadline=None):
    """Theta(n^2): test every pair i < j.  Raises TimeoutError past `deadline`
    (a time.perf_counter() value) so that very large runs can be stopped."""
    n = len(a)
    count = 0
    for i in range(n - 1):
        ai = a[i]
        count += sum(1 for x in a[i + 1:] if ai > x)
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError
    return count


def _sort_and_count(a):
    n = len(a)
    if n <= 1:
        return a[:], 0
    mid = n // 2
    left, cl = _sort_and_count(a[:mid])
    right, cr = _sort_and_count(a[mid:])
    out = []
    count = cl + cr
    i = j = 0
    nl, nr = len(left), len(right)
    while i < nl and j < nr:
        if left[i] <= right[j]:            # equal keys are NOT an inversion
            out.append(left[i])
            i += 1
        else:                              # right[j] jumps over left[i:], all of them are bigger
            out.append(right[j])
            j += 1
            count += nl - i
    out.extend(left[i:])
    out.extend(right[j:])
    return out, count


def inversions_merge(a):
    """Theta(n log n): count inversions while merge-sorting a copy of a."""
    return _sort_and_count(a)[1]


def verify_inversions():
    print("Q3 part 3 - verifying both algorithms agree")
    assert inversions_brute([2, 4, 1, 3, 5]) == inversions_merge([2, 4, 1, 3, 5]) == 3
    assert inversions_brute([]) == inversions_merge([]) == 0
    assert inversions_merge(list(range(100, 0, -1))) == 100 * 99 // 2     # reversed: all pairs
    assert inversions_merge(list(range(100))) == 0                        # sorted: none
    rng = random.Random(SEED)
    sizes = [0, 1, 2, 3, 10, 100, 500, 1000, 2000]
    sizes += [rng.randint(0, 2000) for _ in range(60)]
    for n in sizes:
        perm = rng.sample(range(10 * n + 1), n)                           # distinct values
        dup = [rng.randint(0, 20) for _ in range(n)]                      # many ties
        assert inversions_brute(perm) == inversions_merge(perm)
        assert inversions_brute(dup) == inversions_merge(dup)
    print(f"  hand examples + {len(sizes)} random sizes (0..2000), each with distinct and heavily "
          f"duplicated values: brute force == merge sort in every case  OK")


def run_q3(quick=False):
    verify_inversions()
    sizes = [1000, 2000, 5000, 10000, 20000] if quick else \
        [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    limit = 5 if quick else 60.0
    rng = random.Random(SEED)
    rows = []
    brute_stopped = False
    print(f"\nQ3 part 4 - timing (brute force is stopped once a run exceeds {limit:g} s)")
    print(f"{'n':>10}{'brute force (s)':>18}{'merge based (s)':>18}{'inversions':>18}")
    for n in sizes:
        data = rng.sample(range(10 * n), n)
        t_m, inv_m = timed(inversions_merge, data)
        if not brute_stopped:
            t0 = time.perf_counter()
            try:
                inv_b = inversions_brute(data, deadline=t0 + limit)
                t_b = time.perf_counter() - t0
                assert inv_b == inv_m
                bs = f"{t_b:.4f}"
            except TimeoutError:
                brute_stopped = True
                t_b = None
                bs = f"> {limit:g} (stopped)"
        else:
            t_b = None
            bs = "skipped"
        rows.append((n, t_b, t_m, inv_m))
        print(f"{n:>10,}{bs:>18}{t_m:>18.4f}{inv_m:>18,}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/q3_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "brute_s", "merge_s", "inversions"])
        w.writerows(rows)
    plot_q3(rows)


def plot_q3(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping Q3 plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b = [(r[0], r[1]) for r in rows if r[1] is not None]
    ax.plot([x for x, _ in b], [y for _, y in b], "o-", label="brute force  Θ(n²)")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "s-", label="merge sort based  Θ(n log n)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("time (s)")
    ax.set_title("Counting inversions")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q3_timing.png", dpi=130)
    plt.close(fig)


# ---------- Q3 part 5: Kendall tau distance between two users ----------

def kendall_tau_distance(rank1, rank2):
    """rank1[i], rank2[i] = rank (1 = favourite) that user 1 / user 2 gave item i.

    List the items in user 1's order; read off user 2's ranks along that order.
    Every inversion in that sequence is one pair of movies the two users
    order differently, so the Kendall tau distance = number of inversions.
    """
    order = sorted(range(len(rank1)), key=lambda i: rank1[i])
    return inversions_merge([rank2[i] for i in order])


def _read_ranking(prompt_name, movies):
    n = len(movies)
    while True:
        print(f"\n{prompt_name}: give each movie a distinct rank from 1 (favourite) to {n} (least favourite).")
        ranks = []
        try:
            for m in movies:
                ranks.append(int(input(f"  rank of '{m}': ").strip()))
        except ValueError:
            print("  !! please type whole numbers only - let's start this ranking again.")
            continue
        if sorted(ranks) == list(range(1, n + 1)):
            return ranks
        print(f"  !! each rank 1..{n} must be used exactly once - let's start this ranking again.")


def run_kendall(demo=False):
    N = 10
    if demo:
        movies = ["Inception", "Parasite", "Interstellar", "Coco", "Up",
                  "Whiplash", "Joker", "Titanic", "Avatar", "Frozen"]
        r1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        r2 = [3, 1, 2, 6, 4, 5, 10, 7, 8, 9]
        print("(demo data)")
    else:
        print(f"Enter the titles of {N} movies (one per line).")
        movies = []
        while len(movies) < N:
            t = input(f"  movie {len(movies) + 1}: ").strip()
            if not t:
                print("  !! title cannot be empty")
            elif t.lower() in (m.lower() for m in movies):
                print("  !! already entered")
            else:
                movies.append(t)
        r1 = _read_ranking("User 1", movies)
        r2 = _read_ranking("User 2", movies)

    d = kendall_tau_distance(r1, r2)
    pairs = N * (N - 1) // 2
    print("\n" + "-" * 60)
    print(f"{'movie':<20}{'User 1':>10}{'User 2':>10}")
    for m, a, b in zip(movies, r1, r2):
        print(f"{m:<20}{a:>10}{b:>10}")
    print("-" * 60)
    print(f"Kendall tau distance      d = {d}   (out of {pairs} possible pairs)")
    print(f"Pairs the users agree on      {pairs - d}")
    print(f"Similarity (1 - d/{pairs})        {1 - d / pairs:.3f}   (1 = identical, 0 = exact opposites)")
    print(f"Kendall tau coefficient       {1 - 2 * d / pairs:+.3f}   (+1 identical, -1 reversed)")
    disagreements = [(movies[i], movies[j]) for i, j in itertools.combinations(range(N), 2)
                     if (r1[i] - r1[j]) * (r2[i] - r2[j]) < 0]
    assert len(disagreements) == d                     # brute-force cross-check of the fast answer
    if disagreements:
        print("\nPairs they order differently:")
        for a, b in disagreements:
            print(f"  {a}  vs  {b}")


# =====================================================================
def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="CSF302 Lab 5 - sorting algorithms")
    ap.add_argument("part", choices=["q1", "q2", "q3", "kendall", "all"])
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke test")
    ap.add_argument("--demo", action="store_true", help="kendall: use built-in sample rankings")
    ap.add_argument("--out", default=OUT_DIR, help="output folder (default: results)")
    args = ap.parse_args()
    OUT_DIR = args.out
    if args.part in ("q1", "all"):
        run_q1(args.quick)
    if args.part in ("q2", "all"):
        run_q2(args.quick)
    if args.part in ("q3", "all"):
        run_q3(args.quick)
    if args.part == "kendall":
        run_kendall(args.demo)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()