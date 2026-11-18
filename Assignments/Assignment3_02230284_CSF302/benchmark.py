"""Timing experiments for Assignment 3, Part C.

Usage:   python benchmark.py              (about 3 to 5 minutes)
         python benchmark.py --plot-only  (redraw tables and graphs from results.json)

Experiments
    1. Scaling:     n = 1,000 ... 64,000 on three input types (random, already sorted,
                    many duplicates), basic vs improved vs "random pivot + Lomuto".
                    A crash (RecursionError) is recorded as a result.
    2. Recurrence:  comparison counts and (with the recursion limit raised) running time
                    of quicksort_basic on sorted input, to check T(n) = T(n-1) + Theta(n).
    3. Cutoff:      quicksort_improved at n = 100,000 random values, cutoff 1,4,8,16,32,64.
    4. sorted():    best cutoff vs Python's built-in sorted(), times and comparison counts.

Writes results.json, fig_scaling.png and fig_cutoff.png, and prints Markdown tables.
"""

import gc
import json
import math
import multiprocessing as mp
import os
import platform
import random
import statistics
import sys
import time

from sorts import quicksort_basic, quicksort_improved

SIZES = [1000 * 2 ** k for k in range(7)]          # 1,000 ... 64,000
KINDS = ("random", "sorted", "duplicates")
CUTOFFS = [1, 4, 8, 16, 32, 64]
SEED = 303
LIMIT = 60.0                                        # seconds, per measurement


# ---------------------------------------------------------------- inputs
def make_data(kind, n, seed):
    rng = random.Random(seed)
    if kind == "random":
        return [rng.randint(0, 10 ** 9) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "duplicates":
        return [rng.randint(0, 9) for _ in range(n)]
    raise ValueError(kind)


# ---------------------------------------------------------------- algorithms
def basic(a):
    quicksort_basic(a)


def improved(a):
    quicksort_improved(a, 16)


def lomuto_random(a):
    quicksort_improved(a, 16, "lomuto")


ALGOS = {"basic": basic, "improved": improved, "lomuto_random": lomuto_random}


# ---------------------------------------------------------------- one measurement, in a child process
def _worker(algo, kind, n, seed, reps, queue):
    fn = ALGOS[algo]
    data = make_data(kind, n, seed)
    expected = sorted(data)
    best = math.inf
    for _ in range(reps):
        a = data[:]                                 # the copy is made outside the timer
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            try:
                fn(a)
            except RecursionError:
                queue.put(("crash", "RecursionError"))
                return
            elapsed = time.perf_counter() - start
        finally:
            gc.enable()
        if a != expected:
            queue.put(("wrong", "incorrect output"))
            return
        best = min(best, elapsed)
    queue.put(("ok", best))


def measure(algo, kind, n, reps, limit=LIMIT):
    """Best of `reps` runs in a child process that is stopped after `limit` seconds."""
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(algo, kind, n, SEED + n, reps, queue))
    proc.start()
    proc.join(limit)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"algo": algo, "kind": kind, "n": n, "status": "timeout", "seconds": None}
    try:
        status, value = queue.get(timeout=5)
    except Exception:
        return {"algo": algo, "kind": kind, "n": n, "status": "error", "seconds": None}
    return {"algo": algo, "kind": kind, "n": n, "status": status,
            "seconds": value if status == "ok" else None, "detail": None if status == "ok" else value}


# ---------------------------------------------------------------- experiment 1
def run_scaling():
    records = []
    for kind in KINDS:
        for n in SIZES:
            for algo in ALGOS:
                reps = 5 if n <= 8000 else 3
                if algo == "lomuto_random" and kind == "duplicates" and n >= 16000:
                    reps = 1                         # quadratic: one run is enough
                rec = measure(algo, kind, n, reps)
                records.append(rec)
                shown = f"{rec['seconds']:.4f} s" if rec["status"] == "ok" else rec["status"]
                print(f"{kind:>10} n={n:>6} {algo:>13}: {shown}", flush=True)
    return records


def max_sorted_n_without_crash():
    """Largest n for which quicksort_basic still sorts list(range(n)) with the default recursion limit."""
    lo, hi = 1, 5000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            quicksort_basic(list(range(mid)))
            lo = mid
        except RecursionError:
            hi = mid
    return lo


# ---------------------------------------------------------------- comparison counting
class Counted:
    """A number that counts every `<` and `>` made on it."""
    __slots__ = ("v",)
    count = 0

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        Counted.count += 1
        return self.v < other.v

    def __gt__(self, other):
        Counted.count += 1
        return self.v > other.v


def count_comparisons(fn, data):
    items = [Counted(x) for x in data]
    Counted.count = 0
    result = fn(items)
    if result is not None:                           # sorted() returns a new list
        items = result
    assert [c.v for c in items] == sorted(data)
    return Counted.count


def _sorted_builtin(items):
    return sorted(items)


# ---------------------------------------------------------------- experiment 2
def run_recurrence():
    out = {"max_sorted_n_default_limit": max_sorted_n_without_crash(), "counts": [], "times": []}
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)                     # only here, to see the quadratic time at all
    try:
        for n in (100, 200, 400, 800, 1600):
            row = {"n": n, "n_choose_2": n * (n - 1) // 2}
            for kind in KINDS:
                row[kind] = count_comparisons(quicksort_basic, make_data(kind, n, SEED))
            row["n_log2_n"] = n * math.log2(n)
            out["counts"].append(row)
        for n in (500, 1000, 2000, 4000, 8000):
            data = list(range(n))
            best = math.inf
            for _ in range(3):
                a = data[:]
                gc.collect()
                gc.disable()
                start = time.perf_counter()
                quicksort_basic(a)
                best = min(best, time.perf_counter() - start)
                gc.enable()
            out["times"].append({"n": n, "seconds": best})
    finally:
        sys.setrecursionlimit(old_limit)
    return out


# ---------------------------------------------------------------- experiment 3
def run_cutoff(n=100_000, rounds=9):
    data = make_data("random", n, SEED)
    expected = sorted(data)
    times = {c: [] for c in CUTOFFS}
    for r in range(rounds):                          # interleave, rotating the order, so drift hits all equally
        shift = r % len(CUTOFFS)
        for c in CUTOFFS[shift:] + CUTOFFS[:shift]:
            a = data[:]
            gc.collect()
            gc.disable()
            start = time.perf_counter()
            quicksort_improved(a, c)
            elapsed = time.perf_counter() - start
            gc.enable()
            if r == 0:
                assert a == expected
            times[c].append(elapsed)
    rows = [{"cutoff": c, "best": min(t), "median": statistics.median(t), "worst": max(t)}
            for c, t in times.items()]
    best_cutoff = min(rows, key=lambda r: r["median"])["cutoff"]
    return {"n": n, "rounds": rounds, "rows": rows, "best_cutoff": best_cutoff}


# ---------------------------------------------------------------- experiment 4
def _best_time(fn, data, reps):
    best = math.inf
    for _ in range(reps):
        a = data[:]
        gc.collect()
        gc.disable()
        start = time.perf_counter()
        fn(a)
        best = min(best, time.perf_counter() - start)
        gc.enable()
    return best


def run_vs_sorted(best_cutoff):
    rows = []
    for kind, n in (("random", 100_000), ("random", 64_000), ("sorted", 64_000), ("duplicates", 64_000)):
        data = make_data(kind, n, SEED + 1)
        t_mine = _best_time(lambda a: quicksort_improved(a, best_cutoff), data, 7)
        t_sorted = _best_time(lambda a: sorted(a), data, 21)
        rows.append({"kind": kind, "n": n, "mine": t_mine, "sorted": t_sorted, "ratio": t_mine / t_sorted})
    n = 100_000
    data = make_data("random", n, SEED + 1)
    comps = {
        "improved": count_comparisons(lambda a: quicksort_improved(a, best_cutoff), data),
        "lomuto_random": count_comparisons(lambda a: quicksort_improved(a, best_cutoff, "lomuto"), data),
        "basic": count_comparisons(quicksort_basic, data),
        "sorted": count_comparisons(_sorted_builtin, data),
        "n_log2_n": n * math.log2(n),
    }
    t_basic = _best_time(quicksort_basic, data, 5)
    t_lomuto = _best_time(lambda a: quicksort_improved(a, best_cutoff, "lomuto"), data, 5)
    return {"rows": rows, "comparisons_n": n, "comparisons": comps,
            "t_basic_100k": t_basic, "t_lomuto_100k": t_lomuto}


# ---------------------------------------------------------------- reporting
def slope(xs, ys):
    lx, ly = [math.log10(x) for x in xs], [math.log10(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def cell(rec):
    if rec is None:
        return "n/a"
    if rec["status"] == "ok":
        return f"{rec['seconds']:.4f}"
    if rec["status"] == "crash":
        return "crash*"
    if rec["status"] == "timeout":
        return f"> {LIMIT:.0f} (stopped)"
    return rec["status"]


def lookup(records, kind, algo, n):
    return next((r for r in records if r["kind"] == kind and r["algo"] == algo and r["n"] == n), None)


def print_tables(res):
    sc = res["scaling"]
    print("\n### Table 1: running time in seconds (best of several runs)\n")
    print("| n | random: basic | random: improved | sorted: basic | sorted: improved | "
          "duplicates: basic | duplicates: improved | duplicates: random pivot + Lomuto |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in SIZES:
        cells = [cell(lookup(sc, "random", "basic", n)), cell(lookup(sc, "random", "improved", n)),
                 cell(lookup(sc, "sorted", "basic", n)), cell(lookup(sc, "sorted", "improved", n)),
                 cell(lookup(sc, "duplicates", "basic", n)), cell(lookup(sc, "duplicates", "improved", n)),
                 cell(lookup(sc, "duplicates", "lomuto_random", n))]
        print(f"| {n:,} | " + " | ".join(cells) + " |")
    print("\n*crash = RecursionError (maximum recursion depth exceeded)")

    print("\nlog-log slopes (time ~ n^slope):")
    for kind in KINDS:
        for algo in ALGOS:
            pts = [(r["n"], r["seconds"]) for r in sc
                   if r["kind"] == kind and r["algo"] == algo and r["status"] == "ok"]
            if len(pts) >= 3:
                print(f"  {kind:>10} {algo:>13}: {slope(*zip(*pts)):.2f}")
            else:
                print(f"  {kind:>10} {algo:>13}: only {len(pts)} finished runs "
                      f"({', '.join(sorted({r['status'] for r in sc if r['kind'] == kind and r['algo'] == algo}))})")

    rec = res["recurrence"]
    print(f"\nLargest n at which quicksort_basic sorts list(range(n)) with the default recursion limit: "
          f"{rec['max_sorted_n_default_limit']}")
    print("\n### Table 2: comparisons made by quicksort_basic\n")
    print("| n | n(n-1)/2 | sorted input | random input | duplicates 0..9 | n log2 n |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rec["counts"]:
        print(f"| {r['n']:,} | {r['n_choose_2']:,} | {r['sorted']:,} | {r['random']:,} | "
              f"{r['duplicates']:,} | {r['n_log2_n']:,.0f} |")
    print("\n### Table 2b (text in 3.2): quicksort_basic on sorted input, recursion limit raised\n")
    print("| n | time (s) | time / previous |")
    print("|---:|---:|---:|")
    prev = None
    for r in rec["times"]:
        print(f"| {r['n']:,} | {r['seconds']:.4f} | {'' if prev is None else f'{r['seconds'] / prev:.2f}'} |")
        prev = r["seconds"]

    cut = res["cutoff"]
    print(f"\n### Table 3: quicksort_improved, n = {cut['n']:,} random values "
          f"({cut['rounds']} interleaved rounds)\n")
    print("| cutoff | best (s) | median (s) | worst (s) |")
    print("|---:|---:|---:|---:|")
    for r in cut["rows"]:
        mark = " **(chosen)**" if r["cutoff"] == cut["best_cutoff"] else ""
        print(f"| {r['cutoff']}{mark} | {r['best']:.4f} | {r['median']:.4f} | {r['worst']:.4f} |")

    vs = res["vs_sorted"]
    print(f"\n### Table 4: quicksort_improved (cutoff {cut['best_cutoff']}) vs sorted()\n")
    print("| input | n | quicksort_improved (s) | sorted() (s) | ratio |")
    print("|---|---:|---:|---:|---:|")
    for r in vs["rows"]:
        print(f"| {r['kind']} | {r['n']:,} | {r['mine']:.4f} | {r['sorted']:.5f} | {r['ratio']:.0f}x |")
    c = vs["comparisons"]
    print(f"\nComparisons, n = {vs['comparisons_n']:,} random values:")
    for k in ("sorted", "improved", "lomuto_random", "basic"):
        print(f"  {k:>14}: {c[k]:,}  ({c[k] / c['sorted']:.2f}x sorted(); {c[k] / c['n_log2_n']:.2f} n log2 n)")
    print(f"  n log2 n = {c['n_log2_n']:,.0f}")
    print(f"  random 100,000: basic {vs['t_basic_100k']:.4f} s, random pivot + Lomuto {vs['t_lomuto_100k']:.4f} s")
    mine = next(r for r in vs["rows"] if r["kind"] == "random" and r["n"] == 100_000)
    ratio_total = mine["ratio"]
    ratio_algo = c["improved"] / c["sorted"]
    print(f"\n  total ratio {ratio_total:.1f}x = algorithm (comparisons) {ratio_algo:.2f}x "
          f"x language/implementation {ratio_total / ratio_algo:.1f}x")
    print(f"  time per comparison: improved {mine['mine'] / c['improved'] * 1e9:.0f} ns, "
          f"sorted() {mine['sorted'] / c['sorted'] * 1e9:.0f} ns")


def make_plots(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: graphs skipped (pip install matplotlib)")
        return
    sc = res["scaling"]
    styles = {"basic": ("o-", "tab:red", "basic (last-element pivot)"),
              "improved": ("s-", "tab:blue", "improved (random pivot, Hoare, cutoff 16)"),
              "lomuto_random": ("^--", "tab:green", "random pivot + Lomuto (brief's literal version)")}
    titles = {"random": "random values", "sorted": "already sorted", "duplicates": "many duplicates (0 to 9)"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, kind in zip(axes, KINDS):
        ys_all = []
        for algo, (fmt, color, label) in styles.items():
            pts = [(r["n"], r["seconds"]) for r in sc
                   if r["kind"] == kind and r["algo"] == algo and r["status"] == "ok"]
            if pts:
                ax.loglog(*zip(*pts), fmt, color=color, ms=4, lw=1.4, label=label)
                ys_all += [y for _, y in pts]
        top = max(ys_all) * 6
        for algo, (fmt, color, label) in styles.items():
            bad = [r["n"] for r in sc if r["kind"] == kind and r["algo"] == algo and r["status"] != "ok"]
            if bad:
                what = "RecursionError" if any(r["status"] == "crash" for r in sc
                                               if r["kind"] == kind and r["algo"] == algo) else "stopped after 60 s"
                ax.plot(bad, [top] * len(bad), "x", color=color, ms=8, mew=2, ls="none",
                        label=f"{label.split(' (')[0]}: {what}")
        ax.set_ylim(top=top * 2.5)
        ax.set_title(titles[kind], fontsize=10)
        ax.set_xlabel("number of values n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("running time (seconds)")
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.savefig("fig_scaling.png", dpi=160)
    plt.close(fig)

    cut = res["cutoff"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = list(range(len(cut["rows"])))
    ax.plot(xs, [r["median"] * 1e3 for r in cut["rows"]], "o-", color="tab:blue", label="median of rounds")
    ax.plot(xs, [r["best"] * 1e3 for r in cut["rows"]], "s--", color="tab:green", ms=4, label="best round")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["cutoff"]) for r in cut["rows"]])
    ax.set_xlabel("cutoff (insertion sort for parts of at most this size)")
    ax.set_ylabel("time for n = 100,000 (ms)")
    ax.set_title("Choosing the cutoff", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_cutoff.png", dpi=160)
    plt.close(fig)


def main():
    env = (f"Python {platform.python_version()} ({platform.python_implementation()}), "
           f"{platform.system()} {platform.release()}, {platform.machine()}, "
           f"{os.cpu_count()} logical CPUs")
    print("Environment:", env)
    if "--plot-only" in sys.argv:
        with open("results.json") as f:
            res = json.load(f)
        # JSON turns the cutoff keys into strings only inside dicts; rows are lists, so nothing to fix
    else:
        res = {"environment": env}
        print("\n== Experiment 1: scaling ==")
        res["scaling"] = run_scaling()
        print("\n== Experiment 2: recurrence evidence ==")
        res["recurrence"] = run_recurrence()
        print("\n== Experiment 3: cutoff ==")
        res["cutoff"] = run_cutoff()
        print("\n== Experiment 4: comparison with sorted() ==")
        res["vs_sorted"] = run_vs_sorted(res["cutoff"]["best_cutoff"])
        with open("results.json", "w") as f:
            json.dump(res, f, indent=1)
    print_tables(res)
    make_plots(res)
    print("\nEnvironment:", env)


if __name__ == "__main__":
    main()