"""Timing experiments for Assignment 2, Part C."""

import csv
import gc
import math
import multiprocessing as mp
import os
import platform
import random

import sys
import time

from closest import closest_pair, closest_pair_brute

SIZES = [1000 * 2 ** k for k in range(7)]      # 1,000 ... 64,000
TIME_LIMIT = 60.0                               # seconds, brute force only
SEED = 302


def make_points(n, seed):
    rng = random.Random(seed)
    return [(rng.uniform(0, 10_000), rng.uniform(0, 10_000)) for _ in range(n)]


# ---- brute force runs in a child process so it can be stopped after 60 s ----
def _brute_worker(n, seed, queue):
    pts = make_points(n, seed)
    gc.collect()
    gc.disable()
    start = time.perf_counter()
    d, _, _ = closest_pair_brute(pts)
    elapsed = time.perf_counter() - start
    gc.enable()
    queue.put((elapsed, d))


def brute_with_limit(n, seed, limit=TIME_LIMIT):
    """Return (seconds, d), or None if the run did not finish within `limit`."""
    queue = mp.Queue()
    proc = mp.Process(target=_brute_worker, args=(n, seed, queue))
    proc.start()
    proc.join(limit)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    return queue.get()


def time_dc(pts, reps):
    """Best (minimum) of `reps` runs: on a busy machine other programs can only make
    a run slower, so the minimum is the best estimate of the algorithm's own cost
    (this is also what the standard `timeit` documentation recommends).
    The garbage collector is paused while timing, as `timeit` does, so that
    collection pauses do not add noise."""
    times = []
    result = None
    for _ in range(reps):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            result = closest_pair(pts)
            times.append(time.perf_counter() - start)
        finally:
            gc.enable()
    return min(times), result


def slope(xs, ys):
    """Least-squares slope of log10(y) against log10(x)."""
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


# ---- experiment 1: scaling ----
def run_scaling():
    rows = []
    stopped = False
    for n in SIZES:
        seed = SEED + n
        pts = make_points(n, seed)
        t_dc, (d_dc, _, _) = time_dc(pts, 9)
        t_br = None
        if not stopped:
            runs = []
            reps = 5 if n <= 4000 else 3 if n <= 8000 else 2 if n <= 16000 else 1
            for _ in range(reps):
                out = brute_with_limit(n, seed)
                if out is None:
                    stopped = True
                    break
                runs.append(out)
            if runs and not stopped:
                t_br = min(t for t, _ in runs)
                assert math.isclose(runs[0][1], d_dc, rel_tol=1e-12), "algorithms disagree!"
        rows.append({"n": n, "brute": t_br, "dc": t_dc, "stopped": stopped and t_br is None})
        b = f"{t_br:.4f}" if t_br is not None else f"> {TIME_LIMIT:.0f} (stopped)"
        print(f"n={n:>6}  brute={b:>20}  d&c={t_dc:.4f}", flush=True)
    return rows


# ---- experiment 2: where does divide and conquer overtake brute force? ----
def run_crossover():
    sizes = list(range(2, 40, 2)) + list(range(40, 200, 5)) + list(range(200, 401, 20))
    sets, repeats = 10, 5
    rows = []
    for n in sizes:
        data = [make_points(n, 7000 + 31 * n + k) for k in range(sets)]
        best_b = best_d = math.inf
        for _ in range(repeats):                 # keep the least-disturbed run
            gc.collect()
            gc.disable()
            try:
                s = time.perf_counter()
                for p in data:
                    closest_pair_brute(p)
                best_b = min(best_b, time.perf_counter() - s)
                s = time.perf_counter()
                for p in data:
                    closest_pair(p)
                best_d = min(best_d, time.perf_counter() - s)
            finally:
                gc.enable()
        rows.append({"n": n, "brute": best_b / sets, "dc": best_d / sets})
    return rows


def find_crossover(rows):
    wins = [r["dc"] < r["brute"] for r in rows]
    first = next((r["n"] for r, w in zip(rows, wins) if w), None)
    stable = None
    for k in range(len(rows)):
        if all(wins[k:]):
            stable = rows[k]["n"]
            break
    return first, stable


def make_plots(scaling, crossover, first, stable):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: graphs skipped (pip install matplotlib)")
        return

    ns = [r["n"] for r in scaling]
    bx = [r["n"] for r in scaling if r["brute"] is not None]
    by = [r["brute"] for r in scaling if r["brute"] is not None]
    dc = [r["dc"] for r in scaling]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(bx, by, "o-", color="tab:red", label="brute force, measured")
    ax.loglog(ns, dc, "s-", color="tab:blue", label="divide and conquer, measured")
    ax.loglog(ns, [dc[0] * (n * math.log2(n)) / (ns[0] * math.log2(ns[0])) for n in ns],
              "--", color="tab:blue", lw=0.9, alpha=0.7, label=r"reference $n\log n$")
    ax.loglog(bx, [by[0] * (n / bx[0]) ** 2 for n in bx],
              "--", color="tab:red", lw=0.9, alpha=0.7, label=r"reference $n^2$")
    stopped = [r["n"] for r in scaling if r["stopped"]]
    if stopped:
        ax.plot(stopped[:1], [TIME_LIMIT], "x", color="tab:red", ms=10, mew=2)
        ax.annotate(f"n = {stopped[0]:,}: stopped\nafter {TIME_LIMIT:.0f} s", (stopped[0], TIME_LIMIT),
                    textcoords="offset points", xytext=(-10, 8), ha="right", va="bottom", fontsize=8)
        ax.set_ylim(top=TIME_LIMIT * 6)
    ax.set_xlabel("number of points n")
    ax.set_ylabel("running time (seconds)")
    ax.set_title("Closest pair: running time vs n (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("timing_loglog.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    zoom = [r for r in crossover if r["n"] <= 200]
    ax.plot([r["n"] for r in zoom], [r["brute"] * 1e3 for r in zoom],
            "o-", ms=3, color="tab:red", label="brute force")
    ax.plot([r["n"] for r in zoom], [r["dc"] * 1e3 for r in zoom],
            "s-", ms=3, color="tab:blue", label="divide and conquer")
    if first and stable:
        label = (f"crossover region, n = {first} to {stable}" if first != stable
                 else f"crossover, n = {stable}")
        ax.axvspan(first, stable, color="gray", alpha=0.25, label=label)
    ax.set_xlim(0, 200)
    ax.set_xlabel("number of points n")
    ax.set_ylabel("time per call (milliseconds)")
    ax.set_title("Small-n sweep: where divide and conquer overtakes brute force")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("crossover.png", dpi=160)
    plt.close(fig)


def load_results():
    """Read results_scaling.csv and results_crossover.csv written by an earlier run."""
    with open("results_scaling.csv") as f:
        scaling = [{"n": int(r["n"]),
                    "brute": float(r["brute_seconds"]) if r["brute_seconds"] else None,
                    "dc": float(r["dc_seconds"]),
                    "stopped": not r["brute_seconds"]} for r in csv.DictReader(f)]
    with open("results_crossover.csv") as f:
        crossover = [{"n": int(r["n"]), "brute": float(r["brute_seconds"]),
                      "dc": float(r["dc_seconds"])} for r in csv.DictReader(f)]
    return scaling, crossover


def main():
    env = (f"Python {platform.python_version()} ({platform.python_implementation()}), "
           f"{platform.system()} {platform.release()}, {platform.machine()}, "
           f"{os.cpu_count()} logical CPUs")
    print("Environment:", env)

    if "--plot-only" in sys.argv:             # redraw graphs/tables from saved CSV files
        scaling, crossover = load_results()
    else:
        print("\n== Experiment 1: scaling ==")
        scaling = run_scaling()
        with open("results_scaling.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "brute_seconds", "dc_seconds"])
            for r in scaling:
                w.writerow([r["n"], "" if r["brute"] is None else r["brute"], r["dc"]])

        print("\n== Experiment 2: crossover ==")
        crossover = run_crossover()
        with open("results_crossover.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "brute_seconds", "dc_seconds"])
            for r in crossover:
                w.writerow([r["n"], r["brute"], r["dc"]])
    first, stable = find_crossover(crossover)

    make_plots(scaling, crossover, first, stable)

    print("\n| n | Brute force (s) | Divide and conquer (s) | Speed-up | D&C time / (n log2 n) (µs) |")
    print("|---:|---:|---:|---:|---:|")
    for r in scaling:
        n = r["n"]
        b = f"{r['brute']:.3f}" if r["brute"] is not None else f"> {TIME_LIMIT:.0f} (stopped)"
        sp = f"{r['brute'] / r['dc']:.0f}x" if r["brute"] is not None else "n/a"
        print(f"| {n:,} | {b} | {r['dc']:.4f} | {sp} | {r['dc'] / (n * math.log2(n)) * 1e6:.3f} |")

    bpts = [(r["n"], r["brute"]) for r in scaling if r["brute"] is not None]
    print(f"\nlog-log slope, brute force     : {slope(*zip(*bpts)):.3f}  (theory 2)")
    print(f"log-log slope, divide & conquer: {slope(ns := [r['n'] for r in scaling], [r['dc'] for r in scaling]):.3f}  (theory 1, plus a log factor)")
    print(f"\nFirst n at which D&C is faster      : {first}")
    print(f"Smallest n after which D&C always wins: {stable}")
    print("Environment:", env)


if __name__ == "__main__":
    main()