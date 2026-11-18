#!/usr/bin/env python3
"""
CSF302 - Lab 7: Greedy and Randomized Algorithms
================================================
Q1  Activity selection (3 greedy rules) and knapsack (fractional, greedy 0/1 vs DP)
Q2  Huffman coding
Q3  Dijkstra, Kruskal, Prim (+ Dijkstra failing on a negative edge)
Q4  Randomized quickselect, Miller-Rabin primality test
"""

import argparse
import csv
import heapq
import itertools
import math
import os
import random
import sys
import time
from collections import Counter

OUT_DIR = "results7"
SEED = 2026


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def best_time(fn, repeats=3):
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# =====================================================================
# Q1  ACTIVITY SELECTION AND KNAPSACK
# =====================================================================

def activity_selection(acts):
    """Greedy, earliest finish time first.  O(n log n).  Provably optimal."""
    chosen = []
    last_finish = -math.inf
    for s, f in sorted(acts, key=lambda a: (a[1], a[0])):
        if s >= last_finish:
            chosen.append((s, f))
            last_finish = f
    return chosen


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def greedy_by_rule(acts, rule):
    """Generic greedy: scan in the order given by `rule`, keep an activity if it overlaps nothing chosen so far."""
    keys = {
        "finish": lambda a: (a[1], a[0]),
        "start": lambda a: (a[0], a[1]),
        "duration": lambda a: (a[1] - a[0], a[0], a[1]),
    }
    chosen = []
    for a in sorted(acts, key=keys[rule]):
        if all(not overlaps(a, c) for c in chosen):
            chosen.append(a)
    return chosen


def max_activities_exact(acts):
    """Independent exact answer by trying every subset (only for small n)."""
    n = len(acts)
    best = 0
    for mask in range(1 << n):
        sel = [acts[i] for i in range(n) if (mask >> i) & 1]
        if len(sel) > best and all(not overlaps(a, b) for a, b in itertools.combinations(sel, 2)):
            best = len(sel)
    return best


def read_activities(lines):
    acts = []
    for line in lines:
        line = line.strip().replace("(", " ").replace(")", " ").replace(",", " ")
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        s, f = float(parts[0]), float(parts[1])
        if f < s:
            raise ValueError(f"finish before start in line: {line}")
        acts.append((int(s) if s == int(s) else s, int(f) if f == int(f) else f))
    return acts


def run_activities_cli(path=None):
    if path:
        with open(path, encoding="utf-8") as fh:
            acts = read_activities(fh.readlines())
    else:
        print("Type one activity per line as 'start finish'. Finish with an empty line.")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                break
            lines.append(line)
        acts = read_activities(lines)
    chosen = activity_selection(acts)
    print(f"{len(acts)} activities read, greedy (earliest finish) selects {len(chosen)}:")
    for a in chosen:
        print(f"  ({a[0]}, {a[1]})")


def fractional_knapsack(items, capacity):
    """items = [(value, weight)].  Take items in decreasing value/weight order; the last may be cut.
    Returns (total value, [(item index, fraction taken)])."""
    order = sorted(range(len(items)), key=lambda i: items[i][0] / items[i][1], reverse=True)
    total, room, taken = 0.0, capacity, []
    for i in order:
        v, w = items[i]
        if room <= 0:
            break
        frac = min(1.0, room / w)
        taken.append((i, frac))
        total += v * frac
        room -= w * frac
    return total, taken


def greedy_01(items, capacity):
    """Same ratio rule, but items cannot be cut: take an item whole if it still fits, otherwise skip it."""
    order = sorted(range(len(items)), key=lambda i: items[i][0] / items[i][1], reverse=True)
    total, room, taken = 0, capacity, []
    for i in order:
        v, w = items[i]
        if w <= room:
            taken.append(i)
            total += v
            room -= w
    return total, taken


def greedy_01_fixed(items, capacity):
    """Bonus: best of (ratio greedy, the single most valuable item that fits).  Guaranteed >= 1/2 of optimal."""
    g, _ = greedy_01(items, capacity)
    singles = [v for v, w in items if w <= capacity]
    return max([g] + singles)


def knapsack_dp(items, capacity):
    """Exact 0/1 knapsack, O(n W)."""
    best = [0] * (capacity + 1)
    for v, w in items:
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]


def run_q1(quick=False):
    ensure_out()
    rng = random.Random(SEED)

    print("Q1 part 1 - greedy activity selection (earliest finish time first)")
    acts = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
    print(f"  activities (start, finish): {acts}")
    chosen = activity_selection(acts)
    print(f"  selected {len(chosen)}: {chosen}")
    assert len(chosen) == max_activities_exact(acts) == 4
    print("  matches the exhaustive optimum (4 activities)")

    print("\nQ1 part 2 - random testing of the other two greedy rules")
    trials = 5000 if quick else 30000
    stats = {"start": 0, "duration": 0}
    smallest = {"start": None, "duration": None}
    for _ in range(trials):
        n = rng.randint(3, 8)
        acts = []
        for _ in range(n):
            s = rng.randint(0, 15)
            acts.append((s, s + rng.randint(1, 8)))
        base = len(greedy_by_rule(acts, "finish"))
        if n <= 8:
            assert base == max_activities_exact(acts), "earliest finish must be optimal"
        assert base == len(activity_selection(acts))
        for rule in stats:
            got = len(greedy_by_rule(acts, rule))
            assert got <= base
            if got < base:
                stats[rule] += 1
                cand = (len(acts), sorted(acts))
                if smallest[rule] is None or cand[0] < smallest[rule][0][0] or \
                        (cand[0] == smallest[rule][0][0] and sum(map(sum, cand[1])) < sum(map(sum, smallest[rule][0][1]))):
                    smallest[rule] = (cand, got, base)
    print(f"  {trials:,} random inputs: earliest finish was always optimal (checked by brute force).")
    for rule, label in (("start", "earliest start"), ("duration", "shortest duration")):
        (n, acts), got, base = smallest[rule]
        print(f"  {label:<18}: worse than earliest finish on {stats[rule]:,} inputs ({100 * stats[rule] / trials:.1f}%)")
        print(f"      smallest counterexample ({n} activities): {acts}")
        print(f"      {label} picks {got}: {greedy_by_rule(acts, rule)}")
        print(f"      earliest finish picks {base}: {greedy_by_rule(acts, 'finish')}")

    print("\nQ1 part 3 - fractional knapsack (value/weight ratio first)")
    items = [(60, 10), (100, 20), (120, 30)]
    total, taken = fractional_knapsack(items, 50)
    print(f"  items (value, weight) = {items}, capacity 50")
    for i, fr in taken:
        print(f"    take {fr:.3f} of item {i} (value {items[i][0]}, weight {items[i][1]})")
    print(f"  total value = {total:g}   (expected 240)")
    assert abs(total - 240) < 1e-9
    print(f"  the same greedy rule on 0/1 knapsack gives {greedy_01(items, 50)[0]}, but the true 0/1 optimum is {knapsack_dp(items, 50)}")

    print("\nQ1 part 4 - ratio-greedy vs dynamic programming on 0/1 knapsack")
    trials = 20000 if quick else 200000
    worst = (1.0, None)
    worst_fixed = 1.0
    wrong = 0
    for _ in range(trials):
        n = rng.randint(2, 8)
        items = [(rng.randint(1, 30), rng.randint(1, 20)) for _ in range(n)]
        cap = rng.randint(1, sum(w for _, w in items))
        opt = knapsack_dp(items, cap)
        g = greedy_01(items, cap)[0]
        gf = greedy_01_fixed(items, cap)
        assert g <= opt and gf <= opt
        if g < opt:
            wrong += 1
        if opt > 0 and g / opt < worst[0]:
            worst = (g / opt, (items, cap, g, opt))
        if opt > 0:
            worst_fixed = min(worst_fixed, gf / opt)
    items, cap, g, opt = worst[1]
    print(f"  {trials:,} random instances (2-8 items, weights 1-20, values 1-30)")
    print(f"  greedy not optimal on {wrong:,} instances ({100 * wrong / trials:.1f}%)")
    print(f"  WORST ratio greedy/optimal found: {worst[0]:.4f}  = {g}/{opt}")
    print(f"      items (value, weight) = {items}, capacity {cap}")
    print(f"  the ratio can be made arbitrarily bad: items (value 2, weight 1) and (value W, weight W), capacity W:")
    for W in (10, 100, 1000):
        items = [(2, 1), (W, W)]
        g, opt = greedy_01(items, W)[0], knapsack_dp(items, W)
        print(f"      W = {W:<5} greedy = {g}, optimal = {opt}, ratio = {g / opt:.4f}")
    print(f"  bonus: 'best of greedy and the single best item' never fell below {worst_fixed:.4f} (guarantee: 0.5)")
    assert worst_fixed >= 0.5 - 1e-12


# =====================================================================
# Q2  HUFFMAN CODING
# =====================================================================

class HuffNode:
    __slots__ = ("char", "freq", "left", "right")

    def __init__(self, char, freq, left=None, right=None):
        self.char, self.freq, self.left, self.right = char, freq, left, right


def build_huffman_tree(freq):
    """Repeatedly merge the two least frequent trees (min-heap).  O(k log k) for k distinct symbols."""
    heap = []
    counter = itertools.count()                        # tie-breaker keeps the heap deterministic
    for ch, f in sorted(freq.items()):
        heapq.heappush(heap, (f, next(counter), HuffNode(ch, f)))
    if not heap:
        return None
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, next(counter), HuffNode(None, f1 + f2, a, b)))
    return heap[0][2]


def huffman_codes(root):
    """Left edge = '0', right edge = '1'.  A one-symbol alphabet gets the code '0'."""
    if root is None:
        return {}
    if root.left is None:
        return {root.char: "0"}
    table = {}
    stack = [(root, "")]
    while stack:
        node, code = stack.pop()
        if node.left is None:
            table[node.char] = code
        else:
            stack.append((node.right, code + "1"))
            stack.append((node.left, code + "0"))
    return table


def huffman_encode(text, table):
    return "".join(table[ch] for ch in text)


def huffman_decode(bits, root):
    if root is None:
        return ""
    if root.left is None:                               # single-symbol alphabet
        return root.char * len(bits)
    out, node = [], root
    for b in bits:
        node = node.left if b == "0" else node.right
        if node.left is None:
            out.append(node.char)
            node = root
    assert node is root, "bit string ended in the middle of a code word"
    return "".join(out)


def display_char(ch):
    return {"\n": "\\n", "\t": "\\t", " ": "' '", "\r": "\\r"}.get(ch, ch)


def huffman_report(name, text, show_table=True):
    freq = Counter(text)
    root = build_huffman_tree(freq)
    table = huffman_codes(root)
    bits = huffman_encode(text, table)
    decoded = huffman_decode(bits, root)
    assert decoded == text, "decoded text differs from the original!"
    n = len(text)
    fixed = 8 * n
    avg = len(bits) / n
    entropy = -sum(f / n * math.log2(f / n) for f in freq.values())
    print(f"\n=== {name}: {n:,} characters, {len(freq)} distinct ===")
    if show_table:
        print(f"{'char':>6} {'freq':>8} {'code':<}")
        for ch in sorted(table, key=lambda c: (len(table[c]), -freq[c], c)):
            print(f"{display_char(ch):>6} {freq[ch]:>8,} {table[ch]}")
    print(f"  8-bit fixed-length size : {fixed:,} bits")
    print(f"  Huffman encoded size    : {len(bits):,} bits")
    print(f"  compression ratio       : {len(bits) / fixed:.4f}  (compressed / original)  "
          f"-> {100 * (1 - len(bits) / fixed):.1f}% smaller   ({fixed / len(bits):.2f}:1)")
    print(f"  average code length     : {avg:.4f} bits/char   (entropy lower bound {entropy:.4f})")
    print(f"  decode(encode(text)) == text  ->  OK")
    return {"file": name, "chars": n, "distinct": len(freq), "fixed_bits": fixed, "huffman_bits": len(bits),
            "ratio": len(bits) / fixed, "avg_len": avg, "entropy": entropy}


ENGLISH_SAMPLE = (
    "Dynamic programming and greedy algorithms are two of the most useful ideas in computer science. "
    "A greedy algorithm builds a solution one step at a time and always takes the choice that looks best "
    "right now. When the problem has the greedy-choice property, this simple strategy is also optimal, and "
    "the resulting program is short, fast and easy to reason about. Huffman coding is a famous example. "
    "Given the frequency of every character in a file, it repeatedly merges the two rarest symbols into a "
    "new tree, so that frequent characters end up near the root with short codes while rare characters sink "
    "deeper and receive longer codes. The result is a prefix code: no code word is the beginning of another, "
    "which means the encoded bits can be read from left to right without any separators.\n\n"
    "Students often ask why we do not simply give every letter eight bits. The answer is that real text is "
    "very uneven. In English the letter e appears far more often than the letter z, and the space character "
    "appears more often than any letter. A fixed-length code wastes bits on the common symbols. A variable "
    "length code spends fewer bits where it matters most and therefore shrinks the file, often to about "
    "half of its original size. The same idea powers many real compression formats, usually combined with "
    "other techniques that find repeated phrases before the final Huffman step.\n\n"
    "Of course, a greedy rule is not always correct. Choosing the activity that starts first, or the one "
    "that is shortest, can easily miss the best schedule. Choosing the item with the best value per kilo "
    "works when items can be split, but may fail when they cannot. Part of the craft is learning to prove "
    "when the greedy choice is safe and to find a small counterexample when it is not.\n"
)


def make_demo_files():
    ensure_out()
    rng = random.Random(SEED)
    files = {}
    files["english.txt"] = ENGLISH_SAMPLE * 2
    with open(os.path.abspath(__file__), encoding="utf-8", errors="replace") as fh:
        files["python_source.txt"] = fh.read()
    files["dna.txt"] = "".join(rng.choice("ACGT") for _ in range(8000))
    files["uniform_letters.txt"] = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8000))
    files["skewed.txt"] = "".join(rng.choices("abcdefgh", weights=[60, 20, 8, 5, 3, 2, 1, 1], k=8000))
    paths = []
    for name, text in files.items():
        p = os.path.join(OUT_DIR, name)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        paths.append(p)
    return paths


def run_huffman_files(paths, show_table=True):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            text = fh.read()
        if not text:
            print(f"\n=== {p}: empty file, skipped ===")
            continue
        rows.append(huffman_report(os.path.basename(p), text, show_table))
    return rows


def run_q2(quick=False):
    ensure_out()
    paths = make_demo_files()
    # edge cases first
    for t in ("a", "aaaa", "ab", "hello world", "caf\u00e9 \u2013 na\u00efve"):
        root = build_huffman_tree(Counter(t))
        assert huffman_decode(huffman_encode(t, huffman_codes(root)), root) == t
    print("Q2 edge cases (one symbol, two symbols, non-ASCII) round-trip correctly\n")
    print("Q2 - Huffman coding on five text files (full code table shown for english.txt only)")
    rows = []
    for i, p in enumerate(paths):
        rows += run_huffman_files([p], show_table=(os.path.basename(p) == "english.txt"))
    print("\n" + "-" * 86)
    print(f"{'file':<22}{'chars':>8}{'symbols':>9}{'8-bit bits':>12}{'Huffman bits':>14}{'ratio':>9}{'avg len':>9}")
    for r in rows:
        print(f"{r['file']:<22}{r['chars']:>8,}{r['distinct']:>9}{r['fixed_bits']:>12,}{r['huffman_bits']:>14,}"
              f"{r['ratio']:>9.3f}{r['avg_len']:>9.3f}")
    ensure_out()
    with open(f"{OUT_DIR}/q2_huffman.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# =====================================================================
# Q3  SHORTEST PATHS AND MINIMUM SPANNING TREES
# =====================================================================

def read_graph(path):
    """Directed weighted graph from a text file; returns (vertices in order of appearance, adjacency dict)."""
    adj, order = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            u, v, w = line.split()
            w = float(w)
            w = int(w) if w == int(w) else w
            for x in (u, v):
                if x not in adj:
                    adj[x] = []
                    order.append(x)
            adj[u].append((v, w))
    return order, adj


def dijkstra(adj, source, allow_negative=False):
    """Binary-heap Dijkstra (lazy deletion).  O((V + E) log V).  Each vertex is finalised once."""
    if not allow_negative:
        for u in adj:
            for v, w in adj[u]:
                if w < 0:
                    raise ValueError(f"negative edge {u}->{v} ({w}): Dijkstra requires non-negative weights")
    dist = {u: math.inf for u in adj}
    parent = {u: None for u in adj}
    dist[source] = 0
    heap = [(0, source)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:                                   # stale heap entry
            continue
        done.add(u)
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                parent[v] = u
                heapq.heappush(heap, (dist[v], v))
    return dist, parent


def bellman_ford(adj, source):
    """Reference answer that is also correct with negative edges (no negative cycles assumed)."""
    dist = {u: math.inf for u in adj}
    dist[source] = 0
    for _ in range(len(adj) - 1):
        changed = False
        for u in adj:
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    changed = True
        if not changed:
            break
    return dist


def path_to(parent, target):
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    return list(reversed(path))


def print_shortest_paths(order, adj, source):
    dist, parent = dijkstra(adj, source)
    print(f"Shortest paths from {source}:")
    print(f"  {'vertex':<8}{'distance':>10}   path")
    for v in order:
        if dist[v] == math.inf:
            print(f"  {v:<8}{'inf':>10}   unreachable")
        else:
            print(f"  {v:<8}{dist[v]:>10}   {' -> '.join(path_to(parent, v))}")
    return dist


DEMO_GRAPH = """# directed weighted graph (CLRS figure 24.6)
s t 10
s y 5
t x 1
t y 2
y t 3
y x 9
y z 2
x z 4
z x 6
z s 7
"""


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]     # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal(n, edges):
    """edges = [(w, u, v)].  Sort, then add an edge if it joins two different components.  O(E log E)."""
    uf = UnionFind(n)
    total, tree = 0, []
    for w, u, v in sorted(edges):
        if uf.union(u, v):
            total += w
            tree.append((u, v, w))
            if len(tree) == n - 1:
                break
    return total, tree


def prim(n, adj, start=0):
    """adj[u] = [(v, w)].  Grow one tree from `start` with a binary heap of crossing edges.  O(E log V)."""
    seen = [False] * n
    seen[start] = True
    heap = [(w, start, v) for v, w in adj[start]]
    heapq.heapify(heap)
    total, tree = 0, []
    while heap and len(tree) < n - 1:
        w, u, v = heapq.heappop(heap)
        if seen[v]:
            continue
        seen[v] = True
        total += w
        tree.append((u, v, w))
        for x, wx in adj[v]:
            if not seen[x]:
                heapq.heappush(heap, (wx, v, x))
    return total, tree


def prim_array(n, adj, start=0):
    """Bonus: Prim without a heap - scan an array for the cheapest vertex each round.  O(V^2 + E)."""
    INF = math.inf
    key = [INF] * n
    parent = [-1] * n
    inside = [False] * n
    key[start] = 0
    total, tree = 0, []
    for _ in range(n):
        u, best = -1, INF
        for v in range(n):
            if not inside[v] and key[v] < best:
                u, best = v, key[v]
        if u == -1:
            break
        inside[u] = True
        if parent[u] != -1:
            total += key[u]
            tree.append((parent[u], u, key[u]))
        for v, w in adj[u]:
            if not inside[v] and w < key[v]:
                key[v] = w
                parent[v] = u
    return total, tree


def random_connected_graph(V, E, rng, max_w=10 ** 6):
    """Random connected undirected graph with exactly E edges (E >= V-1)."""
    edges = set()
    labels = list(range(V))
    rng.shuffle(labels)
    for i in range(1, V):                               # random spanning tree guarantees connectivity
        j = rng.randrange(i)
        a, b = labels[i], labels[j]
        edges.add((min(a, b), max(a, b)))
    max_edges = V * (V - 1) // 2
    E = min(E, max_edges)
    while len(edges) < E:
        a, b = rng.randrange(V), rng.randrange(V)
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return [(rng.randint(1, max_w), u, v) for u, v in sorted(edges)]


def to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for w, u, v in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def is_spanning_tree(n, tree):
    if len(tree) != n - 1:
        return False
    uf = UnionFind(n)
    return all(uf.union(u, v) for u, v, _ in tree)


def run_q3(quick=False):
    ensure_out()
    rng = random.Random(SEED)

    gpath = os.path.join(OUT_DIR, "graph.txt")
    with open(gpath, "w", encoding="utf-8") as fh:
        fh.write(DEMO_GRAPH)
    print(f"Q3 part 1 - Dijkstra on the graph file {gpath}\n")
    order, adj = read_graph(gpath)
    dist = print_shortest_paths(order, adj, "s")
    assert dist == bellman_ford(adj, "s") and dist["x"] == 9 and dist["z"] == 7
    print("  (distances equal Bellman-Ford's)")

    print("\nQ3 part 3 - Kruskal vs Prim on random connected graphs")
    ntests = 80 if quick else 400
    for _ in range(ntests):
        V = rng.randint(2, 40)
        E = rng.randint(V - 1, min(V * (V - 1) // 2, 4 * V))
        edges = random_connected_graph(V, E, rng, max_w=rng.choice([5, 20, 1000]))   # small max_w = many ties
        adj_u = to_adj(V, edges)
        wk, tk = kruskal(V, edges)
        wp, tp = prim(V, adj_u)
        wa, ta = prim_array(V, adj_u)
        assert wk == wp == wa, (wk, wp, wa)
        assert is_spanning_tree(V, tk) and is_spanning_tree(V, tp) and is_spanning_tree(V, ta)
        assert sum(w for _, _, w in tk) == wk
    print(f"  {ntests} random connected graphs (2-40 vertices, with many equal weights): "
          f"Kruskal, Prim (heap) and Prim (array) always give the same total weight, and every result "
          f"is a spanning tree with V-1 edges  OK")
    demo = random_connected_graph(8, 14, random.Random(7), max_w=20)
    wk, tk = kruskal(8, demo)
    print(f"  small example (V=8, E=14): MST weight {wk}, edges {sorted((min(u, v), max(u, v), w) for u, v, w in tk)}")

    print("\nQ3 part 4 - timing Kruskal and Prim, V = 1,000 (graph is built before timing; best of 3)")
    V = 300 if quick else 1000
    rows = []
    print(f"{'graph':<22}{'E':>10}{'Kruskal (s)':>14}{'Prim heap (s)':>16}{'Prim array (s)':>17}{'winner (Kruskal vs Prim heap)':>32}")
    for name, E in (("sparse  (E = 3V)", 3 * V), ("dense   (E = V^2/4)", V * V // 4)):
        edges = random_connected_graph(V, E, rng)
        adj_u = to_adj(V, edges)
        tk = best_time(lambda: kruskal(V, edges))
        tp = best_time(lambda: prim(V, adj_u))
        ta = best_time(lambda: prim_array(V, adj_u))
        assert kruskal(V, edges)[0] == prim(V, adj_u)[0] == prim_array(V, adj_u)[0]
        winner = "Kruskal" if tk < tp else "Prim (heap)"
        rows.append((name.strip(), len(edges), tk, tp, ta))
        print(f"{name:<22}{len(edges):>10,}{tk:>14.4f}{tp:>16.4f}{ta:>17.4f}{winner:>32}", flush=True)
    with open(f"{OUT_DIR}/q3_mst_timing.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["graph", "edges", "kruskal_s", "prim_heap_s", "prim_array_s"])
        w.writerows(rows)
    plot_q3(rows, V)

    print("\nQ3 part 5 - Dijkstra with a negative edge")
    neg = {"s": [("a", 2), ("b", 3)], "a": [("c", 1)], "b": [("a", -2)], "c": []}
    print("  edges: s->a 2, s->b 3, b->a -2, a->c 1")
    try:
        dijkstra(neg, "s")
    except ValueError as exc:
        print(f"  dijkstra() refuses: {exc}")
    d_wrong, _ = dijkstra(neg, "s", allow_negative=True)
    d_true = bellman_ford(neg, "s")
    print(f"  {'vertex':<8}{'Dijkstra':>10}{'true (Bellman-Ford)':>22}")
    for v in ("s", "a", "b", "c"):
        flag = "   <-- WRONG" if d_wrong[v] != d_true[v] else ""
        print(f"  {v:<8}{d_wrong[v]:>10}{d_true[v]:>22}{flag}")
    assert d_wrong["c"] != d_true["c"]
    print("  Dijkstra finalises a with distance 2 before it sees b; the edge b->a (-2) later lowers a to 1,")
    print("  but a was already finished, so c is never updated and stays 3 instead of the true 2.")


def plot_q3(rows, V):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r[0] for r in rows]
    x = range(len(rows))
    width = 0.27
    ax.bar([i - width for i in x], [r[2] for r in rows], width, label="Kruskal")
    ax.bar(list(x), [r[3] for r in rows], width, label="Prim (heap)")
    ax.bar([i + width for i in x], [r[4] for r in rows], width, label="Prim (array, O(V^2))")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("time (s)")
    ax.set_title(f"MST algorithms, V = {V:,}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/q3_mst_timing.png", dpi=130)
    plt.close(fig)


# =====================================================================
# Q4  RANDOMIZED ALGORITHMS
# =====================================================================

def quickselect(a, k, rng=random):
    """k-th smallest (k = 1 is the minimum).  Random pivot, three-way split, keep only the part that holds
    the answer.  Expected O(n); the pivot splits are made with list comprehensions (not in place)."""
    if not 1 <= k <= len(a):
        raise ValueError("k out of range")
    a = list(a)
    while True:
        pivot = a[rng.randrange(len(a))]
        lower = [x for x in a if x < pivot]
        if k <= len(lower):
            a = lower
            continue
        equal = a.count(pivot)
        if k <= len(lower) + equal:
            return pivot
        k -= len(lower) + equal
        a = [x for x in a if x > pivot]


def merge_sort(a):
    n = len(a)
    if n <= 1:
        return a[:]
    mid = n // 2
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def run_q4_quickselect(quick, rng):
    print("Q4 part 1 - randomized quickselect vs sorting (median, k = n/2; best of 3 except merge sort)")
    small = random.Random(5)
    for _ in range(300):                                 # correctness, including heavy duplicates
        arr = [small.randint(0, 10) for _ in range(small.randint(1, 40))]
        k = small.randint(1, len(arr))
        assert quickselect(arr, k, small) == sorted(arr)[k - 1]
    print("  correctness: matches sorted(a)[k-1] on 300 random arrays with many duplicates")
    sizes = [10 ** 4, 10 ** 5, 3 * 10 ** 5] if quick else [10 ** 4, 5 * 10 ** 4, 10 ** 5, 5 * 10 ** 5, 10 ** 6]
    rows = []
    print(f"{'n':>10}{'quickselect':>14}{'sorted()[k-1]':>16}{'merge sort[k-1]':>18}{'sorted/quickselect':>20}")
    for n in sizes:
        data = [rng.randint(0, 10 ** 9) for _ in range(n)]
        k = n // 2
        tq = best_time(lambda: quickselect(data, k, rng))
        ts = best_time(lambda: sorted(data)[k - 1])
        t0 = time.perf_counter()
        ms = merge_sort(data)[k - 1]
        tm = time.perf_counter() - t0
        assert quickselect(data, k, rng) == sorted(data)[k - 1] == ms
        rows.append((n, tq, ts, tm))
        print(f"{n:>10,}{tq:>14.4f}{ts:>16.4f}{tm:>18.4f}{ts / tq:>20.2f}", flush=True)
    with open(f"{OUT_DIR}/q4_quickselect.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "quickselect_s", "python_sorted_s", "merge_sort_s"])
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="quickselect (Python)  O(n)")
        ax.plot([r[0] for r in rows], [r[3] for r in rows], "s--", label="merge sort + index (Python)  O(n log n)")
        ax.plot([r[0] for r in rows], [r[2] for r in rows], "^:", label="sorted() + index (C)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("time (s)")
        ax.set_title("k-th smallest element, k = n/2")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/q4_quickselect.png", dpi=130)
        plt.close(fig)
    except ImportError:
        pass


def is_witness(a, d, s, n):
    """True if base a PROVES n composite (strong-probable-prime test fails)."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return False
    return True


def miller_rabin(n, rounds=20, rng=random, presieve=True):
    """Randomized primality test.  'Composite' is always right; 'prime' is wrong with probability <= 4^-rounds.
    presieve=True first tries a few tiny primes (a speed-up); presieve=False runs the bare test, which is
    what the Carmichael experiment needs (561 = 3 x 11 x 17 would otherwise be caught by 561 % 3 == 0)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if presieve:
        for p in (3, 5, 7, 11, 13):
            if n == p:
                return True
            if n % p == 0:
                return False
    if n < 5:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)                       # random base in 2..n-2
        if is_witness(a, d, s, n):
            return False
    return True


def trial_division(n):
    """Returns (is_prime, smallest factor or None).  Tries 2 and the odd numbers up to sqrt(n)."""
    if n < 2:
        return False, None
    if n % 2 == 0:
        return n == 2, (None if n == 2 else 2)
    i = 3
    limit = math.isqrt(n)
    while i <= limit:
        if n % i == 0:
            return False, i
        i += 2
    return True, None


def run_q4_primes(quick, rng):
    print("\nQ4 part 2 - Miller-Rabin vs trial division")
    sieve_limit = 5000 if quick else 20000
    sieve = [True] * sieve_limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(sieve_limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, sieve_limit, i):
                sieve[j] = False
    for n in range(sieve_limit):
        assert miller_rabin(n, 10, rng) == sieve[n] == trial_division(n)[0], n
    print(f"  sanity: Miller-Rabin (10 rounds) = sieve = trial division for every n < {sieve_limit:,}")

    nums = [rng.randrange(10 ** 11, 10 ** 12) for _ in range(20)]
    print("\n  (a) 20 random 12-digit numbers")
    print(f"  {'n':>14}  {'Miller-Rabin':<13}{'trial div.':<12}{'agree':<7}{'t MR (ms)':>11}{'t TD (ms)':>11}  note")
    tot_mr = tot_td = 0
    primes = composites = 0
    for n in nums:
        t0 = time.perf_counter()
        mr = miller_rabin(n, 20, rng)
        t1 = time.perf_counter()
        td, factor = trial_division(n)
        t2 = time.perf_counter()
        assert mr == td
        tot_mr += t1 - t0
        tot_td += t2 - t1
        primes += td
        composites += not td
        note = "prime" if td else f"smallest factor {factor}"
        print(f"  {n:>14}  {'prime' if mr else 'composite':<13}{'prime' if td else 'composite':<12}{'yes':<7}"
              f"{1000 * (t1 - t0):>11.3f}{1000 * (t2 - t1):>11.3f}  {note}")
    print(f"  all 20 agree ({primes} prime, {composites} composite).  total time: Miller-Rabin {1000 * tot_mr:.2f} ms, "
          f"trial division {1000 * tot_td:.2f} ms")
    print("  Most random numbers are composite with a small factor, so trial division stops early here.")

    print("\n  (b) the hard case for trial division: 10 random 12-digit PRIMES (it must test every divisor up to sqrt(n))")
    print(f"  {'n':>14}  {'t MR (ms)':>11}{'t TD (ms)':>12}{'TD / MR':>10}")
    found = []
    while len(found) < 10:
        c = rng.randrange(10 ** 11, 10 ** 12) | 1
        if miller_rabin(c, 20, rng):
            found.append(c)
    tmr = ttd = 0
    for n in found:
        t0 = time.perf_counter()
        mr = miller_rabin(n, 20, rng)
        t1 = time.perf_counter()
        td, _ = trial_division(n)
        t2 = time.perf_counter()
        assert mr and td
        tmr += t1 - t0
        ttd += t2 - t1
        print(f"  {n:>14}  {1000 * (t1 - t0):>11.3f}{1000 * (t2 - t1):>12.1f}{(t2 - t1) / max(t1 - t0, 1e-9):>10,.0f}")
    print(f"  all 10 confirmed prime by both methods.  average: Miller-Rabin {1000 * tmr / 10:.3f} ms, "
          f"trial division {1000 * ttd / 10:.1f} ms")


def run_q4_carmichael(quick, rng):
    n = 561
    print("\nQ4 part 3 - Miller-Rabin with ONE round on the Carmichael number 561 = 3 x 11 x 17")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = range(2, n - 1)
    liars = [a for a in bases if not is_witness(a, d, s, n)]
    fermat = sum(1 for a in bases if pow(a, n - 1, n) == 1)
    trials = 1000
    said_prime = sum(1 for _ in range(trials) if miller_rabin(n, 1, rng, presieve=False))
    print(f"  561 - 1 = {d} * 2^{s}   (bare test: no trial division by small primes, which would expose 561 = 3 x 187)")
    print(f"  {trials} runs, 1 round each: answered 'prime' {said_prime} times -> {100 * said_prime / trials:.1f}%")
    print(f"  exact expectation: {len(liars)} of the {len(bases)} possible bases are strong liars "
          f"-> {100 * len(liars) / len(bases):.2f}%   (theory: at most 25%)")
    print(f"  for comparison, the plain Fermat test a^560 = 1 (mod 561) is fooled by {fermat} of {len(bases)} bases "
          f"-> {100 * fermat / len(bases):.1f}%  (561 is a Carmichael number)")
    print(f"  more rounds (each 1000 runs):")
    for r in (1, 2, 3, 5, 10):
        wrong = sum(1 for _ in range(trials) if miller_rabin(n, r, rng, presieve=False))
        print(f"    {r:>2} rounds: said 'prime' {wrong:>4} times   (expected about {1000 * (len(liars) / len(bases)) ** r:.2f})")
    assert said_prime > 0


def run_q4(quick=False):
    ensure_out()
    rng = random.Random(SEED)
    run_q4_quickselect(quick, rng)
    run_q4_primes(quick, rng)
    run_q4_carmichael(quick, rng)


# =====================================================================
def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="CSF302 Lab 7 - greedy and randomized algorithms")
    ap.add_argument("part", choices=["q1", "q2", "q3", "q4", "all", "activities", "huffman", "dijkstra"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--source", help="dijkstra: source vertex (default: first vertex in the file)")
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke test")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.out
    sys.setrecursionlimit(10000)
    if args.part == "activities":
        run_activities_cli(args.files[0] if args.files else None)
        return
    if args.part == "huffman":
        if not args.files:
            ap.error("huffman needs at least one text file")
        run_huffman_files(args.files)
        return
    if args.part == "dijkstra":
        if not args.files:
            ap.error("dijkstra needs a graph file")
        order, adj = read_graph(args.files[0])
        src = args.source or order[0]
        if src not in adj:
            ap.error(f"unknown source vertex {src!r}")
        print_shortest_paths(order, adj, src)
        return
    for name, fn in (("q1", run_q1), ("q2", run_q2), ("q3", run_q3), ("q4", run_q4)):
        if args.part in (name, "all"):
            if args.part == "all":
                print("\n" + "=" * 78 + f"\n{name.upper()}\n" + "=" * 78)
            fn(args.quick)


if __name__ == "__main__":
    main()