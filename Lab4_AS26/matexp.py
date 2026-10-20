
import random
import sys
import time


def validate_matrix(A, name="A"):
    if (not isinstance(A, list) or not A or any(not isinstance(r, list) for r in A)):
        raise TypeError(f"{name} must be a non-empty list of rows.")
    d = len(A)
    for r in A:
        if len(r) != d:
            raise ValueError(f"{name} must be square ({d} rows but a row has {len(r)} entries).")
        for v in r:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} entries must be integers (got {v!r}).")
    return d


def validate_mod(m):
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError(f"modulus must be an integer (got {m!r}).")
    if m <= 0:
        raise ValueError(f"Invalid modulus {m}: it must be a positive integer.")


def validate_exp(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"exponent must be an integer (got {n!r}).")
    if n < 0:
        raise ValueError(f"Negative exponent {n} is not supported (needs a matrix inverse).")

def identity(d):
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ValueError("dimension must be a positive integer.")
    return [[1 if i == j else 0 for j in range(d)] for i in range(d)]


def mat_mult(A, B, m):
    """(A x B) mod m for d x d matrices; every entry reduced mod m."""
    d = validate_matrix(A, "A")
    if validate_matrix(B, "B") != d:
        raise ValueError("A and B must have the same dimension.")
    validate_mod(m)
    return _mul(A, B, m)


def _mul(A, B, m):
    """Unchecked multiplication (inputs already validated) for speed."""
    d = len(A)
    Bt = list(zip(*B))                      # columns of B
    return [[sum(a * b for a, b in zip(row, col)) % m for col in Bt] for row in A]


def naive_mat_pow(A, n, m, counter=None):
    """A^n mod m by n repeated matrix multiplications, O(n d^3)."""
    d = validate_matrix(A)
    validate_mod(m)
    validate_exp(n)
    base = [[v % m for v in row] for row in A]
    result = [[v % m for v in row] for row in identity(d)]
    for _ in range(n):
        result = _mul(result, base, m)
        if counter is not None:
            counter[0] += 1
    return result


def binary_mat_pow(A, n, m, counter=None):
    """A^n mod m by repeated squaring, O(d^3 log n)."""
    d = validate_matrix(A)
    validate_mod(m)
    validate_exp(n)
    base = [[v % m for v in row] for row in A]
    result = [[v % m for v in row] for row in identity(d)]
    while n:
        if n & 1:                                   # lowest bit set -> multiply
            result = _mul(result, base, m)
            if counter is not None:
                counter[0] += 1
        n >>= 1
        if n:                                        # no useless final squaring
            base = _mul(base, base, m)
            if counter is not None:
                counter[0] += 1
    return result


def random_matrix(d, lo=0, hi=9, rnd=random):
    return [[rnd.randint(lo, hi) for _ in range(d)] for _ in range(d)]


def show(M, title=None):
    if title:
        print(title)
    w = max(len(str(v)) for r in M for v in r)
    for r in M:
        print("  [ " + " ".join(f"{v:>{w}}" for v in r) + " ]")


def compare_matrix_pow(verbose=True, random_tests=300):
    rnd = random.Random(7)
    fixed = [([[1, 1], [1, 0]], 10, 1000), ([[2, 0], [0, 3]], 5, 7), ([[1, 2], [3, 4]], 0, 13),
             ([[1, 2], [3, 4]], 1, 1), ([[5]], 20, 97), ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 12, 1000000007),
             ([[-1, 2], [3, -4]], 7, 11)]
    cases = list(fixed)
    for _ in range(random_tests):
        d = rnd.randint(1, 5)
        cases.append((random_matrix(d, -20, 20, rnd), rnd.randint(0, 200), rnd.randint(1, 10**9)))
    ok = True
    for idx, (A, n, m) in enumerate(cases):
        c1, c2 = [0], [0]
        r1 = naive_mat_pow(A, n, m, c1)
        r2 = binary_mat_pow(A, n, m, c2)
        good = r1 == r2
        ok &= good
        if verbose and idx < len(fixed):
            print(f"\nA = {A}, n = {n}, m = {m}")
            show(r1, "  Naive result:")
            show(r2, "  Binary result:")
            print(f"  matrix multiplications: naive = {c1[0]}, binary = {c2[0]}  ->  {'OK' if good else 'MISMATCH'}")
    if verbose:
        print(f"\n... plus {random_tests} random cases (dimensions 1-5, negative entries, n up to 200, m = 1 included).")
        print("ALL RESULTS MATCH." if ok else "MISMATCH FOUND!")
    return ok


FIB = [[1, 1], [1, 0]]


def fib_mod(n, m):
    """F(n) mod m via [[1,1],[1,0]]^n  (F(0)=0, F(1)=1)."""
    validate_exp(n)
    validate_mod(m)
    return binary_mat_pow(FIB, n, m)[0][1]


def fib_iterative(n, m=None):
    """Reference implementation (loop), used only to verify."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a % m if m else a


def fibonacci_demo():
    print("Verification against the simple loop (n = 0..1000, several moduli):")
    ok = all(fib_mod(n, m) == fib_iterative(n, m)
             for n in range(0, 1001) for m in (1, 2, 10, 1000, 10**9 + 7))
    print("  all match." if ok else "  MISMATCH!")
    print("\nSmall values F(0..15):", [fib_mod(n, 10**9) for n in range(16)])
    m = 10**9 + 7
    print(f"\nLarge Fibonacci numbers mod {m:,} (no earlier term is ever generated):")
    print("-" * 74)
    for n in (10**3, 10**6, 10**9, 10**12, 10**15, 10**18, 10**100):
        c = [0]
        r = binary_mat_pow(FIB, n, m, c)[0][1]
        label = f"10^{len(str(n)) - 1}"
        print(f"  F({label:>6}) mod m = {r:>11}   ({c[0]} 2x2 matrix multiplications)")
    print("-" * 74)
    print("  Known check: F(10^18) mod 1,000,000,007 = 209783453 ->",
          "OK" if fib_mod(10**18, m) == 209783453 else "differs (verify the value independently)")
    print("  Last 10 digits of F(10^15) (m = 10^10):", f"{fib_mod(10**15, 10**10):010d}")


def adjacency_matrix(V, edges):
    """Directed graph on vertices 0..V-1; parallel edges are counted separately."""
    if isinstance(V, bool) or not isinstance(V, int) or V < 1:
        raise ValueError("number of vertices must be a positive integer.")
    A = [[0] * V for _ in range(V)]
    for u, v in edges:
        if not (0 <= u < V and 0 <= v < V):
            raise ValueError(f"edge ({u}, {v}) uses a vertex outside 0..{V - 1}.")
        A[u][v] += 1
    return A


def count_walks(V, edges, u, v, k, m):
    """Number of walks with exactly k edges from u to v, modulo m."""
    if not (0 <= u < V and 0 <= v < V):
        raise ValueError(f"vertices must be in 0..{V - 1}.")
    A = adjacency_matrix(V, edges)
    return binary_mat_pow(A, k, m)[u][v]


def brute_force_walks(V, edges, u, v, k):
    """Reference: count walks by direct recursion (small k only)."""
    adj = [[] for _ in range(V)]
    for a, b in edges:
        adj[a].append(b)

    def go(x, left):
        if left == 0:
            return 1 if x == v else 0
        return sum(go(y, left - 1) for y in adj[x])
    return go(u, k)


def walks_demo():
    V = 4
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)]
    print(f"Directed graph with {V} vertices, edges: {edges}")
    show(adjacency_matrix(V, edges), "Adjacency matrix:")
    print("\nWalks of exactly k edges from 0 to 3 (matrix power vs brute force):")
    print(f"  {'k':>3} | {'matrix power':>12} | {'brute force':>11} | match")
    allok = True
    for k in range(0, 11):
        a = count_walks(V, edges, 0, 3, k, 10**18)
        b = brute_force_walks(V, edges, 0, 3, k)
        allok &= a == b
        print(f"  {k:>3} | {a:>12} | {b:>11} | {'OK' if a == b else 'NO'}")
    print("  all match." if allok else "  MISMATCH!")
    m = 10**9 + 7
    for k in (100, 10**6, 10**18):
        print(f"  walks 0 -> 3 with exactly {k:,} edges, mod {m:,}: {count_walks(V, edges, 0, 3, k, m)}")
    print("  k = 0 means 'no edges': walks(u, u) = 1, walks(u, v) = 0 for u != v:",
          count_walks(V, edges, 2, 2, 0, m), count_walks(V, edges, 2, 3, 0, m))


def _timed(fn, *args):
    t = time.perf_counter()
    r = fn(*args)
    return r, time.perf_counter() - t


def benchmark(m=1_000_000_007):
    print(f"\nNaive vs binary matrix exponentiation (m = {m:,}); times in seconds")
    print("-" * 100)
    print(f"{'dim d':>5} | {'exponent n':>10} | {'naive mults':>11} | {'binary mults':>12} | {'naive (s)':>10} |"
          f" {'binary (s)':>10} | {'speedup':>9} | match")
    print("-" * 100)
    rnd = random.Random(3)
    for d in (2, 4, 8, 16):
        A = random_matrix(d, 0, 9, rnd)
        for n in (10, 100, 1000):
            c1, c2 = [0], [0]
            r1, t1 = _timed(naive_mat_pow, A, n, m, c1)
            reps = 200
            t = time.perf_counter()
            for _ in range(reps):
                c2[0] = 0
                r2 = binary_mat_pow(A, n, m, c2)
            t2 = (time.perf_counter() - t) / reps
            print(f"{d:>5} | {n:>10,} | {c1[0]:>11,} | {c2[0]:>12} | {t1:>10.5f} | {t2:>10.6f} |"
                  f" {t1 / t2:>8.0f}x | {'Yes' if r1 == r2 else 'NO'}")
    print("-" * 100)
    print("\nBinary only (huge exponents, naive impossible):")
    print("-" * 70)
    print(f"{'dim d':>5} | {'exponent n':>10} | {'binary mults':>12} | {'time (s)':>10}")
    print("-" * 70)
    for d in (2, 4, 8, 16, 32):
        A = random_matrix(d, 0, 9, rnd)
        for n in (10**6, 10**12, 10**18):
            c = [0]
            _, t = _timed(binary_mat_pow, A, n, m, c)
            print(f"{d:>5} | {'10^' + str(len(str(n)) - 1):>10} | {c[0]:>12} | {t:>10.5f}")
    print("-" * 70)


def read_int(prompt):
    try:
        return int(input(prompt).strip())
    except ValueError:
        print("Please enter an integer.")
        return None


def read_matrix_input():
    d = read_int("Matrix dimension d: ")
    if d is None or d < 1:
        return None
    M = []
    for i in range(d):
        try:
            row = list(map(int, input(f"  row {i + 1} ({d} integers): ").split()))
        except ValueError:
            print("Integers only.")
            return None
        if len(row) != d:
            print(f"Need exactly {d} integers.")
            return None
        M.append(row)
    return M


def menu():
    while True:
        print("\n===== Topic 7: Matrix Modular Exponentiation =====")
        print("1. Compute A^n mod m (enter a matrix; naive vs binary)")
        print("2. Compare naive vs binary on many test cases")
        print("3. Fibonacci F(n) mod m")
        print("4. Large Fibonacci demo")
        print("5. Count walks of exactly k edges in a directed graph")
        print("6. Benchmark naive vs binary")
        print("7. Exit")
        ch = input("Choice: ").strip()
        try:
            if ch == "1":
                A = read_matrix_input()
                n, m = read_int("exponent n: "), read_int("modulus m: ")
                if A is None or n is None or m is None:
                    continue
                c1, c2 = [0], [0]
                r1 = naive_mat_pow(A, n, m, c1)
                r2 = binary_mat_pow(A, n, m, c2)
                show(r1, "Naive result:")
                show(r2, "Binary result:")
                print(f"Multiplications: naive = {c1[0]}, binary = {c2[0]}.",
                      "Results match." if r1 == r2 else "MISMATCH!")
            elif ch == "2":
                compare_matrix_pow()
            elif ch == "3":
                n, m = read_int("n: "), read_int("modulus m: ")
                if n is not None and m is not None:
                    print(f"F({n}) mod {m} = {fib_mod(n, m)}")
            elif ch == "4":
                fibonacci_demo()
            elif ch == "5":
                V = read_int("Number of vertices: ")
                E = read_int("Number of edges: ")
                if V is None or E is None:
                    continue
                edges = []
                for i in range(E):
                    pair = input(f"  edge {i + 1} (u v, vertices 0..{V - 1}): ").split()
                    edges.append((int(pair[0]), int(pair[1])))
                u, v = read_int("start vertex u: "), read_int("end vertex v: ")
                k, m = read_int("number of edges k: "), read_int("modulus m: ")
                if None in (u, v, k, m):
                    continue
                print(f"Walks of exactly {k} edges from {u} to {v} (mod {m}): {count_walks(V, edges, u, v, k, m)}")
            elif ch == "6":
                benchmark()
            elif ch == "7":
                break
            else:
                print("Invalid choice.")
        except (ValueError, TypeError, IndexError) as e:
            print("Error:", e)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("=== Parts 1-5: naive vs binary matrix power ===")
        print("identity(3) =", identity(3))
        compare_matrix_pow()
        print("\n=== Parts 6-7: Fibonacci ===")
        fibonacci_demo()
        print("\n=== Part 8: walks in a directed graph ===")
        walks_demo()
        print("\n=== Part 9: benchmark ===")
        benchmark()
    else:
        menu()