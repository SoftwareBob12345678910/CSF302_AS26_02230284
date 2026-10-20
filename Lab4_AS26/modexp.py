
import random
import sys
import time


def validate_inputs(a, b, m):
    """Raise TypeError/ValueError with a clear message for invalid input."""
    for name, v in (("base a", a), ("exponent b", b), ("modulus m", m)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{name} must be an integer (got {type(v).__name__}: {v!r}).")
    if m == 0:
        raise ValueError("Invalid modulus: m = 0 (division by zero).")
    if m < 0:
        raise ValueError(f"Invalid modulus: m = {m}; the modulus must be a positive integer.")
    if b < 0:
        raise ValueError(f"Negative exponent b = {b} is not supported: it would need the "
                         f"modular inverse of a, which exists only when gcd(a, m) = 1.")


def naive_modexp(a, b, m, counter=None):
    """a^b mod m by repeated multiplication: b multiplications, O(b)."""
    validate_inputs(a, b, m)
    a %= m                          # also handles negative bases
    result = 1 % m                  # a^0 = 1 (and 0 when m = 1)
    for _ in range(b):
        result = (result * a) % m
        if counter is not None:
            counter[0] += 1
    return result


def binary_modexp(a, b, m, counter=None):
    """a^b mod m by square-and-multiply (uses % 2 and // 2), O(log b)."""
    validate_inputs(a, b, m)
    a %= m
    result = 1 % m
    while b > 0:
        if b % 2 == 1:                       # exponent odd  -> multiply
            result = (result * a) % m
            if counter is not None:
                counter[0] += 1
        a = (a * a) % m                      # square
        if counter is not None:
            counter[0] += 1
        b //= 2                              # halve exponent
    return result


def binary_modexp_bitwise(a, b, m, counter=None):
    """Square-and-multiply using bitwise operations: b & 1 and b >>= 1."""
    validate_inputs(a, b, m)
    a %= m
    result = 1 % m
    while b:
        if b & 1:                            # lowest bit set  <=>  b is odd
            result = (result * a) % m
            if counter is not None:
                counter[0] += 1
        a = (a * a) % m
        if counter is not None:
            counter[0] += 1
        b >>= 1                              # shift right  <=>  b // 2
    return result


def compare_outputs(tests=None, random_tests=2000, verbose=True):
    """Compare naive, binary, bitwise (and Python's pow) on fixed + random cases."""
    fixed = [(2, 10, 1000), (3, 0, 7), (0, 0, 5), (0, 5, 7), (5, 1, 13), (7, 256, 13),
             (2, 100, 1), (-3, 5, 7), (-2, 10, 1000), (10, 9, 6), (123, 456, 789),
             (2, 1000, 1000000007), (12345, 6789, 99991)]
    cases = list(tests or fixed)
    rnd = random.Random(1)
    cases += [(rnd.randint(-10**6, 10**6), rnd.randint(0, 3000), rnd.randint(1, 10**9))
              for _ in range(random_tests)]
    ok = True
    if verbose:
        print(f"{'a':>8} {'b':>6} {'m':>11} | {'naive':>11} {'binary':>11} {'bitwise':>11} | {'pow()':>11} | status")
        print("-" * 86)
    for idx, (a, b, m) in enumerate(cases):
        n1 = naive_modexp(a, b, m)
        b1 = binary_modexp(a, b, m)
        b2 = binary_modexp_bitwise(a, b, m)
        ref = pow(a, b, m)                    # independent reference
        good = n1 == b1 == b2 == ref
        ok &= good
        if verbose and idx < len(fixed):
            print(f"{a:>8} {b:>6} {m:>11} | {n1:>11} {b1:>11} {b2:>11} | {ref:>11} | {'OK' if good else 'MISMATCH'}")
    if verbose:
        print(f"... plus {random_tests} random cases (negative/zero bases, zero exponents, m = 1 included).")
        print("ALL RESULTS MATCH." if ok else "MISMATCH FOUND!")
    return ok


def validation_demo():
    bad = [(2, -3, 7, "negative exponent"), (2, 5, 0, "modulus zero"), (2, 5, -7, "negative modulus"),
           (2.5, 3, 7, "non-integer base"), (2, 3, "7", "string modulus"), (2, True, 7, "boolean exponent")]
    print("Invalid inputs (each must be rejected):")
    for a, b, m, why in bad:
        try:
            binary_modexp(a, b, m)
            print(f"  {why:<20} -> NOT rejected (bug)")
        except (ValueError, TypeError) as e:
            print(f"  {why:<20} -> {type(e).__name__}: {e}")
    print("\nValid edge cases:")
    for a, b, m, why in [(5, 0, 13, "zero exponent"), (0, 0, 13, "0^0 (convention = 1)"),
                         (0, 5, 13, "zero base"), (-3, 5, 7, "negative base"),
                         (-3, 4, 7, "negative base, even exponent"), (9, 9, 1, "modulus 1")]:
        print(f"  {why:<30} ({a})^{b} mod {m} = {binary_modexp(a, b, m)}  (pow: {pow(a, b, m)})")


def _time(fn, a, b, m, repeats):
    t = time.perf_counter()
    for _ in range(repeats):
        fn(a, b, m)
    return (time.perf_counter() - t) / repeats


def timing_table(a=7, m=1_000_000_007):
    print(f"\nTiming for a = {a}, m = {m:,}   (multiplication counts are exact)")
    print("-" * 104)
    print(f"{'exponent b':>12} | {'naive mults':>12} | {'binary mults':>12} | {'bitwise mults':>13} |"
          f" {'naive (s)':>11} | {'binary (s)':>11} | {'bitwise (s)':>11} | speedup")
    print("-" * 104)
    for e in range(1, 8):
        b = 10 ** e
        c1, c2, c3 = [0], [0], [0]
        r1 = naive_modexp(a, b, m, c1)
        r2 = binary_modexp(a, b, m, c2)
        r3 = binary_modexp_bitwise(a, b, m, c3)
        assert r1 == r2 == r3 == pow(a, b, m)
        tn = _time(naive_modexp, a, b, m, 1 if b >= 10**5 else 20)
        tb = _time(binary_modexp, a, b, m, 2000)
        tw = _time(binary_modexp_bitwise, a, b, m, 2000)
        print(f"{b:>12,} | {c1[0]:>12,} | {c2[0]:>12} | {c3[0]:>13} | {tn:>11.6f} | {tb:>11.8f} | {tw:>11.8f} | {tn / tb:>9,.0f}x")
    print("-" * 104)
    print("\nBinary only (naive would never finish) - huge exponents:")
    print("-" * 72)
    print(f"{'exponent b':>14} | {'bits':>5} | {'binary mults':>12} | {'time (s)':>12} | result mod m")
    print("-" * 72)
    for b in (10**12, 10**18, 10**50, 10**100, 10**300):
        c = [0]
        r = binary_modexp(a, b, m, c)
        assert r == pow(a, b, m)
        label = f"10^{len(str(b)) - 1}"
        print(f"{label:>14} | {b.bit_length():>5} | {c[0]:>12} | {_time(binary_modexp, a, b, m, 500):>12.8f} | {r}")
    print("-" * 72)


def read_int(prompt):
    try:
        return int(input(prompt).strip())
    except ValueError:
        print("Please enter an integer.")
        return None


def menu():
    while True:
        print("\n===== Topic 6: Modular Exponentiation =====")
        print("1. Compute a^b mod m (naive, binary, bitwise)")
        print("2. Compare outputs on many test cases")
        print("3. Input-validation demo")
        print("4. Timing / step-count table")
        print("5. Exit")
        ch = input("Choice: ").strip()
        if ch == "1":
            a, b, m = read_int("a: "), read_int("b: "), read_int("m: ")
            if None in (a, b, m):
                continue
            try:
                c1, c2, c3 = [0], [0], [0]
                r1 = naive_modexp(a, b, m, c1)
                r2 = binary_modexp(a, b, m, c2)
                r3 = binary_modexp_bitwise(a, b, m, c3)
            except (ValueError, TypeError) as e:
                print("Error:", e)
                continue
            print(f"Naive   : {r1}   ({c1[0]} multiplications)")
            print(f"Binary  : {r2}   ({c2[0]} multiplications)")
            print(f"Bitwise : {r3}   ({c3[0]} multiplications)")
            print("All match." if r1 == r2 == r3 else "MISMATCH!")
        elif ch == "2":
            compare_outputs()
        elif ch == "3":
            validation_demo()
        elif ch == "4":
            timing_table()
        elif ch == "5":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("=== Part 3: comparison of naive / binary / bitwise ===")
        compare_outputs()
        print("\n=== Part 5: input validation ===")
        validation_demo()
        print("\n=== Part 6: timing ===")
        timing_table()
    else:
        menu()