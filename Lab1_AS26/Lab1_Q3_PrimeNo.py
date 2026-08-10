"""
CSF302_AS26_02230284
Lab1_AS26 - Question 3
Prime Number Generation - Naive Trial Division vs Optimized Trial Division
vs Sieve of Eratosthenes - Performance Comparison
"""

import time
import math


# ----------------------------------------------------------------------
# 1. Naive Prime Checking using Trial Division  -> O(n) per number, O(n^2) total
# ----------------------------------------------------------------------
def is_prime_naive_284(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


def generate_primes_naive_284(limit):
    return [n for n in range(2, limit + 1) if is_prime_naive_284(n)]


# ----------------------------------------------------------------------
# 2. Optimized Trial Division -> only check divisors up to sqrt(n)
# ----------------------------------------------------------------------
def is_prime_optimized_284(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit_284 = int(math.isqrt(n))
    for i in range(3, limit_284 + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_primes_optimized_284(limit):
    return [n for n in range(2, limit + 1) if is_prime_optimized_284(n)]


# ----------------------------------------------------------------------
# 3. Sieve of Eratosthenes -> O(n log log n)
# ----------------------------------------------------------------------
def sieve_of_eratosthenes_284(limit):
    if limit < 2:
        return []
    is_prime_284 = [True] * (limit + 1)
    is_prime_284[0] = is_prime_284[1] = False

    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime_284[i]:
            for multiple_284 in range(i * i, limit + 1, i):
                is_prime_284[multiple_284] = False

    return [i for i, prime_284 in enumerate(is_prime_284) if prime_284]


# ----------------------------------------------------------------------
# Timing helper
# ----------------------------------------------------------------------
def measure_time_284(func, limit, trials=3):
    total_time_284 = 0.0
    for _ in range(trials):
        start_284 = time.perf_counter()
        func(limit)
        end_284 = time.perf_counter()
        total_time_284 += (end_284 - start_284)
    return total_time_284 / trials


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def run_benchmark_284():
    input_sizes_284 = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    # Naive trial division is O(n^2) and becomes impractically slow beyond
    # this point (100,000 already takes ~17s for a SINGLE run), so it is
    # skipped for larger N to keep the benchmark runnable.
    naive_max_n_284 = 100_000

    print(f"{'N':<12}{'Naive (s)':<16}{'Optimized (s)':<16}{'Sieve (s)':<16}")
    print("-" * 60)

    results_284 = []

    for n_284 in input_sizes_284:
        if n_284 <= naive_max_n_284:
            naive_time_284 = measure_time_284(generate_primes_naive_284, n_284, trials=1)
            naive_display_284 = f"{naive_time_284:.6f}"
        else:
            naive_time_284 = None
            naive_display_284 = "SKIPPED"

        optimized_time_284 = measure_time_284(generate_primes_optimized_284, n_284, trials=3)
        sieve_time_284 = measure_time_284(sieve_of_eratosthenes_284, n_284, trials=3)

        results_284.append((n_284, naive_time_284, optimized_time_284, sieve_time_284))
        print(f"{n_284:<12}{naive_display_284:<16}{optimized_time_284:<16.6f}{sieve_time_284:<16.6f}")

    print("\nNote: Naive Trial Division (O(n^2)) is skipped above N = "
          f"{naive_max_n_284:,} because it becomes impractically slow "
          "(hundreds to thousands of seconds).")

    return results_284


if __name__ == "__main__":
    run_benchmark_284()