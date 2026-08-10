"""
CSF302_AS26_02230284
Lab1_AS26 - Question 2
Bubble Sort vs Merge Sort - Performance Comparison
"""

import random
import time


def generate_array_284(size):
    """Generate a list of `size` random integers."""
    return [random.randint(1, size * 10) for _ in range(size)]


def bubble_sort_284(arr):
    """Standard O(n^2) bubble sort. Sorts a COPY, leaves original untouched."""
    a_284 = arr.copy()
    n_284 = len(a_284)
    for i in range(n_284 - 1):
        swapped_284 = False
        for j in range(n_284 - 1 - i):
            if a_284[j] > a_284[j + 1]:
                a_284[j], a_284[j + 1] = a_284[j + 1], a_284[j]
                swapped_284 = True
        if not swapped_284:  # already sorted, stop early
            break
    return a_284


def merge_sort_284(arr):
    """Standard O(n log n) merge sort. Sorts a COPY, leaves original untouched."""
    a_284 = arr.copy()
    if len(a_284) <= 1:
        return a_284

    mid_284 = len(a_284) // 2
    left_284 = merge_sort_284(a_284[:mid_284])
    right_284 = merge_sort_284(a_284[mid_284:])

    return merge_284(left_284, right_284)


def merge_284(left_284, right_284):
    """Merge two sorted lists into one sorted list."""
    result_284 = []
    i_284 = j_284 = 0

    while i_284 < len(left_284) and j_284 < len(right_284):
        if left_284[i_284] <= right_284[j_284]:
            result_284.append(left_284[i_284])
            i_284 += 1
        else:
            result_284.append(right_284[j_284])
            j_284 += 1

    result_284.extend(left_284[i_284:])
    result_284.extend(right_284[j_284:])
    return result_284


def measure_time_284(sort_func, arr, trials=3):
    """Run `sort_func` `trials` times on the SAME original array and return the average time."""
    total_time_284 = 0.0
    for _ in range(trials):
        start_284 = time.perf_counter()
        sort_func(arr)
        end_284 = time.perf_counter()
        total_time_284 += (end_284 - start_284)
    return total_time_284 / trials


def run_benchmark_284():
    input_sizes_284 = [100, 500, 1000, 2000, 4000, 8000]

    print(f"{'Size':<10}{'Bubble Sort (s)':<20}{'Merge Sort (s)':<20}")
    print("-" * 50)

    results_284 = []

    for size_284 in input_sizes_284:
        # Step 1: generate a random array for this size
        array_284 = generate_array_284(size_284)

        # Step 2: measure Bubble Sort (each trial sorts a fresh copy internally)
        bubble_time_284 = measure_time_284(bubble_sort_284, array_284)

        # Step 3: measure Merge Sort (each trial sorts a fresh copy internally)
        merge_time_284 = measure_time_284(merge_sort_284, array_284)

        results_284.append((size_284, bubble_time_284, merge_time_284))
        print(f"{size_284:<10}{bubble_time_284:<20.6f}{merge_time_284:<20.6f}")

    return results_284


if __name__ == "__main__":
    run_benchmark_284()