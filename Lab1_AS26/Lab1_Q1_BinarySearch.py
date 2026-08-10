"""
CSF302_AS26_02230284
Lab1_AS26 - Question 1
Linear Search vs Binary Search - Performance Comparison
"""

import random
import time


def generate_dataset_284(size):
    """Generate a list of `size` random integers."""
    return [random.randint(1, size * 10) for _ in range(size)]


def linear_search_284(arr, target):
    """Standard O(n) linear search. Returns index of target or -1."""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


def binary_search_284(arr, target):
    """Standard O(log n) binary search on a SORTED list. Returns index or -1."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def measure_time_284(search_func, arr, target, trials=5):
    """Run `search_func` `trials` times and return the average execution time."""
    total_time_284 = 0.0
    for _ in range(trials):
        start_284 = time.perf_counter()
        search_func(arr, target)
        end_284 = time.perf_counter()
        total_time_284 += (end_284 - start_284)
    return total_time_284 / trials


def run_benchmark_284():
    input_sizes_284 = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    print(f"{'Size':<12}{'Linear Search (s)':<22}{'Binary Search (s)':<22}")
    print("-" * 56)

    results_284 = []

    for size_284 in input_sizes_284:
        # Step 1: generate random dataset of this size
        dataset_284 = generate_dataset_284(size_284)

        # Pick a target that actually exists in the dataset
        target_284 = random.choice(dataset_284)

        # Step 2: measure Linear Search on the UNSORTED dataset
        linear_time_284 = measure_time_284(linear_search_284, dataset_284, target_284)

        # Step 3: sort the dataset before Binary Search (required step)
        sorted_dataset_284 = sorted(dataset_284)

        # Step 4: measure Binary Search on the SORTED dataset
        binary_time_284 = measure_time_284(binary_search_284, sorted_dataset_284, target_284)

        results_284.append((size_284, linear_time_284, binary_time_284))
        print(f"{size_284:<12}{linear_time_284:<22.8f}{binary_time_284:<22.8f}")

    return results_284


if __name__ == "__main__":
    run_benchmark_284()