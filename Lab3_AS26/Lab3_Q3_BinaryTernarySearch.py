"""
CSF302 - Algorithm Analysis and Design
Lab 3, Question 3: Binary Search vs Ternary Search (Menu-Driven Program)
Objective:
    Menu-driven program to search a sorted array of n integers using
    Binary Search and Ternary Search, and analyze step/frequency counts
    (number of comparisons) for best case, worst case, and increasing n.
"""

import random

# ---------------------------------------------------------------------------
# 1. Array generation / display
# ---------------------------------------------------------------------------
def generate_sorted_array_284(n, low=1, high=1000):
    arr = sorted(random.sample(range(low, high + 1), min(n, high - low + 1)))
    return arr


def display_array_284(arr):
    print(f"\nArray (n={len(arr)}):")
    print(arr)


# ---------------------------------------------------------------------------
# 2. Binary Search  ->  T(n) = T(n/2) + O(1)
# ---------------------------------------------------------------------------
def binary_search_284(arr, key):
    """Returns (index_or_-1, comparison_count)."""
    low, high = 0, len(arr) - 1
    comparisons = 0

    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        if arr[mid] == key:
            return mid, comparisons
        comparisons += 1
        if arr[mid] < key:
            low = mid + 1
        else:
            high = mid - 1

    return -1, comparisons


# ---------------------------------------------------------------------------
# 3. Ternary Search  ->  T(n) = T(n/3) + O(1)
# ---------------------------------------------------------------------------
def ternary_search_284(arr, key):
    """Returns (index_or_-1, comparison_count)."""
    low, high = 0, len(arr) - 1
    comparisons = 0

    while low <= high:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3

        comparisons += 1
        if arr[mid1] == key:
            return mid1, comparisons
        comparisons += 1
        if arr[mid2] == key:
            return mid2, comparisons

        comparisons += 1
        if key < arr[mid1]:
            high = mid1 - 1
        else:
            comparisons += 1
            if key > arr[mid2]:
                low = mid2 + 1
            else:
                low, high = mid1 + 1, mid2 - 1

    return -1, comparisons


# ---------------------------------------------------------------------------
# 4. Step/frequency count experiments
# ---------------------------------------------------------------------------
def best_case_284(arr):
    """Best case: key present near the first place each algorithm checks."""
    if not arr:
        print("Array is empty.")
        return
    mid = len(arr) // 2
    key = arr[mid]  # binary search's first probe -> best case for binary search
    _, b_count = binary_search_284(arr, key)
    _, t_count = ternary_search_284(arr, key)
    print(f"\nBest case (key = {key}, present at middle index {mid}):")
    print(f"  Binary Search comparisons  : {b_count}")
    print(f"  Ternary Search comparisons : {t_count}")


def worst_case_284(arr):
    """Worst case: key absent from the array (forces full search depth)."""
    if not arr:
        print("Array is empty.")
        return
    key = arr[-1] + 1  # guaranteed absent, larger than every element
    _, b_count = binary_search_284(arr, key)
    _, t_count = ternary_search_284(arr, key)
    print(f"\nWorst case (key = {key}, absent from array):")
    print(f"  Binary Search comparisons  : {b_count}")
    print(f"  Ternary Search comparisons : {t_count}")


def comparison_table_284(sizes=(10, 100, 1000, 10000, 100000)):
    print("\n" + "=" * 60)
    print("STEP/FREQUENCY COUNT COMPARISON ACROSS INCREASING n")
    print("=" * 60)
    print(f"{'n':>8} | {'Binary (worst)':>15} | {'Ternary (worst)':>16}")
    print("-" * 45)
    for n in sizes:
        arr = generate_sorted_array_284(n, low=1, high=n * 10)
        key = arr[-1] + 1  # absent -> worst case for both
        _, b_count = binary_search_284(arr, key)
        _, t_count = ternary_search_284(arr, key)
        print(f"{n:>8} | {b_count:>15} | {t_count:>16}")

    # -----------------------------------------------------------------
    # ANALYSIS / CONCLUSION:
    # Binary Search follows T(n) = T(n/2) + O(1), giving O(log2 n) time,
    # and makes 1-2 comparisons per level to decide which half to keep.
    # Ternary Search follows T(n) = T(n/3) + O(1), giving O(log3 n) time
    # in terms of number of LEVELS (fewer levels than binary search),
    # but it must make roughly 2-4 comparisons per level to check both
    # split points (arr[mid1] and arr[mid2]) and decide which third to
    # keep. Mathematically, log3(n) < log2(n), but each ternary "step"
    # costs more comparisons than each binary "step". When the constants
    # are accounted for, Binary Search ends up making FEWER TOTAL
    # comparisons than Ternary Search for the same n in practice, which
    # is exactly what the table above demonstrates. This is why binary
    # search, not ternary search, is the standard choice for searching
    # sorted arrays.
    # -----------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Menu-driven driver
# ---------------------------------------------------------------------------
def print_menu_284():
    print("\n" + "=" * 50)
    print("  BINARY SEARCH vs TERNARY SEARCH - MENU")
    print("=" * 50)
    print("1. Generate n sorted random numbers -> Array")
    print("2. Display Array")
    print("3. Search for a key using Binary Search")
    print("4. Search for a key using Ternary Search")
    print("5. Step/frequency count for BEST case")
    print("6. Step/frequency count for WORST case")
    print("7. Step/frequency count comparison table across increasing n")
    print("8. Exit")


def main_menu_284():
    arr = []
    while True:
        print_menu_284()
        choice = input("Enter choice (1-8): ").strip()

        if choice == "1":
            n = int(input("Enter n (number of elements): "))
            arr = generate_sorted_array_284(n)
            print("Array generated.")

        elif choice == "2":
            display_array_284(arr)

        elif choice == "3":
            if not arr:
                print("Generate the array first (option 1).")
                continue
            key = int(input("Enter key to search: "))
            idx, count = binary_search_284(arr, key)
            if idx != -1:
                print(f"Found at index {idx} in {count} comparisons.")
            else:
                print(f"Not found. Used {count} comparisons.")

        elif choice == "4":
            if not arr:
                print("Generate the array first (option 1).")
                continue
            key = int(input("Enter key to search: "))
            idx, count = ternary_search_284(arr, key)
            if idx != -1:
                print(f"Found at index {idx} in {count} comparisons.")
            else:
                print(f"Not found. Used {count} comparisons.")

        elif choice == "5":
            if not arr:
                print("Generate the array first (option 1).")
                continue
            best_case_284(arr)

        elif choice == "6":
            if not arr:
                print("Generate the array first (option 1).")
                continue
            worst_case_284(arr)

        elif choice == "7":
            comparison_table_284()

        elif choice == "8":
            print("Exiting. Goodbye!")
            break

        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main_menu_284()