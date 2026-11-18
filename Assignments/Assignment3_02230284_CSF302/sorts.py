"""Sorting the bank's nightly transactions (Assignment 3, CSF302).

Two in-place quicksorts for a list of integers, ascending order:

    quicksort_basic(a)                  textbook version: Lomuto partition, last element as pivot
    quicksort_improved(a, cutoff=16)    random pivot + insertion sort for small parts +
                                        recursion only on the smaller part (depth O(log n))

Neither function uses ``sorted()`` or ``list.sort()``; both return ``None`` (like
``list.sort``) and rearrange the list they are given.

Why the basic version fails on real data
----------------------------------------
Considering the final element as the pivot, any pre-sorted array will always be
partitioned into an empty sub-array and another one of size n - 1, hence T(n)
= T(n - 1) + Theta(n) = Theta(n^2) , and n levels of recursion are too much for
Python’s maximum recursion depth limit (RecursionError). This applies to arrays that
have lots of duplicate elements, since using the Lomuto method all identical elements
are grouped to one side of the pivot.

What the improved version changes
---------------------------------
1. Random pivot: There is no fixed input (e.g., "already sorted") that makes splits bad.
2. Hoare partition: Both loops end when values equal to the pivot are encountered and swapped,
   thus splitting the block of identical elements into two equal halves rather than
   leaving it all to one side.  Thus, lists with lots of duplicate values stay at O(n log n).
   (The brief specifies random pivot only; "partition=lomuto" is the literal brief requirement,
   but retained in the report to explain why the additional step is necessary.)
3. Recursive call on the smaller sublist and a loop on the larger sublist: The smaller sublist has at
   most half of the elements, therefore, the number of recursive calls is not more than log2(n).
4. Sublists with ``cutoff`` elements or less are processed by insertion sort, which has less overhead
   than quicksort on small lists.
"""

import random

__all__ = ["quicksort_basic", "quicksort_improved"]

_random = random.random      # seeded by random.seed(); much cheaper to call than randint


def _random_index(lo, hi):
    """A uniformly random index in lo..hi (inclusive)."""
    return lo + int(_random() * (hi - lo + 1))


# --------------------------------------------------------------------------
# Part A.1  Basic quicksort (Lomuto partition, last element as pivot)
# --------------------------------------------------------------------------
def quicksort_basic(a):
    """Sort list ``a`` in place. Last element as pivot, Lomuto partition.

    Expected Theta(n log n) on random data, but Theta(n^2) time and Theta(n)
    recursion depth on sorted data or data with many equal values.
    """
    _quicksort_basic(a, 0, len(a) - 1)


def _quicksort_basic(a, lo, hi):
    if lo < hi:
        p = _lomuto_partition(a, lo, hi)
        _quicksort_basic(a, lo, p - 1)
        _quicksort_basic(a, p + 1, hi)


def _lomuto_partition(a, lo, hi):
    """Partition a[lo..hi] around a[hi]; return the pivot's final index.

    Afterwards a[lo..p-1] < pivot <= a[p+1..hi].  Makes exactly hi - lo comparisons.
    """
    pivot = a[hi]
    i = lo                                   # a[lo..i-1] holds the values < pivot
    for j in range(lo, hi):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]                # put the pivot between the two parts
    return i


# --------------------------------------------------------------------------
# Part A.2  Improved quicksort
# --------------------------------------------------------------------------
def quicksort_improved(a, cutoff=16, partition="hoare"):
    """Sort list ``a`` in place.

    cutoff     parts of at most this many elements are sorted by insertion sort
               (values below 1 are treated as 1).
    partition  "hoare" (default): Hoare partition around a random pivot; copes with
               many equal values.
               "lomuto": Lomuto partition around a random pivot, i.e. exactly what
               the brief describes; quadratic on lists with many equal values.
    """
    if partition == "hoare":
        split = _split_hoare
    elif partition == "lomuto":
        split = _split_lomuto
    else:
        raise ValueError("partition must be 'hoare' or 'lomuto'")
    _quicksort_improved(a, 0, len(a) - 1, max(cutoff, 1), split)


def _quicksort_improved(a, lo, hi, cutoff, split):
    while hi - lo + 1 > cutoff:
        left_hi, right_lo = split(a, lo, hi)     # parts are lo..left_hi and right_lo..hi
        if left_hi - lo < hi - right_lo:         # left part is smaller: recurse on it ...
            _quicksort_improved(a, lo, left_hi, cutoff, split)
            lo = right_lo                        # ... and loop on the larger right part
        else:                                    # right part is smaller: recurse on it ...
            _quicksort_improved(a, right_lo, hi, cutoff, split)
            hi = left_hi                         # ... and loop on the larger left part
    _insertion_sort(a, lo, hi)                   # at most `cutoff` elements remain


def _split_hoare(a, lo, hi):
    """Hoare partition of a[lo..hi] (at least 2 elements) around a random pivot value.

    Returns (j, j + 1) with lo <= j < hi, such that every value in a[lo..j] is <= every
    value in a[j+1..hi].  Both parts are non-empty, so each step makes progress.
    The pivot is taken from lo..hi-1 (never the last element); that is what guarantees
    j < hi.  Both scans stop on values equal to the pivot, which splits runs of equal
    values evenly between the two parts.
    """
    pivot = a[_random_index(lo, hi - 1)]
    i = lo - 1
    j = hi + 1
    while True:
        i += 1
        while a[i] < pivot:                  # stops at a value >= pivot
            i += 1
        j -= 1
        while a[j] > pivot:                  # stops at a value <= pivot
            j -= 1
        if i >= j:
            return j, j + 1
        a[i], a[j] = a[j], a[i]


def _split_lomuto(a, lo, hi):
    """Lomuto partition around a random pivot; returns (p - 1, p + 1) for pivot index p."""
    r = _random_index(lo, hi)
    a[r], a[hi] = a[hi], a[r]                # move the random pivot to the end
    p = _lomuto_partition(a, lo, hi)
    return p - 1, p + 1


def _insertion_sort(a, lo, hi):
    """Insertion sort of a[lo..hi] (inclusive); does nothing if the range has < 2 elements."""
    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while j >= lo and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


if __name__ == "__main__":
    demo = [5, 100, 3, 500, 100, 7, -2, 500, 0]
    b = demo[:]
    quicksort_basic(b)
    c = demo[:]
    quicksort_improved(c)
    print(b)
    print(c)