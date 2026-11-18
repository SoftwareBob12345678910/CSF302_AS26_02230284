"""Tests for sorts.py (Assignment 3, Part B).

Run with either:
    python -m unittest -v test_sorts
    pytest -v test_sorts.py
"""

import ast
import contextlib
import faulthandler
import math
import os
import random
import sys
import unittest

import sorts
from sorts import quicksort_basic, quicksort_improved

# Every variant that must sort correctly. (name, function)
VARIANTS = [
    ("basic", quicksort_basic),
    ("improved", quicksort_improved),                                     # cutoff 16, Hoare
    ("improved_cutoff1", lambda a: quicksort_improved(a, cutoff=1)),      # no insertion sort at all
    ("improved_cutoff2", lambda a: quicksort_improved(a, cutoff=2)),
    ("improved_cutoff5", lambda a: quicksort_improved(a, cutoff=5)),
    ("improved_cutoff64", lambda a: quicksort_improved(a, cutoff=64)),
    ("improved_lomuto", lambda a: quicksort_improved(a, partition="lomuto")),
    ("improved_lomuto_cutoff1", lambda a: quicksort_improved(a, 1, "lomuto")),
]
# Variants that are safe on the large "bad" inputs (sorted, all equal, many duplicates).
ROBUST = [v for v in VARIANTS if v[0] not in ("basic", "improved_lomuto", "improved_lomuto_cutoff1")]


def current_depth():
    """Number of Python frames on the stack right now."""
    frame, depth = sys._getframe(), 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


@contextlib.contextmanager
def recursion_budget(frames):
    """Allow only `frames` more stack frames than the caller is using right now."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(current_depth() + frames)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


WATCHDOG_SECONDS = 30      # a sort that loops forever must fail the run, not freeze it


class SortTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(2026)              # the improved sort draws random pivots
        faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)

    def tearDown(self):
        faulthandler.cancel_dump_traceback_later()

    def assert_same(self, actual, expected, label):
        """assertEqual for lists, but a failure on a huge list reports one line, not a huge diff."""
        if actual != expected:
            if len(expected) <= 20:
                self.fail(f"{label}: got {actual}, expected {expected}")
            i = next((k for k, (x, y) in enumerate(zip(actual, expected)) if x != y),
                     min(len(actual), len(expected)))
            self.fail(f"{label}: result differs from sorted() (first difference at index {i} "
                      f"of {len(expected)})")

    def check(self, data, funcs=VARIANTS):
        """Every function must sort a copy of `data` in place, exactly like sorted()."""
        expected = sorted(data)
        for name, f in funcs:
            a = list(data)
            same_object = a
            result = f(a)
            self.assertIs(a, same_object, name)
            self.assertIsNone(result, f"{name} must sort in place and return None")
            self.assert_same(a, expected, f"{name} on {data if len(data) <= 20 else str(len(data)) + ' items'}")


# --------------------------------------------------------------------------
# Random lists against sorted()   (at least 500 lists per function)
# --------------------------------------------------------------------------
class TestRandomAgainstSorted(SortTestCase):
    def test_600_random_lists_per_function(self):
        rng = random.Random(1)
        lists = 0
        for t in range(600):
            n = rng.randint(0, 60)
            kind = t % 4
            if kind == 0:
                data = [rng.randint(-1000, 1000) for _ in range(n)]
            elif kind == 1:
                data = [rng.randint(0, 9) for _ in range(n)]               # many duplicates
            elif kind == 2:
                data = [rng.randint(-10**9, 10**9) for _ in range(n)]
            else:
                data = [rng.choice([100, 500]) for _ in range(n)]          # Nu 100 / Nu 500 top-ups
            self.check(data)
            lists += 1
        self.assertGreaterEqual(lists, 500)

    def test_every_size_up_to_100(self):
        rng = random.Random(2)
        for n in range(0, 101):
            for _ in range(5):
                self.check([rng.randint(-50, 50) for _ in range(n)])

    def test_larger_random_lists(self):
        rng = random.Random(3)
        for n in (1000, 5000, 20000):
            self.check([rng.randint(-10**6, 10**6) for _ in range(n)], funcs=ROBUST + [VARIANTS[0]])

    def test_all_permutations_of_up_to_6_elements(self):
        from itertools import permutations
        for n in range(0, 7):
            for perm in permutations(range(n)):
                self.check(list(perm))
        for perm in set(permutations([0, 0, 1, 1, 2])):                   # with repeated values
            self.check(list(perm))


# --------------------------------------------------------------------------
# Required edge cases
# --------------------------------------------------------------------------
class TestEdgeCases(SortTestCase):
    def test_empty_list(self):
        self.check([])

    def test_one_element(self):
        self.check([42])
        self.check([-7])

    def test_two_elements(self):
        self.check([1, 2])
        self.check([2, 1])
        self.check([5, 5])

    def test_all_equal(self):
        self.check([7] * 2)
        self.check([7] * 3)
        self.check([7] * 100)
        self.check([0] * 500)

    def test_already_sorted(self):
        self.check(list(range(0, 300)))
        self.check([x * x for x in range(200)])

    def test_reverse_sorted(self):
        self.check(list(range(300, 0, -1)))

    def test_negative_numbers(self):
        self.check([-5, -1, -3, -2, -4])
        self.check([-1, -1, -2, -2, -3])
        self.check([3, -3, 0, -1, 1, -2, 2])
        self.check([-10**9, 10**9, 0, -1, 1])

    def test_sorted_with_few_values_out_of_place(self):
        data = list(range(200))
        data[0], data[-1] = data[-1], data[0]
        data[100], data[101] = data[101], data[100]
        self.check(data)

    def test_saw_tooth_and_organ_pipe(self):
        self.check([i % 10 for i in range(500)])
        self.check(list(range(100)) + list(range(100, 0, -1)))
        self.check(list(range(100, 0, -1)) + list(range(100)))

    def test_list_is_modified_in_place_not_replaced(self):
        for name, f in VARIANTS:
            a = [3, 1, 2]
            alias = a
            f(a)
            self.assertEqual(alias, [1, 2, 3], name)

    def test_works_on_sublist_views_of_larger_data(self):
        # sorting must not touch anything outside the list it is given
        base = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        for name, f in VARIANTS:
            a = base[2:7]
            f(a)
            self.assertEqual(a, sorted(base[2:7]), name)
            self.assertEqual(base, [9, 8, 7, 6, 5, 4, 3, 2, 1], name)

    def test_invalid_partition_name(self):
        with self.assertRaises(ValueError):
            quicksort_improved([3, 2, 1], partition="bogus")

    def test_cutoff_larger_than_list_is_pure_insertion_sort(self):
        rng = random.Random(4)
        data = [rng.randint(0, 99) for _ in range(40)]
        a = data[:]
        quicksort_improved(a, cutoff=1000)
        self.assertEqual(a, sorted(data))

    def test_cutoff_below_one_is_treated_as_one(self):
        for cutoff in (0, -5):
            a = [3, 1, 2, 3, 1]
            quicksort_improved(a, cutoff=cutoff)
            self.assertEqual(a, [1, 1, 2, 3, 3])


# --------------------------------------------------------------------------
# The real-data problem: sorted input and many repeated amounts
# --------------------------------------------------------------------------
class TestBadInputsForBasic(SortTestCase):
    """The textbook version is correct but crashes on the bank's real data."""

    def test_basic_raises_recursion_error_on_large_sorted_input(self):
        with self.assertRaises(RecursionError):
            quicksort_basic(list(range(5000)))

    def test_basic_raises_recursion_error_on_reverse_sorted_input(self):
        with self.assertRaises(RecursionError):
            quicksort_basic(list(range(5000, 0, -1)))

    def test_basic_raises_recursion_error_on_many_duplicates(self):
        rng = random.Random(5)
        with self.assertRaises(RecursionError):
            quicksort_basic([rng.randint(0, 9) for _ in range(30000)])

    def test_basic_is_fine_on_large_random_input(self):
        rng = random.Random(6)
        data = [rng.randint(0, 10**9) for _ in range(20000)]
        a = data[:]
        quicksort_basic(a)
        self.assert_same(a, sorted(data), "basic on random input")


class TestImprovedSurvivesBadInputs(SortTestCase):
    N = 100_000

    def test_recursion_depth_stays_logarithmic(self):
        """Only ~log2(n) + a few frames may be used, far below the default limit of 1000."""
        budget = math.ceil(math.log2(self.N)) + 8
        rng = random.Random(7)
        cases = {
            "sorted": list(range(self.N)),
            "reverse": list(range(self.N, 0, -1)),
            "all equal": [500] * self.N,
            "duplicates 0..9": [rng.randint(0, 9) for _ in range(self.N)],
            "random": [rng.randint(0, 10**9) for _ in range(self.N)],
        }
        for label, data in cases.items():
            expected = sorted(data)
            for name, f in ROBUST:
                a = data[:]
                with recursion_budget(budget + 8):
                    f(a)
                self.assert_same(a, expected, f"{name} on {label}")

    def test_lomuto_version_depth_is_also_logarithmic_on_sorted_input(self):
        n = 50_000
        a = list(range(n))
        with recursion_budget(math.ceil(math.log2(n)) + 16):
            quicksort_improved(a, partition="lomuto")
        self.assert_same(a, list(range(n)), "lomuto on sorted input")

    def test_improved_on_nu_100_and_500_topups(self):
        rng = random.Random(8)
        data = [rng.choice([100, 500, 100, 500, 1000, 250]) for _ in range(50_000)]
        a = data[:]
        quicksort_improved(a)
        self.assert_same(a, sorted(data), "improved on top-up amounts")


# --------------------------------------------------------------------------
# The recurrence T(n) = T(n-1) + n, checked by counting comparisons
# --------------------------------------------------------------------------
class Counted:
    """Wraps a value and counts every `<` and `>` made on it."""
    count = 0
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        Counted.count += 1
        return self.v < other.v

    def __gt__(self, other):
        Counted.count += 1
        return self.v > other.v


class TestComparisonCounts(SortTestCase):
    def count(self, f, data):
        items = [Counted(x) for x in data]
        Counted.count = 0
        f(items)
        self.assert_same([c.v for c in items], sorted(data), "counted sort")
        return Counted.count

    def test_basic_on_sorted_input_makes_exactly_n_choose_2_comparisons(self):
        for n in (1, 2, 10, 100, 400):
            self.assertEqual(self.count(quicksort_basic, list(range(n))), n * (n - 1) // 2, n)

    def test_basic_on_reverse_sorted_input_is_also_quadratic(self):
        n = 300
        self.assertGreaterEqual(self.count(quicksort_basic, list(range(n, 0, -1))), n * (n - 1) // 2)

    def test_basic_on_all_equal_input_is_quadratic(self):
        n = 300
        self.assertEqual(self.count(quicksort_basic, [5] * n), n * (n - 1) // 2)

    def test_improved_on_sorted_input_is_close_to_n_log_n(self):
        n = 4000
        c = self.count(quicksort_improved, list(range(n)))
        self.assertLess(c, 3 * n * math.log2(n))        # n^2/2 would be 8,000,000

    def test_improved_on_duplicates_is_close_to_n_log_n(self):
        rng = random.Random(9)
        n = 4000
        c = self.count(quicksort_improved, [rng.randint(0, 9) for _ in range(n)])
        self.assertLess(c, 3 * n * math.log2(n))

    def test_lomuto_random_pivot_is_quadratic_on_all_equal_input(self):
        n = 1000
        c = self.count(lambda a: quicksort_improved(a, cutoff=1, partition="lomuto"), [5] * n)
        self.assertGreater(c, n * n // 4)


# --------------------------------------------------------------------------
# The rules of the assignment
# --------------------------------------------------------------------------
class TestRules(unittest.TestCase):
    def test_sorts_py_does_not_call_sorted_or_list_sort(self):
        path = os.path.join(os.path.dirname(os.path.abspath(sorts.__file__)), "sorts.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    self.assertNotEqual(fn.id, "sorted")
                if isinstance(fn, ast.Attribute):
                    self.assertNotIn(fn.attr, ("sort", "sorted"))

    def test_public_names(self):
        self.assertEqual(sorted(sorts.__all__), ["quicksort_basic", "quicksort_improved"])
        self.assertEqual(quicksort_improved.__defaults__[0], 16)       # default cutoff is 16


if __name__ == "__main__":
    unittest.main(verbosity=2)