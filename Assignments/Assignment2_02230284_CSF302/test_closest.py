"""Tests for closest.py (Assignment 2, Part B)."""

import copy
import math
import random
import unittest

from closest import closest_pair, closest_pair_brute


def distance(points, i, j):
    dx = points[i][0] - points[j][0]
    dy = points[i][1] - points[j][1]
    return math.sqrt(dx * dx + dy * dy)


class ClosestPairTestCase(unittest.TestCase):
    """Shared helper: compare closest_pair with closest_pair_brute."""

    def check(self, pts):
        """closest_pair must agree with brute force and return a valid pair."""
        before = copy.deepcopy(pts)
        d_brute, _, _ = closest_pair_brute(pts)
        d, i, j = closest_pair(pts)
        msg = f"points={pts if len(pts) <= 12 else str(len(pts)) + ' points'}"

        self.assertAlmostEqual(d, d_brute, places=9, msg=msg)
        self.assertTrue(0 <= i < j < len(pts), msg=f"bad indices {(i, j)}; {msg}")
        # the reported indices really do realise the reported distance
        self.assertAlmostEqual(distance(pts, i, j), d, places=9, msg=msg)
        self.assertEqual(pts, before, "input list must not be modified")
        return d, i, j


# --------------------------------------------------------------------------
# Random comparison against brute force (600 inputs, 2 to 50 points each)
# --------------------------------------------------------------------------
class TestRandomAgainstBrute(ClosestPairTestCase):
    def test_600_random_inputs(self):
        rng = random.Random(2026)
        trials = 0
        for t in range(600):
            n = rng.randint(2, 50)
            kind = t % 3
            if kind == 0:      # tiny integer grid: many duplicates, ties, shared x
                pts = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(n)]
            elif kind == 1:    # wide integer range incl. negatives
                pts = [(rng.randint(-1000, 1000), rng.randint(-1000, 1000)) for _ in range(n)]
            else:              # floating-point coordinates
                pts = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(n)]
            self.check(pts)
            trials += 1
        self.assertGreaterEqual(trials, 500)

    def test_every_size_from_2_to_50(self):
        rng = random.Random(7)
        for n in range(2, 51):
            for _ in range(10):
                self.check([(rng.randint(0, 40), rng.randint(0, 40)) for _ in range(n)])

    def test_larger_input(self):
        rng = random.Random(99)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(2000)]
        self.check(pts)


# --------------------------------------------------------------------------
# Required edge cases
# --------------------------------------------------------------------------
class TestExactlyTwoPoints(ClosestPairTestCase):
    def test_two_points(self):
        d, i, j = closest_pair([(0, 0), (3, 4)])
        self.assertEqual((d, i, j), (5.0, 0, 1))

    def test_two_points_brute(self):
        self.assertEqual(closest_pair_brute([(0, 0), (3, 4)]), (5.0, 0, 1))

    def test_two_points_unsorted_by_x(self):
        # the first point has the larger x, so sorting reorders them
        d, i, j = closest_pair([(10, 10), (7, 6)])
        self.assertEqual((d, i, j), (5.0, 0, 1))

    def test_too_few_points(self):
        for f in (closest_pair, closest_pair_brute):
            with self.assertRaises(ValueError):
                f([])
            with self.assertRaises(ValueError):
                f([(1, 1)])


class TestDuplicatePoints(ClosestPairTestCase):
    def test_two_identical_points(self):
        self.assertEqual(closest_pair([(1, 1), (1, 1)]), (0.0, 0, 1))

    def test_duplicate_among_others(self):
        pts = [(5, 5), (1, 2), (9, 9), (1, 2), (0, 7)]
        self.assertEqual(closest_pair(pts), (0.0, 1, 3))

    def test_three_copies_of_one_point(self):
        d, i, j = self.check([(4, 4), (4, 4), (4, 4)])
        self.assertEqual(d, 0.0)

    def test_duplicates_split_by_the_dividing_line(self):
        # x-sorted: (0,0) (5,5) | (5,5) (9,9): the two (5,5) land in different halves
        d, i, j = self.check([(0, 0), (5, 5), (5, 5), (9, 9)])
        self.assertEqual((d, i, j), (0.0, 1, 2))

    def test_all_points_identical(self):
        d, i, j = self.check([(2.5, -1.5)] * 17)
        self.assertEqual(d, 0.0)

    def test_random_inputs_with_planted_duplicate(self):
        rng = random.Random(11)
        for _ in range(100):
            n = rng.randint(3, 50)
            pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n - 1)]
            pts.insert(rng.randrange(n), rng.choice(pts))
            d, i, j = self.check(pts)
            self.assertEqual(d, 0.0)


class TestVerticalLine(ClosestPairTestCase):
    def test_small_vertical_line(self):
        pts = [(7, 0), (7, 10), (7, 3), (7, 25)]
        d, i, j = self.check(pts)
        self.assertEqual((d, i, j), (3.0, 0, 2))

    def test_random_vertical_lines(self):
        rng = random.Random(5)
        for _ in range(100):
            n = rng.randint(2, 50)
            self.check([(7, rng.randint(-100, 100)) for _ in range(n)])

    def test_long_vertical_line(self):
        ys = [i * i for i in range(1, 801)]
        random.Random(1).shuffle(ys)
        d, _, _ = self.check([(3.0, float(y)) for y in ys])
        self.assertEqual(d, 3.0)           # gap between 1 and 4

    def test_horizontal_line(self):
        rng = random.Random(6)
        for _ in range(50):
            n = rng.randint(2, 50)
            self.check([(rng.randint(-100, 100), -2) for _ in range(n)])


class TestClosestPairStraddlesDividingLine(ClosestPairTestCase):
    def test_fixed_example(self):
        # 8 points; sorted by x the split is between x=30 and x=31.
        pts = [(0, 0), (10, 20), (20, 0), (30, 20),
               (31, 21),
               (40, 0), (50, 20), (60, 0)]
        left, right = pts[:4], pts[4:]
        # make sure the example really tests what it claims to:
        self.assertGreater(closest_pair_brute(left)[0], math.sqrt(2))
        self.assertGreater(closest_pair_brute(right)[0], math.sqrt(2))
        d, i, j = self.check(pts)
        self.assertAlmostEqual(d, math.sqrt(2))
        self.assertEqual((i, j), (3, 4))

    def test_fixed_example_shuffled_input_order(self):
        pts = [(0, 0), (10, 20), (20, 0), (30, 20), (31, 21), (40, 0), (50, 20), (60, 0)]
        random.Random(3).shuffle(pts)
        d, i, j = self.check(pts)
        self.assertAlmostEqual(d, math.sqrt(2))
        self.assertEqual({pts[i], pts[j]}, {(30, 20), (31, 21)})

    def test_random_straddling_instances(self):
        """Build lattices (spacing >= 100) on each side of x=500, then add one
        very close pair, one point on each side of the line. That pair is the
        unique closest pair and straddles the dividing line."""
        rng = random.Random(42)
        for _ in range(200):
            cells = [(a, b) for a in range(5) for b in range(6)]
            k = rng.randint(2, 20)
            left = [(100 * a, 100 * b) for a, b in rng.sample(cells, k)]
            right = [(600 + 100 * a, 100 * b) for a, b in rng.sample(cells, k)]
            y = rng.uniform(0, 500)
            # a very close pair on opposite sides of x = 500 (distance sqrt(5))
            p, q = (499.0, y), (501.0, y + 1.0)
            left.append(p)
            right.append(q)
            pts = left + right
            rng.shuffle(pts)

            # genuinely straddling: each side on its own is worse than the answer
            side_l = [pt for pt in pts if pt[0] < 500]
            side_r = [pt for pt in pts if pt[0] > 500]
            expected = math.hypot(2.0, 1.0)
            self.assertGreater(closest_pair_brute(side_l)[0], expected)
            self.assertGreater(closest_pair_brute(side_r)[0], expected)

            d, i, j = self.check(pts)
            self.assertAlmostEqual(d, expected)
            self.assertEqual({pts[i], pts[j]}, {p, q})

    def test_points_with_same_x_as_dividing_line(self):
        # several points exactly on the dividing line, on both sides of the split
        pts = [(5, 0), (5, 4), (5, 8), (5, 9), (5, 20), (1, 1), (9, 9), (5, 12)]
        self.check(pts)
        rng = random.Random(8)
        for _ in range(100):
            n = rng.randint(4, 40)
            self.check([(rng.choice([4, 5, 5, 5, 6]), rng.randint(0, 30)) for _ in range(n)])


class TestSpecificationExample(ClosestPairTestCase):
    def test_example_from_the_assignment(self):
        pts = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
        for f in (closest_pair_brute, closest_pair):
            d, i, j = f(pts)
            self.assertAlmostEqual(d, math.sqrt(2))
            self.assertEqual((i, j), (0, 5))

    def test_accepts_lists_as_points(self):
        d, i, j = closest_pair([[0, 0], [1, 1], [10, 10]])
        self.assertAlmostEqual(d, math.sqrt(2))
        self.assertEqual((i, j), (0, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)