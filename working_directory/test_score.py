import unittest

from score import classify_score


class TestClassifyScore(unittest.TestCase):
    def test_score_levels(self):
        self.assertEqual(classify_score(95), "A")
        self.assertEqual(classify_score(85), "B")
        self.assertEqual(classify_score(70), "C")
        self.assertEqual(classify_score(50), "D")

    def test_boundary_scores(self):
        self.assertEqual(classify_score(0), "D")
        self.assertEqual(classify_score(100), "A")

    def test_out_of_range_scores_raise_value_error(self):
        for score in (-1, 101):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    classify_score(score)

    def test_non_numeric_score_raises_type_error(self):
        with self.assertRaises(TypeError):
            classify_score("95")


if __name__ == "__main__":
    unittest.main()
