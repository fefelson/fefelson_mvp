# tests/test_metrics.py
import unittest
import numpy as np
from src.tensors.utils import compute_ece, compute_brier_score


######################################################################
######################################################################


class TestECE(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        # Default number of bins


    def test_perfect_predictions(self):
        """Test ECE with perfect predictions (4 samples, 2 classes)."""

        probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        labels = np.array([0, 1, 0, 1], dtype=int)
        ece = compute_ece(probs, labels)
        self.assertAlmostEqual(ece, 0.0, places=6, msg="ECE should be 0 for perfect predictions")


    def test_random_predictions(self):
        """Test ECE with random predictions (5 samples, 5 classes)."""

        probs = np.random.dirichlet(alpha=np.ones(5), size=5)
        labels = np.array([0, 1, 2, 3, 4], dtype=int)
        ece = compute_ece(probs, labels)
        self.assertGreater(ece, 0.0, msg="ECE should be >0 for random predictions")
        self.assertLess(ece, 1.0, msg="ECE should be <1 for reasonable inputs")


    def test_miscalibrated_predictions(self):
        """Test ECE with miscalibrated predictions (3 samples, 4 classes)."""

        probs = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05]
        ])
        labels = np.array([1, 2, 0], dtype=int)
        ece = compute_ece(probs, labels)
        self.assertAlmostEqual(ece, 0.85, places=2, msg="ECE should be ~0.85 for miscalibrated predictions")


    def test_single_sample(self):
        """Test ECE with a single sample (1 sample, 3 classes)."""

        probs = np.array([[1.0, 0.0, 0.0]])
        labels = np.array([0], dtype=int)
        ece = compute_ece(probs, labels)
        self.assertAlmostEqual(ece, 0.0, places=6, msg="ECE should be 0 for a single correct prediction")


    def test_large_array(self):
        """Test ECE with a larger array (20 samples, 10 classes)."""

        probs = np.random.dirichlet(alpha=np.ones(10), size=20)
        labels = np.random.randint(0, 10, size=20, dtype=int)
        ece = compute_ece(probs, labels)
        self.assertGreaterEqual(ece, 0.0, msg="ECE should be non-negative")
        self.assertLess(ece, 1.0, msg="ECE should be <1 for reasonable inputs")


    def test_single_class(self):
        """Test ECE with a single class (5 samples, 1 class)."""

        probs = np.ones((5, 1))
        labels = np.zeros(5, dtype=int)
        ece = compute_ece(probs, labels)
        self.assertAlmostEqual(ece, 0.0, places=6, msg="ECE should be 0 for single class with correct predictions")


    def test_invalid_probs_shape(self):
        """Test ECE with invalid probs shape (1D array)."""

        probs = np.array([0.5, 0.5])
        labels = np.array([0], dtype=int)
        with self.assertRaises(ValueError, msg="Should raise ValueError for 1D probs"):
            compute_ece(probs, labels)


    def test_mismatched_labels(self):
        """Test ECE with mismatched probs and labels shapes."""

        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([0], dtype=int)
        with self.assertRaises(ValueError, msg="Should raise ValueError for mismatched shapes"):
            compute_ece(probs, labels)


    def test_invalid_n_bins(self):
        """Test ECE with invalid n_bins (non-positive)."""

        probs = np.array([[0.5, 0.5]])
        labels = np.array([0], dtype=int)
        with self.assertRaises(AssertionError, msg="Should raise AssertionError for n_bins <= 0") as cm:
            compute_ece(probs, labels, n_bins=0)
        self.assertEqual(str(cm.exception), "n_bins must be positive")


    def test_non_normalized_probs(self):
        """Test ECE with non-normalized probabilities."""

        probs = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1], dtype=int)
        ece = compute_ece(probs, labels)
        self.assertGreaterEqual(ece, 0.0, msg="ECE should handle non-normalized probs gracefully")



######################################################################
######################################################################



class TestBrierScore(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)

    def test_perfect_predictions(self):
        """Test Brier score with perfect predictions (4 samples, 2 classes)."""

        probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        labels = np.array([0, 1, 0, 1], dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertAlmostEqual(brier, 0.0, places=6, msg="Brier score should be 0 for perfect predictions")


    def test_random_predictions(self):
        """Test Brier score with random predictions (5 samples, 5 classes)."""

        probs = np.random.dirichlet(alpha=np.ones(5), size=5)
        labels = np.array([0, 1, 2, 3, 4], dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertGreater(brier, 0.0, msg="Brier score should be >0 for random predictions")
        self.assertLess(brier, 2.0, msg="Brier score should be <2 for valid probabilities")


    def test_miscalibrated_predictions(self):
        """Test Brier score with miscalibrated predictions (3 samples, 4 classes)."""

        probs = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05]
        ])
        labels = np.array([1, 2, 0], dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertAlmostEqual(brier, 1.63, places=2, msg="Brier score should be ~1.63 for miscalibrated predictions")


    def test_single_sample(self):
        """Test Brier score with a single sample (1 sample, 3 classes)."""

        probs = np.array([[1.0, 0.0, 0.0]])
        labels = np.array([0], dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertAlmostEqual(brier, 0.0, places=6, msg="Brier score should be 0 for a single correct prediction")


    def test_large_array(self):
        """Test Brier score with a larger array (20 samples, 10 classes)."""

        probs = np.random.dirichlet(alpha=np.ones(10), size=20)
        labels = np.random.randint(0, 10, size=20, dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertGreaterEqual(brier, 0.0, msg="Brier score should be non-negative")
        self.assertLess(brier, 2.0, msg="Brier score should be <2 for valid probabilities")


    def test_single_class(self):
        """Test Brier score with a single class (5 samples, 1 class)."""

        probs = np.ones((5, 1))
        labels = np.zeros(5, dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertAlmostEqual(brier, 0.0, places=6, msg="Brier score should be 0 for single class with correct predictions")


    def test_invalid_probs_shape(self):
        """Test Brier score with invalid probs shape (1D array)."""

        probs = np.array([0.5, 0.5])
        labels = np.array([0], dtype=int)
        with self.assertRaises(ValueError, msg="Should raise ValueError for 1D probs"):
            compute_brier_score(probs, labels)


    def test_mismatched_labels(self):
        """Test Brier score with mismatched probs and labels shapes."""

        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([0], dtype=int)
        with self.assertRaises(ValueError, msg="Should raise ValueError for mismatched shapes"):
            compute_brier_score(probs, labels)


    def test_non_normalized_probs(self):
        """Test Brier score with non-normalized probabilities."""
        
        probs = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1], dtype=int)
        brier = compute_brier_score(probs, labels)
        self.assertGreaterEqual(brier, 0.0, msg="Brier score should handle non-normalized probs gracefully")