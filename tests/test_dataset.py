# tests/test_dataset.py
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch
from src.at_bat_dataset import AtBatDataset
from src.data_loader import split_dataset, get_data_loader

@pytest.fixture
def mock_db_session():
    """Mock database session and queries."""
    def mock_read_sql(query, bind):
        if "COUNT(*)" in query:
            return pd.DataFrame({"count": [10000]}, index=[0])
        elif "SELECT at_bat_id" in query:
            return pd.DataFrame({"at_bat_id": list(range(10000))})
        elif "AVG" in query:
            return pd.DataFrame({"hit_distance": [50.0], "hit_angle": [30.0]})
        elif "STDDEV" in query:
            return pd.DataFrame({"hit_distance": [10.0], "hit_angle": [5.0]})
        else:
            data = {
                "at_bat_id": list(range(1000)),
                "hit_style": [1] * 1000,
                "hit_distance": np.random.normal(50, 10, 1000),
                "hit_angle": np.random.normal(30, 5, 1000),
                "at_bat_type_id": [0] * 1000
            }
            return pd.DataFrame(data)
    with patch("src.at_bat_dataset.get_db_session") as mock_session:
        mock_session.return_value.__enter__.return_value.bind = None
        mock_session.return_value.__enter__.return_value.read_sql = mock_read_sql
        yield mock_session

@pytest.fixture
def dataset(mock_db_session):
    """Create an AtBatDataset instance."""
    return AtBatDataset(entityId="test", db_batch_size=1000)

def test_dataset_length(dataset):
    """Test dataset length."""
    assert len(dataset) == 10000, "Dataset should have 10000 records"

def test_getitem(dataset):
    """Test __getitem__ returns correct features and label."""
    features, label = dataset[0]
    assert set(features.keys()) == {"batter", "pitcher", "pitch", "hit", "count"}, "Feature keys incorrect"
    assert "hit_style" in features["batter"], "batter should include hit_style"
    assert "hit_distance" in features["pitch"], "pitch should include hit_distance"
    assert "hit_angle" in features["hit"], "hit should include hit_angle"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.dtype == torch.long, "Label should be torch.long"

def test_split_dataset(dataset):
    """Test dataset splitting."""
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(train_dataset) == 8000, "Train set should have 8000 samples"
    assert len(val_dataset) == 1000, "Validation set should have 1000 samples"
    assert len(test_dataset) == 1000, "Test set should have 1000 samples"
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    test_indices = set(test_dataset.indices)
    assert len(train_indices & val_indices) == 0, "Sets should be disjoint"
    assert len(val_indices & test_indices) == 0, "Sets should be disjoint"
    assert len(train_indices & test_indices) == 0, "Sets should be disjoint"

def test_train_loader(dataset):
    """Test training DataLoader."""
    train_dataset, _, _ = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    loader = get_data_loader(train_dataset, split="train")
    batch = next(iter(loader))
    features, labels = batch
    assert len(labels) == 64, "Batch size should be 64"
    assert features["hit_style"].shape == (64,), "hit_style shape incorrect"
    assert features["hit_distance"].shape == (64,), "hit_distance shape incorrect"
    assert labels.shape == (64,), "Labels shape incorrect"

def test_val_loader_small_dataset(mock_db_session):
    """Test validation DataLoader with a small dataset."""
    small_dataset = AtBatDataset(entityId="small", db_batch_size=500)
    small_dataset.total_records = 2000
    small_dataset.valid_indices = list(range(2000))
    _, val_dataset, _ = split_dataset(small_dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    loader = get_data_loader(val_dataset, split="val")
    batch = next(iter(loader))
    features, labels = batch
    assert len(labels) == 64, "Batch size should be 64"
    assert len(val_dataset) == 400, "Validation set should have 400 samples"

def test_cache_alignment(dataset):
    """Test cache alignment in DataLoader."""
    train_dataset, _, _ = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    loader = get_data_loader(train_dataset, split="train")
    batch = next(iter(loader))
    batch_indices = train_dataset.indices[:64]
    chunk_start = (batch_indices[0] // dataset.db_batch_size) * dataset.db_batch_size
    chunk_end = chunk_start + dataset.db_batch_size
    assert all(chunk_start <= idx < chunk_end for idx in batch_indices), "Batch indices should be within db_batch_size chunk"

def test_empty_batch(dataset):
    """Test handling of empty batch."""
    dataset.valid_indices = []  # Simulate empty dataset
    dataset.total_records = 0
    with pytest.raises(ValueError, match="Empty batch fetched"):
        dataset[0]


import unittest
import numpy as np
from torch.utils.data import Dataset, Subset
from unittest.mock import MagicMock

# Assuming split_dataset is defined as above

class MockCustomDataset(Dataset):
    """Mock dataset mimicking CustomDataset for testing."""
    def __init__(self, total_records, db_batch_size=5000):
        self.total_records = total_records
        self.db_batch_size = db_batch_size
        self.valid_indices = np.arange(total_records)

    def __len__(self):
        return self.total_records

    def __getitem__(self, idx):
        # Return a dummy item (not used in split_dataset, but required for Dataset)
        return {"dummy": idx}, idx

class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.seed = 42
        self.db_batch_size = 5000
        self.total_records = 12000  # Creates 2 full chunks + 1 small chunk (2000)
        self.dataset = MockCustomDataset(self.total_records, self.db_batch_size)
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def test_split_sizes(self):
        """Test that split sizes match requested ratios."""
        train_ds, val_ds, test_ds = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        n_train = int(self.total_records * self.train_ratio)
        n_val = int(self.total_records * self.val_ratio)
        n_test = self.total_records - n_train - n_val
        self.assertEqual(len(train_ds), n_train, "Train split size mismatch")
        self.assertEqual(len(val_ds), n_val, "Validation split size mismatch")
        self.assertEqual(len(test_ds), n_test, "Test split size mismatch")

    def test_chunk_contiguity(self):
        """Test that indices within chunks are contiguous."""
        train_ds, val_ds, test_ds = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        # Combine all indices
        all_indices = np.concatenate([train_ds.indices, val_ds.indices, test_ds.indices])
        # Check if indices cover the dataset (allowing for truncation)
        self.assertTrue(len(all_indices) <= self.total_records, "More indices than dataset size")
        # Check contiguity within chunks
        chunks = [all_indices[i:i + self.db_batch_size] for i in range(0, len(all_indices), self.db_batch_size)]
        for chunk in chunks:
            if len(chunk) > 1:
                self.assertTrue(np.all(np.diff(chunk) == 1), "Chunk indices are not contiguous")

    def test_chunk_shuffling(self):
        """Test that chunks are shuffled (not in original order)."""
        train_ds, val_ds, test_ds = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        all_indices = np.concatenate([train_ds.indices, val_ds.indices, test_ds.indices])
        # Reconstruct chunks
        chunks = [all_indices[i:i + self.db_batch_size] for i in range(0, len(all_indices), self.db_batch_size)]
        # Check if chunk starts are not in sequential order
        chunk_starts = [chunk[0] for chunk in chunks if len(chunk) > 0]
        sequential_starts = np.arange(0, self.total_records, self.db_batch_size)
        self.assertFalse(np.array_equal(chunk_starts, sequential_starts[:len(chunk_starts)]),
                        "Chunks are not shuffled")

    def test_small_chunk_inclusion(self):
        """Test that the small chunk is included in the split."""
        train_ds, val_ds, test_ds = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        all_indices = np.concatenate([train_ds.indices, val_ds.indices, test_ds.indices])
        # Check if indices from the small chunk (10000:12000) are present
        small_chunk_indices = np.arange(10000, 12000)
        self.assertTrue(np.any(np.isin(small_chunk_indices, all_indices)),
                        "Small chunk indices not included")

    def test_subset_type(self):
        """Test that returned datasets are Subset instances."""
        train_ds, val_ds, test_ds = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        self.assertIsInstance(train_ds, Subset, "Train dataset is not a Subset")
        self.assertIsInstance(val_ds, Subset, "Validation dataset is not a Subset")
        self.assertIsInstance(test_ds, Subset, "Test dataset is not a Subset")

    def test_invalid_ratios(self):
        """Test that invalid ratios raise an error."""
        with self.assertRaises(ValueError):
            split_dataset(
                self.dataset,
                train_ratio=0.8,
                val_ratio=0.3,
                test_ratio=0.1,
                seed=self.seed
            )

    def test_small_dataset(self):
        """Test behavior with a dataset smaller than db_batch_size."""
        small_dataset = MockCustomDataset(2000, self.db_batch_size)
        train_ds, val_ds, test_ds = split_dataset(
            small_dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 2000,
                         "Small dataset split does not cover all indices")

    def test_single_chunk(self):
        """Test behavior with exactly one chunk."""
        single_chunk_dataset = MockCustomDataset(self.db_batch_size, self.db_batch_size)
        train_ds, val_ds, test_ds = split_dataset(
            single_chunk_dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        self.assertTrue(len(train_ds) > 0, "Train split empty for single chunk")
        self.assertTrue(len(val_ds) > 0, "Validation split empty for single chunk")
        self.assertTrue(len(test_ds) > 0, "Test split empty for single chunk")

    def test_reproducible_shuffling(self):
        """Test that the same seed produces the same split."""
        train_ds1, val_ds1, test_ds1 = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        train_ds2, val_ds2, test_ds2 = split_dataset(
            self.dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        self.assertTrue(np.array_equal(train_ds1.indices, train_ds2.indices),
                        "Train indices not reproducible with same seed")
        self.assertTrue(np.array_equal(val_ds1.indices, val_ds2.indices),
                        "Validation indices not reproducible with same seed")
        self.assertTrue(np.array_equal(test_ds1.indices, test_ds2.indices),
                        "Test indices not reproducible with same seed")

if __name__ == "__main__":
    unittest.main()