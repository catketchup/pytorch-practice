"""
Exercise 03: Datasets & DataLoaders
====================================
Real-world ML needs efficient data pipelines. PyTorch provides Dataset and
DataLoader abstractions to handle batching, shuffling, and data transforms.

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


def exercise_1_tensor_dataset():
    """Use TensorDataset to wrap tensors into a dataset."""

    features = torch.randn(100, 5)
    labels = torch.randint(0, 3, (100,))

    # TODO: Create a TensorDataset from features and labels
    dataset = None

    # TODO: Get the 0-th sample (a tuple of (feature, label))
    sample = None

    assert len(dataset) == 100
    assert sample[0].shape == (5,)
    assert isinstance(sample[1], torch.Tensor)
    print("exercise_1_tensor_dataset passed!")


def exercise_2_custom_dataset():
    """Implement a custom Dataset class.

    A Dataset must implement __len__ and __getitem__.
    """

    class QuadraticDataset(Dataset):
        """Generates (x, y) pairs where y = 3x^2 + 2x + 1 + noise."""

        def __init__(self, n_samples=200, noise_std=0.1):
            self.x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
            noise = torch.randn_like(self.x) * noise_std
            self.y = 3 * self.x ** 2 + 2 * self.x + 1 + noise

        # TODO: Implement __len__
        def __len__(self):
            pass

        # TODO: Implement __getitem__
        def __getitem__(self, idx):
            pass

    dataset = QuadraticDataset()
    assert len(dataset) == 200
    item = dataset[0]
    assert isinstance(item, tuple) and len(item) == 2
    assert item[0].shape == (1,)
    print("exercise_2_custom_dataset passed!")


def exercise_3_dataloader():
    """Use DataLoader to iterate over a dataset in batches."""

    features = torch.randn(100, 5)
    labels = torch.randint(0, 3, (100,))
    dataset = TensorDataset(features, labels)

    # TODO: Create a DataLoader with batch_size=32, shuffle=True
    loader = None

    # TODO: Get the first batch. Iterate over loader and grab the first item.
    first_batch = None

    assert isinstance(first_batch, (tuple, list))
    assert first_batch[0].shape[0] <= 32  # batch dim
    assert first_batch[0].shape[1] == 5    # feature dim

    # TODO: Count how many batches the loader produces
    num_batches = None

    assert num_batches == 4  # ceil(100/32) = 4
    print("exercise_3_dataloader passed!")


def exercise_4_transforms():
    """Use torchvision transforms to preprocess data."""

    # Create a fake "image" tensor: 3x32x32 (C x H x W)
    fake_image = torch.rand(3, 32, 32)

    # TODO: Create a transform pipeline that:
    #   1. Converts a PIL Image or ndarray to tensor
    #   2. Normalizes with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    transform_pipeline = None

    # TODO: Apply the Normalize transform directly on fake_image
    #       Using transforms.Normalize(mean, std)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized = None

    # After normalization with mean=0.5, std=0.5: new_val = (val - 0.5)/0.5
    # Values should now be in [-1, 1] range
    assert normalized.min().item() >= -1.01
    assert normalized.max().item() <= 1.01
    print("exercise_4_transforms passed!")


def exercise_5_collate_fn():
    """Custom collate function for variable-length sequences."""

    class SeqDataset(Dataset):
        def __init__(self):
            self.seqs = [torch.randn(i) for i in [3, 5, 7, 4, 6]]

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            return self.seqs[idx]

    def collate_fn(batch):
        """Custom collate: pad sequences to equal length in a batch.

        Args:
            batch: list of 1D tensors of different lengths

        Returns:
            padded: tensor of shape (batch_size, max_len)
            lengths: tensor of original lengths
        """
        # TODO: Implement this collate function.
        # 1. Find the maximum length in the batch
        # 2. Pad each sequence with zeros to that length
        # 3. Return a stacked tensor and a tensor of original lengths
        pass

    dataset = SeqDataset()
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)

    # TODO: Get one batch from the loader
    batch = None

    assert batch[0].shape[0] == 3              # batch size
    assert batch[0].shape[1] >= 5               # max seq len in first 3 is 5
    assert (batch[1] == torch.tensor([3, 5, 7])).all() or (batch[1] == torch.tensor([3, 5, 7])).all()
    print("exercise_5_collate_fn passed!")


def exercise_6_train_val_split():
    """Split a dataset into training and validation sets."""

    features = torch.randn(1000, 10)
    labels = torch.randint(0, 5, (1000,))
    full_dataset = TensorDataset(features, labels)

    # TODO: Use torch.utils.data.random_split to split 80/20
    train_dataset, val_dataset = None

    assert len(train_dataset) == 800
    assert len(val_dataset) == 200
    print("exercise_6_train_val_split passed!")


if __name__ == "__main__":
    exercise_1_tensor_dataset()
    exercise_2_custom_dataset()
    exercise_3_dataloader()
    exercise_4_transforms()
    exercise_5_collate_fn()
    exercise_6_train_val_split()
    print("\nAll exercises in 03_datasets.py complete!")