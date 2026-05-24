"""Solutions for Exercise 03: Datasets & DataLoaders"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms


def exercise_1_tensor_dataset():
    features = torch.randn(100, 5)
    labels = torch.randint(0, 3, (100,))
    dataset = TensorDataset(features, labels)
    sample = dataset[0]
    assert len(dataset) == 100
    assert sample[0].shape == (5,)
    assert isinstance(sample[1], torch.Tensor)
    print("exercise_1_tensor_dataset passed!")


def exercise_2_custom_dataset():
    class QuadraticDataset(Dataset):
        def __init__(self, n_samples=200, noise_std=0.1):
            self.x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
            noise = torch.randn_like(self.x) * noise_std
            self.y = 3 * self.x ** 2 + 2 * self.x + 1 + noise

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    dataset = QuadraticDataset()
    assert len(dataset) == 200
    item = dataset[0]
    assert isinstance(item, tuple) and len(item) == 2
    assert item[0].shape == (1,)
    print("exercise_2_custom_dataset passed!")


def exercise_3_dataloader():
    features = torch.randn(100, 5)
    labels = torch.randint(0, 3, (100,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    first_batch = next(iter(loader))
    assert isinstance(first_batch, (tuple, list))
    assert first_batch[0].shape[0] <= 32
    assert first_batch[0].shape[1] == 5
    num_batches = len(loader)
    assert num_batches == 4
    print("exercise_3_dataloader passed!")


def exercise_4_transforms():
    fake_image = torch.rand(3, 32, 32)
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized = normalize(fake_image)
    assert normalized.min().item() >= -1.01
    assert normalized.max().item() <= 1.01
    print("exercise_4_transforms passed!")


def exercise_5_collate_fn():
    from torch.utils.data import Dataset, DataLoader

    class SeqDataset(Dataset):
        def __init__(self):
            self.seqs = [torch.randn(i) for i in [3, 5, 7, 4, 6]]

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            return self.seqs[idx]

    def collate_fn(batch):
        lengths = torch.tensor([len(seq) for seq in batch])
        max_len = lengths.max().item()
        padded = torch.zeros(len(batch), max_len)
        for i, seq in enumerate(batch):
            padded[i, :len(seq)] = seq
        return padded, lengths

    dataset = SeqDataset()
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch[0].shape[0] == 3
    assert batch[0].shape[1] >= 5
    assert (batch[1] == torch.tensor([3, 5, 7])).all()
    print("exercise_5_collate_fn passed!")


def exercise_6_train_val_split():
    full_dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    train_dataset, val_dataset = random_split(full_dataset, [800, 200])
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
    print("\nAll solutions in 03_datasets.py verified!")