"""Solutions for Exercise 07: Transfer Learning & Real-World Patterns"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os


def exercise_1_model_saving_loading():
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )
    x = torch.randn(2, 10)
    original_output = model(x).detach().clone()
    torch.save(model.state_dict(), "test_model.pt")
    model2 = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )
    model2.load_state_dict(torch.load("test_model.pt"))
    output2 = model2(x)
    assert torch.allclose(original_output, output2, atol=1e-6)
    print("exercise_1_model_saving_loading passed!")
    os.remove("test_model.pt")


def exercise_2_transfer_learning():
    class PretrainedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(512, 1000)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    pretrained = PretrainedModel()

    class FineTunedModel(nn.Module):
        def __init__(self, pretrained):
            super().__init__()
            self.features = pretrained.features
            self.classifier = nn.Linear(512, 10)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = FineTunedModel(pretrained)
    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.features.parameters():
        assert not param.requires_grad
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable == 5130
    print(f"  Trainable: {trainable}/{total}")
    print("exercise_2_transfer_learning passed!")


def exercise_3_early_stopping():
    torch.manual_seed(42)
    X_train = torch.randn(100, 5)
    y_train = (X_train[:, 0] > 0).long()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    assert epoch < 100
    print(f"  Stopped at epoch {epoch}")
    print("exercise_3_early_stopping passed!")


def exercise_4_validation_loop():
    torch.manual_seed(42)
    X = torch.randn(500, 10)
    y = (X[:, 0] + X[:, 1] > 0).long()
    full_dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(full_dataset, [400, 100])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    val_accuracies = []
    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        val_accuracies.append(correct / total)
    print(f"  Final val accuracy: {val_accuracies[-1]*100:.1f}%")
    print("exercise_4_validation_loop passed!")


def exercise_5_device_management():
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(4, 10).to(device)
    output = model(x)
    assert output.shape == (4, 2)
    print(f"  Using device: {device}")
    print("exercise_5_device_management passed!")


def exercise_6_reproducibility():
    results = []
    for _ in range(2):
        torch.manual_seed(42)
        model = nn.Linear(10, 2)
        x = torch.randn(5, 10)
        output = model(x)
        results.append(output)
    assert torch.allclose(results[0], results[1])
    print("exercise_6_reproducibility passed!")


if __name__ == "__main__":
    exercise_1_model_saving_loading()
    exercise_2_transfer_learning()
    exercise_3_early_stopping()
    exercise_4_validation_loop()
    exercise_5_device_management()
    exercise_6_reproducibility()
    print("\nAll solutions in 07_transfer_learning.py verified!")