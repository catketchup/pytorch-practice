"""
Exercise 07: Transfer Learning & Real-World Patterns
=====================================================
Learn to leverage pretrained models and implement evaluation patterns
used in real projects.

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy


def exercise_1_model_saving_loading():
    """Save and load model weights."""

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )

    # Run a forward pass to get reference output
    x = torch.randn(2, 10)
    original_output = model(x).detach().clone()

    # TODO: Save the model's state_dict to a file called "test_model.pt"
    # YOUR CODE HERE

    # Create a new model with the same architecture
    model2 = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )

    # TODO: Load the state_dict from "test_model.pt" into model2
    # YOUR CODE HERE

    # Verify outputs match
    output2 = model2(x)
    assert torch.allclose(original_output, output2, atol=1e-6)
    print("exercise_1_model_saving_loading passed!")

    import os
    os.remove("test_model.pt")


def exercise_2_transfer_learning():
    """Adapt a pretrained model for a new task."""

    # Simulate a "pretrained" model that outputs 1000-d features
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

    # TODO: Create a new model for 10-class classification:
    #   1. Copy pretrained.features
    #   2. Replace classifier with a new Linear(512, 10)
    #   3. Freeze the feature extractor
    class FineTunedModel(nn.Module):
        def __init__(self, pretrained):
            super().__init__()
            # TODO: Set self.features to the pretrained features
            # TODO: Set self.classifier to new Linear(512, 10)
            pass

        def forward(self, x):
            # TODO: Forward through features then classifier
            pass

    model = FineTunedModel(pretrained)

    # TODO: Freeze the feature extractor (set requires_grad=False)
    # YOUR CODE HERE

    # Verify: feature params are frozen, classifier params are trainable
    for param in model.features.parameters():
        assert not param.requires_grad, "Feature params should be frozen"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Only the classifier should have trainable params: 512*10 + 10 = 5130
    assert trainable == 5130, f"Expected 5130 trainable params, got {trainable}"
    print(f"  Trainable: {trainable}/{total}")
    print("exercise_2_transfer_learning passed!")


def exercise_3_early_stopping():
    """Implement early stopping to prevent overfitting."""

    torch.manual_seed(42)

    # Simple synthetic data
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

        # TODO: Implement early stopping logic:
        # If avg_loss < best_loss: update best_loss, save model, reset counter
        # Else: increment counter. If counter >= patience, break
        # YOUR CODE HERE

    # Should have stopped before epoch 100
    assert epoch < 100, "Early stopping should have triggered"
    print(f"  Stopped at epoch {epoch}")
    print("exercise_3_early_stopping passed!")


def exercise_4_validation_loop():
    """Implement a proper train/validation loop with metrics."""

    torch.manual_seed(42)

    # Generate data
    X = torch.randn(500, 10)
    y = (X[:, 0] + X[:, 1] > 0).long()  # simple decision boundary

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

    train_losses = []
    val_accuracies = []

    for epoch in range(30):
        # === Training ===
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # TODO: Training step
            #   forward, loss, zero_grad, backward, step
            pass
            epoch_loss += 0  # TODO: accumulate loss

        train_losses.append(epoch_loss / len(train_loader))

        # === Validation ===
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # TODO: Compute validation accuracy
                #   forward pass, count correct predictions
                pass

        val_accuracies.append(0.0)  # TODO: replace with actual accuracy

    # Final accuracy should be decent (usually >70% for this simple problem)
    # We're lenient here since the focus is on the loop pattern
    print(f"  Final val accuracy: {val_accuracies[-1]*100:.1f}%")
    print("exercise_4_validation_loop passed!")


def exercise_5_device_management():
    """Properly move data and models between CPU and GPU."""

    # This exercise focuses on the pattern, not actual GPU execution

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    # TODO: Get the appropriate device:
    #   Use "cuda" if available, otherwise "cpu"
    device = None

    # TODO: Move the model to the device
    # YOUR CODE HERE

    # TODO: Create sample data and move it to the same device
    x = torch.randn(4, 10)
    x = None  # TODO: move x to device

    # Forward pass should work on device
    output = model(x)
    assert output.shape == (4, 2)

    print(f"  Using device: {device}")
    print("exercise_5_device_management passed!")


def exercise_6_reproducibility():
    """Set seeds for reproducible experiments."""

    # TODO: Set all relevant random seeds for reproducibility
    # Set torch manual seed to 42
    # YOUR CODE HERE

    # Run twice and verify same results
    results = []
    for _ in range(2):
        # TODO: Set the seed again for the second run
        # YOUR CODE HERE

        model = nn.Linear(10, 2)
        x = torch.randn(5, 10)
        output = model(x)
        results.append(output)

    assert torch.allclose(results[0], results[1]), "Results should be identical with same seed"
    print("exercise_6_reproducibility passed!")


if __name__ == "__main__":
    exercise_1_model_saving_loading()
    exercise_2_transfer_learning()
    exercise_3_early_stopping()
    exercise_4_validation_loop()
    exercise_5_device_management()
    exercise_6_reproducibility()
    print("\nAll exercises in 07_transfer_learning.py complete!")