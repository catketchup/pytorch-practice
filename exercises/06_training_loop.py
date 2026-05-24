"""
Exercise 06: The Training Loop
================================
Learn the core training loop pattern used in all PyTorch projects:
forward pass → compute loss → backward pass → optimizer step.

You'll train real models on synthetic data.

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def exercise_1_optimizer_basics():
    """Understand optimizers and parameter updates."""

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Save initial weights
    initial_weight = model.weight.data.clone()

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1.0]])

    # TODO: Complete the training step:
    # 1. Forward pass
    pred = None
    # 2. Compute MSE loss
    loss = None
    # 3. Zero gradients
    # YOUR CODE HERE
    # 4. Backward pass
    # YOUR CODE HERE
    # 5. Optimizer step
    # YOUR CODE HERE

    # Verify that weights have changed
    assert not torch.equal(model.weight.data, initial_weight), "Weights should update!"
    print("exercise_1_optimizer_basics passed!")


def exercise_2_linear_regression():
    """Train a linear model on synthetic data."""

    # Generate synthetic data: y = 3x + 1 + noise
    torch.manual_seed(42)
    X = torch.randn(200, 1)
    y = 3 * X + 1 + 0.5 * torch.randn(200, 1)

    # TODO: Create a linear model (nn.Linear with appropriate dims)
    model = None

    # TODO: Create an SGD optimizer with lr=0.01
    optimizer = None

    # TODO: Define MSE loss
    criterion = None

    # Training loop
    for epoch in range(200):
        # TODO: Complete each step of the training loop
        # 1. Forward pass
        predictions = None
        # 2. Compute loss
        loss = None
        # 3. Zero gradients
        # YOUR CODE HERE
        # 4. Backward pass
        # YOUR CODE HERE
        # 5. Optimizer step
        # YOUR CODE HERE

    # The model should have learned weight ≈ 3 and bias ≈ 1
    learned_weight = model.weight.data.item()
    learned_bias = model.bias.data.item()

    assert abs(learned_weight - 3.0) < 0.5, f"Weight should be ~3.0, got {learned_weight}"
    assert abs(learned_bias - 1.0) < 0.5, f"Bias should be ~1.0, got {learned_bias}"
    print(f"  Learned: w={learned_weight:.3f}, b={learned_bias:.3f}")
    print("exercise_2_linear_regression passed!")


def exercise_3_minibatch_training():
    """Train with mini-batches using DataLoader."""

    torch.manual_seed(42)
    N = 1000
    X = torch.randn(N, 5)
    # True relationship: y = 2*x1 - 3*x2 + x3 - 0.5*x4 + 1.5
    true_w = torch.tensor([[2.0, -3.0, 1.0, -0.5, 0.0]])
    true_b = 1.5
    y = X @ true_w.T + true_b + 0.1 * torch.randn(N, 1)

    dataset = TensorDataset(X, y)

    # TODO: Create a DataLoader with batch_size=32, shuffle=True
    loader = None

    # TODO: Create a linear model
    model = None

    # TODO: Create an optimizer (SGD with lr=0.01)
    optimizer = None

    criterion = nn.MSELoss()

    # Training loop with mini-batches
    for epoch in range(50):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            # TODO: Complete the training step for each mini-batch
            pass

    # Verify model learned approximately correct weights
    learned_w = model.weight.data
    learned_b = model.bias.data.item()

    # Check first few weights are approximately correct
    assert abs(learned_w[0, 0].item() - 2.0) < 1.0
    assert abs(learned_w[0, 1].item() - (-3.0)) < 1.0
    assert abs(learned_b - 1.5) < 1.0
    print("exercise_3_minibatch_training passed!")


def exercise_4_classification():
    """Train a classifier on a synthetic 2-class problem."""

    torch.manual_seed(42)

    # Generate two clusters
    n = 200
    X0 = torch.randn(n, 2) + torch.tensor([2.0, 2.0])
    X1 = torch.randn(n, 2) + torch.tensor([-2.0, -2.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # TODO: Create an MLP with:
    #   Linear(2, 16) -> ReLU -> Linear(16, 2)
    model = None

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        for batch_X, batch_y in loader:
            # TODO: Complete the training step
            pass

    # Evaluate on full dataset
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()

    assert accuracy > 0.9, f"Accuracy should be >90%, got {accuracy*100:.1f}%"
    print(f"  Classification accuracy: {accuracy*100:.1f}%")
    print("exercise_4_classification passed!")


def exercise_5_learning_rate_scheduling():
    """Use a learning rate scheduler during training."""

    torch.manual_seed(42)
    X = torch.randn(200, 1)
    y = 3 * X + 1 + 0.5 * torch.randn(200, 1)

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # TODO: Create a StepLR scheduler that decays lr by 0.5 every 30 epochs
    scheduler = None

    initial_lr = optimizer.param_groups[0]['lr']

    for epoch in range(90):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: Step the scheduler each epoch
        # YOUR CODE HERE

    final_lr = optimizer.param_groups[0]['lr']

    # The LR should have decayed significantly from the initial 0.1
    assert final_lr < 0.05, f"Expected lr < 0.05, got {final_lr}"
    print(f"  Initial LR: {initial_lr}, Final LR: {final_lr}")
    print("exercise_5_learning_rate_scheduling passed!")


def exercise_6_gradient_clipping():
    """Apply gradient clipping to prevent exploding gradients."""

    torch.manual_seed(42)
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()

    # Check gradient norm before clipping
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # TODO: Clip gradients with max_norm=1.0 using clip_grad_norm_
    # The function above already clips — but practice calling it:
    # YOUR CODE HERE (use clip_grad_norm_ with max_norm=1.0 on model.parameters())

    # After clipping, total norm should be ≤ 1.0
    total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert total_norm <= 1.01, f"Clipped norm should be ≤ 1.0, got {total_norm:.4f}"
    print(f"  Gradient norm before clipping: {grad_norm_before:.4f}")
    print("exercise_6_gradient_clipping passed!")


if __name__ == "__main__":
    exercise_1_optimizer_basics()
    exercise_2_linear_regression()
    exercise_3_minibatch_training()
    exercise_4_classification()
    exercise_5_learning_rate_scheduling()
    exercise_6_gradient_clipping()
    print("\nAll exercises in 06_training_loop.py complete!")