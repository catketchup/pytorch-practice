"""Solutions for Exercise 06: Training Loop"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def exercise_1_optimizer_basics():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_weight = model.weight.data.clone()
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1.0]])
    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(model.weight.data, initial_weight)
    print("exercise_1_optimizer_basics passed!")


def exercise_2_linear_regression():
    torch.manual_seed(42)
    X = torch.randn(200, 1)
    y = 3 * X + 1 + 0.5 * torch.randn(200, 1)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(200):
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    learned_weight = model.weight.data.item()
    learned_bias = model.bias.data.item()
    assert abs(learned_weight - 3.0) < 0.5
    assert abs(learned_bias - 1.0) < 0.5
    print(f"  Learned: w={learned_weight:.3f}, b={learned_bias:.3f}")
    print("exercise_2_linear_regression passed!")


def exercise_3_minibatch_training():
    torch.manual_seed(42)
    N = 1000
    X = torch.randn(N, 5)
    true_w = torch.tensor([[2.0, -3.0, 1.0, -0.5, 0.0]])
    true_b = 1.5
    y = X @ true_w.T + true_b + 0.1 * torch.randn(N, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(50):
        for batch_X, batch_y in loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    learned_w = model.weight.data
    learned_b = model.bias.data.item()
    assert abs(learned_w[0, 0].item() - 2.0) < 1.0
    assert abs(learned_w[0, 1].item() - (-3.0)) < 1.0
    assert abs(learned_b - 1.5) < 1.0
    print("exercise_3_minibatch_training passed!")


def exercise_4_classification():
    torch.manual_seed(42)
    n = 200
    X0 = torch.randn(n, 2) + torch.tensor([2.0, 2.0])
    X1 = torch.randn(n, 2) + torch.tensor([-2.0, -2.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(50):
        for batch_X, batch_y in loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    assert accuracy > 0.9
    print(f"  Classification accuracy: {accuracy*100:.1f}%")
    print("exercise_4_classification passed!")


def exercise_5_learning_rate_scheduling():
    torch.manual_seed(42)
    X = torch.randn(200, 1)
    y = 3 * X + 1 + 0.5 * torch.randn(200, 1)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    initial_lr = optimizer.param_groups[0]['lr']
    for epoch in range(90):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr < 0.05, f"Expected lr < 0.05, got {final_lr}"
    print(f"  Initial LR: {initial_lr}, Final LR: {final_lr}")
    print("exercise_5_learning_rate_scheduling passed!")


def exercise_6_gradient_clipping():
    torch.manual_seed(42)
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert total_norm <= 1.01
    print(f"  Gradient norm before clipping: {grad_norm_before:.4f}")
    print("exercise_6_gradient_clipping passed!")


if __name__ == "__main__":
    exercise_1_optimizer_basics()
    exercise_2_linear_regression()
    exercise_3_minibatch_training()
    exercise_4_classification()
    exercise_5_learning_rate_scheduling()
    exercise_6_gradient_clipping()
    print("\nAll solutions in 06_training_loop.py verified!")