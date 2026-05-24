"""Solutions for Exercise 04: nn.Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_linear_layer():
    linear = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    output = linear(x)
    assert output.shape == (3, 2)
    weight_shape = linear.weight.shape
    assert weight_shape == (2, 4)
    print("exercise_1_linear_layer passed!")


def exercise_2_simple_mlp():
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )
    x = torch.randn(5, 10)
    output = model(x)
    assert output.shape == (5, 3)
    print("exercise_2_simple_mlp passed!")


def exercise_3_custom_module():
    class SimpleAttention(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.W = nn.Parameter(torch.randn(input_dim, input_dim))

        def forward(self, x):
            attn_weights = torch.softmax(x @ self.W, dim=-1)
            return attn_weights * x

    model = SimpleAttention(8)
    x = torch.randn(4, 8)
    output = model(x)
    assert output.shape == (4, 8)
    param_count = sum(1 for _ in model.parameters())
    assert param_count == 1
    print("exercise_3_custom_module passed!")


def exercise_4_forward_pass():
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = TinyCNN()
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    assert output.shape == (2, 10)
    print("exercise_4_forward_pass passed!")


def exercise_5_loss_functions():
    pred = torch.tensor([1.5, 2.1, 3.3])
    target = torch.tensor([1.0, 2.0, 3.0])
    mse_loss = nn.MSELoss()(pred, target)
    assert abs(mse_loss.item() - 0.1166) < 0.01

    logits = torch.tensor([[2.0, 0.5, 0.3]])
    target_class = torch.tensor([0])
    ce_loss = nn.CrossEntropyLoss()(logits, target_class)
    assert ce_loss.item() > 0

    probs = torch.tensor([0.8, 0.3, 0.9])
    targets_binary = torch.tensor([1.0, 0.0, 1.0])
    bce_loss = nn.BCELoss()(probs, targets_binary)
    assert bce_loss.item() > 0
    print("exercise_5_loss_functions passed!")


def exercise_6_model_inspection():
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 517
    assert trainable_params == 517
    print("exercise_6_model_inspection passed!")


def exercise_7_freezing_layers():
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )
    for param in model[0].parameters():
        param.requires_grad = False

    first_layer_frozen = all(not p.requires_grad for p in model[0].parameters())
    second_layer_trainable = all(p.requires_grad for p in model[2].parameters())
    assert first_layer_frozen == True
    assert second_layer_trainable == True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 165
    print("exercise_7_freezing_layers passed!")


if __name__ == "__main__":
    exercise_1_linear_layer()
    exercise_2_simple_mlp()
    exercise_3_custom_module()
    exercise_4_forward_pass()
    exercise_5_loss_functions()
    exercise_6_model_inspection()
    exercise_7_freezing_layers()
    print("\nAll solutions in 04_nn_module.py verified!")