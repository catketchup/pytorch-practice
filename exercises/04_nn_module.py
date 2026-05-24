"""
Exercise 04: nn.Module — Building Neural Network Layers
========================================================
nn.Module is the base class for all neural network components. You define
your architecture in __init__ and the forward pass in forward().

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_linear_layer():
    """Use nn.Linear for a simple linear transformation."""

    # TODO: Create a linear layer with input features=4 and output features=2
    linear = None

    x = torch.randn(3, 4)  # batch of 3 samples, each with 4 features

    # TODO: Pass x through the linear layer
    output = None

    assert output.shape == (3, 2)

    # nn.Linear has weight and bias parameters
    # TODO: Print the shape of the weight matrix
    weight_shape = None

    assert weight_shape == (2, 4)
    print("exercise_1_linear_layer passed!")


def exercise_2_simple_mlp():
    """Build a 2-layer MLP (multi-layer perceptron) using nn.Sequential."""

    # Input: 10 features, Hidden: 32 neurons, Output: 3 classes
    # TODO: Build an MLP with:
    #   Linear(10, 32) -> ReLU -> Linear(32, 3)
    model = None

    x = torch.randn(5, 10)

    # TODO: Pass x through the model
    output = None

    assert output.shape == (5, 3)
    print("exercise_2_simple_mlp passed!")


def exercise_3_custom_module():
    """Define a custom nn.Module with parameters."""

    class SimpleAttention(nn.Module):
        """A minimal attention mechanism:
        attention(x) = softmax(x @ W) * x

        W is a learnable (input_dim, input_dim) matrix.
        """

        def __init__(self, input_dim):
            super().__init__()
            # TODO: Define a learnable weight matrix W of shape (input_dim, input_dim)
            # Hint: use nn.Parameter with a random initialization
            self.W = None

        def forward(self, x):
            # x shape: (batch, input_dim)
            # TODO: Compute attention weights: softmax(x @ W, dim=-1)
            attn_weights = None

            # TODO: Return attn_weights * x (element-wise)
            return None

    model = SimpleAttention(8)
    x = torch.randn(4, 8)

    output = model(x)
    assert output.shape == (4, 8)

    # Verify the module has the parameter
    param_count = sum(1 for _ in model.parameters())
    assert param_count == 1
    print("exercise_3_custom_module passed!")


def exercise_4_forward_pass():
    """Build a model for image classification and run a forward pass."""

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: 1x28x28 (grayscale, like MNIST)
            # TODO: Define:
            #   self.conv1: Conv2d(1, 8, kernel_size=3, padding=1)
            #   self.conv2: Conv2d(8, 16, kernel_size=3, padding=1)
            #   self.fc1: Linear(16*7*7, 64)
            #   self.fc2: Linear(64, 10)
            pass

        def forward(self, x):
            # x: (batch, 1, 28, 28)
            # TODO: Implement the forward pass:
            #   1. conv1 -> ReLU -> MaxPool2d(2)   -> shape: (B, 8, 14, 14)
            #   2. conv2 -> ReLU -> MaxPool2d(2)   -> shape: (B, 16, 7, 7)
            #   3. Flatten                            -> shape: (B, 16*7*7)
            #   4. fc1 -> ReLU                        -> shape: (B, 64)
            #   5. fc2                                 -> shape: (B, 10)
            pass

    model = TinyCNN()
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    assert output.shape == (2, 10)
    print("exercise_4_forward_pass passed!")


def exercise_5_loss_functions():
    """Use common loss functions."""

    # MSE Loss (regression)
    pred = torch.tensor([1.5, 2.1, 3.3])
    target = torch.tensor([1.0, 2.0, 3.0])

    # TODO: Compute MSE loss
    mse_loss = None

    assert abs(mse_loss.item() - 0.1166) < 0.01

    # Cross-Entropy Loss (classification)
    logits = torch.tensor([[2.0, 0.5, 0.3]])
    target_class = torch.tensor([0])

    # TODO: Compute cross-entropy loss
    ce_loss = None

    assert ce_loss.item() > 0  # loss should be positive

    # Binary Cross-Entropy Loss
    probs = torch.tensor([0.8, 0.3, 0.9])
    targets_binary = torch.tensor([1.0, 0.0, 1.0])

    # TODO: Compute binary cross-entropy loss
    # Hint: probs must be in (0, 1) — use torch.sigmoid or clamp
    bce_loss = None

    assert bce_loss.item() > 0
    print("exercise_5_loss_functions passed!")


def exercise_6_model_inspection():
    """Inspect model parameters and layers."""

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )

    # TODO: Count total number of parameters
    total_params = None

    # TODO: Count the number of trainable parameters
    trainable_params = None

    # Hint: p.numel() gives the number of elements in a parameter tensor
    # Linear(10, 32) has 10*32 + 32 = 352 parameters
    # Linear(32, 5) has 32*5 + 5 = 165 parameters
    # Total = 517
    assert total_params == 517
    assert trainable_params == 517
    print("exercise_6_model_inspection passed!")


def exercise_7_freezing_layers():
    """Freeze layers to prevent gradient updates."""

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )

    # TODO: Freeze the first linear layer (set requires_grad=False for its params)
    # YOUR CODE HERE

    # Verify:
    first_layer_frozen = all(not p.requires_grad for p in model[0].parameters())
    second_layer_trainable = all(p.requires_grad for p in model[2].parameters())

    assert first_layer_frozen == True
    assert second_layer_trainable == True

    # TODO: Count trainable parameters after freezing
    trainable_params = None

    assert trainable_params == 165  # only the second layer
    print("exercise_7_freezing_layers passed!")


if __name__ == "__main__":
    exercise_1_linear_layer()
    exercise_2_simple_mlp()
    exercise_3_custom_module()
    exercise_4_forward_pass()
    exercise_5_loss_functions()
    exercise_6_model_inspection()
    exercise_7_freezing_layers()
    print("\nAll exercises in 04_nn_module.py complete!")