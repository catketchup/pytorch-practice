"""
Exercise 05: Convolutional Neural Networks (CNNs)
==================================================
CNNs are the backbone of computer vision. Learn to build them step by step,
from individual conv layers to full architectures.

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_conv1d():
    """Understand 1D convolution basics."""

    # Input: (batch, channels, length) = (1, 1, 6)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])

    # TODO: Create a Conv1d layer with:
    #   in_channels=1, out_channels=1, kernel_size=3, bias=False
    conv = None

    # Manually set the weight to [1, 0, -1] (edge detector)
    with torch.no_grad():
        conv.weight.data = torch.tensor([[[1.0, 0.0, -1.0]]])

    # TODO: Pass x through conv
    output = None

    # Expected output length: 6 - 3 + 1 = 4
    assert output.shape == (1, 1, 4), f"Expected (1,1,4), got {output.shape}"
    print("exercise_1_conv1d passed!")


def exercise_2_conv2d():
    """Understand 2D convolution: filter sizes, stride, and padding."""

    # Input: 1 channel, 5x5 image
    x = torch.randn(1, 1, 5, 5)

    # TODO: Create Conv2d with in_channels=1, out_channels=4, kernel_size=3,
    #       padding=0, stride=1
    conv = None

    # TODO: Pass x through conv
    out = None

    # Output size = (5 - 3 + 0) / 1 + 1 = 3, so (1, 4, 3, 3)
    assert out.shape == (1, 4, 3, 3), f"Expected (1,4,3,3), got {out.shape}"

    # Now with padding=1 to preserve spatial dimensions
    conv_padded = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    out_padded = conv_padded(x)

    # With padding=1: (5 + 2 - 3) / 1 + 1 = 5
    assert out_padded.shape == (1, 4, 5, 5)
    print("exercise_2_conv2d passed!")


def exercise_3_pooling():
    """Understand MaxPool and AvgPool."""

    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    # [[0,  1,  2,  3],
    #  [4,  5,  6,  7],
    #  [8,  9, 10, 11],
    #  [12, 13, 14, 15]]

    # TODO: Create MaxPool2d with kernel_size=2, stride=2
    maxpool = None

    # TODO: Apply max pooling
    max_result = None

    assert max_result.shape == (1, 1, 2, 2)
    # The 4 quadrants' max values: 5, 7, 13, 15
    assert max_result.squeeze().tolist() == [[5.0, 7.0], [13.0, 15.0]]

    # TODO: Create AvgPool2d with kernel_size=2, stride=2
    avgpool = None

    # TODO: Apply average pooling
    avg_result = None

    # The 4 quadrants' averages: 2.5, 4.5, 10.5, 12.5
    expected = [[2.5, 4.5], [10.5, 12.5]]
    for i in range(2):
        for j in range(2):
            assert abs(avg_result[0, 0, i, j].item() - expected[i][j]) < 1e-5
    print("exercise_3_pooling passed!")


def exercise_4_cnn_architecture():
    """Build a full CNN for MNIST classification."""

    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: Define the layers:
            #   conv1: Conv2d(1, 16, 3, padding=1)   -> (B, 16, 28, 28)
            #   conv2: Conv2d(16, 32, 3, padding=1)   -> (B, 32, 28, 28)
            #   pool: MaxPool2d(2, 2)                 -> halves H and W
            #   fc1: Linear(32*7*7, 128)
            #   fc2: Linear(128, 10)
            pass

        def forward(self, x):
            # x: (B, 1, 28, 28)
            # TODO: Implement forward:
            #   conv1 -> ReLU -> pool    -> (B, 16, 14, 14)
            #   conv2 -> ReLU -> pool    -> (B, 32, 7, 7)
            #   flatten                   -> (B, 32*7*7)
            #   fc1 -> ReLU              -> (B, 128)
            #   fc2                       -> (B, 10)
            pass

    model = MNISTNet()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)

    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"

    # Check total parameters (approximate — just verify it's reasonable)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1000, "Model seems too small"
    print("exercise_4_cnn_architecture passed!")


def exercise_5_batch_norm_dropout():
    """Add BatchNorm and Dropout for regularization."""

    class RegularizedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: Define layers with BatchNorm and Dropout:
            #   conv1: Conv2d(1, 16, 3, padding=1)
            #   bn1: BatchNorm2d(16)
            #   conv2: Conv2d(16, 32, 3, padding=1)
            #   bn2: BatchNorm2d(32)
            #   pool: MaxPool2d(2)
            #   fc1: Linear(32*7*7, 128)
            #   dropout: Dropout(0.5)
            #   fc2: Linear(128, 10)
            pass

        def forward(self, x):
            # x: (B, 1, 28, 28)
            # TODO: Forward pass with BN after each conv, dropout before fc2
            #   conv1 -> bn1 -> ReLU -> pool
            #   conv2 -> bn2 -> ReLU -> pool
            #   flatten
            #   fc1 -> ReLU
            #   dropout
            #   fc2
            pass

    model = RegularizedCNN()

    # Set to training mode for dropout to be active
    model.train()
    x = torch.randn(2, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)

    assert out1.shape == (2, 10)
    # Dropout means two forward passes on same input give different outputs
    assert not torch.allclose(out1, out2), "Dropout should cause different outputs in training mode"

    # In eval mode, dropout is disabled
    model.eval()
    out3 = model(x)
    out4 = model(x)
    assert torch.allclose(out3, out4), "Dropout should be off in eval mode"
    print("exercise_5_batch_norm_dropout passed!")


def exercise_6_output_size_formula():
    """Calculate output sizes manually using the conv formula.

    Formula: O = floor((I + 2P - K) / S) + 1
    where I=input_size, P=padding, K=kernel_size, S=stride
    """

    # Input: 32x32 image, 3 channels
    # Conv: 3->16, kernel=5, stride=1, padding=2

    # TODO: Calculate the output spatial dimension
    output_size = None  # Use the formula above

    assert output_size == 32  # With padding=2, size is preserved for 5x5 kernel

    # Now with kernel=5, stride=2, padding=0, input=32
    # TODO: Calculate
    output_size_2 = None

    assert output_size_2 == 14  # (32 + 0 - 5) / 2 + 1 = 14

    # After a 2x2 max pool on a 14x14 feature map
    # TODO: Calculate
    pooled_size = None

    assert pooled_size == 7
    print("exercise_6_output_size_formula passed!")


if __name__ == "__main__":
    exercise_1_conv1d()
    exercise_2_conv2d()
    exercise_3_pooling()
    exercise_4_cnn_architecture()
    exercise_5_batch_norm_dropout()
    exercise_6_output_size_formula()
    print("\nAll exercises in 05_cnn.py complete!")