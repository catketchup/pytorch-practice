"""Solutions for Exercise 05: CNNs"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_conv1d():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight.data = torch.tensor([[[1.0, 0.0, -1.0]]])
    output = conv(x)
    assert output.shape == (1, 1, 4)
    print("exercise_1_conv1d passed!")


def exercise_2_conv2d():
    x = torch.randn(1, 1, 5, 5)
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=0, stride=1)
    out = conv(x)
    assert out.shape == (1, 4, 3, 3)
    conv_padded = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    out_padded = conv_padded(x)
    assert out_padded.shape == (1, 4, 5, 5)
    print("exercise_2_conv2d passed!")


def exercise_3_pooling():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_result = maxpool(x)
    assert max_result.shape == (1, 1, 2, 2)
    assert max_result.squeeze().tolist() == [[5.0, 7.0], [13.0, 15.0]]
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_result = avgpool(x)
    expected = [[2.5, 4.5], [10.5, 12.5]]
    for i in range(2):
        for j in range(2):
            assert abs(avg_result[0, 0, i, j].item() - expected[i][j]) < 1e-5
    print("exercise_3_pooling passed!")


def exercise_4_cnn_architecture():
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = MNISTNet()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1000
    print("exercise_4_cnn_architecture passed!")


def exercise_5_batch_norm_dropout():
    class RegularizedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = RegularizedCNN()
    model.train()
    x = torch.randn(2, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)
    assert out1.shape == (2, 10)
    assert not torch.allclose(out1, out2)
    model.eval()
    out3 = model(x)
    out4 = model(x)
    assert torch.allclose(out3, out4)
    print("exercise_5_batch_norm_dropout passed!")


def exercise_6_output_size_formula():
    output_size = (32 + 2 * 2 - 5) // 1 + 1
    assert output_size == 32
    output_size_2 = (32 + 2 * 0 - 5) // 2 + 1
    assert output_size_2 == 14
    pooled_size = 14 // 2
    assert pooled_size == 7
    print("exercise_6_output_size_formula passed!")


if __name__ == "__main__":
    exercise_1_conv1d()
    exercise_2_conv2d()
    exercise_3_pooling()
    exercise_4_cnn_architecture()
    exercise_5_batch_norm_dropout()
    exercise_6_output_size_formula()
    print("\nAll solutions in 05_cnn.py verified!")