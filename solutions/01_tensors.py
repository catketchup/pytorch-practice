"""Solutions for Exercise 01: Tensor Basics"""

import torch


def exercise_1_creation():
    zeros = torch.zeros(3, 3)
    ones = torch.ones(3, 3)
    sevens = torch.full((2, 4), 7.0)
    from_list = torch.tensor([[1, 2], [3, 4]])
    arange_tensor = torch.arange(10)
    linspace_tensor = torch.linspace(0, 1, 5)
    rand_tensor = torch.rand(2, 3, 2)
    randn_tensor = torch.randn(3, 3)

    assert zeros.shape == (3, 3) and zeros.sum().item() == 0
    assert ones.shape == (3, 3) and ones.sum().item() == 9
    assert sevens.shape == (2, 4) and (sevens == 7.0).all()
    assert from_list.shape == (2, 2) and from_list[1, 0].item() == 3
    assert arange_tensor.shape == (10,) and arange_tensor[5].item() == 5
    assert linspace_tensor.shape == (5,) and abs(linspace_tensor[0].item()) < 1e-6
    assert rand_tensor.shape == (2, 3, 2)
    assert randn_tensor.shape == (3, 3)
    print("exercise_1_creation passed!")


def exercise_2_dtype_device():
    float32_tensor = torch.ones(2, 2, dtype=torch.float32)
    float64_tensor = torch.ones(2, 2, dtype=torch.float64)
    int64_tensor = torch.ones(2, 2, dtype=torch.int64)
    float16_tensor = torch.ones(2, 2, dtype=torch.float16)
    converted = float32_tensor.to(torch.float64)

    assert float32_tensor.dtype == torch.float32
    assert float64_tensor.dtype == torch.float64
    assert int64_tensor.dtype == torch.int64
    assert float16_tensor.dtype == torch.float16
    assert converted.dtype == torch.float64
    print("exercise_2_dtype_device passed!")


def exercise_3_indexing_slicing():
    t = torch.arange(24).reshape(4, 6)
    row_0 = t[0].clone()
    last_col = t[:, -1].clone()
    submatrix = t[2:, 4:].clone()
    evens = t[t % 2 == 0].clone()

    assert row_0.tolist() == [0, 1, 2, 3, 4, 5]
    assert last_col.tolist() == [5, 11, 17, 23]
    assert submatrix.shape == (2, 2) and submatrix[0, 1].item() == 17
    assert (evens % 2 == 0).all()

    t[1] = 0
    assert (t[1] == 0).all()
    print("exercise_3_indexing_slicing passed!")


def exercise_4_reshaping():
    t = torch.arange(12)
    reshaped = t.reshape(3, 4)
    transposed = reshaped.T
    reshaped_4x3 = t.reshape(4, 3)
    unsqueezed = t.unsqueeze(0)
    squeezed = unsqueezed.squeeze()

    a = torch.tensor([[1.0], [2.0], [3.0]])
    b = torch.tensor([[10.0, 20.0, 30.0]])
    broadcast_result = a + b

    assert reshaped.shape == (3, 4)
    assert transposed.shape == (4, 3)
    assert reshaped_4x3.shape == (4, 3)
    assert unsqueezed.shape == (1, 12)
    assert squeezed.shape == (12,)
    assert broadcast_result.shape == (3, 3)
    assert broadcast_result[2, 2].item() == 33.0
    print("exercise_4_reshaping passed!")


def exercise_5_math_ops():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    added = a + b
    multiplied = a * b
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    softmax_a = torch.softmax(a, dim=0)
    mean_a = a.mean()
    std_a = a.std()

    M = torch.randn(3, 4)
    N = torch.randn(4, 2)
    matmul_result = M @ N

    assert (added == torch.tensor([6.0, 8.0, 10.0, 12.0])).all()
    assert (multiplied == torch.tensor([5.0, 12.0, 21.0, 32.0])).all()
    assert abs(dot.item() - 70.0) < 1e-5
    assert abs(norm_a.item() - 5.4772) < 0.01
    assert abs(softmax_a.sum().item() - 1.0) < 1e-5
    assert abs(mean_a.item() - 2.5) < 1e-5
    assert abs(std_a.item() - 1.2909) < 0.01
    assert matmul_result.shape == (3, 2)
    print("exercise_5_math_ops passed!")


def exercise_6_numpy_bridge():
    import numpy as np
    np_array = np.array([1, 2, 3, 4, 5])
    from_numpy = torch.from_numpy(np_array)
    back_to_numpy = from_numpy.numpy()
    shared_tensor = torch.from_numpy(np_array)

    assert type(from_numpy) == torch.Tensor
    assert type(back_to_numpy) == np.ndarray
    print("exercise_6_numpy_bridge passed!")


def exercise_7_in_place_ops():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.clone()
    y.add_(5)
    assert (y == torch.tensor([6.0, 7.0, 8.0])).all()
    print("exercise_7_in_place_ops passed!")


if __name__ == "__main__":
    exercise_1_creation()
    exercise_2_dtype_device()
    exercise_3_indexing_slicing()
    exercise_4_reshaping()
    exercise_5_math_ops()
    exercise_6_numpy_bridge()
    exercise_7_in_place_ops()
    print("\nAll solutions in 01_tensors.py verified!")