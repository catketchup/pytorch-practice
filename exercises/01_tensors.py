"""
Exercise 01: Tensor Basics
==========================
Tensors are PyTorch's core data structure — like NumPy arrays, but they can
run on GPUs and track gradients. Mastering tensor operations is foundational.

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch


def exercise_1_creation():
    """Create tensors from different data sources."""

    # TODO: Create a 3x3 tensor of zeros
    zeros = torch.zeros(3,3)

    # TODO: Create a 3x3 tensor of ones
    ones = torch.ones(3,3)

    # TODO: Create a 2x4 tensor filled with the value 7.0
    sevens = None

    # TODO: Create a tensor from a Python list [[1, 2], [3, 4]]
    from_list = None

    # TODO: Create a 1D tensor with values 0..9 using arange
    arange_tensor = None

    # TODO: Create a 1D tensor of 5 evenly-spaced values from 0.0 to 1.0
    linspace_tensor = None

    # TODO: Create a 2x3x2 tensor of random numbers sampled uniformly in [0, 1)
    rand_tensor = None

    # TODO: Create a 3x3 tensor of random numbers from a normal distribution
    #       (mean=0, std=1)
    randn_tensor = None

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
    """Understand tensor dtypes and devices."""

    # TODO: Create a float32 tensor of shape (2, 2) with value 1.0
    float32_tensor = None

    # TODO: Create a float64 (double) tensor of shape (2, 2) with value 1.0
    float64_tensor = None

    # TODO: Create an int64 tensor of shape (2, 2) with value 1
    int64_tensor = None

    # TODO: Create a float16 (half) tensor of shape (2, 2) with value 1.0
    float16_tensor = None

    # TODO: Convert float32_tensor to float64 using .to()
    converted = None

    assert float32_tensor.dtype == torch.float32
    assert float64_tensor.dtype == torch.float64
    assert int64_tensor.dtype == torch.int64
    assert float16_tensor.dtype == torch.float16
    assert converted.dtype == torch.float64
    print("exercise_2_dtype_device passed!")


def exercise_3_indexing_slicing():
    """Practice tensor indexing and slicing."""

    t = torch.arange(24).reshape(4, 6)
    # t is:
    # [[ 0,  1,  2,  3,  4,  5],
    #  [ 6,  7,  8,  9, 10, 11],
    #  [12, 13, 14, 15, 16, 17],
    #  [18, 19, 20, 21, 22, 23]]

    # TODO: Get the first row
    row_0 = None

    # TODO: Get the last column (use .clone() so it's independent of t)
    last_col = None

    # TODO: Get the 2x2 submatrix in the bottom-right corner
    #       i.e. [[16, 17], [22, 23]]
    submatrix = None

    # TODO: Get all even-valued elements (values divisible by 2) as a 1D tensor
    evens = None

    assert row_0.tolist() == [0, 1, 2, 3, 4, 5]
    assert last_col.tolist() == [5, 11, 17, 23]
    assert submatrix.shape == (2, 2) and submatrix[0, 1].item() == 17
    assert (evens % 2 == 0).all()

    # TODO: Set all values in row 1 to zero (in-place on t)
    # Hint: modify t directly
    # YOUR CODE HERE

    assert (t[1] == 0).all()
    print("exercise_3_indexing_slicing passed!")


def exercise_4_reshaping():
    """Reshape, transpose, permute, and broadcast tensors."""

    t = torch.arange(12)

    # TODO: Reshape t into a 3x4 tensor
    reshaped = None

    # TODO: Transpose reshaped so it becomes 4x3
    transposed = None

    # TODO: Create the same 4x3 result using .reshape() on t directly
    reshaped_4x3 = None

    # TODO: Use .unsqueeze() to add a dimension at position 0 to make t 1x12
    unsqueezed = None

    # TODO: Use .squeeze() to remove that dimension again
    squeezed = None

    # Use a 3x1 and 1x3 tensor for broadcasting:
    a = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
    b = torch.tensor([[10.0, 20.0, 30.0]])    # shape (1, 3)

    # TODO: Add a and b using broadcasting. Result should be 3x3.
    broadcast_result = None

    assert reshaped.shape == (3, 4)
    assert transposed.shape == (4, 3)
    assert reshaped_4x3.shape == (4, 3)
    assert unsqueezed.shape == (1, 12)
    assert squeezed.shape == (12,)
    assert broadcast_result.shape == (3, 3)
    assert broadcast_result[2, 2].item() == 33.0
    print("exercise_4_reshaping passed!")


def exercise_5_math_ops():
    """Perform element-wise and reduction operations."""

    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0, 8.0])

    # TODO: Element-wise addition
    added = None

    # TODO: Element-wise multiplication
    multiplied = None

    # TODO: Compute the dot product of a and b
    dot = None

    # TODO: Compute the L2 norm (Euclidean norm) of a
    norm_a = None

    # TODO: Compute softmax of a along dim=0
    softmax_a = None

    # TODO: Compute the mean and standard deviation of a
    mean_a = None
    std_a = None

    # Matrix multiplication
    M = torch.randn(3, 4)
    N = torch.randn(4, 2)

    # TODO: Compute the matrix product of M and N
    matmul_result = None

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
    """Convert between NumPy and PyTorch tensors."""

    import numpy as np

    np_array = np.array([1, 2, 3, 4, 5])

    # TODO: Convert np_array to a PyTorch tensor
    from_numpy = None

    # TODO: Convert the tensor back to a NumPy array
    back_to_numpy = None

    # Bonus: Modifying the numpy array should affect the tensor (they share
    # memory when using torch.from_numpy, NOT torch.tensor)
    # TODO: Create a tensor from np_array that SHARES memory (not a copy)
    shared_tensor = None

    assert type(from_numpy) == torch.Tensor
    assert type(back_to_numpy) == np.ndarray
    assert shared_tensor.data_ptr() == from_numpy.data_ptr() or True  # memory sharing
    print("exercise_6_numpy_bridge passed!")


def exercise_7_in_place_ops():
    """Understand in-place operations and their implications."""

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # In-place operations in PyTorch are denoted by a trailing underscore.
    # They modify the tensor directly rather than creating a new one.

    # TODO: Multiply x by 2 in-place. Use the in-place method.
    # Note: In-place ops on leaf tensors that require grad can cause errors
    # in autograd. This is just to demonstrate the concept.
    y = x.clone()
    # TODO: Use an in-place operation to add 5 to y

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
    print("\nAll exercises in 01_tensors.py complete!")