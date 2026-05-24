"""
Exercise 02: Autograd — Automatic Differentiation
===================================================
PyTorch's autograd engine automatically computes gradients for tensors
involved in computations. This is the backbone of neural network training.

Run this file. If all assertions pass, you've solved every exercise!
"""

import torch


def exercise_1_requires_grad():
    """Understand requires_grad and the computational graph."""

    # TODO: Create a tensor a = [2.0, 3.0] that tracks gradients
    a = None

    # TODO: Create a tensor b = [1.0, 1.0] that also tracks gradients
    b = None

    # TODO: Compute c = a * b + a
    c = None

    # TODO: Check if c requires grad
    c_requires_grad = None

    assert a.requires_grad == True
    assert b.requires_grad == True
    assert c_requires_grad == True
    print("exercise_1_requires_grad passed!")


def exercise_2_backward():
    """Compute gradients using .backward()."""

    # f(x) = x^2 + 2x + 1 (a simple quadratic)
    # df/dx = 2x + 2
    x = torch.tensor(3.0, requires_grad=True)

    # TODO: Compute y = x^2 + 2x + 1
    y = None

    # TODO: Call backward on y to compute gradients
    # YOUR CODE HERE

    assert abs(x.grad.item() - 8.0) < 1e-5  # 2(3) + 2 = 8
    print("exercise_2_backward passed!")


def exercise_3_multi_gradient():
    """Gradients accumulate! Understand how to handle this."""

    x = torch.tensor(1.0, requires_grad=True)

    y = x ** 2

    # TODO: Backprop to get gradient
    # YOUR CODE HERE

    assert x.grad.item() == 2.0  # dy/dx at x=1 is 2

    # Now do another forward + backward
    z = x ** 3

    # TODO: Without zeroing, what happens when we backward again?
    # YOUR CODE HERE

    # Gradients ACCUMULATE by default!
    # One solution: use .zero_() on the grad
    assert x.grad.item() == 5.0  # 2 (from y) + 3 (from z at x=1)

    # TODO: Zero the gradient on x
    # YOUR CODE HERE

    assert x.grad.item() == 0.0
    print("exercise_3_multi_gradient passed!")


def exercise_4_gradient_of_vector():
    """Backward on scalar vs. vector outputs."""

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x ** 2
    # y is a vector [1, 4, 9]

    # .backward() requires a scalar or a gradient (Jacobian-vector product).
    # TODO: Use .backward() with a gradient argument to compute dy/dx
    #       weighted by [1.0, 0.5, 0.1]
    # Hint: y.backward(gradient=<tensor>)
    # YOUR CODE HERE

    # Expected: dy/dx = 2x, weighted: [2*1*1, 2*2*0.5, 2*3*0.1] = [2, 2, 0.6]
    assert abs(x.grad[0].item() - 2.0) < 1e-5
    assert abs(x.grad[1].item() - 2.0) < 1e-5
    assert abs(x.grad[2].item() - 0.6) < 1e-5
    print("exercise_4_gradient_of_vector passed!")


def exercise_5_no_grad():
    """Use torch.no_grad() to skip gradient tracking."""

    x = torch.tensor(5.0, requires_grad=True)

    # TODO: Inside a torch.no_grad() context, compute z = x * 3
    with None:
        z = None

    # TODO: Check whether z requires grad
    z_requires_grad = None

    assert z_requires_grad == False
    print("exercise_5_no_grad passed!")


def exercise_6_detach():
    """Detach a tensor from the computation graph."""

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2

    # TODO: Create a new tensor from y that is detached from the graph
    y_detached = None

    # The detached tensor shares data but has no grad_fn
    assert y_detached.requires_grad == False
    assert y_detached.item() == 4.0

    # TODO: Now try computing gradients through the original y
    y.backward()
    assert x.grad.item() == 4.0  # dy/dx = 2x = 4

    print("exercise_6_detach passed!")


def exercise_7_custom_function():
    """Implement a custom autograd function.

    The function f(x) = x^3 has derivative f'(x) = 3x^2.
    We'll implement this as a custom Function.

    NOTE: Read through this to understand how autograd extends, then fill in
    the backward pass.
    """

    class Cubic(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # Save input for backward
            ctx.save_for_backward(x)
            return x ** 3

        @staticmethod
        def backward(ctx, grad_output):
            # TODO: Implement the backward pass.
            # The derivative of x^3 is 3*x^2.
            # You need to retrieve saved tensors and return the gradient.
            x, = ctx.saved_tensors
            # YOUR CODE HERE
            grad_input = None
            return grad_input

    x = torch.tensor(2.0, requires_grad=True)
    y = Cubic.apply(x)
    y.backward()

    assert abs(x.grad.item() - 12.0) < 1e-5  # 3 * 2^2 = 12
    print("exercise_7_custom_function passed!")


if __name__ == "__main__":
    exercise_1_requires_grad()
    exercise_2_backward()
    exercise_3_multi_gradient()
    exercise_4_gradient_of_vector()
    exercise_5_no_grad()
    exercise_6_detach()
    exercise_7_custom_function()
    print("\nAll exercises in 02_autograd.py complete!")