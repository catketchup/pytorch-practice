"""Solutions for Exercise 02: Autograd"""

import torch


def exercise_1_requires_grad():
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([1.0, 1.0], requires_grad=True)
    c = a * b + a
    c_requires_grad = c.requires_grad
    assert a.requires_grad == True
    assert b.requires_grad == True
    assert c_requires_grad == True
    print("exercise_1_requires_grad passed!")


def exercise_2_backward():
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1
    y.backward()
    assert abs(x.grad.item() - 8.0) < 1e-5
    print("exercise_2_backward passed!")


def exercise_3_multi_gradient():
    x = torch.tensor(1.0, requires_grad=True)
    y = x ** 2
    y.backward()
    assert x.grad.item() == 2.0
    z = x ** 3
    z.backward()
    assert x.grad.item() == 5.0
    x.grad.zero_()
    assert x.grad.item() == 0.0
    print("exercise_3_multi_gradient passed!")


def exercise_4_gradient_of_vector():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    y.backward(gradient=torch.tensor([1.0, 0.5, 0.1]))
    assert abs(x.grad[0].item() - 2.0) < 1e-5
    assert abs(x.grad[1].item() - 2.0) < 1e-5
    assert abs(x.grad[2].item() - 0.6) < 1e-5
    print("exercise_4_gradient_of_vector passed!")


def exercise_5_no_grad():
    x = torch.tensor(5.0, requires_grad=True)
    with torch.no_grad():
        z = x * 3
    z_requires_grad = z.requires_grad
    assert z_requires_grad == False
    print("exercise_5_no_grad passed!")


def exercise_6_detach():
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    y_detached = y.detach()
    assert y_detached.requires_grad == False
    assert y_detached.item() == 4.0
    y.backward()
    assert x.grad.item() == 4.0
    print("exercise_6_detach passed!")


def exercise_7_custom_function():
    class Cubic(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x ** 3

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            grad_input = 3 * x ** 2 * grad_output
            return grad_input

    x = torch.tensor(2.0, requires_grad=True)
    y = Cubic.apply(x)
    y.backward()
    assert abs(x.grad.item() - 12.0) < 1e-5
    print("exercise_7_custom_function passed!")


if __name__ == "__main__":
    exercise_1_requires_grad()
    exercise_2_backward()
    exercise_3_multi_gradient()
    exercise_4_gradient_of_vector()
    exercise_5_no_grad()
    exercise_6_detach()
    exercise_7_custom_function()
    print("\nAll solutions in 02_autograd.py verified!")