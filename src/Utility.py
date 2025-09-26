import torch

def calc_grad(y, x):
    """
    Calculates the gradient of a tensor y with respect to a tensor x.

    Returns:
        torch.Tensor: The gradient of y with respect to x.
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad