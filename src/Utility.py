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
        retain_graph=True,
        create_graph=True,)[0]
    return grad

def to_require_grad(tensor_list):
    """
    Converts a tensor to require gradients.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with requires_grad set to True.
    """
    return (t.clone().detach().requires_grad_(True) for t in tensor_list)