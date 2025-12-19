from Core.Tensor import Tensor

def new_tensor(data=None, require_grad=False):
    if data is None:
        data = 0.
    return Tensor(data, require_grad=require_grad)

def randn(shape, *args, require_grad=False):
    if not isinstance(shape, tuple):
        shape = (shape, *args)
    return Tensor.randn(*shape, require_grad=require_grad)

def rand(shape, *args, require_grad=False):
    if not isinstance(shape, tuple):
        shape = (shape, *args)
    return Tensor.rand(shape, require_grad=require_grad)

__all__ = ["new_tensor", "randn", "rand"]