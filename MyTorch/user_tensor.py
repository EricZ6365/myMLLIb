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

def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0

    result = []
    i = start

    if step > 0:
        while i < stop:
            result.append(i)
            i += step
    elif step < 0:
        while i > stop:
            result.append(i)
            i += step
    else:
        raise ValueError("step must not be zero")

    return Tensor(result)

__all__ = ["new_tensor", "randn", "rand", "arange"]