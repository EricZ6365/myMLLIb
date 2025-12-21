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
    return Tensor.rand(*shape, require_grad=require_grad)

def zeros(shape, *args, require_grad=False):
    if not isinstance(shape, tuple):
        shape = (shape, *args)

    return Tensor.zeros(*shape, require_grad=require_grad)

def ones(shape, *args, require_grad=False):
    if not isinstance(shape, tuple):
        shape = (shape, *args)

    return Tensor.zeros(*shape, require_grad=require_grad) + 1

def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0

    if step == 0:
        raise ValueError("step must not be zero")

    result = []
    i = start

    # Use a loop that avoids floating point accumulation errors
    n = 0
    if step > 0:
        while True:
            val = start + n * step
            if val >= stop:
                break
            result.append(val)
            n += 1
    else:
        while True:
            val = start + n * step
            if val <= stop:
                break
            result.append(val)
            n += 1

    return Tensor(result)


__all__ = ["new_tensor", "randn", "rand", "arange", "ones", "zeros"]