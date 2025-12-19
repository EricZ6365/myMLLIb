from collections import deque
from Tensor import Tensor
from binding import c_func


def _make_bound(org_function, grad, op_name, tensor):
    def bound_method(*args, **kwargs):
        return getattr(grad, "inner_" + op_name)(tensor, org_function, *args, **kwargs)
    return bound_method

def _matmul_derv(grad_output, a, b):
    return (
        grad_output.matmul(b.transpose(-2, -1)),
        a.transpose(-2, -1).matmul(grad_output),
    )

def _mul_derv(grad_output, a, b):
    return grad_output * b, grad_output * a

def _broadcast_derv(a, wa, shape):
    a_shape = list(a.shape)
    target_shape = list(shape)

    while len(a_shape) < len(target_shape):
        a_shape.insert(0, 1)
    while len(target_shape) < len(a_shape):
        target_shape.insert(0, 1)

    sum_dim = []
    for i, (ai, bi) in enumerate(zip(a_shape, target_shape)):
        if ai == 1 and bi != 1:
            sum_dim.append(i)
        elif ai == bi:
            continue
        else:
            raise IndexError(f"inapplicable shape at dim {i}: {ai} vs {bi}")

    for dim in reversed(sum_dim):
        wa = wa.sum(dim).unsqueeze(dim)


    return [wa]

class MulOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        return _mul_derv(grad_output, self.a, self.b)

class DivOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        grad_a = grad_output / self.b
        grad_b = grad_output * (-self.a / (self.b * self.b))
        return grad_a, grad_b

class PowOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        grad_a = grad_output * self.b * (self.a ** (self.b - 1))
        grad_b = grad_output * output * self.a.log()
        return grad_a, grad_b

class ExpOp:
    __slots__ = [
        "a",
        "inputs"
    ]
    def __init__(self, a):
        self.a = a
        self.inputs = [a]

    def back(self, grad_output, output):
        grad_a = grad_output * output
        return [grad_a]

class LnOp:
    __slots__ = [
        "a",
        "inputs"
    ]
    def __init__(self, a):
        self.a = a
        self.inputs = [a]

    def back(self, grad_output, output):
        return [grad_output / self.a]

class LogOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        grad_a = grad_output / (self.a * self.b.log())
        grad_b = grad_output * -(self.a.log() / (self.b * (self.b.log() ** 2)))
        return grad_a, grad_b

class MatMulOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        return _matmul_derv(grad_output, self.a, self.b)

class SumOp:
    __slots__ = [
        "a",
        "dim",
        "keepdims",
        "inputs"
    ]
    def __init__(self, inp, dim=None, keepdims=False):
        self.a = inp
        self.dim = dim
        self.keepdims = keepdims
        self.inputs = [self.a]

    def back(self, grad_output, output):
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor([grad_output])

        if self.dim is None:
            grad_input = Tensor([grad_output.data[0]] * len(self.a.data))
            grad_input.shape = self.a.shape
            grad_input.stride = grad_input.compute_stride(self.a.shape)
            return [grad_input]

        reshaped_grad = grad_output
        reshaped_grad.shape = grad_output.shape
        reshaped_grad.stride = reshaped_grad.compute_stride(reshaped_grad.shape)
        grad_input = reshaped_grad.broadcast(self.a.shape)
        return [grad_input]


class MeanOp:
    __slots__ = [
        "a",
        "inputs"
    ]
    def __init__(self, inp):
        self.a = inp
        self.inputs = [self.a]

    def back(self, grad_output, output):
        scale = Tensor([1 / len(self.a.data)] * len(self.a.data))
        scale.shape = self.a.shape
        scale.stride = self.a.compute_stride(self.a.shape)
        grad_input = scale * grad_output
        return [grad_input]


class TransposeOp:
    __slots__ = [
        "a",
        "inputs",
        "dim1",
        "dim2"
    ]
    def __init__(self, inp, dim1, dim2):
        self.a = inp
        self.inputs = [self.a]
        self.dim1 = dim1
        self.dim2 = dim2
    def back(self, grad_output, output):
        grad_input = grad_output.transpose(self.dim1, self.dim2)
        return [grad_input]
class MaxOp:
    __slots__ = [
        "a",
        "dim",
        "inputs"
    ]
    def __init__(self, inp, dim):
        self.a = inp
        self.dim = dim
        self.inputs = [self.a]

    def back(self, grad_output, output):
        if self.dim is None:
            max_val = max(self.a.data)
            mask = [1.0 if x == max_val else 0.0 for x in self.a.data]
            grad_input = Tensor([m * grad_output.data[0] for m in mask])
            grad_input.shape = self.a.shape
            grad_input.stride = self.a.compute_stride(self.a.shape)
            return [grad_input]
        else:
            size = len(self.a.data)

            shape = self.a.shape
            stride = self.a.stride
            dim = self.dim

            out_shape = list(shape)
            out_shape[dim] = 1

            grad_input_data = (ctypes.c_float * size)()

            block_size = shape[dim]
            num_blocks = size // block_size

            for block_idx in range(num_blocks):
                start = block_idx * block_size
                block = self.a.data[start:start + block_size]

                max_index = block.index(max(block))

                grad_val = grad_output.data[block_idx]

                grad_input_data[start + max_index] = grad_val

            result = Tensor.__new__(Tensor)
            result.data = grad_input_data
            result.shape = self.a.shape
            result.stride = self.a.compute_stride(self.a.shape)
            result.require_grad = self.a.require_grad
            return [result]

class SubOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        return grad_output, -grad_output

class AddOp:
    __slots__ = [
        "a",
        "b",
        "inputs"
    ]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inputs = [a, b]

    def back(self, grad_output, output):
        return grad_output, grad_output

class AbsOp:
    __slots__ = [
        "a",
        "inputs"
    ]
    def __init__(self, inp):
        self.a = inp
        self.inputs = [self.a]

    def back(self, grad_output, output):
        grad_input = []
        for i, val in enumerate(self.a.data):
            grad_input.append(grad_output.data[i] if val >= 0 else -grad_output.data[i])
        grad_input = Tensor(grad_input)
        grad_input.shape = self.a.shape
        grad_input.stride = self.a.compute_stride(self.a.shape)
        return [grad_input]


class ClampOp:
    __slots__ = [
        "a",
        "min_value",
        "max_value",
        "inputs"
    ]

    def __init__(self, inp, min_value, max_value):
        self.a = inp
        self.min_value = min_value
        self.max_value = max_value
        self.inputs = [self.a]

    def back(self, grad_output, output):
        grad_input = []
        for i, val in enumerate(self.a.data):
            if val <= self.min_value or val >= self.max_value:
                grad_input.append(0)
            else:
                grad_input.append(grad_output.data[i])

        grad_input = Tensor(grad_input)
        grad_input.shape = self.a.shape
        grad_input.stride = self.a.stride
        return [grad_input]

class BroadCastOp:
    __slots__ = [
        "a",
        "target_shape",
        "inputs"
    ]
    def __init__(self, inp, b):
        self.a = inp
        self.target_shape = b
        self.inputs = [inp]

    def back(self, grad_output, output):
        return _broadcast_derv(self.a, grad_output, self.target_shape)

class ReshapeOp:
    __slots__ = [
        "a",
        "new_shape",
        "inputs"
    ]
    def __init__(self, inp, new_shape):
        self.a = inp
        self.new_shape = new_shape
        self.inputs = [inp]

    def back(self, grad_output, output):
        return [grad_output.reshape(*self.a.shape)]

class IndexOp:
    __slots__ = [
        "a",
        "inputs",
        "slices"
    ]
    def __init__(self, inp, *slices):
        self.a = inp
        if len(slices) == 1 and isinstance(slices[0], int):
            self.slices = (slices[0], )
        else:
            self.slices = list(*slices)
        self.inputs = [inp]

    def back(self, grad_output, output):
        mask = Tensor.zeros(*self.a.shape)
        mask[*self.slices] = grad_output
        return [mask]

class UnsqueezeOp:
    __slots__ = [
        "a",
        "inputs",
        "dim"
    ]
    def __init__(self, inp, dim):
        self.a = inp
        self.inputs = [self.a]
        self.dim = dim

    def back(self, grad_output, output):
        return [grad_output.squeeze(self.dim)]

class SqueezeOp:
    __slots__ = [
        "a",
        "inputs",
        "dim"
    ]
    def __init__(self, inp, dim):
        self.a = inp
        self.inputs = [self.a]
        self.dim = dim

    def back(self, grad_output, output):
        return [grad_output.unsqueeze(self.dim)]

class ConcatOp:
    __slots__ = [
        "n",
        "inputs",
        "shapes",
        "dim"
    ]
    def __init__(self, tensors, dim):
        self.dim = dim
        self.inputs = tensors
        self.shapes = [t.shape for t in tensors]
        self.n = len(tensors)

    def back(self, grad_output, output):
        grads = []
        start = 0
        for i, shape in enumerate(self.shapes):
            if not shape:
                grads.append(grad_output[i])
                continue
            length = shape[self.dim]
            slc = [slice(None)] * 3
            slc[self.dim] = slice(start, start + length)
            grads.append(grad_output[tuple(slc)])
            start += length

        return tuple(grads)

class FlattenOp:
    __slots__ = [
        "a",
        "inputs",
        "dim"
    ]
    def __init__(self, inp, dim):
        self.a = inp
        self.inputs = [self.a]
        self.dim = dim

    def back(self, grad_output, output):
        res = Tensor.__new__(Tensor)
        res.data = grad_output.data
        res.shape = self.a.shape
        res.stride = Tensor.compute_stride(res.shape)
        res.require_grad = self.a.require_grad
        return [res]

class NegOp:
    __slots__ = [
        "a",
        "inputs"
    ]
    def __init__(self, a):
        self.a = a
        self.inputs = [a]

    def back(self, grad_output, output):
        return [-grad_output]


import ctypes
from functools import reduce
from operator import mul


class UnfoldOp:
    __slots__ = [
        "a",
        "inputs",
        "dim",
        "win_size",
        "step",
        "input_shape",
        "output_shape",
        "strides"
    ]

    def __init__(self, a, dim, win_size, step):
        self.a = a
        self.dim = dim
        self.win_size = win_size
        self.step = step
        self.inputs = [a]
        self.input_shape = a.shape
        self.strides = a.stride

        orig_dim_size = self.input_shape[dim]
        num_windows = (orig_dim_size - win_size) // step + 1
        self.output_shape = (
            *self.input_shape[:dim],
            num_windows,
            win_size,
            *self.input_shape[dim + 1:]
        )

    def back(self, grad_output, output):
        result = Tensor.__new__(Tensor)
        result.data = c_func["unfold_backward"](grad_output.data, grad_output.shape, self.a.shape,
                                                self.dim, self.win_size, self.step)
        result.shape = self.input_shape
        result.stride = Tensor.compute_stride(result.shape)
        result.require_grad = self.a.require_grad
        return [result]


class GradNode:
    __slots__ = [
        "op",
        "out",
        "grad_a",
        "children"
    ]
    def __init__(self, op, out):
        self.op = op
        self.out = out
        self.grad_a = None
        self.children = []

    def back(self):
        if self.op is None or self.grad_a is None:
            return

        grads = self.op.back(self.grad_a, self.out)
        for i, child in enumerate(self.op.inputs):
            if not isinstance(child, Tensor):
                continue
            if child.grad_node is None:
                continue
            if child.grad_node.grad_a is None:
                child.grad_node.grad_a = grads[i]
            else:
                child.grad_node.grad_a = child.grad_node.grad_a + grads[i]

    def __repr__(self):
        return f"grad: {self.grad_a}"

class Autograd:
    def __init__(self, tensors):
        self.tensor_to_node = {}
        self.no_track = False
        self.cache = set()
        for tensor in tensors:
            self.init_one_tensor(tensor)

    def init_one_tensor(self, tensor):
        self.bind_one_tensor(tensor)
        setattr(tensor, "grad_node", GradNode(None, None))

    def bind_one_tensor(self, tensor):
        bound_ops = {
            "matmul", "mul", "div", "sum", "sub", "pow", "exp", "log", "mean",
            "max", "abs", "neg", "clamp", "add", "broadcast", "index", "transpose",
            "unsqueeze", "concat", "flatten", "reshape", "unfold"
        }
        getattr_func = getattr
        setattr_func = setattr
        grad = self

        for op in bound_ops:
            org_function = getattr_func(tensor.__class__, op)
            setattr_func(tensor, op, _make_bound(org_function, grad, op, tensor))

    def cached_bind(self, result, node, *inputs):
        node.children = [inp.grad_node for inp in inputs if isinstance(inp, Tensor)]
        if result not in self.cache:
            self.tensor_to_node[result] = node
            self.cache.add(result)

        self.bind_one_tensor(result)
        result.grad_node = node

    def inner_reshape(self, T, org_function, *shapes):
        result = org_function(T, *shapes)
        if not self.no_track:
            node = GradNode(ReshapeOp(T, shapes), result)
            self.cached_bind(result, node, T, shapes)

        return result

    def inner_mul(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(MulOp(T, other), result)
            self.cached_bind(result, node, T, other)

        return result

    def inner_div(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(DivOp(T, other), result)
            self.cached_bind(result, node, T, other)
        return result

    def inner_add(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(AddOp(T, other), result)
            self.cached_bind(result, node, T, other)
        return result

    def inner_pow(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(PowOp(T, other), result)
            self.cached_bind(result, node, T, other)
        return result

    def inner_exp(self, T, org_function):
        result = org_function(T, )
        if not self.no_track:
            node = GradNode(ExpOp(T), result)
            self.cached_bind(result, node, T)
        return result

    def inner_log(self, T, org_function, other=None):
        result = org_function(T, other)
        if not self.no_track:
            if other is not None:
                node = GradNode(LogOp(T, other), result)
                self.cached_bind(result, node, T, other)
            else:
                node = GradNode(LnOp(T), result)
                self.cached_bind(result, node, T)

        return result

    def inner_matmul(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(MatMulOp(T, other), result)
            self.cached_bind(result, node, T, other)
        return result

    def inner_mean(self, T, org_function):
        result = org_function(T)
        if not self.no_track:
            node = GradNode(MeanOp(T), result)
            self.cached_bind(result, node, T)

        return result

    def inner_max(self, T, org_function, dim=None):
        result = org_function(T, dim)
        if not self.no_track:
            node = GradNode(MaxOp(T, dim), result)
            self.cached_bind(result, node, T)

        return result

    def inner_abs(self, T, org_function):
        result = org_function(T)
        if not self.no_track:
            node = GradNode(AbsOp(T), result)
            self.cached_bind(result, node, T)

        return result

    def inner_sum(self, T, org_function, dim):
        result = org_function(T, dim)
        if not self.no_track:
            node = GradNode(SumOp(T, dim), result)
            self.cached_bind(result, node, T)

        return result

    def inner_sub(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(SubOp(T, other), result)
            
            self.cached_bind(result, node, T, other)

        return result

    def inner_clamp(self, T, org_function, min_value, max_value):
        result = org_function(T, min_value, max_value)
        if not self.no_track:
            node = GradNode(ClampOp(T, min_value, max_value), result)
            self.cached_bind(result, node, T)

        return result

    def inner_broadcast(self, T, org_function, other):
        result = org_function(T, other)
        if not self.no_track:
            node = GradNode(BroadCastOp(T, other), result)
            self.cached_bind(result, node, T)

        return result

    def inner_index(self, T, org_function, *slices):
        result = org_function(T, *slices)
        if not self.no_track:
            node = GradNode(IndexOp(T, *slices), result)
            self.cached_bind(result, node, T)

        return result

    def inner_unsqueeze(self, T, org_function, dim):
        result = org_function(T, dim)
        if not self.no_track:
            node = GradNode(UnsqueezeOp(T, dim), result)
            self.cached_bind(result, node, T)

        return result

    def inner_squeeze(self, T, org_function, dim, ):
        result = org_function(T, dim)
        if not self.no_track:
            node = GradNode(SqueezeOp(T, dim), result)
            self.cached_bind(result, node, T)

        return result

    def inner_concat(self, T, org_function, *tensors, dim):
        result = org_function(T, *tensors, dim=dim)
        if not self.no_track:
            node = GradNode(ConcatOp([T] + list(tensors), dim), result)
            self.cached_bind(result, node, T, *tensors)

        return result

    def inner_flatten(self, T, org_function, dim):
        result = org_function(T, dim)
        if not self.no_track:
            node = GradNode(FlattenOp(T, dim), result)
            self.cached_bind(result, node, T)

        return result

    def inner_neg(self, T, org_function):
        result = org_function(T)
        if not self.no_track:
            node = GradNode(NegOp(T), result)
            self.cached_bind(result, node, T)

        return result

    def inner_unfold(self, T, org_function, dim, win_size, step):
        result = org_function(T, dim, win_size, step)
        if not self.no_track:
            node = GradNode(UnfoldOp(T, dim, win_size, step), result)
            self.cached_bind(result, node, T)

        return result

    def inner_transpose(self, T, org_function, dim1, dim2):
        result = org_function(T, dim1, dim2)
        if not self.no_track:
            node = GradNode(TransposeOp(T, dim1, dim2), result)
            self.cached_bind(result, node, T)

        return result

    def backward(self, loss_scaler, final_tensor):
        grad_node = getattr(final_tensor, "grad_node", None)
        if grad_node is None:
            raise ValueError("Final tensor has no grad_node")

        self.no_track = True

        grad_node.grad_a = loss_scaler

        queue = deque()
        queue.append(grad_node)

        visited = set()
        visited.add(grad_node)
        while queue:
            node = queue.pop()
            node.back()
            for child in node.children:
                if child is not None and child not in visited:
                    visited.add(child)
                    queue.append(child)


    def zero(self):
        for t, n in self.tensor_to_node.items():
            del t.grad_node
            del n
            t.release()
        self.no_track = False
        self.tensor_to_node.clear()