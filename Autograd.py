import copy
import math
import random

from matplotlib import pyplot as plt

import visualize
from Tensor import Tensor


def _matmul_derv(grad_output, a, b):
    return (
        grad_output.matmul(b.transpose(0, 1)),
        a.transpose(0, 1).matmul(grad_output),
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

    while len(wa.shape) > len(a.shape):
        wa = wa.squeeze(0)
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

        dim = self.dim
        if isinstance(dim, int):
            if dim < 0:
                dim += len(self.a.shape)
            dim = [dim]

        grad_shape = list(self.a.shape)

        for ax in dim:
            grad_shape[ax] = 1
        reshaped_grad = grad_output
        reshaped_grad.shape = tuple(grad_shape)
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

            grad_input_data = [0.0] * size

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
            if self.min_value < val < self.max_value:
                grad_input.append(grad_output.data[i])
            else:
                grad_input.append(0)

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
        return [grad_output.unqueeze(self.dim)]

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
        for tensor in tensors:
            self.init_one_tensor(tensor)

    def init_one_tensor(self, tensor):
        self.bind_one_tensor(tensor)
        setattr(tensor, "grad_node", GradNode(None, None))

    def bind_one_tensor(self, tensor):
        _hook = self._hook_to_op
        bound_ops = {
            "matmul", "mul", "div", "sum", "sub", "pow", "exp", "log", "mean",
            "max", "abs", "neg", "clamp", "add", "broadcast", "_getitem",
            "unsqueeze", "concat", "flatten"
        }
        setattr_func = setattr
        for op in bound_ops:
            setattr_func(tensor, op, _hook(tensor, op))

    def _hook_to_op(self, T, op_name):
        def inner_mul(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(MulOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)
                if isinstance(other, Tensor):
                    node.children.append(other.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
                result.grad_node = node
            return result

        def inner_div(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(DivOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)
                if isinstance(other, Tensor):
                    node.children.append(other.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
                result.grad_node = node
            return result

        def inner_add(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(AddOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)
                if isinstance(other, Tensor):
                    node.children.append(other.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
                result.grad_node = node
            return result

        def inner_pow(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(PowOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)
                if isinstance(other, Tensor):
                    node.children.append(other.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
                result.grad_node = node
            return result

        def inner_exp(**kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, **kwargs)
            if not self.no_track:
                node = GradNode(ExpOp(T), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_log(other=None, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                if other is not None:
                    node = GradNode(LogOp(T, other), result)
                    result.grad_node = node

                    node.children.append(T.grad_node)
                    if isinstance(other, Tensor):
                        node.children.append(other.grad_node)

                    self.bind_one_tensor(result)
                    self.tensor_to_node[result] = node
                    result.grad_node = node
                else:
                    node = GradNode(LnOp(T), result)
                    result.grad_node = node

                    node.children.append(T.grad_node)
                    self.bind_one_tensor(result)
                    self.tensor_to_node[result] = node
                    result.grad_node = node
            return result

        def inner_matmul(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(MatMulOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)
                if isinstance(other, Tensor):
                    node.children.append(other.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
                result.grad_node = node
            return result

        def inner_mean(*args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, *args, **kwargs)
            if not self.no_track:
                node = GradNode(MeanOp(T), result)
                result.grad_node = node
                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
            return result

        def inner_max(dim=None, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, dim, *args, **kwargs)
            if not self.no_track:
                node = GradNode(MaxOp(T, dim), result)
                result.grad_node = node
                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
            return result

        def inner_abs(*args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, *args, **kwargs)
            if not self.no_track:
                node = GradNode(AbsOp(T), result)
                result.grad_node = node
                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_sum(dim, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, dim, *args, **kwargs)
            if not self.no_track:
                node = GradNode(SumOp(T, dim), result)
                result.grad_node = node
                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_sub(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(SubOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)
                if isinstance(other, Tensor):
                    node.children.append(other.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_clamp(min_value, max_value, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, min_value, max_value, *args, **kwargs)
            if not self.no_track:
                node = GradNode(ClampOp(T, min_value, max_value), result)
                result.grad_node = node
                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_broadcast(other, *args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, other, *args, **kwargs)
            if not self.no_track:
                node = GradNode(BroadCastOp(T, other), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
            return result

        def inner_index(*slices, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, *slices, **kwargs)
            if not self.no_track:
                node = GradNode(IndexOp(T, *slices), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_unsqueeze(dim, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, dim, **kwargs)
            if not self.no_track:
                node = GradNode(UnsqueezeOp(T, dim), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_squeeze(dim, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, dim, **kwargs)
            if not self.no_track:
                node = GradNode(SqueezeOp(T, dim), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node

            return result

        def inner_concat(*tensors, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, *tensors, **kwargs)
            if not self.no_track:
                node = GradNode(ConcatOp([T] + list(tensors), **kwargs), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
            return result

        def inner_flatten(dim, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, dim, **kwargs)
            if not self.no_track:
                node = GradNode(FlattenOp(T, dim, **kwargs), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
            return result

        def inner_neg(*args, **kwargs):
            org_function = getattr(T.__class__, op_name)
            result = org_function(T, *args, **kwargs)
            if not self.no_track:
                node = GradNode(NegOp(T, *args, **kwargs), result)
                result.grad_node = node

                node.children.append(T.grad_node)

                self.bind_one_tensor(result)
                self.tensor_to_node[result] = node
            return result

        if op_name == "matmul":
            return inner_matmul

        elif op_name == "sum":
            return inner_sum

        elif op_name == "max":
            return inner_max

        elif op_name == "mul":
            return inner_mul

        elif op_name == "div":
            return inner_div

        elif op_name == "sub":
            return inner_sub

        elif op_name == "mean":
            return inner_mean

        elif op_name == "abs":
            return inner_abs

        elif op_name == "clamp":
            return inner_clamp

        elif op_name == "add":
            return inner_add

        elif op_name == "broadcast":
            return inner_broadcast

        elif op_name == "_getitem":
            return inner_index

        elif op_name == "unsqueeze":
            return inner_unsqueeze

        elif op_name == "squeeze":
            return inner_squeeze

        elif op_name == "concat":
            return inner_concat

        elif op_name == "flatten":
            return inner_flatten

        elif op_name == "pow":
            return inner_pow

        elif op_name == "log":
            return inner_log

        elif op_name == "neg":
            return inner_neg

        elif op_name == "exp":
            return inner_exp

    def backward(self, loss_scalar, final_tensor):
        if not hasattr(final_tensor, "grad_node"):
            raise ValueError("Final tensor has no grad_node")

        self.no_track = True
        final_tensor.grad_node.grad_a = loss_scalar

        queue = [final_tensor.grad_node]
        visited = set()

        while queue:
            node = queue.pop()
            if node in visited or node is None:
                continue
            visited.add(node)
            node.back()
            for child in node.children:
                queue.append(child)

        self.no_track = False

    def zero(self):
        for t, n in self.tensor_to_node.items():
            del t.grad_node
            t.release()
        self.no_track = True
        self.tensor_to_node.clear()
