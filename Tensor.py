import math
import random
from functools import reduce
from operator import mul

import visualize

_eps = 1e-12

class Tensor:
    def __init__(self, data, require_grad=True):
        self.data = self.flat(data)
        self.shape = self._get_shape(data)
        assert len(self.data) == reduce(mul, self.shape, 1)
        self.require_grad = require_grad
        self.grad_node = None
        self.stride = self.compute_stride(self.shape)

    def value(self):
        assert self.shape == (1, ) or self.shape == (), "must be a single element tensor to convert back to float"
        res = self.data[0]
        return res

    @staticmethod
    def flat(data):
        res = []
        if isinstance(data, (int, float)):
            return [data]
        if isinstance(data, (list, tuple)):
            for li in data:
                res.extend(Tensor.flat(li))
        return res

    @staticmethod
    def randn(*sizes, require_grad=True):
        randn_tensor = Tensor.__new__(Tensor)
        randn_tensor.data = [random.normalvariate() for i in range(reduce(mul, sizes, 1))]
        randn_tensor.shape = sizes
        randn_tensor.shape = sizes
        randn_tensor.stride = Tensor.compute_stride(sizes)
        randn_tensor.require_grad = require_grad
        return randn_tensor

    @staticmethod
    def rand(*sizes, require_grad=True):
        randn_tensor = Tensor.__new__(Tensor)
        randn_tensor.data = [random.random() for i in range(reduce(mul, sizes, 1))]
        randn_tensor.shape = sizes
        randn_tensor.stride = Tensor.compute_stride(sizes)
        randn_tensor.require_grad = require_grad
        return randn_tensor

    @staticmethod
    def zeros(*sizes, require_grad=True):
        randn_tensor = Tensor.__new__(Tensor)
        randn_tensor.data = [0 for i in range(reduce(mul, sizes, 1))]
        randn_tensor.shape = sizes
        randn_tensor.stride = Tensor.compute_stride(sizes)
        randn_tensor.require_grad = require_grad
        return randn_tensor

    @staticmethod
    def _get_shape(data):
        shape = []
        while isinstance(data, list):
            shape.append(len(data))
            if len(data) == 0:
                break
            data = data[0]
        return tuple(shape)

    @staticmethod
    def compute_stride(shape):
        stride = []
        acc = 1
        for size in reversed(shape):
            stride.insert(0, acc)
            acc *= size
        return tuple(stride)

    def get_grad(self):
        if self.grad_node is None:
            return None

        return self.grad_node.grad_a
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other):
        assert len(self.shape) == 2 and len(other.shape) == 2, "Only 2D matmul supported"
        assert self.shape[1] == other.shape[0], (f"Shapes not aligned for matmul "
                                                 f"source shape: {self.shape}, other shape: {other.shape}")

        out_rows, out_cols = self.shape[0], other.shape[1]
        result_data = [0] * (out_rows * out_cols)

        for i in range(out_rows):
            for j in range(out_cols):
                val = 0
                for k in range(self.shape[1]):
                    val += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j]
                result_data[i * out_cols + j] = val

        result = Tensor.__new__(Tensor)
        result.data = result_data
        result.shape = (out_rows, out_cols)
        result.stride = result.compute_stride(result.shape)
        result.require_grad = self.require_grad or other.require_grad
        return result

    def transpose(self, dim1, dim2):
        assert dim1 < len(self.shape) and dim2 < len(self.shape), "Dimension out of range"

        new_shape = list(self.shape)
        new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
        new_stride = list(self.stride)
        new_stride[dim1], new_stride[dim2] = new_stride[dim2], new_stride[dim1]
        new_data = [0] * len(self.data)

        def unravel_index(idx, shape):
            res = []
            for s in reversed(shape):
                res.insert(0, idx % s)
                idx //= s
            return res

        def ravel_index(indices, stride):
            return sum(i * s for i, s in zip(indices, stride))

        for new_flat_idx in range(len(new_data)):
            new_multi_idx = unravel_index(new_flat_idx, new_shape)
            old_multi_idx = list(new_multi_idx)
            old_multi_idx[dim1], old_multi_idx[dim2] = old_multi_idx[dim2], old_multi_idx[dim1]
            old_flat_idx = ravel_index(old_multi_idx, self.stride)
            new_data[new_flat_idx] = self.data[old_flat_idx]

        result = Tensor.__new__(Tensor)
        result.data = new_data
        result.shape = tuple(new_shape)
        result.stride = tuple(new_stride)
        result.require_grad = self.require_grad
        return result

    def unsqueeze(self, dim):
        result = Tensor.__new__(Tensor)
        result.data = self.data
        result.shape = tuple(list(self.shape[:dim]) + [1] + list(self.shape[dim:]))
        result.stride = self.compute_stride(self.shape)
        result.require_grad = self.require_grad

        return result


    def squeeze(self, dim):
        if self.shape[dim] == 1:
            result = Tensor.__new__(Tensor)
            result.data = self.data
            result.shape = tuple(list(self.shape[:dim]) + list(self.shape[dim + 1:]))
            result.stride = self.compute_stride(self.shape)
            result.require_grad = self.require_grad
            return result
        else:
            raise IndexError(f"dim {dim} is not singleton")

    def _getitem(self, *slices):
        if len(slices) == 1 and isinstance(slices[0], tuple):
            slices = slices[0]
        full_slices = list(slices) + [slice(None)] * (len(self.shape) - len(slices))

        norm_slices = []
        new_shape = []
        for i, s in enumerate(full_slices):
            if isinstance(s, slice):
                start, stop, step = s.start, s.stop, s.step
                dim = self.shape[i]
                step = 1 if step is None else step
                start = 0 if start is None else (start + dim if start < 0 else start)
                stop = dim if stop is None else (stop + dim if stop < 0 else stop)
                new_shape.append((stop - start + (step - 1)) // step)
            elif isinstance(s, int):
                if s < 0:
                    s = self.shape[i] + s
                start, stop, step = s, s + 1, 1
            else:
                raise IndexError("invalid index: ", s)
            norm_slices.append(slice(start, stop, step))

        def rec(i, offset):
            if i == len(norm_slices):
                return [self.data[offset]]
            s = norm_slices[i]
            step = s.step
            acc = []
            for j in range(s.start, s.stop, step):
                acc.extend(rec(i + 1, offset + j * self.stride[i]))
            return acc

        result = Tensor.__new__(Tensor)
        result.data = rec(0, 0)
        result.shape = tuple(new_shape)
        result.stride = result.compute_stride(result.shape)
        result.require_grad = self.require_grad
        return result

    def flatten(self, dim=1):
        result = Tensor.__new__(Tensor)
        result.data = self.data
        if dim == -1:
            dim += dim + len(self.shape)

        flat_size = 1
        for d in self.shape[dim:]:
            flat_size *= d

        result.shape = tuple(list(self.shape[:dim]) + [flat_size])
        result.stride = result.compute_stride(result.shape)
        result.require_grad = self.require_grad
        return result

    def _setitem(self, slices, value):
        if not isinstance(slices, tuple):
            slices = (slices,)

        full_slices = list(slices) + [slice(None)] * (len(self.shape) - len(slices))
        norm_slices = []
        shape = []
        for i, s in enumerate(full_slices):
            if isinstance(s, slice):
                start, stop, step = s.start, s.stop, s.step
                dim = self.shape[i]
                step = 1 if step is None else step
                start = 0 if start is None else (start + dim if start < 0 else start)
                stop = dim if stop is None else (stop + dim if stop < 0 else stop)
                norm_slices.append(slice(start, stop, step))
                shape.append((stop - start + (step - 1)) // step)
            elif isinstance(s, int):
                if s < 0:
                    s = self.shape[i] + s
                start, stop, step = s, s + 1, 1
                norm_slices.append(slice(start, stop, step))
            else:
                raise IndexError("invalid index: ", s)
        shape = tuple(shape)

        if isinstance(value, Tensor):
            if not ((value.shape == shape) or
                    (value.shape == () and shape == (1,)) or
                    (value.shape == (1,) and shape == (1,))):
                raise ValueError(f"shape mismatch: {value.shape} vs {shape}")
            value_data = value.data
        else:
            value_data = [value]

        def rec(i, offset, val_offset):
            if i == len(norm_slices):
                self.data[offset] = value_data[val_offset % len(value_data)]
                return val_offset
            s = norm_slices[i]
            for j in range(s.start, s.stop, s.step):
                val_offset = rec(i + 1, offset + j * self.stride[i], val_offset)
            return val_offset

        rec(0, 0, 0)

    def neg(self):
        return self._unary_op(lambda x: -x)

    def abs(self):
        return self._unary_op(abs)

    def exp(self):
        return self._unary_op(math.exp)

    def _unary_op(self, func):
        result = Tensor.__new__(Tensor)
        result.data = [func(x) for x in self.data]

        result.shape = self.shape
        result.stride = self.compute_stride(self.shape)
        result.require_grad = self.require_grad
        return result

    def __pow__(self, power, modulo=None): self._bind_broadcast(power, "pow") if power.shape != self.shape else self.pow(power)
    def __neg__(self): return self.neg()
    def __abs__(self): return self.abs()
    def __add__(self, other): return self._bind_broadcast(other, "add") if other.shape != self.shape else self.add(other)
    def __sub__(self, other): return self._bind_broadcast(other, "sub") if other.shape != self.shape else self.sub(other)
    def __mul__(self, other): return self._bind_broadcast(other, "mul") if other.shape != self.shape else self.mul(other)
    def __truediv__(self, other): return self._bind_broadcast(other, "div") if other.shape != self.shape else self.div(other)
    def __getitem__(self, *args, **kwargs): return self._getitem(*args)
    def __setitem__(self, key, value): return self._setitem(key, value)

    @staticmethod
    def _bi_broadcast(a, b, op_name):
        def _inner(a, b):
            sa, sb = tuple(a.shape), tuple(b.shape)
            max_len = max(len(sa), len(sb))

            for _ in range(max_len - len(sa)):
                a = a.unsqueeze(0)
                sa = a.shape

            for _ in range(max_len - len(sb)):
                b = b.unsqueeze(0)
                sb = b.shape

            for a_axis, b_axis in zip(sa, sb):
                if a_axis != b_axis:
                    if a_axis == 1:
                        return a.broadcast(b.shape), b
                    elif b_axis == 1:
                        return a, b.broadcast(a.shape)
                    else:
                        raise ValueError(f"Incompatible broadcast shapes: {a.shape} and {b.shape}")

            return a, b

        a_norm, b_norm = _inner(a, b)
        return getattr(a_norm, op_name)(b_norm)

    def _bind_broadcast(self, other, op_name):
        return Tensor._bi_broadcast(self, other, op_name)


    def add(self, other):

        return self._binary_op(other, lambda a, b: a + b)

    def sub(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def mul(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def div(self, other):
        return self._binary_op(other, lambda a, b: a / b if b != 0 else 1.)

    def pow(self, other):
        return self._binary_op(other, lambda a, b: a ** b)

    def log(self, other=None):
        if other is None:
            return self._unary_op(lambda a: math.log(max(a, 1e-6)))  # natural log

        if not hasattr(other, "shape"):
            safe_data = self.clamp(1e-6, float("inf"))
            return safe_data._unary_op(lambda a: math.log(a) / math.log(other))

        safe_data = self.clamp(1e-6, float("inf"))
        return safe_data._binary_op(other, lambda a, b: math.log(a) / math.log(max(b, 1e-6)))

    def broadcast(self, shape):
        shape_self = list(self.shape)
        shape_other = list(shape)
        ndim_self = len(shape_self)
        ndim_other = len(shape_other)

        if ndim_self < ndim_other:
            shape_self = shape_self + [1] * (ndim_other - ndim_self)
        elif ndim_other < ndim_self:
            shape_other = shape_other + [1] * (ndim_self - ndim_other)

        out_shape = []
        for s, o in zip(shape_self, shape_other):
            if s == o or s == 1 or o == 1:
                out_shape.append(max(s, o))
            else:
                raise ValueError(f"Incompatible broadcast shapes: {self.shape} and {shape}")

        def expand_data(data, shape, out_shape):
            if shape == out_shape:
                return data
            if len(shape) == 1:
                if shape[0] == 1:
                    return data * out_shape[0]
                return data
            size = shape[0]
            sub_shape = shape[1:]
            sub_out_shape = out_shape[1:]
            expanded_subs = []
            sub_size = len(data) // size
            for i in range(size):
                sub_data = data[i * sub_size:(i + 1) * sub_size]
                expanded_subs.append(expand_data(sub_data, sub_shape, sub_out_shape))
            expanded_data = expanded_subs * (out_shape[0] // size)
            return [item for sub in expanded_data for item in sub]

        new_data = expand_data(self.data, shape_self, out_shape)

        result = Tensor.__new__(Tensor)
        result.data = new_data
        result.shape = tuple(out_shape)
        result.stride = self.compute_stride(result.shape)
        result.require_grad = self.require_grad
        return result

    def _binary_op(self, other, func):
        result = Tensor.__new__(Tensor)

        def apply_func(a, b):
            return func(a, b)

        if isinstance(other, Tensor):
            if self.shape == other.shape:
                result.data = [apply_func(a, b) for a, b in zip(self.data, other.data)]
            else:
                raise ValueError(f"Incompatible broadcast shapes: {self.shape} and {other.shape}")
        else:
            result.data = [apply_func(a, other) for a in self.data]

        result.require_grad = self.require_grad
        result.shape = self.shape
        result.stride = self.compute_stride(result.shape)
        return result

    def sum(self, dim=None):
        if dim is None:
            result = Tensor.__new__(Tensor)
            result.data = [sum(self.data)]
            result.shape = ()
            result.stride = ()
            result.require_grad = self.require_grad
            return result

        if dim < 0:
            dim += len(self.shape)

        assert 0 <= dim < len(self.shape), f"dim {dim} is out of bounds for shape {self.shape}"

        new_shape = self.shape[:dim] + self.shape[dim + 1:]

        dim_size = self.shape[dim]
        inner_block = 1
        for s in self.shape[dim + 1:]:
            inner_block *= s
        outer_block = 1
        for s in self.shape[:dim]:
            outer_block *= s

        result_data = []
        for outer in range(outer_block):
            base = outer * dim_size * inner_block
            for inner in range(inner_block):
                val = 0
                for a in range(dim_size):
                    val += self.data[base + a * inner_block + inner]
                result_data.append(val)

        result = Tensor.__new__(Tensor)
        result.data = result_data
        result.shape = tuple(new_shape) if new_shape else ()
        result.stride = self.compute_stride(result.shape) if new_shape else ()
        result.require_grad = self.require_grad
        return result

    def mean(self):
        result = Tensor.__new__(Tensor)
        result.data = [sum(self.data) / len(self.data)]
        result.shape = ()
        result.stride = ()
        result.require_grad = self.require_grad
        return result

    def argmax(self):
        pass

    def max(self, dim=None):
        result = Tensor.__new__(Tensor)
        if dim is None:
            result.data = [max(self.data)]
            result.shape = ()
            result.stride = ()
        else:
            inner_block = self.stride[dim - 1] if dim > 0 else 1
            step = self.shape[dim]
            outer_size = len(self.data) // inner_block
            inner_size = inner_block // step

            max_values = []
            for i in range(outer_size):
                for j in range(0, inner_block, step):
                    start = i * inner_block + j
                    block = self.data[start:start + step]
                    max_values.append(max(block))

            result.data = max_values
            result.shape = self.shape[:dim] + self.shape[dim + 1:]
            result.stride = self.stride[:dim] + self.stride[dim + 1:]

        result.require_grad = self.require_grad
        return result

    def min(self):
        result = Tensor.__new__(Tensor)
        result.data = min(self.data)
        result.shape = ()
        result.stride = ()
        result.require_grad = self.require_grad
        return result

    def clamp(self, min_value=float("-inf"), max_value=float("inf")):
        clamped_data = []

        for val in self.data:
            if min_value is not None and val < min_value:
                val = min_value
            if max_value is not None and val > max_value:
                val = max_value
            clamped_data.append(val)

        result = Tensor.__new__(Tensor)
        result.data = clamped_data
        result.require_grad = self.require_grad
        result.shape = self.shape
        result.stride = self.stride

        return result


    def concat(self, *tensors, dim=0):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]

        if len(tensors) == 0:
            raise ValueError("concat expects at least one tensor")

        tensors = [self] + list(tensors)

        base_shape = tensors[0].shape
        rank = len(base_shape)

        if rank == 0:
            rank += 1
            tensors = [t.unsqueeze(0) for t in tensors]
            base_shape = tuple([1] + list(base_shape))

        if dim < 0:
            dim += rank


        for t in tensors:
            if len(t.shape) != rank:
                raise ValueError("All tensors must have the same rank")
            for d in range(rank):
                if d == dim:
                    continue
                if t.shape[d] != base_shape[d]:
                    raise ValueError(
                        f"Sizes of tensors must match except in dimension {dim}. "
                        f"Expected {base_shape[d]} at dim {d}, got {t.shape[d]}."
                    )

        if not (0 <= dim < rank):
            raise ValueError(f"dim out of range (got {dim}, expected in [{-rank}, {rank - 1}])")

        out_shape_list = list(base_shape)
        out_shape_list[dim] = sum(t.shape[dim] for t in tensors)
        out_shape = tuple(out_shape_list)

        def prod(it):
            p = 1
            for x in it:
                p *= x
            return p

        outer = prod(base_shape[:dim]) if dim > 0 else 1
        inner = prod(base_shape[dim + 1:]) if dim < rank - 1 else 1

        out_numel = prod(out_shape)
        out_data = [0] * out_numel

        outer_stride_out = out_shape_list[dim] * inner
        dst_base = 0
        for o in range(outer):
            dst_ptr = o * outer_stride_out
            for t in tensors:
                size_d = t.shape[dim]
                block = size_d * inner
                src_ptr = o * (t.shape[dim] * inner)
                out_data[dst_ptr:dst_ptr + block] = t.data[src_ptr:src_ptr + block]
                dst_ptr += block

        result = Tensor.__new__(Tensor)
        result.data = out_data
        result.shape = out_shape
        result.stride = result.compute_stride(out_shape)
        result.require_grad = any(getattr(t, "require_grad", False) for t in tensors)

        return result


