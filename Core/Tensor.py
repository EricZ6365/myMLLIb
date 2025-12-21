import ctypes
import random
import weakref
from Binding.binding import c_func
from functools import reduce
from operator import mul

class Pool:
    pool = set()
    in_use = weakref.WeakSet()

_eps = 1e-12

class Tensor:
    __slots__ = [
        "data",
        "shape",
        "require_grad",
        "grad_node",
        "stride",
        "velocity",
        "__weakref__",
        "__dict__",
    ]

    def __new__(cls, *args, **kwargs):
        if len(Pool.pool) > 0:
            tensor = Pool.pool.pop()
        else:
            tensor = super().__new__(Tensor)

        Pool.in_use.add(tensor)
        return tensor

    def release(self):
        if self in Pool.in_use:
            Pool.in_use.remove(self)
            Pool.pool.add(self)
        else:
            print("Tensor not in use, cannot release")
        del self.data

    def __init__(self, data, require_grad=True):
        self.data = self.to_c(self.flat(data))
        self.shape = self._get_shape(data)
        assert len(self.data) == reduce(mul, self.shape, 1)
        self.require_grad = require_grad
        if require_grad:
            self.grad_node = None
        self.stride = self.compute_stride(self.shape)

    def value(self):
        total = reduce(mul, self.shape, 1)
        assert total == 1;
        res = self.data[0]
        return res

    def to_list(self):
        res = []
        def build(i, offset, ctx):
            for idx in range(self.shape[i]):
                if i == len(self.shape) - 1:
                    ctx.append(float(self.data[offset + idx]))
                else:
                    new_ctx = []
                    build(i + 1, offset + idx * self.stride[i], new_ctx)
                    ctx.append(new_ctx)

            return ctx

        build(0, 0, res)
        return res

    @staticmethod
    def to_c(data):
        return (ctypes.c_float * len(data))(*data)

    @staticmethod
    def flat(data):
        res = []
        try:
            iterator = iter(data)
        except TypeError:
            return [float(data)]
        else:
            for item in iterator:
                res.extend(Tensor.flat(item))
        return res

    @staticmethod
    def randn(*sizes, require_grad=True):
        randn_tensor = Tensor.__new__(Tensor)
        total = reduce(mul, sizes, 1)
        randn_tensor.data  = c_func["randn"](total)
        randn_tensor.shape = sizes
        randn_tensor.shape = sizes
        randn_tensor.stride = Tensor.compute_stride(sizes)
        randn_tensor.require_grad = require_grad
        return randn_tensor

    @staticmethod
    def rand(*sizes, require_grad=True):
        randn_tensor = Tensor.__new__(Tensor)
        total = reduce(mul, sizes, 1)
        out = (ctypes.c_float * total)()
        for i in range(total):
            out[i] = random.random()
        randn_tensor.data = out
        randn_tensor.shape = sizes
        randn_tensor.stride = Tensor.compute_stride(sizes)
        randn_tensor.require_grad = require_grad
        return randn_tensor

    @staticmethod
    def zeros(*sizes, require_grad=True):
        randn_tensor = Tensor.__new__(Tensor)
        total = reduce(mul, sizes, 1)
        randn_tensor.data = (ctypes.c_float * total)()
        randn_tensor.shape = sizes
        randn_tensor.stride = Tensor.compute_stride(sizes)
        randn_tensor.require_grad = require_grad
        return randn_tensor

    @staticmethod
    def _get_shape(data):
        shape = []
        while isinstance(data, list) or isinstance(data, tuple):
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
        return f"Tensor(shape={self.shape}, data_length={len(self.data)}, data={list(self.data[:5]) if len(self.data) > 5 else list(self.data)})"

    def matmul(self, other):
        assert self.shape[-1] == other.shape[-2], (
            f"Shapes not aligned for matmul: {self.shape} x {other.shape}"
        )

        # [*, M, K] x [*, K, N] => [*, M, N]
        m = self.shape[-2]
        n = other.shape[-1]

        batch_shape = self.shape[:-2] if len(self.shape) > 2 else ()
        result_shape = tuple(batch_shape + (m, n))

        out_data = c_func["matmul"](self.data, other.data, self.shape, other.shape)

        result = Tensor.__new__(Tensor)
        result.data = out_data
        result.shape = result_shape
        result.stride = Tensor.compute_stride(result.shape)
        result.require_grad = self.require_grad or other.require_grad
        return result

    def transpose(self, dim1, dim2):
        assert dim1 < len(self.shape) and dim2 < len(self.shape), "Dimension out of range"
        if dim1 < 0:
            dim1 += len(self.shape)

        if dim2 < 0:
            dim2 += len(self.shape)

        new_shape = list(self.shape)
        new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
        new_data = c_func["transpose"](self.data, self.shape, dim1, dim2)

        result = Tensor.__new__(Tensor)
        result.data = new_data
        result.shape = tuple(new_shape)
        result.stride = Tensor.compute_stride(new_shape)
        result.require_grad = self.require_grad
        return result

    def unsqueeze(self, dim):
        if dim < 0:
            dim += len(self.shape) + 1

        result = Tensor.__new__(Tensor)
        result.data = self.data
        result.shape = tuple(list(self.shape[:dim]) + [1] + list(self.shape[dim:]))
        result.stride = self.compute_stride(self.shape)
        result.require_grad = self.require_grad

        return result


    def squeeze(self, dim):
        if dim < 0:
            dim += len(self.shape)
        if self.shape[dim] == 1:
            result = Tensor.__new__(Tensor)
            result.data = self.data
            result.shape = tuple(list(self.shape[:dim]) + list(self.shape[dim + 1:]))
            result.stride = self.compute_stride(self.shape)
            result.require_grad = self.require_grad
            return result
        else:
            raise IndexError(f"dim {dim} is not singleton")

    def index(self, *slices):
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
                start = max(0, min(start, dim))
                stop = max(0, min(stop, dim))

                size = (stop - start + (step - 1)) // step
                new_shape.append(size)

            elif isinstance(s, int):
                if s < 0:
                    s = self.shape[i] + s
                start, stop, step = s, s + 1, 1
            else:
                raise IndexError("invalid index: ", s)
            norm_slices.append(slice(start, stop, step))


        result = Tensor.__new__(Tensor)
        result.data = c_func["index"](self.data, self.shape, new_shape, norm_slices)
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
        return self._unary_op(c_func["neg"])

    def abs(self):
        return self._unary_op(c_func["abs"])

    def exp(self):
        return self._unary_op(c_func["exp"])

    def _unary_op(self, func):
        result = Tensor.__new__(Tensor)
        result.data = func(self.data, len(self.shape))

        result.shape = self.shape
        result.stride = self.compute_stride(self.shape)
        result.require_grad = self.require_grad
        return result

    def __pow__(self, power, modulo=None): return self._bind_broadcast(power, "pow")
    def __neg__(self): return self.neg()
    def __abs__(self): return self.abs()
    def __add__(self, other): return self._bind_broadcast(other, "add")
    def __sub__(self, other): return self._bind_broadcast(other, "sub")
    def __mul__(self, other): return self._bind_broadcast(other, "mul")
    def __truediv__(self, other): return self._bind_broadcast(other, "div")
    def __getitem__(self, *args, **kwargs): return self.index(*args)
    def __setitem__(self, key, value): return self._setitem(key, value)
    def __matmul__(self, other): return self._bind_matmul_broadcast(other)

    @staticmethod
    def _bi_broadcast(a, b, op_name):
        func = getattr(a, op_name)

        if not isinstance(b, Tensor) or a.shape == b.shape:
            return func(b)

        def _inner(a, b):
            sa, sb = tuple(a.shape), tuple(b.shape)
            max_len = max(len(sa), len(sb))
            pad_a = [1] * (max_len - len(sa))
            pad_b = [1] * (max_len - len(sb))

            a.shape = tuple(pad_a + list(a.shape))
            a.stride = a.compute_stride(a.shape)
            b.shape = tuple(pad_b + list(b.shape))
            b.stride = b.compute_stride(b.shape)

            for a_axis, b_axis in zip(a.shape, b.shape):
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

    @staticmethod
    def _matmul_broadcast(a, b):
        func = getattr(a, "matmul")
        if len(a.shape) == len(b.shape):
            return func(b)

        def _inner(a, b):
            sa, sb = tuple(a.shape), tuple(b.shape)

            if len(sa) < 2:
                sa = tuple([1] + list(sa))
            if len(sb) < 2:
                sb = tuple(list(sb) + [1])

            if sa[-1] != sb[-2]:
                raise ValueError(
                    f"Incompatible matmul dimensions: {sa[-1]} and {sb[-2]}"
                )

            batch_a, mat_a = sa[:-2], sa[-2:]
            batch_b, mat_b = sb[:-2], sb[-2:]

            len_diff = len(batch_b) - len(batch_a)
            pad_a = [1] * (len_diff if len_diff > 0 else 0)
            pad_b = [1] * (-len_diff if len_diff < 0 else 0)
            out_batch = []
            for da, db in zip(pad_a + list(batch_a), pad_b + list(batch_b)):
                if da == db:
                    out_batch.append(da)
                elif da == 1:
                    out_batch.append(db)
                elif db == 1:
                    out_batch.append(da)
                else:
                    raise ValueError(
                        f"Incompatible broadcast shapes: {sa} and {sb}"
                    )

            out_batch = tuple(out_batch)

            a_view = a.broadcast(out_batch + mat_a)
            b_view = b.broadcast(out_batch + mat_b)

            return a_view, b_view

        a_norm, b_norm = _inner(a, b)
        return getattr(a_norm, "matmul")(b_norm)

    def _bind_broadcast(self, other, op_name):
        return Tensor._bi_broadcast(self, other, op_name)

    def _bind_matmul_broadcast(self, other):
        return Tensor._matmul_broadcast(self, other)

    def add(self, other):
        return self._binary_op(other, c_func["add"])

    def sub(self, other):
        return self._binary_op(other, c_func["sub"])

    def mul(self, other):
        return self._binary_op(other,  c_func["mul"])

    def div(self, other):
        return self._binary_op(other,  c_func["div"])

    def pow(self, other):
        return self._binary_op(other, c_func["pow"])

    def rsub(self, other):
        if isinstance(other, Tensor):
            return
        return Tensor(other, require_grad=False) - self

    def rtruediv(self, other):
        if isinstance(other, Tensor):
            return
        return Tensor(other, require_grad=False) / self

    def radd(self, other):
        if isinstance(other, Tensor):
            return
        return Tensor(other, require_grad=False) + self

    def rmul(self, other):
        if isinstance(other, Tensor):
            return
        return Tensor(other, require_grad=False) * self

    def __rsub__(self, other):
        return self.rsub(other)

    def __radd__(self, other):
        return self.radd(other)

    def __rtruediv__(self, other):
        return self.rtruediv(other)

    def __rmul__(self, other):
        return self.rmul(other)


    def log(self, other=None):
        if other is None:
            return self._unary_op(c_func["ln"])
        safe_data = self.clamp(1e-6, float("inf"))

        if not hasattr(other, "shape"):
            return safe_data._binary_op(other, c_func["log"])

        return safe_data._binary_op(other, c_func["log"])

    def broadcast(self, shape):
        shape_self = list(self.shape)
        shape_other = list(shape)

        len_diff = len(shape_other) - len(shape_self)
        if len_diff > 0:
            shape_self =  [1] * len_diff + shape_self
        elif len_diff < 0:
            shape_other = [1] * (-len_diff) + shape_other

        out_shape = []
        for s, o in zip(shape_self, shape_other):
            if s == o or s == 1 or o == 1:
                out_shape.append(max(s, o))
            else:
                raise ValueError(f"Incompatible broadcast shapes: {self.shape} and {shape}")

        out_data = c_func["broadcast"](self.data, shape_self, out_shape)
        result = Tensor.__new__(Tensor)
        result.data = out_data
        result.shape = tuple(out_shape)
        result.stride = self.compute_stride(result.shape)
        result.require_grad = self.require_grad
        return result

    def _binary_op(self, other, func):
        result = Tensor.__new__(Tensor)

        def apply_func(a, b, a_dim):
            return func(a, b, a_dim)

        if isinstance(other, Tensor):
            if self.shape == other.shape:
                result.data = apply_func(self.data, other.data, len(self.data))
            else:
                raise ValueError(f"Incompatible broadcast shapes: {self.shape} and {other.shape}")
        else:
            total = reduce(mul, self.shape, 1)
            b_arr = (ctypes.c_float * total)()
            for i in range(total):
                b_arr[i] = other
            result.data = apply_func(self.data, b_arr, len(self.shape))

        result.require_grad = self.require_grad
        result.shape = self.shape
        result.stride = self.compute_stride(result.shape)
        return result


    def sum(self, dim=None):
        if dim is None:
            # Full reduction case - sum all elements
            result = Tensor.__new__(Tensor)
            result.data = (ctypes.c_float * 1)(sum(self.data))
            result.shape = ()
            result.stride = ()
            result.require_grad = self.require_grad
            return result

        # Handle negative dimensions
        if dim < 0:
            dim += len(self.shape)

        if not 0 <= dim < len(self.shape):
            raise ValueError(f"dim {dim} is out of bounds for shape {self.shape}")

        new_shape = self.shape[:dim] + self.shape[dim + 1:]
        if not new_shape:
            new_shape = (1,)  # Maintain as rank-1 tensor for scalar-like results
        else:
            new_shape = self.shape[:dim] + self.shape[dim + 1:]

        dim_size = self.shape[dim]
        outer_block = 1
        for s in self.shape[:dim]:
            outer_block *= s
        inner_block = 1
        for s in self.shape[dim + 1:]:
            inner_block *= s

        result_size = outer_block * inner_block
        result_data = (ctypes.c_float * result_size)()

        for i in range(outer_block):
            for j in range(inner_block):
                total = 0.0
                for k in range(dim_size):
                    idx = i * dim_size * inner_block + k * inner_block + j
                    total += self.data[idx]
                result_idx = i * inner_block + j
                result_data[result_idx] = total

        result = Tensor.__new__(Tensor)
        result.data = result_data
        result.shape = new_shape
        result.stride = self.compute_stride(new_shape) if new_shape else (1,)
        result.require_grad = self.require_grad
        return result

    def mean(self):
        result = Tensor.__new__(Tensor)
        result.data = (ctypes.c_float * 1)(sum(self.data) / len(self.data))
        result.shape = ()
        result.stride = ()
        result.require_grad = self.require_grad
        return result

    def argmax(self):
        pass

    def max(self, dim=None):
        result = Tensor.__new__(Tensor)
        if dim is None:
            result.data = (ctypes.c_float * 1)(max(self.data))
            result.shape = ()
            result.stride = ()
        else:
            inner_block = self.stride[dim - 1] if dim > 0 else 1
            step = self.shape[dim]
            outer_size = len(self.data) // inner_block
            inner_size = inner_block // step

            max_values = (ctypes.c_float * (len(self.data) // step))()
            for i in range(outer_size):
                for j in range(0, inner_block, step):
                    start = i * inner_block + j
                    block = self.data[start:start + step]
                    max_values[start // step] = max(block)

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
        total = reduce(mul, self.shape, 1)
        out = (ctypes.c_float * total)()
        for idx in range(total):
            val = self.data[idx]
            if min_value is not None and val < min_value:
                val = min_value
            if max_value is not None and val > max_value:
                val = max_value
            out[idx] = val
        result = Tensor.__new__(Tensor)
        result.data = out
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
        out_data = (ctypes.c_float * out_numel)()

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

    def reshape(self, *shape):
        out_shape = []
        wild_idx = reduce(mul, self.shape, 1) // reduce(mul, shape, 1)
        if wild_idx < 0:
            wild_idx = -wild_idx

        used_all = False
        for i, _slice in enumerate(shape):
            if _slice == -1:
                if used_all:
                    raise ValueError("at most one -1 can used in reshaping")
                out_shape.append(wild_idx)
                used_all = True
                continue

            out_shape.append(_slice)

        assert len(self.data) == reduce(mul, out_shape, 1)

        result = Tensor.__new__(Tensor)
        result.data = self.data
        result.shape = tuple(out_shape)
        result.stride = result.compute_stride(shape)
        result.require_grad = self.require_grad

        return result

    def unfold(self, dim, win_size, step):
        unfold_count = (self.shape[dim] - win_size) // step + 1
        out_shape = tuple(list(self.shape[:dim]) + [unfold_count, win_size] + list(self.shape[dim + 1:]))
        result = Tensor.__new__(Tensor)
        result.data = c_func["unfold"](self.data, self.shape, out_shape, dim, win_size, step)

        result.shape = out_shape
        result.stride = result.compute_stride(result.shape)
        result.require_grad = self.require_grad
        return result

