import os
import sys
import ctypes
from functools import reduce
from operator import mul
from pathlib import Path


def bin_op_wrap(func):
    def inner(a, b, dim):
        total = len(a)

        ArrayType = ctypes.c_float * total
        out_arr = ArrayType()

        func(a, b, total, dim, out_arr)
        return out_arr

    return inner

def unary_op_wrap(func):
    def inner(a, dim):
        total = len(a)

        ArrayType = ctypes.c_float * total
        out_arr = ArrayType()

        func(a, total, dim, out_arr)

        return out_arr
    return inner

def broadcast_op_wrap(func):
    def inner(a, shape_a, shape_out):
        dim = len(shape_a)
        ShapeType = ctypes.c_int * dim
        shape_a_arr = ShapeType(*shape_a)
        shape_out_arr = ShapeType(*shape_out)
        OutType = ctypes.c_float * reduce(mul, shape_out, 1)
        out_arr = OutType()
        func(a, shape_a_arr, shape_out_arr, out_arr, dim)

        return out_arr

    return inner


def transpose_op_wrap(func):
    def inner(a, shape_a, dim1, dim2):
        dim = len(shape_a)
        ShapeType = ctypes.c_int * dim
        shape_a_arr = ShapeType(*shape_a)
        OutType = ctypes.c_float * reduce(mul, shape_a, 1)
        out_arr = OutType()
        func(a, shape_a_arr, dim1, dim2, out_arr, dim)

        return out_arr
    return inner

def matmul_op_wrap(func):
    def inner(a, b, shape_a, shape_b):
        dim = len(shape_a)
        assert len(shape_a) == len(shape_b)
        assert shape_a[-1] == shape_b[-2]
        if len(shape_a) > 2:
            assert shape_a[:-2] == shape_b[:-2]

        ShapeType = ctypes.c_int * dim
        shape_a_arr = ShapeType(*shape_a)
        shape_b_arr = ShapeType(*shape_b)
        shape_out_arr = ShapeType()

        batch = 1
        for i in range(dim - 2):
            batch *= shape_a[i]
        m = shape_a[dim - 2]
        n = shape_b[dim - 1]
        total_out = batch * m * n

        OutType = ctypes.c_float * total_out
        out_arr = OutType()

        func(a, b, shape_a_arr, shape_b_arr, shape_out_arr, out_arr, dim)
        return out_arr
    return inner

def index_op_wrap(func):
    def inner(a, shape_a, shape_out, slices):
        dim = len(shape_a)
        st, stop, step = [], [], []
        for _slice in slices:
            st.append(_slice.start)
            stop.append(_slice.stop)
            step.append(_slice.step)

        slice_part_type = (ctypes.c_int * len(slices))
        shape_type = (ctypes.c_int * len(slices))

        st = slice_part_type(*st)
        stop = slice_part_type(*stop)
        step = slice_part_type(*step)
        shape_a_arr = shape_type(*shape_a)

        out = (ctypes.c_float * reduce(mul, shape_out, 1))()
        func(a, shape_a_arr, st, stop, step, out, dim)

        return out
    return inner

def unfold_op_wrap(func):
    def inner(a, a_shape, out_shape, dim, win_size, step):
        total_out = reduce(mul, out_shape, 1)

        out = (ctypes.c_float * total_out)()
        a_shape = (ctypes.c_int * len(a_shape))(*a_shape)
        func(a, a_shape, dim, win_size, step, out, len(a_shape))
        return out
    return inner


def unfold_back_op_wrap(func):
    def inner(a, a_shape, out_shape, dim, win_size, step):
        total_out = reduce(mul, out_shape, 1)

        out = (ctypes.c_float * total_out)()
        a_shape = (ctypes.c_int * len(a_shape))(*a_shape)
        out_shape = (ctypes.c_int * len(out_shape))(*out_shape)

        func(a, a_shape, out_shape, dim, win_size, step, out, len(a_shape), len(out_shape))

        return out
    return inner

bin_op_sig = ((ctypes.POINTER(ctypes.c_float),
               ctypes.POINTER(ctypes.c_float),
               ctypes.c_int,
               ctypes.c_int,
               ctypes.POINTER(ctypes.c_float)),
               None)

unary_op_sig = ((ctypes.POINTER(ctypes.c_float),
               ctypes.c_int,
               ctypes.c_int,
               ctypes.POINTER(ctypes.c_float)),
               None)

matmul_op_sig = ((
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
), None)

broadcast_op_sig = ((
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
), None)

transpose_op_sig = ((
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
), None)

index_op_sig = ((
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
), None)

unfold_op_sig = ((
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
), None)

unfold_back_op_sig = ((
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
), None)
names = [
    ["add", *bin_op_sig, bin_op_wrap],
    ["sub", *bin_op_sig, bin_op_wrap],
    ["div", *bin_op_sig, bin_op_wrap],
    ["mul", *bin_op_sig, bin_op_wrap],
    ["log", *bin_op_sig, bin_op_wrap],
    ["pow", *bin_op_sig, bin_op_wrap],
    ["matmul", *matmul_op_sig, matmul_op_wrap],
    ["broadcast", *broadcast_op_sig, broadcast_op_wrap],
    ["unfold", *unfold_op_sig, unfold_op_wrap],
    ["transpose", *transpose_op_sig, transpose_op_wrap],
    ["index", *index_op_sig, index_op_wrap],
    ["neg", *unary_op_sig, unary_op_wrap],
    ["ln", *unary_op_sig, unary_op_wrap],
    ["abs", *unary_op_sig, unary_op_wrap],
    ["exp", *unary_op_sig, unary_op_wrap],
    ["unfold_backward", *unfold_back_op_sig, unfold_back_op_wrap],
]

c_func = {}
base_dir = "build"


def load_clib(lib_name, directory="."):
    ext = {
        "linux": ".so",
        "darwin": ".dylib",  # macOS
        "win32": ".dll"
    }

    platform = sys.platform
    if platform.startswith("linux"):
        suffix = ext["linux"]
    elif platform.startswith("darwin"):
        suffix = ext["darwin"]
    elif platform.startswith("win32"):
        suffix = ext["win32"]
    else:
        raise RuntimeError(f"Unsupported platform: {platform}")

    lib_path = Path(directory) / f"{lib_name}{suffix}"
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


os.makedirs(base_dir, exist_ok=True)
lib = load_clib("c_utils", base_dir)
for name, arg_types, res_type, wrap in names:
    function = getattr(lib, name + "_py")
    function.argtypes = arg_types
    function.restype = res_type
    c_func[name] = wrap(function)
