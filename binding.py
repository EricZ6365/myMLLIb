import os
import sys
import ctypes
from pathlib import Path
bin_op_sig = ((ctypes.POINTER(ctypes.c_int),
              ctypes.POINTER(ctypes.c_int),
              ctypes.POINTER(ctypes.c_int),
              ctypes.c_int),
              None)
names = [
    ["add", *bin_op_sig],
    ["sub", *bin_op_sig],
    ["div", *bin_op_sig],
    ["mul", *bin_op_sig]
]
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
for name, arg_types, res_type in names:
    function = getattr(lib, name + "_py")
    function.argtypes = arg_types
    function.restype = res_type

    a = (ctypes.c_int * 3)(1, 2, 3)
    b = (ctypes.c_int * 3)(1, 2, 3)
    out = (ctypes.c_int * 3)()
    shape = (ctypes.c_int * 1)(3)
    dim=1
    function(a, b, shape, dim, out)
    for i in range(3):
        print(out[i])


