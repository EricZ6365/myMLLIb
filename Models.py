import ctypes
import math
from typing import NewType

import matplotlib.pyplot as plt

import visualize
from Autograd import Autograd
from Tensor import Tensor


class Model:
    def __init__(self):
        self._tensors = []
        self.params = {}
        forward_func = getattr(self, "forward")
        self.grad = Autograd([])
        setattr(self, "forward", self._forward_hook(forward_func))

    def register_tensor(self, T):
        self._tensors.append(T)
        self.grad.init_one_tensor(T)

    def register_tensors(self, Ts):
        for T in Ts:
            self.register_tensor(T)

    def register_param(self, T, name):
        self.params[name] = T
        self.register_tensor(T)

    def register_params(self, Ts, names):
        for i, name in enumerate(names):
            self.params[name] = Ts[i]
            self.register_tensor(Ts[i])

    def register_module(self, M, name):
        self.register_tensors(M._tensors)
        self.register_params(list(M.params.values()), list(map(lambda x: name + "." + x, M.params.keys())))

    def get_state(self):
        return {k: list(p.data) for k, p in self.params.items()}

    def load_state(self, state):
        for name, T in state.items():
            obj = getattr(self, name.split(".")[0])
            for path_off in name.split(".")[1:]:
                obj = getattr(obj, path_off)

            obj.data = (ctypes.c_float * len(T))(*T)

    def _forward_hook(self, forward_func):
        def wrap(*args, **kwargs):
            tmp_tensor = []
            for arg in args:
                if isinstance(arg, Tensor):
                    tmp_tensor.append(arg)
            for value in kwargs.values():
                if isinstance(value, Tensor):
                    tmp_tensor.append(value)

            self.grad = Autograd(self._tensors + tmp_tensor)
            return forward_func(*args, **kwargs)
        return wrap

    def backward(self, loss, lr):
        self.grad.backward(loss, loss)
        for T in self._tensors:
            if T.grad_node is None:
                continue
            T.data = (T - T.get_grad() * lr).data

    def save(self, path):
        with open(path, "w") as f:
            for name, value in self.params.items():
                s = f"{name}->{str(value.data)}\n"
                f.write(s)

    def zero(self):
        self.grad.zero()

class Module:
    def __init__(self, grad):
        self._tensors = []
        self.params = {}
        self.grad = grad

    def register_tensor(self, T):
        self._tensors.append(T)

    def register_tensors(self, Ts):
        self._tensors.extend(Ts)

    def register_param(self, T, name):
        self.params[name] = T
        self.register_tensor(T)

    def register_params(self, Ts, names):
        for i, name in enumerate(names):
            self.params[name] = Ts[i]
        self.register_tensors(Ts)

    def __call__(self, *args, **kwargs):
        return getattr(self, "forward")(*args, **kwargs)

class Conv1d(Module):
    """
    Fast(ish) naive 1D convolution implemented via im2col + matmul.
    - weights kept as a single Tensor of shape (out_c, in_c, kernel_size)
    - forward expects inputs shaped (B, in_c, L)
    - stride only (no padding/dilation here, but trivial to add)
    """
    def __init__(self, grad, in_c, out_c, kernel_size, stride=1, bias=True):
        super().__init__(grad)
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride

        scale = (in_c * kernel_size) ** 0.5
        self.weight = Tensor.randn(out_c, in_c, kernel_size) / scale
        self.bias = Tensor.zeros(out_c) if bias else None

        self.register_param(self.weight, "weight")
        self.register_param(self.bias, "bias")

    def forward(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        assert C == self.in_c, f"expected {self.in_c} channels, got {C}"
        k, s = self.kernel_size, self.stride
        L_out = (L - k) // s + 1
        assert L_out > 0, "input too short for given kernel_size/stride"
        x_unfold = x.unfold(2, k, s)
        x_reshaped = x_unfold.transpose(1, 2).reshape(B, L_out, -1)
        W_reshaped = self.weight.reshape(self.out_c, -1)
        y = x_reshaped @ W_reshaped.unsqueeze(0).broadcast((B, self.out_c, self.in_c * self.kernel_size)).transpose(1, 2)
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, self.out_c)

        return y.transpose(1, 2)
    def visualize_output(self, x):
        y_out = self.forward(x)[0]
        out_c = y_out.shape[0]
        fig, axs = plt.subplots(out_c, 1)
        for i in range(out_c):
            data = []
            step_size = int(math.sqrt(y_out.shape[1]))
            for idx in range(0, y_out.shape[1] - step_size, step_size):
                data.append(y_out[i, idx:idx + step_size].data)
            axs[i].imshow(data)
        plt.show()



class Linear(Module):
    def __init__(self, grad, in_feature, out_feature):
        super().__init__(grad)
        self.linear = Tensor.randn(in_feature, out_feature) / math.sqrt(in_feature * out_feature)
        self.bias = Tensor.randn(1, out_feature)
        self.register_param(self.linear, "linear")
        self.register_param(self.bias, "bias")

    def forward(self, x):
        return x @ self.linear + self.bias