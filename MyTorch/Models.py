import ctypes
import math

from Core.Autograd import Autograd
from Core.Tensor import Tensor

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
            T.data = (T - T.get_grad() * Tensor(lr)).data

    def __call__(self, *args, **kwargs):
        return getattr(self, "forward")(*args, **kwargs)

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

class Linear(Module):
    def __init__(self, grad, in_feature, out_feature):
        super().__init__(grad)
        self.linear = Tensor.randn(in_feature, out_feature) / math.sqrt(in_feature * out_feature)
        self.bias = Tensor.randn(1, out_feature)
        self.register_param(self.linear, "linear")
        self.register_param(self.bias, "bias")

    def forward(self, x):
        return x @ self.linear + self.bias

__all__= ["Module", "Model", "Linear"]