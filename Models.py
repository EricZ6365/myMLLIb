import visualize
from Autograd import Autograd
from Tensor import Tensor


class Model:
    def __init__(self):
        self._tensors = []
        forward_func = getattr(self, "forward")
        self.grad = Autograd([])
        setattr(self, "forward", self._forward_hook(forward_func))

    def register_tensor(self, T):
        self._tensors.append(T)
        self.grad.init_one_tensor(T)

    def register_module(self, M):
        for T in M._tensors:
            self._tensors.append(T)
            self.grad.init_one_tensor(T)

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

    def backward(self, loss, lr, momentum=0.9):
        self.grad.backward(loss.value(), loss)
        for T in self._tensors:
            if T.grad_node is None:
                continue

            if not hasattr(T, "velocity"):
                T.velocity = [0.0 for _ in T.data]

            new_data = []

            for i, (w, gw) in enumerate(zip(T.data, T.grad_node.grad_a.data)):
                T.velocity[i] = momentum * T.velocity[i] + (1 - momentum) * gw
                new_data.append(w - lr * T.velocity[i])
            T.data = new_data


class Module:
    def __init__(self, grad):
        self._tensors = []
        self.grad = grad

    def register_tensor(self, T):
        self._tensors.append(T)

    def register_tensors(self, Ts):
        self._tensors.extend(Ts)

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

        self.register_tensor(self.weight)
        self.register_tensor(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        assert C == self.in_c, f"expected {self.in_c} channels, got {C}"
        k, s = self.kernel_size, self.stride
        L_out = (L - k) // s + 1
        assert L_out > 0, "input too short for given kernel_size/stride"

        out = []
        W = self.weight.unsqueeze(0)  # (1, out_c, C, k)

        for i in range(L_out):
            start = i * s
            end = start + k
            x_win = x._getitem((slice(None), slice(None), slice(start, end)))  # (B, C, k)
            x_win = x_win.unsqueeze(1).broadcast((B, self.out_c, C, k))
            Wb = W.broadcast((B, self.out_c, C, k))

            # Sum over channels and kernel
            y = x_win.mul(Wb).sum(dim=2).sum(dim=2)  # (B, out_c)

            if self.bias is not None:
                y = y + self.bias.unsqueeze(0).broadcast((B, self.out_c))  # (B, out_c)

            # Add dimension for length
            y = y.unsqueeze(2)  # (B, out_c, 1)
            out.append(y)

        return out[0].concat(*out[1:], dim=2)  # (B, out_c, L_out)


class Linear(Module):
    def __init__(self, grad, in_feature, out_feature):
        super().__init__(grad)
        self.linear = Tensor.randn(in_feature, out_feature)
        self.bias = Tensor.randn(1, out_feature)
        self.register_tensor(self.linear)
        self.register_tensor(self.bias)

    def forward(self, x):
        first = None
        tensors = []
        if len(x.shape) == 3:
            for i in range(x.shape[0]):
                lin = x[i] @ self.linear
                if first is None:
                    first = lin
                tensors.append(lin)
        else:
            return x @ self.linear + self.bias
        return first.concat(tensors, dim=0) + self.bias