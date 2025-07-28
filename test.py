import math
import random

from tqdm import tqdm

import visualize
from Tensor import Tensor
from Autograd import Autograd, GradNode
from Models import Model, Conv1d, Linear
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# ---------- NEW: PyTorch ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# -------------------------------------------------
# Your tiny-autograd model (unchanged)
# -------------------------------------------------
class SimpleConvModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(self.grad, in_c=1, out_c=2, kernel_size=5)
        self.conv2 = Conv1d(self.grad, in_c=2, out_c=2, kernel_size=5)
        self.conv3 = Conv1d(self.grad, in_c=2, out_c=4, kernel_size=5)
        self.fc = Linear(self.grad, in_feature=4 * (15 * 15 - 12), out_feature=10)

        self.register_module(self.conv1)
        self.register_module(self.conv2)
        self.register_module(self.conv3)
        self.register_module(self.fc)

    def forward(self, x):
        x = self.conv1(x).clamp(0, float("inf"))
        x = self.conv2(x).clamp(0, float("inf"))
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def one_hot_encode(label, num_classes=10):
    vec = [0.] * num_classes
    vec[label] = 1.
    return vec

def load_mnist(batch_size=16, limit=60):
    transform = transforms.Compose([
        transforms.Resize((15, 15)),
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda x: x.reshape(1, 15 * 15))
    ])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    X_data, y_data = [], []
    for i in range(limit):
        img, label = mnist[i]
        X_data.append(img.numpy().tolist())
        y_data.append(one_hot_encode(label))

    # Create batches
    X_batched, y_batched = [], []
    for i in range(0, len(X_data) - batch_size, batch_size):
        batch_x = Tensor([X_data[j + i] for j in range(batch_size)])  # (B, 1, 100)
        batch_y = Tensor([y_data[j + i] for j in range(batch_size)])  # (B, 10)
        X_batched.append(batch_x)
        y_batched.append(batch_y)

    return X_batched, y_batched


# -------------------------------------------------
# NEW: helpers to measure accuracy for tiny-autograd
# -------------------------------------------------
def tensor_argmax_lastdim(t: Tensor):
    # assuming t.value() -> numpy array
    return np.argmax(t.value(), dim=-1)

def numpy_argmax_lastdim(arr: np.ndarray):
    return np.argmax(arr, dim=-1)

# -------------------------------------------------
# NEW: equivalent PyTorch model
# -------------------------------------------------
class TorchConv1dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=2)
        self.conv3 = nn.Conv1d(16, 10, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.mean(dim=2)
        return x
def log_softmax(x, dim=1):
    x_max = x.max(dim=dim)

    shifted = x - x_max
    logsumexp = (shifted.exp()).sum(dim=dim).log()
    return shifted - logsumexp


def cross_entropy_loss(pred, target):
    logit = log_softmax(pred, dim=1)
    loss = -(logit * target).sum(dim=1)
    return loss.mean()

def get_torch_loader(batch_size=64, limit=200):
    transform = transforms.Compose([
        transforms.Resize((10, 10)),
        transforms.ToTensor(),  # -> (1, 10, 10)
    ])
    full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset = Subset(full, list(range(limit)))
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)


def train_torch(limit=200, epochs=10, batch_size=64, lr=1e-2, device="cpu"):
    loader = get_torch_loader(batch_size=batch_size, limit=limit)
    model = TorchConv1dModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    torch_losses, torch_accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_count = 0.0, 0, 0

        for x, y in tqdm(loader, desc=f"[Torch] Epoch {epoch}"):
            x = x.view(x.size(0), 1, -1).to(device)  # (B, 1, 100)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_count += x.size(0)

        ep_loss = running_loss / running_count
        ep_acc = running_correct / running_count
        torch_losses.append(ep_loss)
        torch_accs.append(ep_acc)
        print(f"[Torch] Epoch {epoch:3d} | Loss = {ep_loss:.4f} | Acc = {ep_acc*100:.2f}%")

    return model, torch_losses, torch_accs

def get_acc(pred, y, pred_axis=-1):
    if pred_axis < 0:
        pred_axis += len(pred.shape)

    def argmax(arr):
        return arr.index(max(arr))

    correct, total = 0, 0

    for b in range(pred.shape[0]):
        if argmax(pred[b].data) == argmax(y[b].data):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.

if __name__ == "__main__":
    # a = Tensor([
    #     [1, 2, 3]
    # ])
    # grad = Autograd([a])
    #
    # pred = a.log()
    # target = Tensor([[1, 0, 0]])
    # target.grad_node = GradNode(None, None)
    # loss = cross_entropy_loss(pred, target)
    # grad.backward(Tensor(1), loss)
    # print(a.get_grad())
    # visualize.visualize_DCG(loss)
    # # exit()
    random.seed(42)
    batch_size = 2
    X_batched, y_batched = load_mnist(batch_size=batch_size, limit=100)

    model = SimpleConvModel()
    tiny_losses, tiny_accs = [], []

    for epoch in range(100):
        epoch_loss = 0.0
        acc = 0
        for b_X, b_y in tqdm(zip(X_batched, y_batched), total=len(X_batched), desc=f"[Tiny] Epoch {epoch}"):
            pred = model.forward(b_X)
            b_y.grad_node = GradNode(None, None)
            loss = cross_entropy_loss(pred, b_y)
            epoch_loss += loss.value()
            # visualize.visualize_DCG(loss)
            model.backward(loss, lr=1e-3, momentum=0.)
            acc += get_acc(pred, b_y)

        tiny_losses.append(epoch_loss)
        tiny_accs.append(acc / len(X_batched))
        print(f"[Tiny]  Epoch {epoch:3d} | Loss = {epoch_loss:.4f} | Acc = {100 * acc / len(X_batched):.2f}%")

    # ------------------- PyTorch -------------------
    _, torch_losses, torch_accs = train_torch(limit=200, epochs=10, batch_size=64, lr=1e-2)

    # ------------------- Visualization -------------------
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.plot(tiny_losses, label="tiny-autograd")
    plt.plot(torch_losses, label="PyTorch")
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(tiny_accs, label="tiny-autograd")
    plt.plot(torch_accs, label="PyTorch")
    plt.title("Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()
