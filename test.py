import math
import os
import random
import sys
from profiler import SimpleProfiler
import torchvision.io.image
from tqdm import tqdm

import visualize
from Tensor import Tensor
from Autograd import Autograd, GradNode
from Models import Model, Conv1d, Linear
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
    def __init__(self, num_class):
        super().__init__()
        self.conv1 = Conv1d(self.grad, in_c=3, out_c=4, kernel_size=3)
        self.conv2 = Conv1d(self.grad, in_c=4, out_c=4, kernel_size=3)
        self.conv3 = Conv1d(self.grad, in_c=4, out_c=8, kernel_size=3)
        self.fc = Linear(self.grad, in_feature=8 * (30 * 30 - 6), out_feature=num_class)
        self.fc2 = Linear(self.grad, in_feature=num_class, out_feature=num_class)

        self.register_module(self.conv1)
        self.register_module(self.conv2)
        self.register_module(self.conv3)
        self.register_module(self.fc)
        self.register_module(self.fc2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x).clamp(0, float("inf"))
        x = self.fc2(x)
        return x

    def inference(self, x):
        self.grad.no_track = True
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x).clamp(0, float("inf"))
        x = self.fc2(x)
        return x

def one_hot_encode(label, num_classes=10):
    vec = [0.] * num_classes
    vec[label] = 1.
    return vec

def load_mnist(batch_size=16, limit=60):
    transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda x: x.reshape(3, 30 * 30))
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

def load_sneaker(batch_size, limit, image_size, split=0.8, data_dir="data/sneakers"):
    data_pair = []
    num_classes = len(os.listdir(data_dir))
    resize_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x: x.reshape(3, image_size[0] * image_size[1]))
    ])

    total_loaded = 0
    for data_idx, class_dir in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if total_loaded > limit:
            break
        if not os.path.isdir(class_path):
            continue
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            try:
                image = torchvision.io.read_image(file_path).float() / 255.0  # Normalize
                image = resize_transform(image).tolist()

                data_pair.append([image, data_idx])
                total_loaded += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    random.shuffle(data_pair)
    data_pair = data_pair[:limit]
    split_idx = int(len(data_pair) * split)
    train_data = data_pair[:split_idx]
    val_data = data_pair[split_idx:]

    def make_batches(data):
        return [[[a, one_hot_encode(b, data_idx)] for a, b in data[i:i + batch_size]] for i in range(0, len(data), batch_size)]

    train_batches = make_batches(train_data)
    val_batches = make_batches(val_data)

    return train_batches, val_batches, data_idx



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
    return loss.sum(None)

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
    import random
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    random.seed(42)
    batch_size = 8
    image_size = 30, 30
    train_batch, val_batch, num_class = load_sneaker(batch_size=batch_size, limit=500, image_size=image_size)

    model = SimpleConvModel(num_class)
    tiny_train_losses, tiny_val_losses = [], []
    tiny_train_accs, tiny_val_accs = [], []

    for epoch in range(10):
        # ---------- Training ----------
        train_loss = 0.0
        train_acc = 0

        for batch in tqdm(train_batch, total=len(train_batch), desc=f"[Tiny] Epoch {epoch} (train)"):
            b_X, b_y = zip(*batch)
            b_X = Tensor(b_X)
            b_y = Tensor(b_y)

            pred = model.forward(b_X)
            b_y.grad_node = GradNode(None, None)
            print(pred.shape, b_y.shape)
            loss = cross_entropy_loss(pred, b_y)

            train_loss += loss.value()
            model.backward(loss, lr=1e-4, momentum=0.5)
            model.zero()
            train_acc += get_acc(pred, b_y)

        avg_train_acc = train_acc / len(train_batch)
        avg_train_loss = train_loss / len(train_batch)
        tiny_train_losses.append(avg_train_loss)
        tiny_train_accs.append(avg_train_acc)
        val_loss = 0.0
        val_acc = 0
        for batch in tqdm(val_batch, total=len(val_batch), desc=f"[Tiny] Epoch {epoch} (val)"):
            b_X, b_y = zip(*batch)

            b_X = Tensor(b_X)
            b_y = Tensor(b_y)

            pred = model.inference(b_X)
            loss = cross_entropy_loss(pred, b_y)

            val_loss += loss.value()
            val_acc += get_acc(pred, b_y)

        avg_val_acc = val_acc / len(val_batch)
        avg_val_loss = val_loss / len(val_batch)
        tiny_val_losses.append(avg_val_loss)
        tiny_val_accs.append(avg_val_acc)

        # ---------- Output ----------
        print(f"[Tiny] Epoch {epoch:3d} | "
              f"Train Loss = {avg_train_loss:.4f}, Train Acc = {100 * avg_train_acc:.2f}% | "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {100 * avg_val_acc:.2f}%")

        # Optional visualization
        if epoch % 2 == 0:
            inter = model.conv1.forward(b_X).clamp(0, float("inf"))
            inter = model.conv2.forward(inter).clamp(0, float("inf"))
            model.conv3.visualize_output(inter)
