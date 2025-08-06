import math
import os
import random
import multiprocessing as mp
import time
from collections import defaultdict

import torch
import torchvision.io.image
from tqdm import tqdm

from Autograd import GradNode
from Tensor import Tensor
from Models import Model, Conv1d, Linear
import numpy as np
from torchvision import transforms

from profiler import SimpleProfiler


class OptimizedConvModel(Model):
    def __init__(self, num_class, image_size):
        super().__init__()
        # Use larger kernel sizes and more channels for better feature extraction
        self.conv1 = Conv1d(self.grad, in_c=3, out_c=4, kernel_size=11)
        self.conv2 = Conv1d(self.grad, in_c=4, out_c=4, kernel_size=11)
        self.conv3 = Conv1d(self.grad, in_c=4, out_c=8, kernel_size=11)
        self.num_class = num_class
        self.image_size = image_size

        conv_output_size = 8 * (image_size[0] * image_size[1] - 30)  # Adjusted for padding

        # More powerful fully connected layers with dropout simulation
        self.fc1 = Linear(self.grad, in_feature=conv_output_size, out_feature=100)
        self.fc2 = Linear(self.grad, in_feature=100, out_feature=50)
        self.fc3 = Linear(self.grad, in_feature=50, out_feature=num_class)

        self.register_module(self.conv1, "conv1")
        self.register_module(self.conv2, "conv2")
        self.register_module(self.conv3, "conv3")
        self.register_module(self.fc1, "fc1")
        self.register_module(self.fc2, "fc2")
        self.register_module(self.fc3, "fc3")

    def forward(self, x):
        x = self.conv1(x).clamp(0, float("inf"))  # ReLU effect
        x = self.conv2(x).clamp(0, float("inf"))
        x = self.conv3(x).clamp(0, float("inf"))
        x = x.flatten(1)

        x = self.fc1(x).clamp(0, float("inf"))
        x = self.fc2(x).clamp(0, float("inf"))
        x = self.fc3(x)
        return x

    def inference(self, x):
        self.grad.no_track = True
        return self.forward(x)

def log_softmax(x, dim=1):
    x_max = x.max(dim=dim).unsqueeze(dim)
    shifted = x - x_max
    logsumexp = shifted.exp().sum(dim=dim).log().unsqueeze(dim)
    return shifted - logsumexp

def cross_entropy_loss(pred, target):
    logit = log_softmax(pred, dim=1)
    loss = -(logit * target).sum(-1)
    return loss.sum(None)

def one_hot_encode(label, num_classes):
    vec = [0.] * num_classes
    vec[label] = 1.
    return vec

def get_acc(pred, y, pred_axis=-1):
    if pred_axis < 0:
        pred_axis += len(pred.shape)

    def argmax(arr):
        max_val, idx = float("-inf"), 0
        for i, val in enumerate(arr):
            if val > max_val:
                idx = i
                max_val = val
        return idx


    correct, total = 0, 0
    for b in range(pred.shape[0]):
        d = pred[b].data
        if argmax(d) == argmax(y[b].data):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.


def load_sneaker_optimized(batch_size, limit, image_size, split=0.8, data_dir="data/sneakers"):
    # Enhanced data loading with better augmentation
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda x: x.reshape(3, image_size[0] * image_size[1]) / 255.0),
    ])

    data_pair = []
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    num_classes = len(class_dirs)

    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        loaded = 0

        for file_name in os.listdir(class_path)[:limit]:
            try:
                image_path = os.path.join(class_path, file_name)
                image = torchvision.io.read_image(image_path).float()
                image = transform(image)
                data_pair.append([image.tolist(), class_idx])
                loaded += 1
            except Exception as e:
                print(f"Skipping {file_name}: {e}")

    random.shuffle(data_pair)
    split_idx = int(len(data_pair) * split)
    train_data = data_pair[:split_idx]
    val_data = data_pair[split_idx:]

    def make_batches(data):
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if len(batch) < batch_size:
                continue
            images = [item[0] for item in batch]
            labels = [one_hot_encode(item[1], num_classes) for item in batch]
            batches.append([images, labels])
        return batches

    return make_batches(train_data), make_batches(val_data), num_classes

if __name__ == "__main__":
    batch_size = 8
    image_size = (50, 50)
    train_batch, val_batch, num_class = load_sneaker_optimized(
        batch_size=batch_size,
        limit=1,
        image_size=image_size
    )

    model = OptimizedConvModel(num_class, image_size)
    for epoch in range(50):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        for batch in tqdm(train_batch):
            b_X, b_y = batch
            b_X = Tensor(b_X)
            b_y = Tensor(b_y)
            pred = model.forward(b_X)
            loss = cross_entropy_loss(pred, b_y)

            model.backward(loss, 1e-3)
            train_loss += loss.value() / len(train_batch)
            train_acc += get_acc(pred, b_y) / len(train_batch)
            b_y.release()

        for batch in tqdm(val_batch):
            b_X, b_y = batch
            b_X = Tensor(b_X)
            b_y = Tensor(b_y)
            b_y.grad_node = GradNode(None, None)
            pred = model.inference(b_X)
            loss = cross_entropy_loss(pred, b_y)

            val_loss += loss.value() / len(val_batch)
            val_acc += get_acc(pred, b_y) / len(val_batch)
            b_y.release()

        model.conv1.visualize_output(b_X)

        print(f"\nEpoch {epoch:3d} | "
              f"Train Loss = {train_loss:.4f}, Train Acc = {100 * train_acc:.2f}% | "
              f"Val Loss = {val_loss:.4f}, Val Acc = {100 * val_acc:.2f}%")