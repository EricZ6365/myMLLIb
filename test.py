import os
import random

import matplotlib.pyplot as plt
import torchvision.io.image
from tqdm import tqdm

import visualize
from Autograd import GradNode, Autograd
from Tensor import Tensor
from Models import Model, Conv1d, Linear
from torchvision import transforms

from profiler import SimpleProfiler


class OptimizedConvModel(Model):
    def __init__(self, num_class, image_size):
        super().__init__()
        # Use larger kernel sizes and more channels for better feature extraction
        self.conv1 = Conv1d(self.grad, in_c=3, out_c=4, kernel_size=25)
        self.conv2 = Conv1d(self.grad, in_c=4, out_c=8, kernel_size=25)
        self.conv3 = Conv1d(self.grad, in_c=8, out_c=16, kernel_size=25)
        self.num_class = num_class
        self.image_size = image_size

        conv_output_size = 16 * (image_size[0] * image_size[1] - 72)  # Adjusted for padding

        self.fc1 = Linear(self.grad, in_feature=conv_output_size, out_feature=200)
        self.fc3 = Linear(self.grad, in_feature=200, out_feature=num_class)

        self.register_module(self.conv1, "conv1")
        self.register_module(self.conv2, "conv2")
        self.register_module(self.conv3, "conv3")
        self.register_module(self.fc1, "fc1")
        self.register_module(self.fc3, "fc3")


    def forward(self, x):
        x = self.conv1(x).clamp(0, float("inf"))
        x = self.conv2(x).clamp(0, float("inf"))
        x = self.conv3(x).clamp(0, float("inf"))
        x = x.flatten(1)
        x = self.fc1(x).clamp(0, float("inf"))
        x = self.fc3(x)
        return x

    def inference(self, x):
        self.grad.no_track = True
        return self.forward(x)

def L1Loss(pred, target):
    return (pred - target).abs().mean()

def one_hot_encode(label, num_classes):
    vec = [0.] * num_classes
    vec[label] = 10
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


def load_sneaker_optimized(batch_size, limit, image_size, class_limit, split=0.8, data_dir="data/sneakers"):
    transform = transforms.Compose([
        transforms.Resize(image_size),
    ])

    data_pair = []
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))][:class_limit]
    num_classes = len(class_dirs)

    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        loaded = 0

        for file_name in os.listdir(class_path)[:limit]:
            try:
                image_path = os.path.join(class_path, file_name)
                image = torchvision.io.read_image(image_path)
                image = image.float() / 255
                image = transform(image).reshape(3, image_size[0] * image_size[1])
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
    batch_size = 1
    image_size = (20, 20)

    train_batch, val_batch, num_class = load_sneaker_optimized(
        batch_size=batch_size,
        limit=2,
        class_limit=2,
        image_size=image_size
    )

    epochs = 50

    model = OptimizedConvModel(num_class, image_size)
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        lr = 5e-4
        for batch in tqdm(train_batch):
            b_X, b_y = batch
            b_X = Tensor(b_X)
            b_y = Tensor(b_y)
            pred = model.forward(b_X)

            loss = L1Loss(pred, b_y)
            model.backward(loss, lr)
            train_loss += loss.value() / len(train_batch)
            train_acc += get_acc(pred, b_y) / len(train_batch)
            b_y.release()
            model.zero()

        for batch in tqdm(val_batch):
            b_X, b_y = batch
            b_X = Tensor(b_X)
            b_y = Tensor(b_y)
            pred = model.inference(b_X)
            loss = L1Loss(pred, b_y)

            val_loss += loss.value() / len(val_batch)
            val_acc += get_acc(pred, b_y) / len(val_batch)

            b_y.release()

        print(f"\nEpoch {epoch:3d} | "
              f"Train Loss = {train_loss:.4f}, Train Acc = {100 * train_acc:.2f}% | "
              f"Val Loss = {val_loss:.4f}, Val Acc = {100 * val_acc:.2f}%")

        model.save("test.txt")