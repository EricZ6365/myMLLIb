import math
import os
import random
import multiprocessing as mp

import torch
import torchvision.io.image
from tqdm import tqdm
from Tensor import Tensor
from Models import Model, Conv1d, Linear
import numpy as np
from torchvision import transforms


class OptimizedConvModel(Model):
    def __init__(self, num_class, image_size):
        super().__init__()
        # Use larger kernel sizes and more channels for better feature extraction
        self.conv1 = Conv1d(self.grad, in_c=3, out_c=16, kernel_size=7,)
        self.conv2 = Conv1d(self.grad, in_c=16, out_c=32, kernel_size=5)
        self.conv3 = Conv1d(self.grad, in_c=32, out_c=64, kernel_size=3)
        self.num_class = num_class
        self.image_size = image_size

        conv_output_size = 64 * (image_size[0] * image_size[1] - 12)  # Adjusted for padding

        # More powerful fully connected layers with dropout simulation
        self.fc1 = Linear(self.grad, in_feature=conv_output_size, out_feature=20)
        self.fc2 = Linear(self.grad, in_feature=20, out_feature=10)
        self.fc3 = Linear(self.grad, in_feature=10, out_feature=num_class)

        self.register_module(self.conv1, "conv1")
        self.register_module(self.conv2, "conv2")
        self.register_module(self.conv3, "conv3")
        self.register_module(self.fc1, "fc1")
        self.register_module(self.fc2, "fc2")
        self.register_module(self.fc3, "fc3")

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

        x = self.conv1(x).clamp(0, float("inf"))  # ReLU effect
        x = self.conv2(x).clamp(0, float("inf"))
        x = self.conv3(x).clamp(0, float("inf"))
        x = x.flatten(1)

        if not self.grad.no_track:
            mask = Tensor([random.random() > 0.5 for _ in range(x.shape[1])])
            x = x * mask * 2

        x = self.fc1(x).clamp(0, float("inf"))
        x = self.fc2(x).clamp(0, float("inf"))
        x = self.fc3(x)
        return x

    def inference(self, x):
        self.grad.no_track = True
        return self.forward(x)

def log_softmax(x, dim=1):
    x_max = x.max(dim=dim)
    shifted = x - x_max
    logsumexp = shifted.exp().sum(dim=dim).log()
    return shifted - logsumexp


def cross_entropy_loss(pred, target):
    logit = log_softmax(pred, dim=1)
    loss = -(logit * target).sum(dim=1)
    return loss.sum(None)

def one_hot_encode(label, num_classes=10):
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
        transforms.Lambda(lambda x: x.reshape(3, image_size[0] * image_size[1])),
        transforms.Lambda(lambda x: (x + torch.rand_like(x) * 0.01).clamp(0, 1))
    ])

    data_pair = []
    num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for class_idx, class_dir in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        loaded = 0
        for file_name in os.listdir(class_path)[:limit]:
            try:
                image = torchvision.io.read_image(os.path.join(class_path, file_name)).float() / 255.0
                image = transform(image).tolist()
                data_pair.append([image, class_idx])
                loaded += 1
            except Exception as e:
                print(f"Skipping {file_name}: {e}")

    # Stratified shuffle split
    random.shuffle(data_pair)
    split_idx = int(len(data_pair) * split)
    train_data = data_pair[:split_idx]
    val_data = data_pair[split_idx:]

    # Batch creation with balanced classes
    def make_batches(data):
        batches = []
        class_buckets = [[] for _ in range(num_classes)]
        for img, label in data:
            class_buckets[label].append([img, one_hot_encode(label, num_classes)])

        min_bucket_size = min(len(b) for b in class_buckets)
        balanced_data = []
        for bucket in class_buckets:
            balanced_data.extend(bucket[:min_bucket_size])

        random.shuffle(balanced_data)
        return [balanced_data[i:i + batch_size] for i in range(0, len(balanced_data), batch_size)]

    return make_batches(train_data), make_batches(val_data), num_classes


class OptimizedTrainer:
    def __init__(self, model, num_workers, learning_rate=0.001, momentum=0.9):
        self.model = model
        self.num_workers = num_workers
        self.lr = learning_rate
        self.momentum = momentum
        self.manager = mp.Manager()
        self.loss_queue = self.manager.Queue()
        self.acc_queue = self.manager.Queue()

        # Learning rate scheduling
        self.lr_decay = 0.95
        self.lr_min = 1e-5

    def adjust_learning_rate(self):
        self.lr = max(self.lr * self.lr_decay, self.lr_min)

    @staticmethod
    def worker_process(batch_data, model_state, num_class, image_size, is_training,
                       loss_queue, acc_queue, lr, momentum):
        local_model = OptimizedConvModel(num_class, image_size)
        local_model.load_state(model_state)

        b_X, b_y = zip(*batch_data)
        b_X = Tensor(b_X)
        b_y = Tensor(b_y)

        pred = local_model.forward(b_X)  # Uses forward/inference automatically
        loss = cross_entropy_loss(pred, b_y)
        acc = get_acc(pred, b_y)

        if is_training:
            local_model.backward(loss, lr=lr, momentum=momentum)
            local_model.zero()

        loss_queue.put(loss.value())
        acc_queue.put(acc)
        return local_model.get_state() if is_training else None

    def process_batches(self, batches, is_training=True, desc=""):
        total_batches = len(batches)
        model_state = self.model.get_state()

        with mp.Pool(self.num_workers) as pool:
            args = [(batch, model_state, self.model.num_class, self.model.image_size,
                     is_training, self.loss_queue, self.acc_queue, self.lr, self.momentum)
                    for batch in batches]

            results = []
            with tqdm(total=total_batches, desc=desc) as pbar:
                for res in pool.imap_unordered(self.worker_process, args):
                    results.append(res)
                    pbar.update()

            if is_training:
                avg_state = None
                for state in [r for r in results if r]:
                    if avg_state is None:
                        avg_state = state
                    else:
                        for k in avg_state:
                            avg_state[k] = (avg_state[k] + state[k]) / 2
                self.model.load_state(avg_state)

        # Collect metrics
        losses, accs = [], []
        while not self.loss_queue.empty():
            losses.append(self.loss_queue.get())
        while not self.acc_queue.empty():
            accs.append(self.acc_queue.get())

        return np.mean(losses), np.mean(accs)


if __name__ == "__main__":
    batch_size = 32
    image_size = (28, 28)
    train_batch, val_batch, num_class = load_sneaker_optimized(
        batch_size=batch_size,
        limit=10,
        image_size=image_size
    )

    model = OptimizedConvModel(num_class, image_size)
    trainer = OptimizedTrainer(model, num_workers=4, learning_rate=0.001)

    best_val_acc = 0
    for epoch in range(50):  # More epochs
        train_loss, train_acc = trainer.process_batches(
            train_batch,
            is_training=True,
            desc=f"Epoch {epoch} Training"
        )

        val_loss, val_acc = trainer.process_batches(
            val_batch,
            is_training=False,
            desc=f"Epoch {epoch} Validation"
        )

        trainer.adjust_learning_rate()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 5
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered")
                break

        print(f"\nEpoch {epoch:3d} | "
              f"Train Loss = {train_loss:.4f}, Train Acc = {100 * train_acc:.2f}% | "
              f"Val Loss = {val_loss:.4f}, Val Acc = {100 * val_acc:.2f}% | "
              f"LR = {trainer.lr:.2e}")