
import argparse
import csv
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from models import ResNet20, PlainNet20


DATA_DIR = "data/cifar-10-batches-py"
OUTPUT_DIR = "outputs"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file, encoding="bytes")


def get_value(data, key):
    return data[key] if key in data else data[key.decode()]


def load_cifar_batches(paths):
    images, labels = [], []

    for path in paths:
        batch = read_pickle(path)
        images.append(get_value(batch, b"data"))
        labels.extend(get_value(batch, b"labels"))

    images = np.concatenate(images, axis=0)
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, np.asarray(labels, dtype=np.int64)


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(
            self.images[index].copy()
        ).permute(2, 0, 1).float() / 255.0

        if self.augment:
            # Pad by 4 pixels, then randomly crop to 32 x 32.
            image = F.pad(
                image, (4, 4, 4, 4), mode="constant", value=0
            )

            top = random.randint(0, 8)
            left = random.randint(0, 8)
            image = image[:, top:top + 32, left:left + 32]

            # Random horizontal flip.
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])

        image = (image - self.mean) / self.std
        label = int(self.labels[index])

        return image, label


def prepare_data(batch_size, workers):
    train_paths = [
        os.path.join(DATA_DIR, f"data_batch_{i}")
        for i in range(1, 6)
    ]
    test_path = os.path.join(DATA_DIR, "test_batch")

    for path in train_paths + [test_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing CIFAR-10 file: {path}")

    train_images, train_labels = load_cifar_batches(train_paths)
    test_images, test_labels = load_cifar_batches([test_path])

    # Reproducible 45,000 / 5,000 training-validation split.
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(train_labels), generator=generator
    ).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    augmented_dataset = CIFAR10Dataset(
        train_images, train_labels, augment=True
    )
    clean_dataset = CIFAR10Dataset(
        train_images, train_labels, augment=False
    )
    test_dataset = CIFAR10Dataset(
        test_images, test_labels, augment=False
    )

    train_set = Subset(augmented_dataset, train_indices)
    val_set = Subset(clean_dataset, val_indices)

    loader_options = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, **loader_options
    )

    print(f"Training images:   {len(train_set)}")
    print(f"Validation images: {len(val_set)}")
    print(f"Test images:       {len(test_dataset)}")

    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def run_experiment(
    model_name,
    model_factory,
    train_loader,
    val_loader,
    test_loader,
    device,
    steps,
    output_dir,
    eval_every,
):
    print(f"\n{'=' * 60}")
    print(f"Starting {model_name}")
    print(f"{'=' * 60}")

    model = model_factory().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Iteration-based learning-rate schedule.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[32000, 48000],
        gamma=0.1,
    )

    model_dir = os.path.join(output_dir, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, "best.pt")
    log_path = os.path.join(model_dir, "metrics.csv")

    best_val_accuracy = -1.0
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    global_step = 0

    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step", "learning_rate", "train_loss",
            "train_accuracy", "val_loss", "val_accuracy"
        ])

        train_iterator = iter(train_loader)
        model.train()

        while global_step < steps:
            try:
                images, labels = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                images, labels = next(train_iterator)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            running_correct += (
                logits.detach().argmax(dim=1) == labels
            ).sum().item()
            running_total += labels.size(0)

            if global_step % eval_every == 0 or global_step == steps:
                train_loss = running_loss / max(
                    1, running_total / train_loader.batch_size
                )
                train_accuracy = 100.0 * running_correct / max(
                    1, running_total
                )

                val_loss, val_accuracy = evaluate(
                    model, val_loader, device
                )

                lr = optimizer.param_groups[0]["lr"]

                print(
                    f"{model_name} | step {global_step}/{steps} "
                    f"| train_loss={train_loss:.4f} "
                    f"| train_acc={train_accuracy:.2f}% "
                    f"| val_acc={val_accuracy:.2f}% "
                    f"| lr={lr:.5g}"
                )

                writer.writerow([
                    global_step, lr, train_loss,
                    train_accuracy, val_loss, val_accuracy
                ])
                csvfile.flush()

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_accuracy": val_accuracy,
                            "model_name": model_name,
                        },
                        checkpoint_path,
                    )

                running_loss = 0.0
                running_correct = 0
                running_total = 0
                model.train()

    # Evaluate the best validation checkpoint on the test set.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = evaluate(model, test_loader, device)

    print(f"\n{model_name} finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test loss:                {test_loss:.4f}")
    print(f"Test accuracy:            {test_accuracy:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics:    {log_path}")

    return {
        "model": model_name,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet", "plain"],
        default=["resnet", "plain"],
    )
    args = parser.parse_args()

    if args.steps < 1 or args.eval_every < 1:
        parser.error("--steps and --eval-every must be positive")

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = prepare_data(
        args.batch_size, args.workers
    )

    factories = {
        "resnet": ("ResNet-20", ResNet20),
        "plain": ("Plain-20", PlainNet20),
    }

    results = []

    for key in args.models:
        name, factory = factories[key]
        result = run_experiment(
            name,
            factory,
            train_loader,
            val_loader,
            test_loader,
            device,
            args.steps,
            OUTPUT_DIR,
            args.eval_every,
        )
        results.append(result)

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nExperiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
