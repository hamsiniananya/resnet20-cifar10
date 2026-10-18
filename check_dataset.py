
import os
from torchvision.datasets import CIFAR10

print("Data folder contents:", os.listdir("./data"))

try:
    dataset = CIFAR10(
        root="./data",
        train=True,
        download=False,
    )
    print("Training images:", len(dataset))
    print("Classes:", dataset.classes)
    print("First image size:", dataset[0][0].size)

except Exception as e:
    print("Dataset verification failed:", repr(e))
