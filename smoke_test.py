
import torch
import torch.nn as nn
from models import ResNet20, PlainNet20

device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()

# Synthetic CIFAR-shaped data: tests the training mechanics only.
images = torch.randn(128, 3, 32, 32, device=device)
labels = torch.randint(0, 10, (128,), device=device)

for name, model_fn in [
    ("ResNet-20", ResNet20),
    ("Plain-20", PlainNet20),
]:
    model = model_fn().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    model.train()

    for step in range(100):
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0:
            print(
                f"{name} | step {step + 1}/100 | "
                f"loss={loss.item():.4f}"
            )

    print(f"{name}: 100 updates completed successfully\n")

print("GPU smoke test passed!")
print("CUDA memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")
