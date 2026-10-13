import torch
from models import ResNet20, PlainNet20

x = torch.randn(2, 3, 32, 32, device="cuda")

for name, model_fn in [
    ("ResNet-20", ResNet20),
    ("Plain-20", PlainNet20),
]:
    model = model_fn().cuda()
    y = model(x)

    params = sum(p.numel() for p in model.parameters())

    print(name)
    print("Output shape:", y.shape)
    print("Parameters:", params)
    print()