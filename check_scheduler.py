
import torch

# Match the optimizer configuration in train.py
model = torch.nn.Linear(10, 10)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[32000, 48000],
    gamma=0.1,
)

checkpoints = {1, 31999, 32000, 32001, 47999, 48000, 48001, 64000}

for step in range(1, 64001):
    # Simulate one optimizer update
    optimizer.step()
    scheduler.step()

    if step in checkpoints:
        lr = optimizer.param_groups[0]["lr"]
        print(f"Iteration {step:>5}: learning rate = {lr:.6f}")

print("\nScheduler test completed.")
