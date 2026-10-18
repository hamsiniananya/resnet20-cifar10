
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --------------------------------------------------
# Publication-quality CIFAR-10 comparison plots
# --------------------------------------------------

OUTPUT = Path("outputs")
FIGURES = OUTPUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ResNet-20": "#3977F6",  # Electric blue
    "Plain-20": "#F47B55",   # Warm coral
}

LR_DROPS = [32000, 48000]
TOTAL_STEPS = 64000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FAFBFF",
    "axes.edgecolor": "#DCE2EC",
    "axes.linewidth": 1.0,
    "grid.color": "#D9E0EA",
    "grid.alpha": 0.65,
    "grid.linewidth": 0.8,
    "savefig.facecolor": "#FFFFFF",
    "savefig.bbox": "tight",
})


def load_metrics(model_name, folder):
    path = OUTPUT / folder / "metrics.csv"

    with path.open("r", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No metric rows found in {path}")

    return {
        "steps": [int(r["step"]) for r in rows],
        "loss": [float(r["train_loss"]) for r in rows],
        "accuracy": [float(r["val_accuracy"]) for r in rows],
    }


data = {
    "ResNet-20": load_metrics("ResNet-20", "resnet-20"),
    "Plain-20": load_metrics("Plain-20", "plain-20"),
}


def style_axes(ax):
    ax.set_xlim(0, TOTAL_STEPS)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8000))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k")
    )

    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DCE2EC")
    ax.spines["bottom"].set_color("#DCE2EC")

    ax.tick_params(colors="#475569", length=4)


def add_lr_annotations(ax):
    # Shade the three learning-rate phases subtly.
    ax.axvspan(
        0, 32000, color="#3977F6", alpha=0.035, zorder=0
    )
    ax.axvspan(
        32000, 48000, color="#14B8A6", alpha=0.045, zorder=0
    )
    ax.axvspan(
        48000, 64000, color="#A78BFA", alpha=0.055, zorder=0
    )

    for step, label in [
        (32000, "LR → 0.01"),
        (48000, "LR → 0.001"),
    ]:
        ax.axvline(
            step,
            color="#64748B",
            linestyle="--",
            linewidth=1.3,
            alpha=0.85,
            zorder=2,
        )

        ax.annotate(
            label,
            xy=(step, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(6, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.5,
            color="#475569",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "white",
                "ec": "#DCE2EC",
                "alpha": 0.95,
            },
        )


def save_figure(fig, name):
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {name}.png and {name}.pdf")


# --------------------------------------------------
# Figure 1: Training loss
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.5))
add_lr_annotations(ax)

for name, values in data.items():
    x = values["steps"]
    y = values["loss"]
    color = COLORS[name]

    ax.plot(
        x, y,
        color=color,
        linewidth=2.8,
        label=name,
        solid_capstyle="round",
        zorder=4,
    )

    ax.scatter(
        x, y,
        color=color,
        s=24,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )

ax.set_title(
    "Training Loss on CIFAR-10",
    loc="left",
    pad=16,
    color="#172033",
)
ax.set_xlabel("Training iterations", labelpad=9)
ax.set_ylabel("Training loss", labelpad=9)

ax.set_ylim(bottom=0)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
style_axes(ax)

ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#DCE2EC",
    framealpha=0.97,
)

fig.tight_layout()
save_figure(fig, "training_loss")


# --------------------------------------------------
# Figure 2: Validation accuracy
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.5))
add_lr_annotations(ax)

for name, values in data.items():
    x = values["steps"]
    y = values["accuracy"]
    color = COLORS[name]

    ax.plot(
        x, y,
        color=color,
        linewidth=2.8,
        label=name,
        solid_capstyle="round",
        zorder=4,
    )

    ax.scatter(
        x, y,
        color=color,
        s=32,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    # Highlight the best recorded validation checkpoint.
    best_index = max(range(len(y)), key=lambda i: y[i])
    best_x = x[best_index]
    best_y = y[best_index]

    ax.scatter(
        [best_x], [best_y],
        s=115,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=7,
    )

    ax.annotate(
        f"{best_y:.2f}%",
        xy=(best_x, best_y),
        xytext=(0, 12 if name == "ResNet-20" else -23),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": color,
            "alpha": 0.96,
        },
        zorder=8,
    )

ax.set_title(
    "Validation Accuracy on CIFAR-10",
    loc="left",
    pad=16,
    color="#172033",
)
ax.set_xlabel("Training iterations", labelpad=9)
ax.set_ylabel("Validation accuracy (%)", labelpad=9)

ax.set_ylim(60, 95)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
style_axes(ax)

ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="#DCE2EC",
    framealpha=0.97,
)

fig.tight_layout()
save_figure(fig, "validation_accuracy")

print("\nAll plots generated successfully.")
print(f"Figures folder: {FIGURES.resolve()}")
