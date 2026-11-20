import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--log",
    type=str,
    required=True,
    help="Path to the training log.",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    required=True,
    help="Learning rate used during training.",
)
parser.add_argument(
    "--beta-1",
    type=float,
    default=0.9,
    help="Beta 1 value used during training.",
)
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Plot title and the filename.",
)

args = parser.parse_args()

df = pd.read_csv(args.log)
steps = df["step"]

fig, ax1 = plt.subplots(figsize=(21, 9))
ax1.set_title(args.name)
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Sharpness")

if "Preconditioned Sharpness" not in df.columns:
    threshold = 2.0 / args.learning_rate
    ax1.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"2/η (η={args.learning_rate})",
    )

sharpness = df["Sharpness"]
mask = sharpness.notna()
ax1.scatter(
    steps[mask],
    sharpness[mask],
    color="tab:red",
    alpha=0.5,
    s=10,
    label="Sharpness",
)

ax2 = ax1.twinx()
ax2.set_ylabel("Training Loss", color="tab:gray")

loss = df["Training Loss"]
mask = loss.notna()
ax2.plot(
    steps[mask],
    loss[mask],
    color="tab:gray",
    linewidth=1.5,
    alpha=0.7,
    label="Training Loss",
)
ax1.plot([], [], color="tab:gray", label="Training Loss")

ax1.legend()
plt.savefig(
    f"./logs/{args.name}.pdf", transparent=True, bbox_inches="tight", format="pdf"
)
plt.close()


if "Preconditioned Sharpness" in df.columns:
    fig, ax1 = plt.subplots(figsize=(21, 9))
    ax1.set_title(args.name)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Preconditioned Sharpness")
    ax1.set_yscale("log")

    threshold = (2 + 2 * args.beta_1) / ((1 - args.beta_1) * args.learning_rate)
    ax1.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"38/η (η={args.learning_rate})",
    )

    sharpness = df["Preconditioned Sharpness"]
    mask = sharpness.notna()
    ax1.scatter(
        steps[mask],
        sharpness[mask],
        color="tab:red",
        alpha=0.5,
        s=10,
        label="Preconditioned Sharpness",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Loss", color="tab:gray")

    loss = df["Training Loss"]
    mask = loss.notna()
    ax2.plot(
        steps[mask],
        loss[mask],
        color="tab:gray",
        linewidth=1.5,
        alpha=0.7,
        label="Training Loss",
    )
    ax1.plot([], [], color="tab:gray", label="Training Loss")

    ax1.legend()
    plt.savefig(
        f"./logs/{args.name} Precond.pdf",
        transparent=True,
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()
