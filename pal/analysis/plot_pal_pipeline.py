import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

steps = [
    ("SMT(LRA)\nconstraints", "Logical formula over\nlinear inequalities"),
    ("LRAProblem", "PAL constraint\nobject"),
    ("SplineSQ2D\nBuilder", "Piecewise spline\ndensity builder"),
    ("Polynomial\nrepresentation", "Local polynomial\npieces"),
    ("Integrate\ndistribution", "Constrained symbolic\nintegration"),
    ("Partition\nfunction", "Normalising\nconstant"),
    ("Normalised\ndensity", "Constraint-satisfying\nprobability density"),
]

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n = len(steps)
margin_x = 0.035
gap = 0.018
box_w = (1 - 2 * margin_x - (n - 1) * gap) / n
box_h = 0.48
y0 = 0.29
y_mid = y0 + box_h / 2

for i, (title, subtitle) in enumerate(steps):
    x0 = margin_x + i * (box_w + gap)

    box = FancyBboxPatch(
        (x0, y0),
        box_w,
        box_h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.1,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)

    ax.text(
        x0 + box_w / 2,
        y0 + box_h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        linespacing=0.95,
    )

    ax.text(
        x0 + box_w / 2,
        y0 + box_h * 0.32,
        subtitle,
        ha="center",
        va="center",
        fontsize=7.2,
        linespacing=0.95,
    )

    if i < n - 1:
        start_x = x0 + box_w + 0.004
        end_x = x0 + box_w + gap - 0.004

        arrow = FancyArrowPatch(
            (start_x, y_mid),
            (end_x, y_mid),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="black",
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

plt.savefig(output_dir / "fig_pal_pipeline.pdf", bbox_inches="tight")
plt.savefig(output_dir / "fig_pal_pipeline.png", dpi=300, bbox_inches="tight")
plt.show()