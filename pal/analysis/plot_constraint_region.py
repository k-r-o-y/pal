import numpy as np
import matplotlib.pyplot as plt

# Grid over the unit square
n = 500
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# SMT(LRA) running example:
# outer square AND NOT central obstacle
outer = (X >= 0) & (X <= 1) & (Y >= 0) & (Y <= 1)
obstacle = (X >= 0.4) & (X <= 0.6) & (Y >= 0.4) & (Y <= 0.6)
feasible = outer & (~obstacle)

plt.figure(figsize=(5, 5))
plt.imshow(
    feasible,
    extent=[0, 1, 0, 1],
    origin="lower",
    interpolation="nearest",
    aspect="equal",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"Feasible region $C_\phi$")
plt.tight_layout()
plt.savefig("fig_running_constraint_region.pdf", bbox_inches="tight")
plt.savefig("fig_running_constraint_region.png", dpi=300, bbox_inches="tight")
plt.show()