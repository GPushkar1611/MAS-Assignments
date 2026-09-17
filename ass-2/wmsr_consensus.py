import numpy as np
import matplotlib.pyplot as plt

# Graph definition

neighbors = {
    1: [5, 7, 6],
    2: [5, 6, 8, 4],
    3: [5, 7, 8, 4],
    4: [2, 3, 8],
    5: [1, 2, 3, 6, 7, 8],
    6: [1, 2, 5, 7, 8],
    7: [1, 3, 5, 6, 8],
    8: [2, 3, 4, 5, 6, 7],
}

N = 8
F = 1
ADVERSARY = 8                # node acting as the adversary
NORMAL_NODES = [i for i in range(1, N + 1) if i != ADVERSARY]

T = 60                       # number of iterations
rng = np.random.default_rng(seed=42)

# Random initial conditions
x = rng.uniform(-10, 10, size=N + 1)   # index 0 unused, nodes are 1..8
history = np.zeros((T + 1, N + 1))
history[0] = x.copy()

print("Initial values:")
for i in range(1, N + 1):
    print(f"  Node {i}: {x[i]:.4f}")

# W-MSR update rule for one normal node
def wmsr_update(node, x_curr, F):
    own_val = x_curr[node]
    nbr_vals = [x_curr[j] for j in neighbors[node]]

    # values strictly larger / smaller than own value
    larger  = sorted([v for v in nbr_vals if v > own_val], reverse=True)
    smaller = sorted([v for v in nbr_vals if v < own_val])
    middle  = [v for v in nbr_vals if v == own_val]   # ties, kept as-is

    # remove up to F largest and up to F smallest
    larger  = larger[F:]   if len(larger)  > F else []
    smaller = smaller[F:]  if len(smaller) > F else []

    kept = [own_val] + larger + smaller + middle
    return float(np.mean(kept))


# Adversary behavior (node 8)
def adversary_update(t):
    return 15 * np.sin(0.5 * t) + rng.normal(0, 1.0)

# Simulation loop

for t in range(1, T + 1):
    x_new = x.copy()

    for node in NORMAL_NODES:
        x_new[node] = wmsr_update(node, x, F)

    x_new[ADVERSARY] = adversary_update(t)

    x = x_new
    history[t] = x.copy()

# Results / inference

final_normal_vals = [history[T, i] for i in NORMAL_NODES]
spread = max(final_normal_vals) - min(final_normal_vals)

print("\nFinal values (t = {}):".format(T))
for i in range(1, N + 1):
    tag = " (ADVERSARY)" if i == ADVERSARY else ""
    print(f"  Node {i}: {history[T, i]:.4f}{tag}")

print(f"\nSpread among normal nodes at final step: {spread:.6f}")
if spread < 1e-3:
    print("=> Normal nodes reached resilient consensus, despite node 8's malicious behavior.")
else:
    print("=> Normal nodes have not fully converged yet (try increasing T).")

plt.figure(figsize=(9, 5.5))
for i in range(1, N + 1):
    if i == ADVERSARY:
        plt.plot(history[:, i], 'k--', linewidth=2, label=f"Node {i} (adversary)")
    else:
        plt.plot(history[:, i], linewidth=1.8, label=f"Node {i}")

plt.xlabel("Iteration")
plt.ylabel("State value")
plt.title("W-MSR Resilient Consensus (F=1, Node 8 = Adversary)")
plt.legend(loc="upper right", ncol=2, fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("wmsr_consensus.png", dpi=150)
print("\nPlot saved to wmsr_consensus.png")