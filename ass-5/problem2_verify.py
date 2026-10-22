import numpy as np

H = 6
p0 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
E = [(0, 1), (1, 2), (2, 3), (3, 0)]
err = lambda P: [np.abs(P[i] - P[j] - (p0[i] - p0[j])).max() for i, j in E]   # formation errors

g1 = p0 + [0, 2]
g2 = np.array([[0, 2], [2, 2], [2, 3], [0, 3]], float)

# phi_2: the target itself violates the formation constraint (limit 0.2)
print("phi_2 edge errors at target:", err(g2))

# phi_1: all robots move up 0.5 for 4 steps, then stop
u = np.zeros((4, H, 2))
u[:, :4, 1] = 0.5
P = p0[:, None] + np.concatenate([np.zeros((4, 1, 2)), np.cumsum(u, 1)], 1)   # shape (4, H+1, 2)

for i in range(4):
    print(f"robot {i+1}:", P[i].tolist())
print("input ok     :", np.abs(u).max() <= 0.5)
print("always ok    :", all(max(err(P[:, l])) <= 0.2 + 1e-9 for l in range(H + 1)))
print("eventually at l =", [l for l in range(H + 1) if np.abs(P[:, l] - g1).max() <= 0.1])
