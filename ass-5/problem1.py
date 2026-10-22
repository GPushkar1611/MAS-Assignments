import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
S = 7341                                  # your random integer < 10^4
k = S % 5 + 1
G1, G2 = (1, k), (5, 6 - k)
ALPHA, GAMMA, EPS_MIN, DECAY = 0.2, 0.95, 0.05, 0.995
N_EP, H, W = 2000, 30, 50
MOVES = {"Up": (0, 1), "Down": (0, -1), "Left": (-1, 0), "Right": (1, 0), "Stay": (0, 0)}
NAMES = list(MOVES)
rng = np.random.default_rng(0)


def move(p, a):
    x, y = p[0] + MOVES[NAMES[a]][0], p[1] + MOVES[NAMES[a]][1]
    return (x, y) if 1 <= x <= 5 and 1 <= y <= 5 else p       # wall: stay in place


def step(p1, p2, a1, a2):
    p1, p2 = move(p1, a1), move(p2, a2)
    if (p1, p2) in [(G1, G2), (G2, G1)]:
        return p1, p2, 19, True                                 # +20 - 1
    return p1, p2, (-5 if p1 == p2 else -1), False


def idx(p1, p2):                                                # joint state -> 0..624
    return ((p1[0] - 1) * 5 + p1[1] - 1) * 25 + (p2[0] - 1) * 5 + p2[1] - 1


# ---- IQL training: one Q-table per agent over the joint state ----
Q = np.zeros((2, 625, 5))
rets, succ, eps = [], [], 1.0
for ep in range(N_EP):
    p1, p2, G, done = (1, 1), (5, 5), 0, False
    for t in range(H):
        s = idx(p1, p2)
        a = [rng.integers(5) if rng.random() < eps else Q[i, s].argmax() for i in range(2)]
        p1, p2, r, done = step(p1, p2, *a)
        s2 = idx(p1, p2)
        for i in range(2):
            Q[i, s, a[i]] += ALPHA * (r + GAMMA * Q[i, s2].max() * (not done) - Q[i, s, a[i]])
        G += r
        if done:
            break
    rets.append(G)
    succ.append(float(done))
    eps = max(EPS_MIN, DECAY * eps)

# ---- Plots ----
ma = lambda x: np.convolve(x, np.ones(W) / W, "valid")
for data, name, ylabel in [(rets, "return", "Return"), (succ, "success", "Success rate")]:
    plt.figure(figsize=(7, 4))
    plt.plot(range(W, N_EP + 1), ma(np.array(data)))
    plt.xlabel("Episode"); plt.ylabel(f"{ylabel} (moving avg, window {W})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"p1_{name}.png"), dpi=200); plt.close()

# ---- Greedy evaluation (ties broken randomly) ----
greedy = lambda i, s: int(rng.choice(np.flatnonzero(Q[i, s] == Q[i, s].max())))
print(f"S={S}, k={k}, g1={G1}, g2={G2}, last-{W} success={np.mean(succ[-W:]):.2f}")
for e in range(1, 4):
    p1, p2, G = (1, 1), (5, 5), 0
    print(f"\nEpisode {e}\nt= 0 | A1 {p1} | A2 {p2}")
    for t in range(1, H + 1):
        s = idx(p1, p2)
        a1, a2 = greedy(0, s), greedy(1, s)
        p1, p2, r, done = step(p1, p2, a1, a2)
        G += r
        print(f"t={t:2d} | A1 {p1} {NAMES[a1]:5s} | A2 {p2} {NAMES[a2]:5s} | r={r}")
        if done:
            break
    print(f"Return={G}, success={done}")
