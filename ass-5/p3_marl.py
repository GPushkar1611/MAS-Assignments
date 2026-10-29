import os
import numpy as np
import matplotlib.pyplot as plt
from p3_env import *

OUT = os.path.dirname(os.path.abspath(__file__))
ALPHA, GAMMA, EPS_MIN, DECAY, N_EP, N_SEEDS = 0.1, 0.9, 0.05, 0.985, 400, 20
LEGAL = [[0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0]]       # A1={0,+1}, A5={-1,0}
rng = np.random.default_rng(0)


def state(o):                                                        # discretise (Q_{i-1}, Q_i, Q_{i+1}, q)
    l, c, r, q = o
    b = lambda x: int(np.digitize(x, [1, 3, 6, 11]))                 # queue bins: 0 | 1-2 | 3-5 | 6-10 | 11+
    return (b(l), b(c), b(r), int(np.digitize(q, [-0.2, 0.5])))      # satisfaction bins: low | mid | high


def act(Q, i, s, eps):                                               # eps-greedy, ties -> keep (0)
    acts = sorted(LEGAL[i], key=abs)
    return int(rng.choice(acts)) if rng.random() < eps else max(acts, key=lambda a: Q[i][s][a + 1])


def train(theta):
    Q, eps, rets = np.zeros((5, 5, 5, 5, 3, 3)), 1.0, []            # one table per department agent
    for ep in range(N_EP):
        env = Env(theta, seed=ep)                                    # training seeds 0..N_EP-1
        o, done = env.obs(), False
        while not done:
            s = [state(x) for x in o]
            a = [act(Q, i, s[i], eps) for i in range(5)]
            o, r, done = env.step(a)
            s2 = [state(x) for x in o]
            for i in range(5):                                       # IQL update, shared reward r_d
                nxt = max(Q[i][s2[i]][b + 1] for b in LEGAL[i])
                Q[i][s[i]][a[i] + 1] += ALPHA * (r / 100 + GAMMA * nxt * (not done) - Q[i][s[i]][a[i] + 1])
        rets.append(env.result()["R"][-1])
        eps = max(EPS_MIN, DECAY * eps)
    return Q, rets


systems = {"Fixed": lambda o: [0] * 5,
           "D2->D3 always": lambda o: [0, 1, 0, 0, 0],                # simple hand-made reference
           "Learned": None}
names = ["Received", "Completed", "Same-day fraction", "Avg delay (days)", "Unfinished (day 365)", "R_365", "q_365"]
curves, mean_R, staff = {}, {}, {}

for n, th in SCEN.items():
    Q, curves[n] = train(th)
    systems["Learned"] = lambda o: [act(Q, i, state(o[i]), 0.0) for i in range(5)]   # greedy
    print(f"\n=== {n} (theta={th}) | {N_SEEDS} seeds, mean +- std ===")
    for sname, pol in systems.items():
        runs = [run(th, 1000 + i, pol) for i in range(N_SEEDS)]       # evaluation seeds 1000..1019
        res = np.array([stats(r) for r in runs])
        print(f"--- {sname}")
        for j, name in enumerate(names):
            print(f"{name:22s} {res[:, j].mean():10.2f} +- {res[:, j].std(ddof=1):7.2f}")
        m = np.mean([r["m"].mean(0) for r in runs], 0)
        print("Avg staffing D1..D5   ", np.round(m, 2))
        mean_R[n, sname] = np.mean([r["R"] for r in runs], 0)
        if sname == "Learned":
            staff[n] = m

# ---- Plots ----
days = np.arange(1, DAYS + 1)
ma = lambda x, w=10: np.convolve(x, np.ones(w) / w, "valid")
plt.figure(figsize=(7, 4))
for n in SCEN:
    plt.plot(range(10, N_EP + 1), ma(curves[n]), label=n)
plt.xlabel("Training episode (1 episode = 365 days)"); plt.ylabel("$R_{365}$ (moving avg, window 10)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(OUT, "p3_marl_train.png"), dpi=200); plt.close()

fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, n in zip(axs.flat, SCEN):
    for sname in systems:
        ax.plot(days, mean_R[n, sname], label=sname)
    ax.set_title(f"{n} (theta={SCEN[n]})"); ax.grid(alpha=0.3)
axs[0, 0].legend(); fig.supxlabel("Day"); fig.supylabel("$R_d$ (mean over seeds)"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "p3_marl_reward.png"), dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
for ax, n in zip(axs, SCEN):
    ax.bar(DEPTS, staff[n]); ax.axhline(3, color="k", ls="--", lw=1); ax.set_title(n); ax.set_ylim(2, 4.5)
axs[0].set_ylabel("Avg officers (learned)"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "p3_marl_staffing.png"), dpi=200); plt.close(fig)
