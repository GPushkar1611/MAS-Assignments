import os
import heapq
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
T = [20, 30, 60, 40, 30]            # service minutes at D1..D5
M = [3, 3, 3, 3, 3]                 # officers per department (fixed staffing)
DAY, DAYS, N_SEEDS = 480, 365, 20
SCEN = {"S1": 0.75, "S2": 1.0, "S3": 1.25, "S4": 1.5}
DEPTS = ["D1", "D2", "D3", "D4", "D5"]
days = np.arange(1, DAYS + 1)


def simulate(theta, seed, feedback):
    rng = np.random.default_rng(seed)
    arr, comp, rem, stage = [], [], [], []               # per-application records
    qs = [[] for _ in range(5)]                          # FIFO queue of application ids per dept
    q, p_prev = 0.0, 0.5
    Q, r_list, q_list, same = [], [], [], 0
    for d in range(1, DAYS + 1):
        A = rng.poisson(16 * theta * (1 + 0.5 * q) if feedback else 16 * theta)
        for _ in range(A):
            qs[0].append(len(arr)); arr.append(d); comp.append(0); rem.append(T[0]); stage.append(0)
        ready = [0] * len(arr)                           # minute at which an app becomes available today
        cs = 0
        for i in range(5):
            free, left = [0] * M[i], []                  # officer free-times (heap), apps left for tomorrow
            order = sorted(qs[i], key=lambda a: (ready[a], a))        # FIFO
            for n, a in enumerate(order):
                start = max(free[0], ready[a])
                if start >= DAY:                         # no overtime
                    left += order[n:]
                    break
                end = min(start + rem[a], DAY)
                heapq.heapreplace(free, end)
                rem[a] -= end - start                    # partial work is preserved
                if rem[a] > 0:
                    left.append(a)
                elif i == 4:
                    comp[a] = d
                    cs += arr[a] == d
                else:
                    stage[a], rem[a], ready[a] = i + 1, T[i + 1], end
                    qs[i + 1].append(a)
            qs[i] = left
        b = [len(x) for x in qs]
        p = cs / A if A else p_prev
        p_prev = p
        q = 0.8 * q + 0.2 * (2 * p - 1)
        Q.append(b); r_list.append(5 * cs - 10 * sum(b)); q_list.append(q); same += cs
    Q = np.array(Q)
    return dict(Q=Q, B=Q.sum(1), R=np.cumsum(r_list), q=np.array(q_list),
                arr=np.array(arr), comp=np.array(comp), same=same)


def stats(r):
    done = r["comp"] > 0
    n = len(r["arr"])
    return [n, done.sum(), r["same"] / n, (r["comp"] - r["arr"])[done].mean(), r["B"][-1], r["R"][-1], r["q"][-1]]


def make_plots(runs, tag, keys):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (n, r) in zip(axs.flat, runs.items()):
        ax.plot(days, r["Q"]); ax.set_title(f"{n} (theta={SCEN[n]})"); ax.grid(alpha=0.3)
    axs[0, 0].legend(DEPTS); fig.supxlabel("Day"); fig.supylabel("$Q_i(d)$"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"p3_{tag}_queues.png"), dpi=200); plt.close(fig)
    for key, label in keys:
        plt.figure(figsize=(7, 4))
        for n, r in runs.items():
            plt.plot(days, r[key], label=n)
        plt.xlabel("Day"); plt.ylabel(label); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"p3_{tag}_{key}.png"), dpi=200); plt.close()


print("Daily capacity per dept:", [round(m * DAY / t, 1) for m, t in zip(M, T)])

# ---- Part 1: constant demand (no feedback) ----
static = {n: simulate(th, 0, False) for n, th in SCEN.items()}
make_plots(static, "static", [("B", "$B_d$"), ("R", "$R_d$")])

# ---- Part (e): do more arrivals mean more completions? ----
print("\nConstant demand, mean of", N_SEEDS, "seeds")
for n, th in SCEN.items():
    s = np.mean([stats(simulate(th, i, False)) for i in range(N_SEEDS)], axis=0)
    print(f"{n}: arrivals={s[0]:.0f}, completed={s[1]:.0f}, unfinished={s[4]:.0f}")

# ---- Part 2: with public-satisfaction feedback ----
fb = {n: simulate(th, 0, True) for n, th in SCEN.items()}
make_plots(fb, "feedback", [("B", "$B_d$"), ("R", "$R_d$"), ("q", "$q_d$")])

# ---- Final evaluation: fixed staffing + feedback, 20 seeds ----
names = ["Received", "Completed", "Same-day fraction", "Avg delay (days)", "Unfinished (day 365)", "R_365", "q_365"]
res = {n: np.array([stats(simulate(th, 1000 + i, True)) for i in range(N_SEEDS)]) for n, th in SCEN.items()}
print("\nFixed staffing, feedback ON, mean +- std over", N_SEEDS, "seeds")
print(f"{'':22s}" + "".join(f"{n:>22s}" for n in SCEN))
for j, name in enumerate(names):
    print(f"{name:22s}" + "".join(f"{res[n][:, j].mean():12.2f} +-{res[n][:, j].std(ddof=1):7.2f}" for n in SCEN))
print("Avg staffing per dept (fixed):", [float(m) for m in M])
