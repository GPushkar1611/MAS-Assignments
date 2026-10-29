import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from p3_env import *

OUT = os.path.dirname(os.path.abspath(__file__))
days = np.arange(1, DAYS + 1)


def queue_plot(runs, tag):                                           # one row per department, own y-scale
    fig, axs = plt.subplots(5, 4, figsize=(14, 9), sharex=True)
    for j, (n, r) in enumerate(runs.items()):
        for i in range(5):
            axs[i, j].plot(days, r["Q"][:, i], color=f"C{i}", lw=0.8); axs[i, j].grid(alpha=0.3)
            if j == 0: axs[i, j].set_ylabel(DEPTS[i])
        axs[0, j].set_title(f"{n} (theta={SCEN[n]})")
    fig.supxlabel("Day"); fig.supylabel("$Q_i(d)$"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"p3_{tag}_queues.png"), dpi=200); plt.close(fig)


def line_plot(runs, key, label, tag):                                # left: all scenarios, right: without S4
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, names in zip(axs, [list(runs), ["S1", "S2", "S3"]]):
        for n in names:
            ax.plot(days, runs[n][key], label=n, lw=0.9)
        ax.set_xlabel("Day"); ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend()
    axs[0].set_title("S1-S4"); axs[1].set_title("S1-S3 (zoom)"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"p3_{tag}_{key}.png"), dpi=200); plt.close(fig)


# ---- Fix: clearer plots (constant demand, then feedback) ----
for tag, fb in [("static", False), ("feedback", True)]:
    runs = {n: run(th, 0, feedback=fb) for n, th in SCEN.items()}
    queue_plot(runs, tag)
    for key, label in [("B", "$B_d$"), ("R", "$R_d$")] + ([("q", "$q_d$")] if fb else []):
        line_plot(runs, key, label, tag)

# ---- Fix: per-application records (S3, constant demand, seed 0) ----
r = run(SCEN["S3"], 0, feedback=False)
pend = r["comp"] == 0
df = pd.DataFrame(dict(app=np.arange(len(r["arr"])), arrival_day=r["arr"], current_dept=r["stage"] + 1,
                       remaining_min=r["rem"], completion_day=np.where(pend, -1, r["comp"])))
df.to_csv(os.path.join(OUT, "p3_applications.csv"), index=False)     # all records (-1 = unfinished)
late = df[(df.completion_day > df.arrival_day)].head(5)             # a few delayed ones
print("5 same-day + 5 delayed applications:\n", pd.concat([df.head(5), late]).to_string(index=False))
print("\nUnfinished at end of day 365:\n", df[pend].to_string(index=False))
print("\nConsistency: unfinished records =", pend.sum(), "| B_365 =", r["B"][-1],
      "| per-dept from records:", [int(((r["stage"] == i) & pend).sum()) for i in range(5)],
      "| Q_i(365):", [int(x) for x in r["Q"][-1]])

# ---- Analysis for 'popular scheme can fail': feedback ON, mean over 20 seeds ----
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
print("\nMean over 20 seeds (feedback ON): first 30 days vs last 30 days")
print(f"{'':4s}{'arrivals/day':>18s}{'B_d':>16s}{'q_d':>16s}{'days A>24':>12s}")
for n, th in SCEN.items():
    rs = [run(th, 1000 + i) for i in range(20)]
    A, B, q = (np.mean([x[k] for x in rs], 0) for k in ("A", "B", "q"))
    for ax, y, lab in zip(axs, (q, B, A), ("$q_d$", "$B_d$", "Arrivals $A_d$")):
        ax.plot(days, y, label=n, lw=0.9); ax.set_xlabel("Day"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    over = np.mean([(x["A"] > 24).mean() for x in rs])
    f = lambda y: f"{y[:30].mean():6.2f} -> {y[-30:].mean():6.2f}"
    print(f"{n:4s}{f(A):>18s}{f(B):>16s}{f(q):>16s}{over:12.2f}")
axs[2].axhline(24, color="k", ls="--", lw=1, label="D3 capacity (24)"); axs[2].legend(); axs[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "p3_feedback_dynamics.png"), dpi=200)

# ---- Part (e): constant demand, mean over 20 seeds ----
print("\nConstant demand, mean of 20 seeds")
for n, th in SCEN.items():
    s = np.mean([stats(run(th, i, feedback=False)) for i in range(20)], axis=0)
    print(f"{n}: arrivals={s[0]:.0f}, completed={s[1]:.0f}, unfinished={s[4]:.0f}")