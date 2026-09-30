import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

NET_FILE = "network.csv"
DATA_FILE = "epidemic_data.csv"
STATE_MAP = {"S": 0, "I": 1, "R": 2}
DT = 1.0
B = 15
N_RUNS = 200
T_SIM = 200
rng = np.random.default_rng(0)


# ---------- Load data ----------
df = pd.read_csv(DATA_FILE)
df["state"] = df["state"].str.strip().str.upper().map(STATE_MAP)
wide = df.pivot(index="day", columns="node", values="state").sort_index()
states = wide.values.astype(int)
node_ids = list(wide.columns)
node_index = {n: idx for idx, n in enumerate(node_ids)}
N = len(node_ids)
T = states.shape[0]

edges = pd.read_csv(NET_FILE)
A = np.zeros((N, N))
for u, w in zip(edges["source"], edges["target"]):
    if u in node_index and w in node_index and u != w:
        A[node_index[u], node_index[w]] = 1
A = np.maximum(A, A.T)

print("Loaded: %d days, %d individuals, %d edges" % (T, N, int(A.sum() / 2)))


# ---------- Q1: S, I, R over time ----------
S = (states == 0).sum(axis=1).astype(float)
I = (states == 1).sum(axis=1).astype(float)
R = (states == 2).sum(axis=1).astype(float)
days = np.arange(T)

plt.figure()
plt.plot(days, S, label="S")
plt.plot(days, I, label="I")
plt.plot(days, R, label="R")
plt.xlabel("Day")
plt.ylabel("Individuals")
plt.title("SIR counts from data")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("q1_sir_data.png", dpi=150)

ever_infected = ((states == 1) | (states == 2)).any(axis=0).sum()
print("\nQ1(a) max infected:", int(I.max()))
print("Q1(b) day of peak:", int(I.argmax()))
print("Q1(c) total ever infected:", int(ever_infected))
print("Q1(d) fraction infected: %.3f" % (ever_infected / N))


# ---------- Q2: gamma ----------
K = [k for k in range(T - 1) if I[k] > 0]
gamma_hat = np.mean([(R[k + 1] - R[k]) / (I[k] * DT) for k in K])
print("\nQ2 gamma_hat = %.4f per day" % gamma_hat)
print("Q2 avg infectious duration = %.2f days" % (1 / gamma_hat))


# ---------- Q3: beta and R0 ----------
s, i = S / N, I / N
Kb = [k for k in range(T - 1) if S[k] * I[k] > 0]
beta_hat = np.mean([(s[k] - s[k + 1]) / (s[k] * i[k] * DT) for k in Kb])
R0 = beta_hat / gamma_hat
print("\nQ3 beta_hat = %.4f per day" % beta_hat)
print("Q3 R0 = %.3f" % R0)
print("Q3 infection initially", "grows" if R0 > 1 else "dies out")


# ---------- Q4: simulate continuous SIR ----------
def sir_ode(t, y):
    S_, I_, R_ = y
    return [-beta_hat * S_ * I_, beta_hat * S_ * I_ - gamma_hat * I_, gamma_hat * I_]


t_eval = np.linspace(0, T - 1, 400)
sol = solve_ivp(sir_ode, (0, T - 1), [s[0], i[0], R[0] / N], t_eval=t_eval)

plt.figure()
for idx, (name, data) in enumerate(zip("SIR", (S, I, R))):
    plt.plot(sol.t, sol.y[idx] * N, label=name + " (model)")
    plt.plot(days, data, "--", alpha=0.6, label=name + " (data)")
plt.xlabel("Day")
plt.ylabel("Individuals")
plt.title("Fitted SIR model vs data")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("q4_sir_model.png", dpi=150)

model_I = np.interp(days, sol.t, sol.y[1] * N)
rmse = np.sqrt(np.mean((model_I - I) ** 2))
print("\nQ4 RMSE of I(t): %.2f individuals" % rmse)
print("Q4 model peak: %.1f on day %.1f" % (sol.y[1].max() * N, sol.t[sol.y[1].argmax()]))
print("Q4 model final infected fraction: %.3f" % (1 - sol.y[0, -1]))


# ---------- Q5: epidemic evidence ----------
print("\nQ5 evidence")
print("  R0 = %.3f" % R0)
print("  peak infected = %d (%.1f%% of population)" % (I.max(), 100 * I.max() / N))
print("  final fraction infected = %.3f" % (ever_infected / N))
print("  epidemic:", "YES" if R0 > 1 and I.max() > I[0] else "NO")


# ---------- Q6: vaccination on the network ----------
deg = A.sum(axis=1)
beta_edge = beta_hat / deg.mean()
p_inf = 1 - np.exp(-beta_edge * DT)
p_rec = 1 - np.exp(-gamma_hat * DT)
init_infected = np.where(states[0] == 1)[0]


def simulate(vaccinated):
    st = np.zeros(N, dtype=int)
    st[vaccinated] = 3
    st[init_infected] = 1
    curve = np.zeros(T_SIM)
    for t in range(T_SIM):
        infected = (st == 1)
        curve[t] = infected.sum()
        if curve[t] == 0:
            break
        n_inf_nbrs = A @ infected
        new_inf = (st == 0) & (rng.random(N) < 1 - (1 - p_inf) ** n_inf_nbrs)
        new_rec = infected & (rng.random(N) < p_rec)
        st[new_inf] = 1
        st[new_rec] = 2
    total_infected = (st == 1).sum() + (st == 2).sum()
    return curve, total_infected


candidates = np.setdiff1d(np.arange(N), init_infected)


def pick_random():
    return rng.choice(candidates, B, replace=False)


def pick_degree():
    return candidates[np.argsort(-deg[candidates])[:B]]


eigvals, eigvecs = np.linalg.eigh(A)
v = np.abs(eigvecs[:, -1])


def pick_eigen():
    return candidates[np.argsort(-v[candidates])[:B]]


strategies = {
    "No vaccination": lambda: np.array([], dtype=int),
    "Random": pick_random,
    "Degree": pick_degree,
    "Eigenvector": pick_eigen,
}

results = {}
plt.figure()
print("\nQ6 results (mean over %d runs)" % N_RUNS)
print("%-16s %8s %8s %10s" % ("Strategy", "Imax", "Tpeak", "Ninfected"))
for name, pick in strategies.items():
    curves, totals = [], []
    for _ in range(N_RUNS):
        c, tot = simulate(pick())
        curves.append(c)
        totals.append(tot)
    mean_curve = np.mean(curves, axis=0)
    Imax = np.mean([c.max() for c in curves])
    Tpeak = np.mean([c.argmax() for c in curves])
    Ninf = np.mean(totals)
    results[name] = (Imax, Tpeak, Ninf)
    print("%-16s %8.1f %8.1f %10.1f" % (name, Imax, Tpeak, Ninf))
    plt.plot(mean_curve, label=name)

plt.xlabel("Day")
plt.ylabel("Infected individuals")
plt.title("Infection trajectories under vaccination strategies (B=%d)" % B)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("q6_vaccination.png", dpi=150)

best = min((k for k in results if k != "No vaccination"), key=lambda k: results[k][2])
print("\nQ6(a) best strategy:", best)

top_deg = set(pick_degree())
top_eig = set(pick_eigen())
print("Q6(c) overlap between degree and eigenvector picks: %d of %d" % (len(top_deg & top_eig), B))
print("      degree of degree-picks: mean %.1f | degree of eigenvector-picks: mean %.1f"
      % (deg[list(top_deg)].mean(), deg[list(top_eig)].mean()))
print("      network mean degree: %.1f, lambda_max: %.2f" % (deg.mean(), eigvals[-1]))

plt.show()