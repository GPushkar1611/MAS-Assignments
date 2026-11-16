import numpy as np
import matplotlib.pyplot as plt

N, K, c, L, T_iso = 500, 150, 4, 5, 10         # agents, days, avg contacts/day, tracing window, isolation days
bE, bI, sig, gam, p_sym = 0.04, 0.05, 0.2, 0.1, 0.3   # p_sym: daily chance a symptomatic agent gets tested
Imax, Isafe, Tsup, Twait = 40, 5, 90, 14
S, E, I, R = 0, 1, 2, 3

def run(policy, seed=1):
    rng = np.random.default_rng(seed)
    x = np.full(N, S); x[rng.choice(N, 5, replace=False)] = E
    know_inf = np.full(N, -99)                   # day agent tested positive
    notified = np.full(N, -99)                   # day of last exposure notification
    tested = np.zeros(N, bool)                   # tested after the notification
    hist, out = [], []
    for k in range(K):
        A = np.triu(rng.random((N, N)) < c / N, 1); A = A | A.T
        K_inf = (know_inf >= 0) & (x != R)                         # K_i Infected_i
        K_exp = k - notified <= L                                  # K_i Exposure_i
        u_iso = K_inf & (k - know_inf < T_iso) & (policy != 'none')
        Au = A & ~u_iso[:, None] & ~u_iso[None, :]                 # A^u = D A D
        hist = (hist + [Au])[-L:]
        out.append([np.sum(x == E), np.sum(x == I), K_inf.sum(), K_exp.sum(), (K_exp & tested).sum()])

        # testing: symptomatic agents (+ notified agents who are unsure, for the epistemic policy)
        u_test = (x == I) & (know_inf < 0) & (rng.random(N) < p_sym) & (policy != 'none')
        if policy == 'epistemic':
            u_test |= K_exp & ~K_inf & ~tested
        tested |= u_test
        pos = u_test & ((x == E) | (x == I))
        know_inf[pos] = k
        if policy == 'epistemic':                                  # contact tracing
            contacts = np.any([h[pos].any(0) for h in hist], axis=0)
            new = contacts & (k - notified > L)
            notified[new], tested[new] = k, False

        # SEIR transitions on the controlled graph
        nE, nI = Au @ (x == E), Au @ (x == I)
        p_inf = 1 - (1 - bE) ** nE * (1 - bI) ** nI
        r = rng.random(N)
        x = np.where((x == S) & (r < p_inf), E, np.where((x == E) & (r < sig), I, np.where((x == I) & (r < gam), R, x)))
    return np.array(out)

res = {p: [run(p, s) for s in range(10)] for p in ['none', 'symptoms', 'epistemic']}
for p, runs in res.items():
    NI = [o[:, 1] for o in runs]
    rho5 = [np.min(Imax - n) for n in NI]
    rho6 = [max(np.min(Isafe - n[t:t + Twait + 1]) for t in range(Tsup + 1)) for n in NI]
    print(f'{p:10s} mean peak N_I={np.mean([n.max() for n in NI]):5.1f}  '
          f'spec5 satisfied {np.sum(np.array(rho5) > 0)}/10  spec6 satisfied {np.sum(np.array(rho6) > 0)}/10  '
          f'mean rho5={np.mean(rho5):6.1f}  mean rho6={np.mean(rho6):5.1f}')

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for p, runs in res.items():
    ax[0].plot(np.mean([o[:, 1] for o in runs], 0), label=p)
ax[0].axhline(Imax, color='r', ls='--', label='$I_{max}$'); ax[0].axhline(Isafe, color='g', ls='--', label='$I_{safe}$')
ax[0].set(xlabel='day $k$', ylabel='$N_I[k]$', title='Symptomatic infected (mean of 10 runs)'); ax[0].legend()
o = res['epistemic'][0]
for q, lab in enumerate(['$N_E$ (true)', '$N_I$ (true)', '$N_{KI}$: know infected', '$N_{KE}$: know exposed', '$N_{KET}$: exposed & tested']):
    ax[1].plot(o[:, q], label=lab)
ax[1].set(xlabel='day $k$', yscale='log', title='Epistemic policy (one run): truth vs knowledge'); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig('p4_sim.png', dpi=200)
