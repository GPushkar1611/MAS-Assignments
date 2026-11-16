import numpy as np
import matplotlib.pyplot as plt

# Edges (from, to): approximate distance in km
D = {('v01', 'v4'): 116, ('v02', 'v3'): 61, ('v4', 'v5'): 118, ('v4', 'v1'): 157, ('v4', 'v2'): 174,
     ('v3', 'v5'): 225, ('v3', 'v2'): 111, ('v5', 'v1'): 80, ('v2', 'v1'): 65,
     ('v1', 'v6'): 70, ('v1', 'v7'): 95, ('v1', 'v8'): 173}
order = ['v01', 'v02', 'v4', 'v3', 'v5', 'v2', 'v1', 'v6', 'v7', 'v8']   # topological order
town = {'v4': 'Uttarkashi', 'v3': 'Joshimath', 'v5': 'Tehri', 'v2': 'Rudraprayag',
        'v1': 'Devprayag', 'v6': 'Rishikesh', 'v7': 'Haridwar', 'v8': 'Bijnor'}
pop = {'v4': 0.5, 'v3': 0.75, 'v5': 0.6, 'v2': 1.0, 'v1': 0.3, 'v6': 0.7, 'v7': 1.2, 'v8': 0.8}  # millions
E1, delay, v = 240, 10, 0.24        # evacuate 1M people (min), detect+communicate (min), 4 m/s in km/min
tstar, Tend = 60, 36 * 60           # burst time and horizon (min)
out = {n: [b for (a, b) in D if a == n] for n in order}
tau = {e: round(D[e] / v) for e in D}

def reach(start):                   # earliest flood arrival (min), DAG shortest path
    R = dict.fromkeys(order, np.inf); R.update(start)
    for a in order:
        for b in out[a]:
            if b not in start:
                R[b] = min(R[b], R[a] + tau[(a, b)])
    return R

def simulate(src):                  # flow q_i[k] in million litres/s
    q = {n: np.zeros(Tend) for n in order}
    for k in range(Tend):
        for n in order:
            if n in ('v01', 'v02'):
                q[n][k] = 35 if (n in src and k >= tstar) else 9
            else:
                q[n][k] = sum(q[a][max(k - tau[(a, b)], 0)] / len(out[a]) for (a, b) in D if b == n)
    return q

# Part 3: FlashFloodReach and evacuation slack for each burst scenario
for src in [['v01'], ['v02'], ['v01', 'v02']]:
    R = reach({s: 0 for s in src})
    print('Burst at', src)
    for n in town:
        E = pop[n] * E1
        print(f'  {n} {town[n]:12s} FFR={R[n] / 60:5.2f} h  Evac={E / 60:4.2f} h  slack={(R[n] - delay - E) / 60:5.2f} h')

# Part 4: v4 reports the flood 60 min earlier than predicted -> downstream update
R = reach({'v01': 0})
Rn = reach({'v01': 0, 'v4': R['v4'] - 60})
print('Update:', {n: f'{R[n] / 60:.2f}->{Rn[n] / 60:.2f} h' for n in town if R[n] < np.inf})

# Simulation: burst at v01, check arrival times match FlashFloodReach
q = simulate(['v01'])
t = np.arange(Tend) / 60
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for n in town:
    arr = np.argmax(q[n] > q[n][0] + 1e-9) if (q[n] > q[n][0] + 1e-9).any() else None
    print(f'{n}: simulated arrival={arr}, predicted={tstar + R[n]}')
    ax[0].plot(t, q[n], label=f'{n} {town[n]}')
    if arr:
        ax[1].barh(town[n], pop[n] * E1 / 60, left=(tstar + delay) / 60, color='tab:green')
        ax[1].plot((tstar + R[n]) / 60, town[n], 'rv', ms=10)
ax[0].axvline(tstar / 60, color='k', ls='--')
ax[0].set(xlabel='time (h)', ylabel='flow (million L/s)', title='Flow at each town (burst at $v_{01}$, $t^*=1$ h)')
ax[0].legend(fontsize=8)
ax[1].set(xlabel='time (h)', title='Evacuation (green) vs flood arrival (red)')
plt.tight_layout(); plt.savefig('p3_sim.png', dpi=200)
