import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N, d, m, sigma, K = 20, 20, 500, 0.1, 300
alpha, rho = 0.15, 0.3

# Data
theta_star = np.random.randn(d)
A = np.random.randn(N, m, d)
b = A @ theta_star + sigma * np.random.randn(N, m)

# Graph: ring + random chords (always connected), Metropolis-Hastings weights
Adj = np.zeros((N, N))
for i in range(N):
    Adj[i, (i + 1) % N] = Adj[(i + 1) % N, i] = 1
R = np.triu(np.random.rand(N, N) < 0.1, 2).astype(float)
Adj = np.maximum(Adj, R + R.T)
deg = Adj.sum(1)
W = Adj / (1 + np.maximum.outer(deg, deg))
W += np.diag(1 - W.sum(1))

H = np.einsum('nki,nkj->nij', A, A) / m       # A_i^T A_i / m
g = np.einsum('nki,nk->ni', A, b) / m         # A_i^T b_i / m
grad = lambda T: np.einsum('nij,nj->ni', H, T) - g
F = lambda th: np.mean(np.sum((A @ th - b) ** 2, 1)) / (2 * m)
theta_ls = np.linalg.solve(H.sum(0), g.sum(0))  # centralized solution (reference only)

def record(T, h):
    tb = T.mean(0)
    h.append([np.sqrt(np.mean(np.sum((T - tb) ** 2, 1))), np.linalg.norm(tb - theta_star), F(tb),
              np.linalg.norm(tb - theta_ls)])

hist = {'DGD': [], 'GT': [], 'ADMM': []}

T = np.zeros((N, d))                          # DGD
for k in range(K + 1):
    record(T, hist['DGD'])
    T = W @ T - alpha * grad(T)

T = np.zeros((N, d)); S = grad(T)             # Gradient Tracking
for k in range(K + 1):
    record(T, hist['GT'])
    Tn = W @ T - alpha * S
    S = W @ S + grad(Tn) - grad(T)
    T = Tn

T = np.zeros((N, d)); P = np.zeros((N, d))    # Decentralized ADMM
Minv = np.linalg.inv(H + 2 * rho * deg[:, None, None] * np.eye(d))
for k in range(K + 1):
    record(T, hist['ADMM'])
    T = np.einsum('nij,nj->ni', Minv, g - P + rho * (deg[:, None] * T + Adj @ T))
    P += rho * (deg[:, None] * T - Adj @ T)

# Plots
titles = ['Consensus error $E_{con}[k]$', r'Error to true parameter $E_{true}[k]$',
          r'Global objective $F(\bar\theta[k])$', r'Distance to optimum $\|\bar\theta[k]-\theta_{LS}\|$']
refs = [None, np.linalg.norm(theta_ls - theta_star), F(theta_ls), None]
for q in range(4):
    plt.figure(figsize=(6, 4))
    for name, h in hist.items():
        plt.semilogy(range(1, K + 1), np.array(h)[1:, q], label=name)
    if refs[q]:
        plt.axhline(refs[q], color='k', ls='--', label='centralized LS')
    plt.xlabel('iteration $k$'); plt.title(titles[q]); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'p2_fig{q + 1}.png', dpi=200, bbox_inches='tight')

# Summary
print('edges:', int(Adj.sum() / 2), ' lambda_2(W):', round(np.sort(np.linalg.eigvalsh(W))[-2], 3))
print(f'LS: E_true={refs[1]:.2e}  F*={refs[2]:.5f}')
for name, h in hist.items():
    h = np.array(h)
    it = np.argmax(h[:, 3] < 1e-6) if (h[:, 3] < 1e-6).any() else '-'
    print(f'{name}: E_con={h[-1, 0]:.1e}  E_true={h[-1, 1]:.2e}  dist_LS={h[-1, 3]:.1e}  iters to 1e-6={it}')
