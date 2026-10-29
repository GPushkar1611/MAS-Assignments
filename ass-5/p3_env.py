import heapq
import numpy as np

T = [20, 30, 60, 40, 30]                      # service minutes at D1..D5
DAY, DAYS = 480, 365
SCEN = {"S1": 0.75, "S2": 1.0, "S3": 1.25, "S4": 1.5}
DEPTS = ["D1", "D2", "D3", "D4", "D5"]


class Env:
    """365-day administration simulator. step(a) takes one lending action per department."""

    def __init__(self, theta, seed, feedback=True):
        self.theta, self.fb = theta, feedback
        self.rng = np.random.default_rng(seed)
        self.d, self.q, self.p_prev = 0, 0.0, 0.5
        self.arr, self.comp, self.rem, self.stage = [], [], [], []   # per-application records
        self.qs = [[] for _ in range(5)]                             # FIFO queue of app ids per dept
        self.Q, self.m_hist, self.r_list, self.q_list, self.A_list, self.same = [], [], [], [], [], 0
        self._arrivals()

    def _arrivals(self):                                             # arrivals of day d use q_{d-1}
        self.d += 1
        lam = 16 * self.theta * (1 + 0.5 * self.q) if self.fb else 16 * self.theta
        self.A = self.rng.poisson(lam)
        for _ in range(self.A):
            self.qs[0].append(len(self.arr))
            self.arr.append(self.d); self.comp.append(0); self.rem.append(T[0]); self.stage.append(0)

    def obs(self):                                                   # (Q_{i-1}, Q_i, Q_{i+1}, q_{d-1})
        b = [len(x) for x in self.qs]
        return [(b[i - 1] if i > 0 else 0, b[i], b[i + 1] if i < 4 else 0, self.q) for i in range(5)]

    def step(self, a):
        a = list(a); a[0] = max(a[0], 0); a[4] = min(a[4], 0)        # boundary departments
        m = [3 - (a[i] != 0) + (i > 0 and a[i - 1] == 1) + (i < 4 and a[i + 1] == -1) for i in range(5)]
        d, ready, cs = self.d, [0] * len(self.arr), 0
        for i in range(5):
            free, left = [0] * m[i], []                              # officer free-times, apps for tomorrow
            order = sorted(self.qs[i], key=lambda k: (ready[k], k))  # FIFO
            for n, k in enumerate(order):
                start = max(free[0], ready[k])
                if start >= DAY:                                     # no overtime
                    left += order[n:]
                    break
                end = min(start + self.rem[k], DAY)
                heapq.heapreplace(free, end)
                self.rem[k] -= end - start                           # partial work is preserved
                if self.rem[k] > 0:
                    left.append(k)
                elif i == 4:
                    self.comp[k] = d
                    cs += self.arr[k] == d
                else:
                    self.stage[k], self.rem[k], ready[k] = i + 1, T[i + 1], end
                    self.qs[i + 1].append(k)
            self.qs[i] = left
        b = [len(x) for x in self.qs]
        p = cs / self.A if self.A else self.p_prev
        self.p_prev = p
        self.q = 0.8 * self.q + 0.2 * (2 * p - 1)
        r = 5 * cs - 10 * sum(b)
        self.Q.append(b); self.r_list.append(r); self.q_list.append(self.q)
        self.m_hist.append(m); self.A_list.append(self.A); self.same += cs
        done = d == DAYS
        if not done:
            self._arrivals()
        return self.obs(), r, done

    def result(self):
        Q = np.array(self.Q)
        return dict(Q=Q, B=Q.sum(1), R=np.cumsum(self.r_list), q=np.array(self.q_list), A=np.array(self.A_list),
                    m=np.array(self.m_hist), arr=np.array(self.arr), comp=np.array(self.comp),
                    stage=np.array(self.stage), rem=np.array(self.rem), same=self.same)


def run(theta, seed, policy=lambda o: [0] * 5, feedback=True):
    env = Env(theta, seed, feedback)
    o, done = env.obs(), False
    while not done:
        o, _, done = env.step(policy(o))
    return env.result()


def stats(r):                                                        # items (a)-(g)
    done = r["comp"] > 0
    n = len(r["arr"])
    return [n, done.sum(), r["same"] / n, (r["comp"] - r["arr"])[done].mean(), r["B"][-1], r["R"][-1], r["q"][-1]]
