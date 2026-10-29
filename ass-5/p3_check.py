import numpy as np
from p3_env import *

# sanity: reproduce old fixed-staffing table (seeds 1000..1019) and check records vs queues
for n, th in SCEN.items():
    res = np.array([stats(run(th, 1000 + i)) for i in range(20)])
    print(n, np.round(res.mean(0), 2))
r = run(1.5, 0)
pend = r["comp"] == 0
print("records match queues:", pend.sum() == r["B"][-1],
      [int(((r["stage"] == i) & pend).sum()) for i in range(5)] == list(r["Q"][-1]))
