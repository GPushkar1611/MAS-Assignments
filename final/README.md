# AI3403 Multi-Agent Systems: Final Exam

Submission: `ai24btech11012Exam.pdf` (exam first page + all 5 problems + code + AI usage declaration).

## Problem 1: Start-up Pitch (25 pts)
- Pitched **NETRA** (Networked Epistemic–Temporal Runtime Assurance): a safety layer on every agent of a robot/drone fleet.
- Physical rules are written in STL and knowledge rules in Epistemic Logic, then enforced by a distributed STL–EL MPC on each agent.
- Logo and architecture diagram drawn in TikZ (no code).
- Includes the ₹1 crore budget, a 2-year turnover estimate (≈ ₹1.8 Cr) and a 6-month work package.

## Problem 2: Distributed Linear Regression (75 pts) — `p2.py`
- 20 agents on a ring + random chords graph (33 edges), Metropolis–Hastings mixing matrix W.
- Implemented **DGD**, **Gradient Tracking** and **decentralized (edge-based) ADMM** with the same data, graph and start (θ = 0).
- Plots: consensus error, error to θ*, global objective, and distance to the exact least-squares solution (`p2_fig1–4.png`).
- Result: DGD stalls near the optimum (constant-step bias), GT converges exactly in 213 iterations, ADMM in 60.
- **Preferred ADMM**: fastest, exact, and sends one vector per iteration.

## Problem 3: Ganges Flash Flood Warning (50 pts) — `p3.py`
- Mapped nodes to towns (Uttarkashi, Joshimath, Tehri, Rudraprayag, Devprayag, Rishikesh, Haridwar, Bijnor), using Google Maps road distances as a stand-in for river distances.
- **Model:** flow splits equally at each node and travels with a delay of distance / 4 m/s. Glaciers give 9 ML/s each (normal) and 35 ML/s each (burst).
- **5 specs:** DetectBurst (MTL), AlertDownstream (temporal-epistemic), EvacuatePopulation (MTL), NoPeopleInFlood (STL), SafeReturn (STL + EL).
- **FlashFloodReach:** earliest arrival time over all paths. Slack = arrival time − warning time − evacuation time. All towns have positive slack; Joshimath is tightest (1.07 h).
- **Update rule:** when an upstream town reports the actual arrival time, downstream towns recompute along the network.
- **Simulation (bonus):** burst at Gangotri. Simulated arrival times match the formula exactly (`p3_sim.png`).
- **ImmediateDrill:** with no alert system, towns act on the *possibility* of a flood (M_i) rather than knowledge (K_i).

## Problem 4: COVID Control with STL + Epistemic Logic (100 pts)
- No code needed; all 10 parts are logic encodings.
- Covers: Infected ⇔ x ∈ {E, I}; knowledge encodings (K_i, M_i, nested KⱼKᵢ); infection probability over the controlled neighbours; a two-world argument for why K_i Exposure ⇏ K_i Infected; STL specs for case limits, suppression, isolation and testing; A^u = D A D; and counting signals N_KI, N_KE, N_KET.
- `p4.py` / `p4_sim.png` are an optional extra simulation and are **not** used in the submission.

## Problem 5: Scalability Under Percolation (50 pts)
- No code needed.
- Wrote Ā and M̄ for all 4 changes (remove/add node, remove/add edge).
- Proved using condition (9) by reusing the same vector v:
  - (a) deleting an isolated node: drop the last entry of v;
  - (b) removing an edge: each row only gets more negative;
  - (c) adding an edge: only row i changes, which gives exactly condition (10).

## Running the code
```
python p2.py   # Problem 2 plots + numbers
python p3.py   # Problem 3 tables + simulation plot
```
