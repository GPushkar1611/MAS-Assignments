# AI3403 Multi-Agent Systems — Final Exam: Plan & Checklist

**Posted:** Mon 16 Nov 2026, 06:00 · **Due:** Tue 17 Nov 2026, **06:00 (hard, no extensions)** · **Total:** 300 pts

## 0. Submission rules (do not lose marks here)

- [ ] **One PDF** named **`ai24btech11012Exam.pdf`**, uploaded to Google Classroom.
- [ ] **Attach the exam's first page** (the rules and score-table page), and write your **Name + Roll No.** on it.
- [ ] Show **all the work** that leads to each answer, not just the final results.
- [ ] **Attach all code** inside the same PDF (LaTeX `listings`/`minted` appendix). Keep the `.py` files too.
- [ ] **Declare AI-tool usage openly**: add a short "AI Usage Declaration" section saying what you used and for what.
- [ ] Typeset it in LaTeX (strongly encouraged).
- [ ] Problem 1: say whether you're **seriously interested** in pursuing the startup. The professor offered an introduction to the IITH incubation cell.

### Suggested folder layout (`final/`)

```
final/
├── README.md                  ← this file
├── IIT_Hyderabad_Multi_Agent_Systems_Final_Exam.pdf
├── report/
│   ├── main.tex               ← first page + 5 problem sections + AI declaration + code appendix
│   └── figs/                  ← logo, startup diagram, plots, Ganga network figure
├── p2_dlr.py                  ← DGD / Gradient Tracking / ADMM
├── p3_flood_sim.py            ← (optional/bonus) flash-flood simulation + spec monitor
├── p4_covid_sim.py            ← (optional) SEIR + epistemic signals demo
└── p5_check.py                ← (optional) numerical sanity check of the P5 proofs
```

### Suggested time budget (~22 h left from 08:00 Mon)

| Order | Problem | Pts | Type | Est. time |
|---|---|---|---|---|
| 1 | P2: Distributed linear regression | 75 | Code + plots + discussion | 3–4 h |
| 2 | P4: COVID with STL + epistemic logic | 100 | Logic encodings (no code required) | 3–4 h |
| 3 | P5: Percolation, scalability | 50 | Matrix algebra + proofs | 1.5–2 h |
| 4 | P3: Ganga flash-flood warning | 50 | Modelling + formal specs (+ bonus sim) | 3–4 h |
| 5 | P1: Startup pitch | 25 | 2-page proposal | 1.5 h |
| 6 | Assemble LaTeX, code appendix, proofread, upload | — | — | 2 h |

**Aim to upload by ~04:00 Tue** so you have a buffer before the deadline.

---

## Problem 1: MAS + Formal Verification startup pitch (25 pts)

**Required:** a proposal of **at most 2 pages** pitched to an angel investor for **₹1 crore**, covering all 7 items below.

### Steps
1. **Pick an idea** at the intersection of *multi-agent systems* and *formal verification*. Some directions to consider:
   - Runtime verification / "safety shields" for **multi-agent LLM workflows** (agents that call tools, pay, or write code), with STL/LTL monitors that block unsafe actions.
   - **Certified drone/robot swarms** for disaster response or warehouses, with mission specs in STL and correct-by-construction controllers. This ties in nicely with P3.
   - Verified **multi-robot fleet coordination** for factories and ports (deadlock and collision freedom proved before deployment).
   - Verified **smart-grid / EV-charging** agent coordination.
2. **Name + acronym + tagline + logo.** The logo can be hand-drawn and scanned, or made in TikZ.
3. **Core idea, novelty and excellence:** what exists today, what's missing (guarantees), and your edge (formal guarantees + scalability).
4. **Target domain and audience:** who pays (the customer) and who uses it, plus the market size in one line.
5. **Picture of the idea:** an architecture diagram (agents → spec/monitor layer → actions), in draw.io or TikZ.
6. **Budget for ₹1,00,00,000:** use a table whose rows add up exactly. A sample split:

   | Head | ₹ (lakh) |
   |---|---|
   | Team (3–4 engineers + 1 FM researcher, 12–18 mo) | 45 |
   | Compute / cloud / LLM APIs | 10 |
   | Hardware (drones/robots/test rig) | 15 |
   | Pilot deployments & customer trials | 8 |
   | IP, legal, incorporation, compliance | 5 |
   | Marketing / BD / events | 7 |
   | Contingency | 10 |
   | **Total** | **100** |
7. **Expected turnover in 2 years:** a simple year-1 / year-2 projection (number of pilots, revenue per pilot or subscription), stated with its assumptions.
8. **6-month work package:** a table or Gantt chart with M1–M6 milestones and deliverables (for example, M1–2 spec language and monitor prototype, M3–4 simulator demo, M5 first pilot MoU, M6 MVP and first paying customer).
9. Add the line saying whether you're serious about pursuing it.

---

## Problem 2: Distributed Linear Regression (75 pts), code required

**Required:** implement **DGD, Gradient Tracking (GT) and ADMM** on the *same* data, graph and initialisation. Plot, for all three methods on the same axes: (1) consensus error, (2) ‖θ̄−θ*‖, (3) F(θ̄). Say which method you prefer and justify it with your results.

> Note: the PDF says θ* ∈ ℝᴺ, which is a typo; it should be **θ* ∈ ℝᵈ (d = 20)**. Mention this in the report.

### Steps (`p2_dlr.py`)
1. **Setup:** `N=20, d=20, m=500, sigma=0.1`, fix `np.random.seed(...)`.
   - `theta_star ~ N(0, I_d)`, `A_i ~ N(0,1)^{m×d}`, `b_i = A_i θ* + ε_i`.
2. **Graph:** a connected undirected graph, e.g. a ring plus a few random chords, or Erdős–Rényi with p≈0.2, with a check that it is connected (`networkx.is_connected`). Plot it for the report.
3. **Mixing matrix W:** use **Metropolis–Hastings** weights: `w_ij = 1/(1+max(d_i,d_j))` for each edge, `w_ii = 1 − Σ_j w_ij`. Check numerically that W is symmetric, has rows and columns summing to 1, and has spectral gap > 0 (report λ₂(W)).
4. **Reference solution:** compute the centralised least-squares θ_LS = (Σ AᵢᵀAᵢ)⁻¹ Σ Aᵢᵀbᵢ and F* = F(θ_LS). This gives you a floor for each plot (‖θ_LS−θ*‖ ≈ σ√(d/Nm) is the best any method can reach).
5. **Step size:** L = maxᵢ λ_max(AᵢᵀAᵢ)/m (≈1.4–1.5 here). Use the **same α** (e.g. α = 1/L, or tune) for DGD and GT. Same init for all three: θᵢ⁰ = 0.
6. **Algorithms** (stack all agents as a matrix Θ ∈ ℝ^{N×d}; `g(Θ)` = per-agent gradients `(1/m)Aᵢᵀ(Aᵢθᵢ − bᵢ)`):
   - **DGD:** `Θ ← WΘ − α g(Θ)`. With a constant α it converges only to an O(α) neighbourhood (a biased fixed point), so you should see a plateau.
   - **Gradient Tracking:** `Θ⁺ = WΘ − αY`, `Y⁺ = WY + g(Θ⁺) − g(Θ)`, with `Y⁰ = g(Θ⁰)`. It converges exactly and linearly.
   - **Decentralised ADMM** (Shi et al. 2014 / Boyd style), with penalty ρ, duals φᵢ⁰ = 0:
     - θᵢ⁺ = (AᵢᵀAᵢ/m + 2ρdᵢI)⁻¹ [Aᵢᵀbᵢ/m − φᵢ + ρ Σ_{j∈Nᵢ}(θᵢ + θⱼ)]
     - φᵢ⁺ = φᵢ + ρ Σ_{j∈Nᵢ}(θᵢ⁺ − θⱼ⁺)
     - Pre-factorise the d×d matrices once, since d=20 makes this cheap. Tune ρ over a few values (e.g. 0.1, 1, 10) and report the one you choose.
   - **Raw data (Aᵢ, bᵢ) is never exchanged.** Only θ (and Y for GT) goes to neighbours. State this explicitly.
7. **Record every iteration** (K ≈ 1000–3000): θ̄[k], E_con[k], E_true[k], F(θ̄[k]). Also log **F(θ̄)−F\***, because it shows convergence more clearly on a log scale.
8. **Plots** (log y-axis, all 3 methods per figure, with legend, axis labels and the θ_LS floor drawn as a dashed line): `p2_consensus.png`, `p2_true_error.png`, `p2_objective.png` (plus optionally a plot of the optimality gap).
9. **Discussion:** compare
   - iterations to reach tolerance, final accuracy (DGD bias vs exact convergence for GT/ADMM),
   - **communication per iteration** (DGD: 1 vector, GT: 2 vectors, ADMM: 1 vector),
   - **computation per iteration** (gradient vs local solve), and sensitivity to α and ρ.
   - Give a **clear preference** backed by your numbers, e.g. a table of "iterations to reach ‖θ̄−θ*‖ within 1% of the floor".
10. Optional extras: a sweep over α and ρ, or a comparison across graph densities.

---

## Problem 3: Ganges flash-flood warning system (50 pts)

**Required:** answer all 5 sub-parts as if writing a real proposal. Bonus marks for a flood simulation.

### Step 0: Fix the network and assumptions (one table in the report)
- **Directed edges** (water flows downward), read from the figure. Double-check them:
  `v01→v4, v02→v3, v4→{v5, v1, v2}, v3→{v5, v2}, v5→v1, v2→v1, v1→{v6, v7, v8}`
  (This matches the PDF's example, where v4 splits equally into v1, v2 and v5.)
- **Equal split rule:** outflow of node i = inflow / out-degree(i).
- **Name the towns** using Google Maps, for example:
  - Bhagirathi: v01 Gangotri glacier → v4 Uttarkashi → v5 Tehri
  - Alaknanda: v02 (Satopanth/Bhagirath-Kharak glacier; the PDF says Gangotri) → v3 Joshimath → v2 Rudraprayag
  - v1 Devprayag (confluence) → v6, v7, v8 = e.g. Rishikesh, Haridwar, Bijnor (or Kanpur, Prayagraj, Varanasi)
- **Distances dᵢⱼ** for every edge, measured along the river or road on Google Maps. Make up plausible values for the artificial edges (v4→v2, v3→v5). Put them in a table.
- **Speeds:** 3 m/s normally, 4 m/s in a flood. Travel time τᵢⱼ = dᵢⱼ / v (e.g. 100 km at 4 m/s ≈ 6.9 h).
- **Flows:** 18 ML/s total normally (e.g. 9 ML/s from each glacier) and 70 ML/s in a flood.
- **Population Pᵢ** per town (e.g. v2 = 1 M, v3 = 0.75 M, ...). **EvacuateTime** for 1 M people (e.g. 6 h), scaled linearly: `EvacuateTime_i = (P_i / 1M) × EvacuateTime`.

### 3.1 Network dynamics model
- **Discrete time** step Δt (e.g. 1 min). The state of node i at time k, xᵢ[k], could include:
  - the flow/volume through the node qᵢ[k] (L/s), the water level hᵢ[k] (an optional rating curve h = c·q^β),
  - a flood flag (qᵢ > q_thr), the alert status, the evacuated fraction eᵢ[k] ∈ [0,1], the estimated arrival time.
- **Inputs:** glacier flows u₀₁[k], u₀₂[k] as step functions: normal value before the burst time t\*, flood value after.
- **Delayed routing equation:**
  `q_j[k] = Σ_{i ∈ parents(j)} q_i[k − τ_ij] / outdeg(i)`, where τᵢⱼ depends on the regime (3 or 4 m/s).
- Show that it satisfies **conservation** (inflow = outflow at each junction) and that the abrupt jump at t\* propagates downstream.

### 3.2 Five safety specifications (in words, then formal logic, with a reason for each logic choice)
Suggested labels (keep or change them):
1. `DetectBurst`: G(burst → F_[0,δ_det] alarm). **MTL**, because it has a real-time bound.
2. `AlertDownstream`: G(alarm(t\*) → ∧_{i∈Down(src)} F_[0,δ_comm] alertᵢ). **MTL / first-order over nodes.**
3. `EvacuatePopulation`: G(alertᵢ → F_[0, EvacuateTimeᵢ] evacuatedᵢ), with the deadline alert_timeᵢ + EvacuateTimeᵢ ≤ FlashFloodReach(i,t\*). **MTL.**
4. `NoPeopleInFlood`: G(qᵢ(t) ≥ q_flood → occupancyᵢ(t) = 0). **STL**, because qᵢ is a real-valued signal and STL gives a robustness margin.
5. `SafeReturn` / `AlertPersistence`: alertᵢ U G_[0,T_clear](qᵢ < q_safe), i.e. no return until the flow has been safe for T_clear. **STL with Until.**
- **Reasons for the choices:** LTL handles only ordering (no time bounds); MTL adds deadlines; STL adds real-valued signals and quantitative robustness. Optionally, epistemic logic covers "town i *knows* a flood is coming", which links to P4.

### 3.3 FlashFloodReach(i, t\*)
- Define it as the **earliest arrival time** over all paths from the burst source(s):
  `FlashFloodReach(i,t*) = t* + min_{paths p: src→i} Σ_{(u,w)∈p} d_uw / v_flood`
  (for a burst at both glaciers, also take the min over both sources).
- **Warning rule:** issue the alert to i at t\* + δ_det + δ_comm and require **slack**
  `Sᵢ = FlashFloodReach(i,t*) − (t* + δ_det + δ_comm) − EvacuateTimeᵢ ≥ 0`.
  If Sᵢ < 0, flag i as **infeasible**, which triggers ImmediateDrill (3.5) and a priority evacuation.
- Give a **table** for each scenario (burst at v01 only, v02 only, both): arrival time, EvacuateTime, slack, and feasible yes/no for every town.

### 3.4 Update logic using the topology, plus the optional simulation
- When upstream town u reports its **observed** arrival time t_obs(u) and flow q_obs(u), each downstream town j recomputes along the DAG **in topological order**:
  `FFR(j) = min_{i∈parents(j)} [ FFR(i) + d_ij / v ]` with FFR(u) := t_obs(u). This is a Bellman–Ford / DAG shortest-path relaxation.
  The expected magnitude is updated too: `q_j ≈ Σ q_i / outdeg(i)`. Also re-estimate v from the observed travel times if you like.
  Write this as a specification as well (e.g. G(msg_u → F_[0,δ] updated_j) for j ∈ Desc(u)).
- **Bonus `p3_flood_sim.py`:** simulate the routing equation, inject a burst at v01 and/or v02, and plot
  (a) qᵢ(t) for every node, (b) a timeline of alerts, evacuations and flood arrivals per town, and (c) **a check that every spec holds**
  (compute each STL robustness value by hand with min/max over windows, and confirm the simulated arrival times match the 3.3 formula).

### 3.5 ImmediateDrill (no alert system, like Nepal)
- The town has only **local** information, so the specification triggers on local sensing:
  - `G( (dhᵢ/dt > r_thr ∨ hᵢ > h_warn ∨ unusual_turbidity ∨ quake_felt ∨ upstream_comm_lost) → F_[0,T_drill] evacuated_riverbank_i )`
  - plus standing rules: `G(occupancy_floodplain_i ≤ cap)`, routes pre-assigned (`G F_[0,T_period] drill_conducted`), sirens tested.
- Encode it in the **same logic** you chose above and justify it (a worst-case assumption: arrival could be "now").

---

## Problem 4: COVID control with STL + Epistemic Logic (100 pts)

**Required:** answer and encode all 10 parts (including the sub-parts) by combining STL and EL. Code is not required. A small simulation is a nice extra if time allows.

### Setup to write first
- Define Kᵢφ ("agent i knows φ") using Kripke / possible-worlds semantics: Kᵢφ holds iff φ is true in every world i considers possible. State the **S5 axioms**, especially **T (veridicality): Kᵢφ → φ**.
- State how the Boolean signals are formed: Infectedᵢ[k], Exposureᵢ[k], Kᵢ(·)[k] become 0/1 signals, so STL can operate on their sums (N_E, N_I, ...).
- Testing changes only the epistemic state, never xᵢ.

### Part-by-part plan
1. **Infectedᵢ ⇔ xᵢ[k] ∈ {E, I}.** (S and R are not infected.)
2. Encodings:
   - (a) Infectedᵢ ∧ ¬Kᵢ Infectedᵢ
   - (b) Kᵢ Infectedᵢ (by axiom T this already implies Infectedᵢ; writing Infectedᵢ ∧ Kᵢ Infectedᵢ is fine too)
   - (c) ¬Kᵢ Infectedᵢ ∧ ¬Kᵢ ¬Infectedᵢ
   - (d)(i) ¬Kᵢ ( ∧_{j∈Nᵢ^inf} Infectedⱼ ∧ ∧_{j∈Nᵢ∖Nᵢ^inf} ¬Infectedⱼ )
   - (d)(ii) i knows *some* 2-subset of its 5 neighbours is infected but not which one:
     Kᵢ( ∨_{S⊆Nᵢ,|S|=2} [∧_{j∈S} Infⱼ ∧ ∧_{j∉S} ¬Infⱼ] ) ∧ ∧_{S,|S|=2} ¬Kᵢ[∧_{j∈S} Infⱼ ∧ ∧_{j∉S} ¬Infⱼ]   (C(5,2) = 10 subsets; write one of them out explicitly)
   - (d)(iii) Nᵢ^knows = { j ∈ Nᵢ : Kⱼ Infectedᵢ ∧ ¬Kⱼ Kᵢ Infectedᵢ ∧ ¬Kⱼ ¬Kᵢ Infectedᵢ }   (j is unsure about i's awareness)
   - (d)(iv) Nᵢ^knows² = { j ∈ Nᵢ : Kⱼ Kᵢ Infectedᵢ }   (second-order knowledge)
3. **p_i^infected[k]**, for a susceptible i, using the controlled graph Aᵘ:
   `p_i^inf[k] = 1 − Π_{j∈Nᵢ[k]} (1 − β_E·1{xⱼ=E})(1 − β_I·1{xⱼ=I}) = 1 − (1−β_E)^{n_E,i[k]} (1−β_I)^{n_I,i[k]}`
   State the assumption that transmissions are independent, and note that it is 0 if i is isolated.
4. Kᵢ Exposureᵢ ∧ ¬Kᵢ Infectedᵢ ∧ ¬Kᵢ ¬Infectedᵢ. **Why Kᵢ Exposureᵢ does not imply Kᵢ Infectedᵢ:** transmission succeeds only with probability p < 1, so among the worlds i considers possible there is one where exposure happened but transmission did not. Knowledge needs truth in *all* accessible worlds. Give a 2-world Kripke counterexample.
5. **G_[0,∞) G_[0,T] (I_max − N_I[k] > 0)**, i.e. for every t the window [t, t+T] stays below I_max. Note that it reduces to G(N_I < I_max); explain why.
6. **F_[0,T_suppress] G_[0,T_wait] (I_safe − N_I[k] > 0).**
7. **G( Kᵢ Infectedᵢ → G_[0,T_isolate] (uᵢ^isolate = 1) ).** Discuss whether to count from the first time Kᵢ becomes true, e.g. trigger on the rising edge.
8. **G( (Kᵢ Exposureᵢ ∧ ¬Kᵢ Infectedᵢ ∧ ¬Kᵢ¬Infectedᵢ) → F_[0,T_test] (uᵢ^test = 1) ).**
9. **Controlled graph:** aᵢⱼᵘ[k] = aᵢⱼ[k] (1 − uᵢ^iso[k]) (1 − uⱼ^iso[k]). In matrix form, **Aᵘ = D A D** with D = diag(1 − u^iso). Isolation removes *all* edges of the node (its row and column).
10. Counting signals (STL over the summed Boolean signals):
    - (a) N_KI[k] = Σᵢ 1{Kᵢ Infectedᵢ[k]}, where Kᵢ Infectedᵢ[k] is true iff i tested positive at some k' ≤ k (and has not since been confirmed recovered)
    - (b) N_KE[k] = Σᵢ 1{Kᵢ Exposureᵢ[k]}, true iff i received a contact-tracing notification (a contact who tested positive within the last L steps)
    - (c) N_KE,T[k] = Σᵢ 1{Kᵢ Exposureᵢ ∧ ∃k'∈[k_notifᵢ, k]: uᵢ^test[k'] = 1}
    - Optionally add STL requirements on these, e.g. G(N_KE,T ≥ 0.9·N_KE).
- **Optional `p4_covid_sim.py`:** a small SEIR simulation on a random time-varying graph with testing, contact tracing and isolation, plotting N_I, N_KI and N_KE and checking specs 5–8 with robustness values. Run it once with and once without epistemic-aware control to show that it **improves** control, which is the stated goal of the question.

---

## Problem 5: Scalability & robustness under percolation (50 pts)

**Required:** (1) write Ā and M̄ for each of the 4 elementary changes, and (2) give three proofs.

### 5.1 New matrices (E := [I_{N−1} 0] ∈ ℝ^{(N−1)×N}, eᵢ = unit vector)
| Change | Ā | M̄ |
|---|---|---|
| Remove isolated node N | E A Eᵀ | E M Eᵀ |
| Remove edge (i,j) | A | M − mᵢⱼ eᵢ eⱼᵀ |
| Add isolated node N+1 | diag(A, a_{N+1}) | diag(M, 0) = [M 0; 0 0] |
| Add edge (i,j), weight mᵢⱼ>0 | A | M + mᵢⱼ eᵢ eⱼᵀ |

### 5.2 Proofs, all using the LP certificate (9): Σ is γ-robust ⇔ ∃v with −(A−M)v + 1 ≤ 0, 0 < v ≤ γ1
- **Key observation:** −(A−M) is Metzler, and a v > 0 with (A−M)v ≥ 1 > 0 makes A−M a non-singular M-matrix, so −(A−M) is Hurwitz. That is why (9) alone gives stability. Cite the equivalence the PDF states.
- **(a) Deleting an isolated node:** since node N has no edges, A−M = diag(Ā−M̄, a_N) is block-diagonal. Take v̄ = E v. Then −(Ā−M̄)v̄ + 1 = E(−(A−M)v + 1) ≤ 0, and 0 < v̄ ≤ γ1. So Σ̄ is γ-robust for every γ that works for Σ, which means the change is scalable.
- **(b) Removing an edge:** keep the same v. −(A−M̄)v + 1 = −(A−M)v + 1 − mᵢⱼvⱼ eᵢ ≤ −mᵢⱼvⱼ eᵢ ≤ 0, since mᵢⱼ > 0 and vⱼ > 0. The bound v ≤ γ1 is unchanged, so Σ̄ is γ-robust.
- **(c) Adding an edge (i,j):** keep the same v. Only row i changes:
  −aᵢvᵢ + Σ_{k∈Nᵢ} mᵢₖvₖ + mᵢⱼvⱼ + 1 ≤ 0 ⇔ aᵢ ≥ (Σ_{k∈Nᵢ} mᵢₖvₖ + mᵢⱼvⱼ + 1)/vᵢ, which is exactly (10). All other rows are unchanged and still satisfy (9), so v certifies γ-robustness of Σ̄.
  Point out that Nᵢ means the neighbourhood *before* the addition, and that vᵢ > 0 keeps the division valid.
- **Optional `p5_check.py`:** for a random γ-robust network, compute u = (A−M)⁻¹1, verify Hurwitz and u ≤ γ1 before and after each change, and show a counterexample where (10) fails and robustness is lost. This takes about 30 lines of numpy.

---

## Final assembly checklist
- [ ] First page of the exam attached, with Name and Roll No. filled in
- [ ] P1 is at most 2 pages, contains all 7 items, and the budget adds up to ₹1 crore
- [ ] P2: 3 plots (log scale, all methods on each), W shown to be doubly stochastic, a clear preference with numbers
- [ ] P3: assumptions table, dynamics equation, 5 specs with logic justification, FFR formula and table, update rule, ImmediateDrill (+ bonus sim)
- [ ] P4: all parts 1–10, including 2(d)(i)–(iv) and 10(a)–(c)
- [ ] P5: matrices table plus proofs (a), (b), (c)
- [ ] AI Usage Declaration
- [ ] Code appendix (all `.py` files) in the same PDF
- [ ] File named **`ai24btech11012Exam.pdf`** and uploaded **before 06:00 Tue 17 Nov**
