# PNDS v1.4 — Complete parallel information-flow formulation for a native TAC-OSM candidate

**Architecture status:** NEW UNTRAINED CANDIDATE, *not* observationally equivalent to the native recurrent DIAGKEY004 implementation. The original 101,527 trainable parameters, embedding tables, gates, 8×8 associative value-by-key store, old-memory-read semantics, readout/codebook and learned correction maps are reused; the *fast/slow recurrence is replaced* by a causal parallel hierarchical summary and a subsequent read-refinement stage. The source model is never overwritten. No new training, real qualification, checkpoint gain or action/intervention claim follows.

**Mathematical authority:** PNDS/ParallelMemory.lean. **Executable graph authority:** TAC-OSM PR #389, src/tac_osm/variants/full_time_parallel_native.py. Source anchor: native DIAGKEY004 commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8, but the candidate is a semantically distinct graph from this source.

## Explicit typed input→output graph

For batch B, observed sequence length T, event IDs c_t, key IDs i_t and value IDs j_t, define learned e_t∈R^48, normalized key k_t∈R^8 and normalized value v_t∈R^8. Let \(\mathcal F\) and \(\mathcal S\) denote the unchanged learned 2×88-output fast/slow linear maps, with candidate/gate splitting; G the source two-output memory-gates linear map; D the source readout MLP. Operations with subscript t here are independent B×T tensor batches except causal prefix reductions.

**Stage P0 — parallel perception**

\[
e_t=\tanh(W_e[\mathrm{emb}_c(c_t)+\mathrm{emb}_k(i_t)+\mathrm{emb}_v(j_t)]+b_e),\qquad
k_t=\operatorname{normalize}(\mathrm{address}(i_t)),\
v_t=\operatorname{normalize}(\mathrm{content}(j_t)).
\]

**Stage P1 — causal time-parallel fast and slow candidate summaries**

\[
a_t = \tanh(\mathcal F_p(e_t,0,0)),\qquad
c_t^{\rm past}= \frac{\sum_{i<t} a_i}{\max(t,1)}.
\]

A prefix-sum is parallelizable with O(log T) arithmetic depth and O(T) work in an efficient work-conserving implementation; actual torch.cumsum implementation costs depend on backend.

\[
(f_t^0,g_t^f)=\mathrm{split}(\mathcal F(e_t,c_t^{\rm past},0)),
\quad
\bar f_t=(1-\sigma(g_t^f))c_t^{\rm past}+
\sigma(g_t^f)\tanh(f_t^0).
\]

\[
(\delta_t^s,g_t^s)=\mathrm{split}(\mathcal S(\bar f_t,c_t^{\rm past},0)),
\quad
\bar s_t=(1-\tfrac14\sigma(g_t^s))c_t^{\rm past}
+\tfrac14\sigma(g_t^s)\tanh(c_t^{\rm past}+\tanh(\delta_t^s)).
\]

Unlike the original, these fast and slow summaries do **not** recursively depend on their previous learned registers. The prefix only reads observed earlier events; no future leakage or instantaneous memory cycle.

**Stage P2 — fully known parallel memory gate schedule**

\[
(\alpha_t^{\rm raw},\beta_t)=\sigma(G(e_t,\bar f_t,\bar s_t)),
\quad
\alpha_t=\alpha_t^{\rm raw}(1-\beta_t/R),\quad R=4.
\]

For writable events define
\[
P_t=\alpha_t(I-\beta_t k_t k_t^\top),\qquad
Q_t=\beta_t v_t k_t^\top,
\]
and for read-only events use \(P_t=I,Q_t=0\), exactly preserving an old memory in real arithmetic.

**Stage P3 — associative chronological prefix scan**

The memory update is \(M_t=M_{t-1}P_t+Q_t\), \(M_0=0\). If \(A\) represents an earlier contiguous interval and \(B\) follows it, define

\[
 (P_A,Q_A)\diamond(P_B,Q_B)
 =
 (P_AP_B,\ Q_AP_B+Q_B).
\]

Associativity is proved as **a matrix operator law**, not assumed. Rebalancing chronological parentheses (never reversing event order) computes all \(M_t\) in O(log T) scan stages using batched 8×8 matrix products; the specific Hillis–Steele implementation has O(T log T) matrix-multiply work and O(T) stored states, not O(T) work. A work-efficient alternative may be engineered separately. A causal prefix-max gather makes read-only entries bitwise equal to the last actual write despite floating-point reassociation in the scanned matrix products.

**Stage P4 — time-parallel read-before-write and refinement**

\[
r_t=M_{t-1}k_t;\qquad
f_t=\operatorname{lerp}(\bar f_t,\tanh(\mathcal F_p(e_t,\bar f_t,r_t)),\sigma(\mathcal F_g(\ldots))),
\]
and
\[
s_t=\operatorname{lerp}(\bar s_t,
\tanh(\bar s_t+\tanh(\mathcal S_\delta(f_t,\bar s_t,r_t))),
\tfrac14\sigma(\mathcal S_g(\ldots))).
\]

Both are parallel across observed time because the *already-computed* old memory prefix is read, not fed back into the gate schedule. This is a deliberate architectural separation. It limits iterative hypothesis correction; any looped refinement would need explicit depth and a new computation contract.

**Stage P5 — independent final decoder**

\[
r_T^{\rm final}=M_T k_T,\qquad
\ell = D([f_T,s_T,r_T^{\rm final},e_T])
 + \exp(\operatorname{clamp}(\mathrm{read\_gain},-2,4))
    \left(r_T^{\rm final} C^\top\right),
\]
where C is the normalized original learned value-codebook. This output has 32 class logits, just like the original DIAGKEY004 source.

This is a **complete** input-to-output time-parallel mathematical DAG for the *candidate*. It is not a completed mathematical description for the enterprise, counterfactual or real physical versions with different inputs, missingness, actions, continuous-time dynamics or measured cost.

## Projection-free bounded associative memory

Assume \(0\le\alpha_t^{raw},\beta_t\le1\), \(R\ge1\), \(\|k_t\|_2,\|v_t\|_2\le1\).
Since \(I-\beta_t k_t k_t^\top\) has eigenvalues in [0,1], its spectral norm ≤1. Hence for writable events,

\[
\begin{aligned}
\|M_t\|_F&\le\alpha_t\|M_{t-1}\|_F+
\beta_t\|v_t k_t^\top\|_F\\
&\le\alpha_t\|M_{t-1}\|_F+\beta_t.
\end{aligned}
\]

If \(\|M_{t-1}\|_F\le R\), then

\[
\|M_t\|_F\le\alpha_t^{raw}(R-\beta_t)+\beta_t\le R.
\]

Thus from zero memory the radius-R budget is invariant WITHOUT the source's global norm projection. This is a mathematical sufficiency certificate for an alternative retention law, not evidence that old information persists longer: applying the budget to every write can increase forgetting. PNDS/ParallelMemory.lean proves the generic right-affine composition, identity, chunk-fold semantics, and the **conditional scalar capacity inequality** under an explicit norm recurrence premise. The matrix spectral-norm inequality needs its own Lean norm-operator formalization for full machine-checked status; here it is an analytical lemma and PyTorch test obligation.

## Computational complexity and honest parallelism

- Trainable parameters: **exactly the same 101,527 source parameters**, reused in the candidate. No invented padding parameters, no new weights.
- Work: learned dense projections O(BT·P) for fixed hidden widths (with repeated source fast/slow maps), 8×8 prefix scan O(BT log T·8³) using Hillis–Steele, and full output codebook read O(B·32·8). The original nonlinear recurrent transition remains O(T) serial depth.
- Candidate critical path: parallel input GEMMs, O(log T) causal prefix reduction, two provisional learned stages, O(log T) gated-delta affine scan, one read-refinement stage, and the decoder. No Python T-length recurrence loop appears in its forward pass.
- Memory: O(BT(88+8²)) for training activations and scan intermediates; full scan can cost more compute/memory than sequential on a CPU. Parallelizable does not mean measured faster.
- Batch, entity and horizon independence are separate from legal chronological grouping. Floating-point grouping changes last-bit results; validate numerically and preserve source read-only identity.

## Empirical capability is unqualified

No new training has been run. Because the fast/slow recurrence and gate timing differ, original pretrained checkpoint behavior is **not preserved**; a freshly trained candidate must be compared fairly to original native TAC and real structured/GRU/Transformer baselines on authentic longitudinal data. Matching parameter count alone is insufficient: exposure, data/time split, optimizer, **training compute** and prediction targets must also match or a superiority claim is prohibited. CPU wall-clock time is an efficiency measurement, not an equal-budget fairness requirement.

Original DIAGKEY004 synthetic immediate recall, ME094 stationarity failures, ESWAM low-speed failures, DEV long-memory failures and all 17 indexed variants are retained verbatim. Demonstrating mathematical graph structure is not demonstrating intelligence.

## Engineering levels

- **S0** exact typed candidate graph and source anchoring: implemented.
- **S1** exact real associative matrix chronological composition and conditional budget: Lean proof + explicit assumptions.
- **S2** actual PyTorch forward/gradient/causal invariants and original-source *partitioned path* equivalence: subject to verified CI.
- **S3** full network Lean tensor/FP contract, spectral/stability and measured parallel speed: OPEN.
- **S4** authentic real-data generalization, state-usefulness, decision consequences against matched baselines: OPEN.

The above is the strongest sound whole-graph parallelization construction currently justified by source and formal mathematics. An exactly output-identical time-parallel replacement for the original nonlinear native recurrence is **not** established.
