# TAC-OSM Dual-Mode Mathematical Engineering v1.5 — parallel batch + finite-state inference

Date: 2026-10-25. **Verified algebraic/implementation engineering, not complete mathematical proof of all neural tensors or independently qualified new skill.**

TAC-OSM source branch: [draft PR #389](https://github.com/FPC-effortless/TAC-OSM/pull/389).
PNDS proofs: [draft PR #7](https://github.com/FPC-effortless/pnds-lean/pull/7).
These PRs deliberately preserve original negative experiment records and the 17 source-pinned historical variant links.

## I. Nonnegotiable architecture identity

TAC-OSM is a Task–Action–Counterfactual Observational State Model family. **Do not replace the native model with a Mamba SSM** or rebrand a categorical keyed-recall classifier as a trained real-world world model. Separate two programs:
- **Source DIAGKEY004 (101,527 parameters)**: synthetic categorical role/key/value task, original nonlinear fast→slow→gated delta state. Its exact outputs are maintained only by its separate parallel-input preprojection implementation. The *fully parallel native-weight candidate* has deliberately changed those nonlinear state semantics.
- **Continuous robot pilot (93,324 TAC vs 93,454 GRU parameters)**: new action-conditioned next-observation model with train-only normalizations/residual scale and genuine 14D observed state and 14D logged action. It has its own parallel and streaming paths and did not inherit any original trained source model weights.

Do not claim causal interventions from observational logged actions.

## II. Source-shaped finite-state dual-mode TAC mathematical engine

Let the *streaming sufficient state* for one model instance be

\[
Z_t=(A_t,M_t,n_t),\qquad A_t\in\mathbb R^{88},\
M_t\in\mathbb R^{8\times8},\quad n_t\in\mathbb N.
\]

\(A_t\) is the accumulated causal input-only fast evidence, not a stored list of all past observations; \(M_t\) is the bounded learned value-by-key associative state; \(n_t\) is an event counter. This is **152 FP32 values = 608 bytes per sample plus the integer counter**, excluding model parameters, outputs, optimizer state and deployment-runtime memory. Fixed-size hidden state ≠ unlimited information capacity.

For observed encoded event \(e_t\), normalized learned key \(k_t\) and value \(v_t\), define input-only basis \(b_t=\tanh(\mathcal F_p(e_t,0,0))\), and exclusive causal context

\[
c_t=A_t/\max(n_t,1),\quad A_{t+1}=A_t+b_t,\quad n_{t+1}=n_t+1.
\]

Provisional fast/slow states are fully determined from \(e_t,c_t\) and the original learned maps, without reading \(M_t\). Compute known gate schedule

\[
(\bar\alpha_t,\beta_t)=\sigma(G(e_t,\bar f_t,\bar s_t)),\qquad
\alpha_t=\bar\alpha_t(1-\beta_t/R),\ R=4.
\]

For a writable input and key/value vector norm at most one,

\[
P_t=\alpha_t(I-\beta_tk_tk_t^{\!\top}),\quad
Q_t=\beta_tv_tk_t^{\!\top},\quad
M_{t+1}=M_tP_t+Q_t.
\]

Equivalently, streaming implements the lower-rank (O(8²)) form,

\[
M_{t+1}=\alpha_tM_t+
 \beta_t(v_t-\alpha_tM_tk_t)k_t^{\!\top}.
\]

Its **read-before-write** evidence is \(r_t=M_tk_t\), used in read-refined fast/slow representations and in downstream original-style readout. A read-only observation explicitly selects \(P_t=I,Q_t=0\) and must preserve memory exactly. The continuous physical pilot writes each valid measured event; it predicts \(o_{t+1}\) from the available \(o_t,a_t\).

The batch execution computes all \(b_t,c_t\), provisional states, gates and pairs \((P_t,Q_t)\) independently or through causal prefix reductions. It uses the chronological associative operator

\[
(P_A,Q_A)\diamond(P_B,Q_B)=
(P_AP_B,\ Q_AP_B+Q_B).
\]

PNDS/ParallelMemory.lean proves operator application, associativity, identity, chronological chunk grouping and conditional radius-budget inequality. PNDS/DualModeMemory.lean proves **for an arbitrary finite input schedule in exact real arithmetic** that applying the summarized chronological affine operator equals stepping the matrix through the same list, and that a resumed chunk fold equals processing concatenated events. Lean does *not* yet prove all 48D/88D neural maps, floating-point kernels, gradients and nonlinear source-to-candidate equivalence.

## III. Stability, capacity and real-time complexity

For \(0\le\bar\alpha,\beta\le1\), \(R\ge1\) and normalized \(\|k\|_2,\|v\|_2\le1\), the spectral norm of \(I-\beta kk^\top\) is ≤1. Hence

\[
\|M_{t+1}\|_F\le\alpha_t\|M_t\|_F+\beta_t\le R.
\]

The scalar last step is Lean-certified *conditional* on the matrix norm inequality; a full Lean operator-norm proof for actual 8×8 tensors and corresponding floating-point refinement remain OPEN. The bound is a boundedness certificate, **not** long-term useful information preservation: retention can still erase old facts. Original DIAGKEY004's global projection additionally can attenuate orthogonal memories and cannot be treated as its own parallel affine scan.

Training/batch: balanced Blelloch chronological scan does O(T) fixed-dimension affine-matrix multiplication work and O(log T) scan depth, with O(T) intermediate activation storage. Streaming: only \(Z_t\) is carried, the rank-one associative update takes O(8²) per event and the reused dense maps are fixed-cost. GPU kernel fusion, memory bandwidth, multi-GPU scalability and training FLOPs are not yet measured or proved equivalent to Mamba performance.

## IV. Verification — this is what is supported

- Original categorical variant: **334 verified batch-to-step checks**, FP32/FP64, all parameter-gradient comparisons, state traces, mixed read-only events and 2,048-event fixed-state tests. Maximum compared intermediate absolute discrepancy \(3.2782554626464844\times10^{-7}\).
- Continuous physical world variant: **146 verified batch-to-step and gradient checks**, 2,048-event memory-size invariance and fixed same-parameter stream; 93,324 TAC active trainable parameters.
- CPU source-runtime experiment: 4 parallel samples, single PyTorch CPU thread, 80 observed forward-step measurements per cold/warm history regime. Medians at histories 0,128,2048 were **0.652790,0.656021,0.654352 milliseconds per whole batch step**, respectively. Not a certified embedded deadline or measured GPU advantage. Primary fixed-size footprint is 608 FP32 bytes/sample, excluding weights and Python allocator overhead.
- Bounded entity memory is a **separate optional O(SD) dynamic admission** module requiring actual observed stable entity IDs, observable relevance [0,1] and explicit protected slots; see PNDS/EntityAdmission.lean for isolated write and replacement theorem. Dynamic replacement generally depends on prior slot state and is *not* automatically time-parallel scan-compatible. It must be evaluated on original real actor tracks and integrated only with a separately proven routing schedule.
- ALOHA cabinet genuine action-conditioned world pilot: original direct-state decoder **failed persistence badly**. Revised train-only residual state anchoring improves development behavior. The original heldout episodes were re-observed during model redesign; later errors from them are **exposed-test descriptive** and cannot qualify a fresh generalization success. GRU trainable parameters match to 0.139%; identical training windows/steps but **training FLOPs not certified equal**, so full fair comparison is prohibited.
- Existing ESWAM012 and QUAL001 immutable traffic evidence shows native 256-frame memory did *not* beat 8-frame native. These source failures remain binding and forbid assuming longer observation automatically improves decisions.

## V. What remains open — explicitly

- Proof of the entire neural tensor graph and gradients in Lean across each historical architecture.
- Real-world **independent** prediction skill with previously untouched domain/site/splits and full parameter/data/training FLOP parity against native, GRU and attention.
- Training/validation of an **integrated** admission-and-retention policy on original traffic identities under a legally parallelizable schedule or an explicit serial state-dependent component.
- Validated cross-entity admission costs, collisions, forgetting, decision loss, closed-loop correction and stable persistent world-state semantics.
- Production CPU/GPU low-latency kernels, power, memory allocator overhead, extreme lengths and reproducible hardware P50/P95.
- Conversion of the 17 historical variants. A proven shared algebraic interface is NOT 17 source-verified whole-model proofs.

**Promotion gate:** do not merge untrained neural architecture into TAC master, nor promote status S2 to real skill S4 until an immutable fair, authentic, independent test passes. When comparing model architectures, match active parameter counts, real source data and splits, observation/action exposure and measured training compute; CPU wall-clock equality is *not* compulsory.
