# PNDS v1.8: Six prospective models, one parallel information architecture

This is the mathematical contract accompanying [TAC consolidation](https://github.com/FPC-effortless/TAC-OSM/tree/research/parallel-six-consolidation-2026-10-25/research/parallel-six). It **does not** convert the 17 archived models into six trained models, prove six networks, or establish new real-world skill.

## Authority, objects, and six typed model specializations

PNDS defines architecture/proof obligations. TAC-OSM implements runtime and owns authentic source training/negative evidence. The source-pinned `foundation/v1/registry.json`, `foundation/v1/variant-math-links.json`, `foundation/v1/archive-gaps.json` and TAC reciprocal registry remain historically immutable. This prospective portfolio lives beside, not instead of, those research records. `active-variants.json` is intentionally byte-identical to TAC's copied file on the corresponding branch.

Let **family** v belong to exactly {system1, physical, enterprise, temporal, edge_lite, counterfactual}. Each model declares explicit typed observation space X_v, logged action space A_v (never assumed available), event time, missingness, identity schema and task output Y_v. The mathematical core is `F_θ(e_t,Z_t)` with a shared *operator family*, not shared learned weights or necessarily equal inputs/outputs across domains.

| Family | Observed stream | Output and missing formal head |
| --- | --- | --- |
| system1 | authenticated current observation, known-at-time context | P(risk/response_{t+1}|observed history); no executable authority or calibrated risk head |
| physical (including ES-WAM) | measured position/joint/actor state and actually logged actions/identities | p(s_{t+h}|past observed s,a) or uncertainty; correct identity admission and moving/quiet regime protection not proved |
| enterprise | real timestamped, permissioned activity/case event prefixes | p(next event, waiting time, event sequences), source record end not true completion |
| temporal | evidence with valid-time and recorded-time, including authenticated corrections | source-preserving as-known-at/as-valid-at reconstruction plus prospective state; late revisions require correct replay |
| edge_lite | task-specific typed input plus immutable deployment artifact | same qualified source-model output within measured numerical/runtime error; precision quantization alone is no new model |
| counterfactual | real observed actions/outcomes, logged availability and possibly propensity | observational P(Y|S,A), not P(Y|S,do(A)) absent identification, no permission to intervene |

For each v, the specialist head H_v and any adapter/missingness semantics must be given complete typed tensor construction, Lean statements, executable Python, PyTorch forward/backward/tolerance tests, source pin and authentic data contract. Presently those six source-equivalent, task-specialized neural heads are **OPEN**; no theorem here treats them as proved.

## Shared native-inspired dual-mode memory, conditional source law

For input encoding e_t, bounded causal features, known provisional gates, k_t,v_t ∈ R⁸ with ||k_t||₂,||v_t||₂ ≤ 1, and M_t ∈ R^{8×8},

```
Z_t = (A_t in R^88, M_t in R^(8x8), n_t in N)
c_t = A_t / max(n_t,1);  A_(t+1) = A_t + b_theta(e_t)
alpha_t = sigmoid(g_alpha(e_t,provisional_fast,provisional_slow))
beta_t  = sigmoid(g_beta(e_t,provisional_fast,provisional_slow))
alpha_eff_t = alpha_t * (1 - beta_t/4)
P_t = alpha_eff_t * (I - beta_t k_t k_t^T)
Q_t = beta_t v_t k_t^T
M_(t+1) = M_t P_t + Q_t
r_t = M_t k_t  (READ BEFORE WRITE; read-only: P=I, Q=0)
```

Chronological composition for first pair (P₁,Q₁) then (P₂,Q₂) is (P₁P₂, Q₁P₂+Q₂). Associativity permits balanced time-prefix scan under **precomputable, memory-independent gates**: O(T) work, O(log T) idealized depth for fixed matrices; a real kernel may differ. Streaming uses the exact same ordered update with rank-one O(8²) event work and a fixed 88+64=152 FP32-value state = 608 bytes per batch element plus counter. See existing `PNDS/ParallelMemory.lean`, `PNDS/DualModeMemory.lean`, `foundation/v1/DUAL_MODE_STREAMING_V1_5.md` for actual *conditional* theorems and finite-precision caveats. The accompanying Python reference here checks chronological matrix algebra only, not neural model equivalence.

If admission, eviction, a missing-data correction, causal verification or any action decision reads prior mutable state to choose a gate, naive advance scheduling is illegal. Either define an explicit known schedule, use a sequential step, or prove another associative lifted operator. The original native fast→slow→memory nonlinear recurrent graph is not exactly this scan; do not imply weight source semantics are preserved when moving to it.

## Preserved negative evidence and prospective qualifying tests

Original historical row statuses and direct source commits remain as-is. Source ALOHA v1.7 real DEV experiments showed four-timescale and controlled-retention changes underperformed the unmodified baseline; retention longer was not evidence of useful retrieved information. ESWAM012 256-frame memory lagged shorter-horizon 8-frame baseline, and 8 slots rejected most original actors. No lossless 'information fusion' assertion without capacity, interference and admission test.

Six families must pass **separate** source-authenticated tasks, fair matched original real datasets/training FLOPs/parameter count (95k–105k each, ≤1% between head-to-head models) and held-out previously untouched outcomes. CPU clock time is not an equality gate. Synthetic only for diagnostics, never evidence of real skill. Require good unchanged strong persistence/servo, GRU, attention and other relevant classical controls. Per-family operational/effectiveness gate remains CLOSED.

This is a no-fit consolidation, not six completed mathematical formalizations or six trained models. Each new mathematical component must have source-equivalent Lean and Python per repo's compulsory rule before engineering maturity may be promoted.
