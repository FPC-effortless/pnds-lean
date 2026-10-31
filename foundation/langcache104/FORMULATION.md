# LANGCACHE104: deep associative carried cache

Status: **candidate formulation, no trained capability or GPU speed result**.
Original source: `FPC-effortless/TAC-OSM` implementation commit
`b567f4c68a73106781555496c44145939efd1d92`, with exact file hashes in
`SOURCE_MAP.json`. Source binding does not complete mathematical qualification.
The ancestral LANGMATH098 positive associative mechanism is preserved as one
component. This graph additionally contains residual depth, independent learned
heads, rank-R writes, a separate signed field, and bounded recent local attention.

## Observation and ownership boundary

The input is nonempty observed byte IDs in `[0,255]`, shape `[B,T]`, with ordered,
unique nonempty stream-owner identifiers. Output `[B,T,256]` predicts the next
byte after each observed byte; it is not a prediction of its own current input.
No target, loss, future byte, reward, candidate answer or outcome enters forward.
An optional Boolean write mask is an explicit intervention affecting bank and
field updates. It does not hide a byte from residual computation or recent KV.

Each stream carries a clock and L layer states. Each layer has two bank pairs
`memory[B,H,V,K]`, `mass[B,H,K]`, a separate signed `rotary[B,H,V]`, and recent
keys `[B,H,min(clock,W-1),K]` / values `[B,H,min(clock,W-1),V]`. When the signed
branch is disabled its zero carry remains part of this version's state schema.
The model validates owner, clock, configuration signature, shape, dtype, device
and finite values. The signature is SHA256 of canonical configuration JSON;
it prevents reinterpreting same-shaped state under different RoPE/window or
other settings. It does not identify checkpoint weights, which remains OPEN.
Serialization preserves all layers; truncated BPTT requires explicit `detach`.
Detached inference resident tensor storage is bounded independently of history.
Autograd training graphs and intermediate activations are not included in that
resident-state bound and grow with the differentiated history.

## Full neural layer graph

Let `D,H,K,V,R,F,L,W` be residual width, head count, address dimension, value
dimension, write rank, FFN dimension, depth and recent window. For layer l,

`u = gamma_memory * x / sqrt(mean(x^2) + 1e-6)`.

Each of its two banks separately computes, for every head,

`q = softmax(Wq u)`, `k_r = softmax((Wk u)_r)`,
`v_r = tanh((Wv u)_r)`, `alpha = sigmoid(Wa u + ba)`,
`eligibility = sigmoid(We u + be)`.

Under a true write mask, `a=alpha` and `w=(1-a)*eligibility`; under a false mask,
`a=1,w=0`. Retention biases initialize to 2.2 (fast) and 5.3 (slow), but are
learned and do not constrain their eventual timescales. The inclusive update is

`M_t = a_t M_(t-1) + (w_t/R) sum_r v_(t,r) k_(t,r)^T`,

`z_t = a_t z_(t-1) + (w_t/R) sum_r k_(t,r)`,

`e_t = z_t^T q_t`, `r_t = M_t q_t / (e_t + 1e-4)`.

The current observed byte may write before the model predicts the next byte.
No previous/future target byte is supplied as a separate privileged value.

The optional signed field computes `a_f=sigmoid(Wa_f u + ba_f)`,
`theta=pi*tanh(Wtheta u + btheta)` per adjacent coordinate pair,
`offer=tanh(Woffer u + boffer)`. Its update is

`f_t = a_f R(theta_t) f_(t-1) + (1-a_f) offer_t`,

where `R(theta)(x,y)=(cos(theta)x-sin(theta)y,sin(theta)x+cos(theta)y)`.
A false write mask sets `a_f=1,theta=0` so the field freezes exactly.
This signed dynamics does not use the positive bank's cone invariant.

The recent lane projects `q_local=Wq_local u`, `k_local=Wk_local u`,
`v_local=Wv_local u`. With `local_rope=true`, adjacent key/query pairs rotate
by absolute clock position times fixed frequencies
`exp(-log(10000)*2j/K)`; retained keys are already rotated. Query position t
attends only to the current and at most W-1 previous observed positions, with
scores `q_t^T k_i / sqrt(K)` and softmax weights. This is causal local attention,
not exact key matching or guaranteed exact copy. Tiled evaluation prevents a
full T-by-T mask. Absolute positions originate as int64 and are converted to
FP64 for phase/trigonometry; cosine/sine then cast to the activation dtype.
The implementation rejects a RoPE clock plus input length above 2^53. This
avoids FP32 adjacent-clock aliasing; it is not a long-horizon float-error proof.
The cache owns a copied suffix rather than a prefill view.

For each head, the exact concatenation order is

`[r_fast, r_slow, r_fast-r_slow, r_fast*r_slow, (f if enabled), r_local,
  e_fast, e_slow]`.

The head axis is flattened only after preserving those perspectives. With
`s=1/sqrt(2L)`,

`y = x + s*W_perspective concat`,

`n = RMSNorm_ffn(y)`,

`x_next = y + s*W_ffn_out[(W_ffn_value n) * SiLU(W_ffn_gate n)]`.

The full model embeds bytes, applies L blocks in order, applies final RMSNorm,
then a 256-way linear head, optionally tied to the byte embedding. Every layer
has its own parameters and state. Later layers can condition writes on earlier
layer reads. A bank's own nonlinear current-state read does not feed its own
write coefficients within that same layer and time scan.

## Invariants and capacity

For a valid bank state `0<=z_j<=1` and `|M_ij|<=z_j`, nonnegative L1 keys,
bounded tanh values, `0<=a<=1` and `0<=w<=1-a` preserve those inequalities.
Nonnegative normalized queries and positive epsilon then imply `|read_i|<=1`.
The Lean coordinate theorem expresses a sufficient one-coordinate cone step;
lifting it to full rank-R source tensors and the normalized read remains OPEN.
The invariant is a stability statement, not a useful-information theorem.

The signed field instead satisfies a Euclidean norm envelope
`m_t=a_t*m_(t-1)+(1-a_t)*||offer_t||_2`, with `||f_t||_2<=m_t` when initialized
validly. Neither this envelope nor the bank cone proves whole-model stability:
residual projections, learned RMS weights and logits are not globally bounded.

For fixed geometry and numerical precision, carried state has finite capacity.
Different histories can collide. Forgotten or overwritten facts cannot be
reconstructed exactly merely because the state is bounded. Heads, wider bank
geometry, rank and depth are separately tunable hypotheses, not demonstrated
general reasoning or autonomous structure discovery.

## Parallel mathematical evaluation

For independently precomputable scalar retention, a chunk's affine bank
transition composes as `(a2,b2) o (a1,b1)=(a2*a1,a2*b1+b2)`. Inclusive contribution
from write i to read t has weight `w_i/R * product_(j=i+1..t) a_j`; the incoming
state has weight `product_(j=chunk_start..t) a_j`. These decay weights multiply
rank-wise query/key dot products and values using batched matrix products.
Chunks can prepare all local work independently, then use affine scan for their
incoming-state boundary conditions. Full per-token `[T,H,V,K]` state trajectories
are unnecessary for this evaluation. Zero retention requires correct product
semantics; inverse-prefix-product formulas are not justified.

For fixed chunk C, bank work scales linearly in T, with O(T*C*H*R*(K+V)) local
comparison/matmul work plus O(T*H*R*V*K) summary construction. Training activation
memory still grows with T; O(1) refers only to inference carried state. The
signed vector has pairwise scaled-rotation affine closure, allowing a distinct
work-efficient prefix scan. Causal local attention uses O(T*W) score work.

Layers depend on lower-layer outputs, and unknown autoregressive bytes depend
on previous predictions. This is teacher-forced within-layer sequence
parallelism plus batched/within-step inference, not unrestricted parallel
generation. GPU wall-clock advantage, tensor-core utilization, kernel launch
costs and mixed-precision training remain measured engineering questions.

## Exact discrete API and biological interpretation

`exact_store.py` is an independently budgeted explicit byte-key/byte-value
sidecar. It verifies complete keys, ownership, causal timestamps and retention.
Conditional on requesting a correct resident key, it returns the preserved
payload exactly. It is not part of the current differentiable logits graph;
learned language-to-record addressing remains OPEN. Neural local attention and
compressed associative retrieval do not inherit its exactness guarantee.

The design supplies distributed representations, nonlinear depth, gated
plastic memory, multiple timescales and selective access. Those are testable
computational conditions. Biological equivalence or brain-like intelligence
does not follow from naming those mechanisms.

## Evidence boundary and remaining obligations

`math_reference.py` independently expresses the full neural graph as scalar/list
Python, including its parameter names, RoPE, state and write masks. Its tests
are no-fit fixtures. Original-source parity, all-parameter/input-state gradient
checks, future poisoning, owner isolation, state replay and precision tests
must retain exact results from actual execution. Lean contains typed bank,
rotary, recent-access and layer-interface definitions plus scoped algebraic
proof scripts; it is not a complete lowering of every parameter operation.
No Lean compiler is present in the authoring environment, so kernel build and
axiom audit are explicitly OPEN until CI runs. No `sorry` or new axioms count
as proof. Pinned cross-repository peer verification, full tensor proof,
float-error analysis, signed independent
real-source training registration and trained matched-architecture capability
all remain separate gates.
