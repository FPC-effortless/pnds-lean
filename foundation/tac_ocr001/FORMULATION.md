# TAC-OCR-001 — typed graph and mathematical scope

All three candidates are **untrained**. This formulation is partial, not
a universal theorem about PyTorch, trained OCR or unrestricted documents.

## Inputs, identity and the image bridge

The encoder consumes observed RGB image regions only. The retained pinned
TrOCR/DeiT encoder supplies `F[B,N,384]` and normalized spatial boxes
`P[B,N,4]`, after excluding its CLS and distillation tokens. It is frozen;
its unused pooler and text decoder are not deployed. Every deployed learned
encoder weight is counted separately from the custom decoder.

When N exceeds K=128, label-free adaptive average pooling reduces both F
and P along raster order to K entries. Otherwise all N entries remain.
Pooling is not learned relevance and need not preserve small characters.
Let C<=K be the resulting reference length:

`V = RMSNorm(F W_F + P W_P) [B,C,128]`.

`K_v, V_v = split(V W_KV)`, reshaped `[B,4,C,32]`. These projected references
are computed once per region, shared across layers and reused across output
steps. They do not include target text, sample filenames or gold labels.
Page/owner IDs enforce state ownership; their string contents are not embedded.

The text vocabulary is all 256 UTF-8 byte values plus BOS/EOS/PAD. For a
supervised transcript, decoder input is BOS followed by the prior bytes;
next-byte targets are the transcript followed by EOS. No current/future
target byte may condition its own prediction. UTF-8 bytes are not normalized.

## Complete high-level layer graph

`x_0 = E(previous_bytes)` with tied output embedding E.

At layer l, first compute

`q_l = reshape(RMSNorm_l(x_l) W_Ql) [B,4,T,32]`;

`v_l = softmax(q_l K_v^T / sqrt(32)) V_v`;

`u_l = x_l + reshape(v_l) W_Ol / sqrt(4)`.

Next apply that family's actual core and feed-forward residual. All layers
are on the prediction path. Output is `RMSNorm(x_final) E^T [B,T,259]`.
Cross-attention can see every patch in the observed current region, but not
future transcript bytes. Entire region visibility is legal OCR conditioning.

For TAC, the core is the actual inherited LANGCACHE104 `Block104` with four
heads, key/value width 16, write rank 1, two positive associative banks,
128-byte local RoPE attention, and FFN width 384. Its separate oscillatory
memory branch is disabled. The no-op rotary state placeholder remains in
the source-compatible state type and its bytes are counted.

For each positive bank, source-defined projections create softmax q/k,
tanh v, retention alpha=sigmoid(Au+b), and eligibility e=sigmoid(Bu+c).
For rank R and a legal write:

`w=(1-alpha)e`;

`M_t=alpha_t M_(t-1)+(w_t/R) sum_r v_(t,r) k_(t,r)^T`;

`z_t=alpha_t z_(t-1)+(w_t/R) sum_r k_(t,r)`;

`read_t=M_t q_t/(z_t dot q_t+1e-4)`.

The block concatenates fast read, slow read, their difference/product,
local attention and evidence values. A projection supplies a residual
update, followed by RMSNorm/SwiGLU. Source scale is `1/sqrt(2*layers)`.
The inherited independent LANGCACHE104 reference covers its original core;
whole new bridge/core composition correspondence remains OPEN.

The Transformer core uses fused QKV, double-phase RoPE, causal local SDPA
tiles with at most W+C-1 keys per chunk, residual projection and RMSNorm /
SwiGLU FFN width 520. The GRU core is a residual stacked PyTorch GRU with
RMSNorm/SwiGLU FFN width 433. Its exact PyTorch reset/update/candidate gate
semantics are authoritative; an independently lowered full GRU graph has
not been verified here.

## Carried shapes and storage

Per TAC layer: two `M[B,4,16,16]`, two `z[B,4,16]`, inactive placeholder
`[B,4,16]`, and recent keys/values `[B,4,min(clock,127),16]`. Per Transformer
layer: recent keys/values `[B,4,min(clock,127),32]`. Per GRU layer: hidden
`[1,B,128]`. Every completed inference cache owns compact physical storage.
Autograd retention during training is outside that inference-only bound.

For B=1 and a filled cache, TAC carries 295,936 tensor bytes, Transformer
520,192 and GRU 2,048. All share 131,072 bytes of projected visual K/V at
128 references in FP32. These counts exclude weights, raw pixels/features,
encoder activations, temporary workspaces and runtime/library metadata.
Output and exact source documents/features live outside the working state;
their disk usage grows with document length.

Both recurrent and chunk TAC paths are legal along the token axis, with
different floating-point association. Local attention never forms T-by-T
full-history attention. Its temporal cost depends on fixed window/chunk
sizes. Decoder per-token complexity remains bounded in past document length
for fixed shapes; this does not guarantee information survival or constant
total processing time. Total document work must grow with recognized text.

## Optional frozen step and evidence boundaries

The steady kernel accepts one token, a filled W-1 local cache, current
projected visual references, scalar double clock, and flattened owned state.
It uses the same source parameters and positive-bank functions. One-token
local attention can use all W keys without a causal mask. The clock remains
a tensor input; freezing it to the trace-time integer would be incorrect.
Owner/configuration/shape/precision/device and parameter-version checks sit
outside that fixed-geometry tensor graph.

FP64 fixtures compare eager and frozen logits/all state across multiple
steps, a changed region and a large clock; FP32 profiling samples 12 further
steps. These numerical observations are not all-input equivalence proofs.
Mixed precision, quantization and GPU lowering remain unqualified.

`math_reference.py` independently implements one visual attention head using
scalar lists and stabilized softmax. TAC tests compare actual SDPA and a
selected derivative coordinate against it. `PNDS/TACOCR001.lean` separately
defines typed interfaces, Float formulas and a list-capacity theorem. No
Lean theorem proves full source lowering, finite-precision stability,
compression sufficiency, faithful image recognition or trained superiority.

Source partition authenticity, trained checkpoint replay, real OCR quality,
all-weight/backward correspondence and full-model mathematical completion
remain OPEN regardless of a scoped source/hash or list-bound check passing.
