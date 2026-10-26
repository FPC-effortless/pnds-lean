# LATFUSE002 mathematically source-pinned fixed-state candidate

**Source TAC implementation SHA:** `c63a4bb457e7af5f3a04fdbb55a6ef1397f9d5fb`, path `src/tac_osm/latent_fusion_002.py` on draft branch `research/latfuse002-fixed-state-training-2026-10-26`. **Parent candidate:** `LATFUSE001` source SHA `068c3c81bef766eb97660ba5f01891275460287f`. No trained model or real-world result is claimed in this mathematical record.

## Exact tensor changes

Let `h=80, d_k=d_v=16, N=16, D_o=13, D_a=6`. Original fast/slow each 80 scalar FP32; associative matrix [16,16], entity slots [16,80], flags [16] bool. **There are exactly 1,696 carried FP32 scalars plus 16 Boolean occupancy flags = 6,800 nominal bytes/stream**, unchanged from H80 LATFUSE001. No additional persistent recurrent allocations.

- **Initial retention:** `alpha_t = sigmoid(W_alpha u_t + b_alpha)`; initialize `W_alpha=0, b_alpha=logit(.995)` (fully trainable).
- **Initial write:** `beta_t = sigmoid(W_beta u_t + b_beta)`; initialize `W_beta=0, b_beta=logit(.1)` (fully trainable).
- **Gated delta same operator:** `M_t=alpha_t M_(t-1)+beta_t (v_t-alpha_t M_(t-1) k_t) k_t^T`, normalized `k_t`. Not lossless for more than memory rank allows; high alpha does not guarantee exact late recall.
- **Multirate slow within existing dimensions:** `s_t[j]=rho[j]s_(t-1)[j]+(1-rho[j])tanh(W_s u_t+b_s)[j]` with `rho` geometrically interpolating from0.9 to0.99999 across 80 coordinates. Direct homogeneous sensitivity `rho_j^H`, conditional on fixed intervening inputs. Long-lived channels do not by themselves guarantee relevance-selective memory survival.
- **Contextual query:** `q_t=normalize(tanh(W_c c_t+b_c) + W_b [f_t,s_t]+b_b)` for source-authorized *future-known calendar* `c_t` and current belief. `r_t=M_tq_t`. The learned contextual query adds 2,576 trainable weights/biases only—not recurrent bytes—and yields **98,775 active/deployed** FP32 model parameters at registered shape.
- **Prediction:** as in LATFUSE001, read-only current fast, slow, exact source entity slot, associative recall and query concatenated into existing nonlinear mean/positive-scale decoder. Gaussian interpretation/calibration are not proven; a timestamp feature is not `do(action)` and has no causal action claim.

`PNDS/LatentFusion002.lean` defines exact-real slow update/attenuation, delta erase gates, belief query and state count theorems. `foundation/v1/latent_fusion_002_math.py` is independent Python mathematical reference for these local operators; `tests/test_latent_fusion_002_math.py` runs 5 checks. Local operator CI build is required; this is *not* source-identical full Torch forward/backward/FP32/IEEE/probability formalization.

The 17-variant canonical historical register is preserved: validator hardcodes original source PR5 and symbols. This new candidate is **source-pinned with explicit files here but NOT registered as a completely verified 18th canonical historical variant**. Coordinated append-only schema migration remains OPEN.

## Proof / scientific status

- Exact-real local kernel Lean propositions: defined, CI proof status to be checked.
- Python scalar reference: defined, CI proof status to be checked.
- Entire model Lean and actual PyTorch automatic differentiation / finite precision refinement: OPEN.
- Dynamic key collision, memory update boundedness under learned time-varying gates, information survival: OPEN.
- Original source authenticity, model input boundary, verified outcome source, reward hacking and eligibility: TAC owner; this math module alone provides no evidence.
- Genuine-data fit on 2013-2014 and real exposed 2015 development scoring: requires prospective TAC study registration and actual stored checkpoints before reporting.
- Protected 2016-2017 evaluation, calibrated reliability and counterfactual policy utility: NOT CLAIMED.
