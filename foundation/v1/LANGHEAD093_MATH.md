# LANGHEAD093 exact scoped decoder math

**TAC source**: FPC-effortless/TAC-OSM at commit `de954a177aa5bf5bcd162acf7bd5797fb5bc75cf`, experimental draft PR #646, path `research/mathematical-engineering/langhead093/model093.py`. Derived from HYBGEOM092 original five-seed frozen hypothesis; original TAC state transitions are unchanged.

## Types and causal boundary
Token: integer x in {0,...,255}, observed at t. Trainable byte embedding matrix E: (256,32) real-valued; current observation embedding e_x and each candidate e_y. Native TAC state, input/time dependency and original model logits L_t: untouched original 100,521 parameter graph with fast/slow 8×8, aligned memory, wave, concepts and recurrent upper path. Additional gain scalar g=0.02*sigmoid(raw), 0<g<0.02. Output L_t(y)+g*S_y, where S centered over 256 candidates. Predicted target y_{t+1} is NEVER passed to S. No human semantic labels, elapsed time or physical action inputs.

## Five alternatives
Original: unmodified L. Direct dot: S_y=2 e_x dot e_y. Dot+norm: S_y=2 e_x dot e_y - ||e_y||^2. Norm-only: S_y=-||e_y||^2. Shuffled control: S_y=2 e_x dot e_{(73y+41) mod 256}. All new arms add one active scalar. Centering subtracts a candidate-independent constant.

Exact Euclidean identity:
 -||e_y-e_x||²=2 e_x dot e_y-||e_y||²-||e_x||².
The final term is candidate-independent; after centering, distance and tied dot+norm are the same exact-real logits. The Lean theorem proves **one-coordinate pairwise candidate differences** and common shift invariance. The Python reference tests **whole 32D** candidate-centering equality and negative shuffled controls. This does **not** prove original full native TAC PyTorch forward/backward, floating-point bitwise equality, gradient equivalence, whole-model stability, or model capability.

Training and evaluation live only in TAC-OSM PR #646, with original Git-backed real source receipt and potentially failed benchmark gates preserved there. No training, promotion or quality/safety claim occurs in PNDS. Still OPEN: Lean theorem and source correspondence for the 32D full computation graph; actual PyTorch AD gradient comparison; source-to-output state/shape/mask/time identity; floating point error bounds; full model Lean4 proofs; training-FLOPs matched controls; independent publisher source transfers.
