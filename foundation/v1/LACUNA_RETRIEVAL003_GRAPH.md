# LACUNA-RETRIEVAL003 — conditional read diagnostic, PARTIAL

TAC PR639, parent f92842b83a42fa82366612e15f17a0f1b992fe4b; original
model source SHA2567179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1.
Original100521 learned parameters,1920B tensors+8B clock; three unchanged
LANGMEM087640 checkpoints. No model fits. Full inherited observation→embedding,
position/lexical context→writer/query/gates→fast/slow/aligned/wave/basis/concepts
→fast/slow features→evidence/concepts→norm→upper/FFN→256 logits follows
LANGMEM086/087/Survival001 source graph, timing and open obligations.

For clock t, source query q is [8], source old/current slow components H,C
are [8,8]. Fixed source path gives S+=H+C. A/B read Hq/Ha. C/D reconstruct
keys K[n,8], native values V[n,8] from a256B ring (8clock+248sourcebytes),
then r=softmax(8Kaddress)^T V. Every key uses four bytes preceding its value;
no observed historical target value is in its address. Supplied a and chosen j
are evaluator privileges. Native upper/query states retain original trajectory.

All old reads are scaled to ||Hq||, zero vectors stayzero, then Cq is added.
Original read mixture m_f*r_f+m_s*r_s is replaced at three consumers:
fast inputlast8 of96, slow inputlast8 of104, norminput96:104 of160.
All other state and direct current write stay fixed; counterfactual next states
are discarded. Native A identity and slow_zero correspondence are source-tested.
Local VJP/finite difference concerns only this read leaf; weights never updated.

Conditional transported coefficients A[n,8] yield H=V^T A. A cyclic value-row
permutation preserves values/keys/transport coefficients as separate multisets,
not their associations. Native includes all old writes, ring only retained
records; no identical record-set claim. These arrays are evaluator workspace,
not additional actor memory. Byte-budget matching concerns banks only; no
equal-full-state autonomous rollout or architecture superiority is established.

Lean proves scale squared-norm algebra, unchanged-current read decomposition,
constant-value convex read conditional on normalized weights and integerpayloads.
Python independently implements stable attention, normmatching and transported
reads. Native source tests compare actual consumer patches and reconstruction;
random float64 reference parity is numerical evidence only.

OPEN: whole NN source equivalence, IEEE roundoff proofs, all-parameter AD,
softmax formal normalization/exponentials, source document authenticity theorem,
cohort selection proof, learned addressing efficacy, causal field utility,
independent qualification, production cost and canonical registry merge.
Original failed gates stayfailed; these partial proofs never authorize training.

## Original diagnostic evidence (not a theorem)

Original TAC execution a74627fc1d8d1eaf209ff6eb5be828fdbf326768,
run41007483485/artifact12956147521.678measurements repeat226source/query/bin
cells across3frozen models, with154current source positions. Two registered
publisher documents, now exposed DEV. Supplied-key/raw-history historical-byte
accuracy34.52%/21.31%/21.45% exceeds supplied-key/matrix5.95%/5.15%/6.27%,
but no condition passes the preregistered next-byte utility gate. Short-age
favorable averages and unfavorable seed/document strata are preserved. In
particular, supplied-key raw D improves short-age mean NLL in all3seeds but
worsens weakref. No training gate reopens. See TAC original saved report for
all controls, memory/workspace costs, integrity checks and inference limits.
