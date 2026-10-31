# LANGCAP103 source-bound graph formulations

This record represents six actual untrained TAC/Transformer/GRU configurations
in TAC-OSM PR 681 at commit `afa07568975a8e55ed59a79a50c9bfd28ef9c40b`.
It preserves the seventeen-entry historical foundation and its original scale,
partial proofs and negative findings. The separate prospective graph index is
not a retrospective promotion of any historical model.

`PNDS/LANGCAP103.lean` defines shape-indexed exact-real input-to-logit equations
for the eight-layer contextual associative stack, four-layer GRU and causal
pre-norm Transformer. Address partitions, previous-key writes, first-stream
write suppression, mass normalization, read mixing, residuals, gated head,
GRU reset placement, causal attention and sinusoidal positions are explicit.
Parameters for packed PyTorch affine maps are represented by their row/head
partitions. Dropout is zero; there is no learned-tokenizer or pretrained model.

The five proved facts concern structural recurrent prefix causality, streaming
state composition, future-mask exclusion, conditional bank-mass nonnegativity,
and positive read denominators under stated hypotheses. These facts do not
prove the backend, IEEE rounding, learned information survival or intelligence.
The full nonlinear graph is specified; universal whole-graph source equivalence
and finite-precision/learning guarantees remain OPEN.

`reference.py` independently unrolls the entire three forward graphs from named
weight tensors. It never calls a TAC model forward, PyTorch recurrent module,
or attention module. Primitive Torch arithmetic kernels are shared. TAC uses
serial per-time updates; GRU uses explicit reset/update/new gates; Transformer
uses explicit causal QK-softmax-V matrices. This permits measured forward,
state and parameter-gradient comparisons against the actual source, including
parallel-scan and chunked-stream variants. Passing finite cases is bounded
numerical evidence, not universal source-equivalence proof.

`SOURCE_MAP.json` freezes actual source hashes, configurations, equation hashes,
precision tolerances and open obligations. The source repository executes
the comparison on original FIT bytes and records replay evidence. The paired
registries index this record without changing the legacy variant set.

No optimizer, protected TEST, dataset-derived model selection, signed evaluator
receipt, full-compute parity or mathematical release qualification is created
by compiling this file or passing numerical probes.

## Executed evidence

The exact-real module and all five theorem axiom sets passed in PNDS run
41226944193; paired root build and reciprocal manifest verification passed
in 41227438282. Original Lean output and artifact identity are preserved
under `evidence/`.

TAC-OSM run 41228185975 passed **1,211 repository tests** and all **24**
full-graph forward/state/parameter-gradient cases on original FIT prefixes.
Worst FP32 logit/parameter-gradient absolute errors were approximately
4.77e-7 / 1.87e-9; FP64 errors were 1.34e-15 / 5.64e-18. Actual native
serial/parallel and chunked carry checks, wrong-readout and future-poison
falsifiers, generic state purity and entity-isolation probes passed.
Transformer integer external history is now inspected and counted with
exact discrete identity semantics, rather than described as stateless.

These finite initialized-model checks do not prove universal backend or
IEEE correspondence and do not establish learned capability. Independent
prefit evidence, full compute accounting, permitted T4 execution and final
training registration remain separate requirements.
