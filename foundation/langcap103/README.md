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
