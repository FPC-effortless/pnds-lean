# ME109 inward role guard

**Status: PARTIAL conditional vector projection and source-pinned utility parity; full IEEE/PyTorch refinement OPEN.** This prospective no-fit training utility is specified by `PNDS.InwardRoleGuard` and the independent executable reference in `inward_role_guard.py`. It does not change the ME109 model, training recipe, checkpoint, or historical results. Historical ME109 gates remain **FAIL**.

The source correspondence is the utility `src/tac_osm/inward_role_guard.py` at TAC-OSM commit `f1cf90213cdc9daee161e99e017c55fdcbfb10cc`. The regression test reads that immutable Git blob, verifies that the local TAC source body matches it, and compares a nonfallback FP32 runtime candidate to the Python mathematical reference using the margin recorded by that runtime call. The zero-gradient block is exercised through the runtime's own zero-norm path; the positive-norm reference deliberately rejects zero norm.

For each block, let `p` be the current FP32 weights, `g` the point-priority gradient, and `raw` the raw FP32 proposal. The runtime forms `d = raw - p` in FP64, `norm = Σg²`, `dot = Σg d`, `scale = Σ|g|(|p|+|raw|)`, and

```text
gamma = ((n+8) eps64) / (1 - (n+8) eps64)
margin = (4 eps32 + 8 gamma) scale
alpha = max(0, (dot + margin) / norm)
z = float32(raw - alpha g)
```

It then checks the realized candidate with

```text
actual = Σg(z-p)
error = gamma Σ|g|(|z|+|p|) / (1-gamma)
actual + error <= 0
```

If the check fails, nonzero-gradient coordinates revert to `p`, while zero-gradient coordinates retain their raw proposal. The Lean theorem proves the exact-real projection inequality for positive squared norm. Its scalar rounding theorem is conditional on an externally supplied bound. The runtime's FP64/FP32 envelope is an implemented conditional check; a full proof that the envelope covers every relevant IEEE/PyTorch operation, device, and reduction is **OPEN**. The tests establish utility-level source parity at the pinned blob, not whole-model source equivalence, full backward/gradient equivalence, finite-step loss descent, new capability, or real-world superiority.

Operation cost is linear in the number of scalar parameters in a block and requires blockwise FP64 reductions plus one FP32 cast. This contract makes no claim about performance or legal parallel execution on a particular device. No fitting or training run was performed for this repair.
