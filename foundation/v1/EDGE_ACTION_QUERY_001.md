# EDGE-ACTION-001 — observational command-query purity

**Engineering and mathematical interface only, 2026-10-25. NO newly trained model or useful-control certification.** This is a formal counterpart to [TAC-OSM draft PR401](https://github.com/FPC-effortless/TAC-OSM/pull/401) source commit `3cec4a41700563e36c3dde16b7f5749a0be665ed`. The original ME099/ME101 neural source, failed gates, and PNDS 17 historical foundation records remain immutable. This appendix is not a replacement for a complete formal network graph.

## Source and mathematical operator contract

With observed, source-timestamped recorded history `H_t={(x_i,u_(i-1)) | i<=t}`, calculate the **already implemented** prior-read-independent affine memory scan for the 8x8 matrix, with fixed 88D prefix evidence. Exact-real right-affine composition is inherited from `PNDS/ParallelMemory.lean` / `DualModeMemory`; the actual full inherited neural graph was independently tested in PyTorch. No new parallel-affine theorem is claimed here.

The new read-only query consumes a candidate `u_t` and **same factual** state after observing `x_t`, returning `(predict(x_t,H_t,u_t), H_t)`. Under the declared function's type:
- **Purity:** projection/proposed commands do not modify factual source evidence or the memory scan.
- **Query alternatives:** replacing one candidate with another cannot alter that carried state.
- **Source-clock authority:** no repeated/older observation hour or foreign owner may be admitted by the typed predicate.

`PNDS/EdgeActionQuery.lean` provides Lean 4 proofs over an `ObservedEvidence` structure (88 real basis values, 8x8 real matrix, count, owner and hour) and `Fin 14→ℝ` command/forecast vectors. These proofs hold for the *functional interface*, not for arbitrary mutation inside a neural network, for physical causality or for complete IEEE software semantics. A real source action remains an **observationally logged signal** unless its execution timing and behavior policy are independently documented.

## Evidence and next obligations

Real fitted ALOHA source preflight [GitHub Action run 40187233556](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40187233556) validates 59 original FIT episodes, 88,382 adjacent observed command/source-state triples, 14D states/actions and a ~20ms median frame interval. It does not confirm the meaning of a logged `action[t]` as executed motor effect over `[t,t+1]`. This semantic fact is necessary before controlled model training and never follows from a mathematical theorem.

CI on the original pinned TAC implementation runs its actual candidate-query memory-purity, chunk-state, parameter-active and gradient tests using CPU PyTorch; all are engineering tests. No protected TEST labels are scored or source-only simulator outcomes used for real-world qualification.

The original SIM001 [PR399](https://github.com/FPC-effortless/TAC-OSM/pull/399) is an **OOD MuJoCo synthetic diagnostic** with corrected last-valid sensor-gap protocol, not a real-world robot/car result. Zero interval width under `valid=False` was an invalid calibration test. Corrected two-joint simulated arm gap 90%-coverage 1.0%, simplified vehicle *proxy* 11.4%. The pretrained ME099 checkpoint was NOT trained on real current-command-conditioned dynamics. No safe actuation.

**Open:** source action execution meaning, complete source-to-output mathematical tensor specification in Lean, source-aligned true-current-action training, real observed uncertainty distribution and its calibration, parameter/data/training FLOP-matched baselines, independent site and hardware safety qualification. No eighth model, losslessly shared trained weights, autonomous motor permission or submillisecond SLA is granted.
