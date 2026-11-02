# LANGSCALE001 mathematical counterpart

Dimension-specialized complete exact-real TAC graphs for [TAC-OSM PR 695](https://github.com/FPC-effortless/TAC-OSM/pull/695), source commit `3193566e93bd3794539ef2fefac82da0ace128d9`. This builds on the generic graph at PNDS commit `623a14f75c6e5f166bcbd3c2770fa2aaa624a043` ([PR 103](https://github.com/FPC-effortless/pnds-lean/pull/103)); both remain draft, unmerged work.

The TAC forward/update equations are unchanged; c/k/v/width/head shapes are specialized for four memory reallocations. `PNDS/LANGSCALE001.lean` defines their complete typed input-to-logit/state steps by specializing `PNDS.LANGCAP103.tacStep`. `reference.py` independently unrolls that graph from named weights and infers geometry. `SOURCE_MAP.json` pins the actual constructor, inherited forward source, Python reference, generic Lean graph and specialization bytes.

The new elementary count theorems concern nonlexical state coordinates (1536, 2560, 2560, 9216), not information capacity, trainability, stability or language capability. Existing generic exact-real causality/composition and conditional mass/denominator lemmas retain their own hypotheses. No `sorry` or unsupported axiom is introduced.

Source-side initialized-model evidence contains 32 FP32/FP64 cases, including length 128, all parameter and carried-state gradients, full/chunk carry, serial/parallel, future poison, owner/entity isolation and negative controls. Those are bounded numerical tests, not a universal IEEE/PyTorch theorem. No new model was trained; original DEV/TEST and checkpoints were unchanged. Source authentication of 560 historical FIT files/order happened separately without model exposure.

Local content-pin checks passed; Lean/kernel and axiom evidence must come from the dedicated workflow, not this statement. Global IEEE equivalence and trained capability remain OPEN. The source training registration intentionally fails canonical preflight until missing independently qualified evaluation contracts, full compute and signed source/reward/ACL evidence exist. No mathematical proof upgrades those missing empirical gates.

The historical registry is append-only. Four exact variant rows are added on both sides after these formulation/source commits exist; no historical version is recertified or removed. This PR is stacked on #103 and must be reviewed separately from source PR #695.

Both initial runs passed on formulation commit `ff3c9fc3814853188f3f9cddc5cb3efebadd76d8`: [specialization/axiom run 41400440538](https://github.com/FPC-effortless/pnds-lean/actions/runs/41400440538), [full-root run 41400440515](https://github.com/FPC-effortless/pnds-lean/actions/runs/41400440515). SOURCE_MAP preserves initial registration status; REMOTE_EXECUTION.json records the subsequent successful execution without recertifying global IEEE or trained capability.
