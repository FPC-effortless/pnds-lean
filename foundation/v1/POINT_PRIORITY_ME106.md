# ME106 — PointPriorityTraining

**Status: PARTIAL (exact finite-vector training rule only).** Variant `point_priority_training`, a training-only descendant of ME105 `protected_sensor`. The inference graph is inherited from `PNDS.SensorGraph105.forward`; `PointPrioritySensorWorld` wraps the same parameter record and delegates to that definition unchanged. ME105's 100,349 trainable scalar count and its specified 33 learned parameter tensors are inherited. No tensor, forecast head, or forward computation is added here.

## Training roles

Let `P = w_point * normalized_raw_median_MAE` be the point-priority objective. Let

```text
U = proper_NLL + 0.1*location_MSE
                + 0.3*masked_current_reconstruction_NLL
                + 0.25*proper_interval_score.
```

The finite exact-real gradient vectors over all model parameters are `gP = ∇P` and `gU = ∇U`. Their coordinate space is `Fin 100349 → ℝ`, using the inherited parameter count. Define `dot = Σᵢ gPᵢ*gUᵢ` and `n2 = Σᵢ gPᵢ²`. If `dot < 0` and `n2 > 0`, the asymmetric point-priority update projects only the uncertainty gradient:

```text
gU' = gU - (dot/n2) gP
gTotal = gP + gU'.
```

Otherwise `gU' = gU`. The Lean module proves, under `n2 > 0`, `dot(gP,gU') ≥ 0` and `dot(gP,gTotal) ≥ ||gP||²`. In the conflicting case the projected inner product is exactly zero. The zero-vector point gradient leaves `gU` unchanged, zero-norm input leaves `gU` unchanged, nonconflicting gradients leave `gU` unchanged, and any coordinate with `gPᵢ = 0` is unchanged by projection. These are exact finite-vector statements.

This is an **asymmetric point-priority** version inspired by Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020 (PCGrad). It is not a new discovery or a claim of novel projection theory. The result concerns the gradient combination before the optimizer. AdamW preconditioning, weight decay, clipping, finite precision, and finite update steps invalidate an automatic point-descent guarantee; that guarantee remains OPEN.

## Median uncertainty role and AD distinction

The uncertainty-side location expression is `mu - (median_correction - median_correction.detach())`. The scalar Lean contract uses an explicit snapshot correction. Its value equals `mu` when live and snapshot corrections match. This is a forward-value identity, not an identity of classical derivatives. With the snapshot treated as an independent frozen real constant, the classical derivative of `mu - (correction - snapshot)` with respect to `correction` is `-1`. Under the custom autodiff rule, `detach(correction)` has zero tangent, so `correction - detach(correction)` has tangent `dc`, not zero. Since `mu = base + correction`, the full uncertainty expression has tangent `(dBase + dc) - dc = dBase`: the median correction gradient cancels only across the composed expression. The Lean module records these custom tangent equations separately from its scalar forward identity. Their match to actual framework autograd and the source snapshot path remains OPEN.

## Formalization and evidence boundary

`PNDS/PointPriorityTraining.lean` imports the cached `PNDS.SensorGraph105` graph and adds only the wrapper, loss-role contract, snapshot value identity, and finite-vector projection. The gradient projection uses real arithmetic. Existing ME105 graph and proof files are unchanged. No `sorry` or new axiom is used by this module; the imported graph has its existing limited Lean dependencies.

The theorem does not prove the neural derivatives of either loss, objective normalization/masks, source-to-Lean equation correspondence, PyTorch forward/backward equivalence, gradient aggregation precision, or empirical behavior. The specified production aggregation accumulates `dot` and `n2` in FP64, computes the projection coefficient in FP64, applies the correction in FP64 before rounding, and does not use higher-order gradients; implementation/source verification is OPEN. Optimizer effects described above are OPEN.

At initial drafting, the repository two-formulation/source-link items were pending: an ME106 Python reference and Python tests, source-pinned TAC implementation SHA/path, reciprocal TAC manifest entry, and source forward/backward/finite-precision checks remain OPEN. The upstream source SHA is pending the mainline freeze. This record is a partial mathematical contract, not a whole-model or capability claim.

## Source freeze and executable formulation

TAC implementation source: `adb71873a8ddf9939f9c185b47e24a68927b5122`.
Forward class: `src/tac_osm/variants/point_priority_sensor_world.py`; training
operators: `scripts/me106_gradient_roles.py`; full independent serial/loss/
projected-gradient reference: `foundation/v1/point_priority_reference.py`
(byte-identical `scripts/me106_graph_reference.py` in both repositories).
Actual-source FP32/FP64 loss/assigned-gradient tests pass; inherited complete
forward state/input/weight-gradient/chunk/distribution tests are also reused.
The source-pinned workflow checks the immutable commit and byte equality.
Reciprocal registries include prospective ME106; the 17 historical rows stay
unchanged. Earlier pending-source wording above records obligations that are now
met by these explicit artifacts, not by a whole-neural Lean proof.
