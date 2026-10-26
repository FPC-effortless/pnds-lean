# ME111 coherent conditional belief

Status: **PARTIAL; no training or sealed-data evaluation.** Native affine TAC
state transition and current-mean/forecast-median computation are inherited from
ME109/ME110. ME111 changes conditional forecast scales and explicitly supervises
696 current reconstruction-mean coordinates. Prior negative gates and measured
ME110 outcomes remain unchanged. This is not a full neural proof or an empirical
calibration claim.

TAC source is pinned to `0e712a798ce943b9f25c306bec1250e0b1e29031` in `variant-math-links.json`,
with spec `research/mathematical-engineering/me111/MATHEMATICS.md`. Source ownership, training receipts and evaluation remain TAC-OSM.

## Observation-to-output contract

At the last current observation, each of the six forecast channels carries an
observed bit, standardized value, and native affine state response `r`.
`m=1-observed` is discrete observed missingness, not a differentiable parameter.
Inherited native traces provide positive learned future variance `vf`, positive
reconstructed current variance `vc`, current mean, and twelve raw side-width
outputs. Tensor shape for `vf,vc,r,m` is `[batch,6]`; raw width and width factors
have shape `[batch,12]`, first six left and final six right. In the inherited
source these width factors already equal `exp(log(4)*tanh(raw))`.

For each channel and side:

```
b² = 1e-6 + vf + r² * vc * m
w  = exp(log(4) * tanh(raw))
sigma_side = sqrt(b²) * w
```

There is no empirical age floor and no `max(0.8,w)` in this new scale law. The
old model floor remains only an inherited provenance buffer/computation. State,
current mean and forecast location at initialization are inherited unchanged;
scales intentionally change. `b²` is a component sum, **not** the final variance
of the asymmetric split distribution. The interval result concerns nonnegative
tail multipliers and positive side scales; it does not prove calibration,
coverage, normalization of the entire decoder or causal identifiability.

## Training roles and gradient paths

Flatten the 45 parameter tensors row-major (104663 coordinates). Keep the ME110
binary uncertainty mask `U` of size 4410 and its complement `P` of size 100253.
The binary reconstruction-mean mask `R` selects first six output rows of
`current_head.weight/bias` and `current_residual.output.weight/bias`, totaling
696 coordinates, all inside P. Every other coordinate has R=0.

With raw gradients `gp`, `gr`, `gu`, first form

```
gP = P*gp + .3*R*gr
gU = U*gu
g  = clip(gP) + clip(gU)
clip(x) = x * min(1, 1/(||x||₂ + 1e-6))
```

The source rejects any raw point gradient on U. `gr` is the unweighted
reconstruction loss gradient; source applies .3 once. `gu` already differentiates
balanced NLL + .25 interval score + .3 balanced reconstruction NLL. Reconstruction
uses recorded deliberately hidden values; natural gaps do not fabricate labels.
Masking and the P reconstruction addition precede independent role norm clipping.
The actual AdamW candidate remains inward guarded using the **original raw point
gradient**; this reference does not replace or prove that optimizer guard.

## Verification and precise proof limits

`coherent_conditional_belief_reference.py` independently implements scalar scales,
analytic VJPs in vf/r/vc/raw (no m derivative), quantile order and mask/routing
operators. Reference tests include deterministic randomized central differences,
observed independence from r/vc, response symmetry and positivity.
`test_coherent_conditional_belief_source.py` compares fp32/fp64 scales and all
input VJPs on both sides and all 104663 source mask/routing coordinates. These
synthetic tests do not establish full neural forward/backward equivalence.

`PNDS/CoherentConditionalBelief.lean` states the actual Mathlib real-number
variance positivity, side-scale positivity, observed-case reduction, interval
order and preclipping mean-role algebra. Its local build is OPEN because the
Mathlib binary cache endpoint returned HTTP 403. CI must compile it and audit
axioms before claiming those real theorems as checked.
`PNDS/CoherentConditionalBeliefStandalone.lean` separately builds with Lean Init
and proves the same positivity/order implications under explicit primitive
operation laws, plus structural binary-mask facts. It does not construct real
numbers or instantiate the primitive laws; its empty axiom audit cannot replace
the Mathlib theorem check. Both are deliberately limited; full neural graph,
AD/optimizer, finite-precision global error, and calibration remain OPEN.

Point-only neural paths and uncertainty-invariance beyond the tested source
operators, full distribution normalization, legal whole-model parallelism and
matched-compute empirical comparisons are inherited/open obligations. Channel
and batch scalar equations are independent; no new legal time parallelism is
asserted. Neither an algebra theorem nor synthetic parity upgrades capabilities.
