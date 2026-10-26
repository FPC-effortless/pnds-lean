# ME109 retained-parent residual role decoder

Status: prospective, no-fit mathematical candidate. Exact TAC source pin is
`452a1a977275823d21b3f04265a26ff836518ce6`; the reciprocal variant manifests
and source-check workflow bind that immutable implementation. ME107
replaced three parent affine heads with randomly initialized FFNs; ME108's fixed
five-model follow-up failed the full gate in every seed. Those original results
remain unchanged. ME109 retains all parent heads and adds zero-start residuals;
no mathematical result here establishes predictive improvement.

## Typed observation-to-output graph

Inherited input is finite normalized values/ages `[B,T,13]`, boolean observed
mask `[B,T,13]`, calendar `[B,T,6]`, station IDs `[B]`, chronological recorded
hours `[B,T]`, and optional inherited native TAC state. Values remain subject to
the inherited absolute bound 8; ages are normalized to `[0,1]`. Missing values
are masked before encoding; a missing value still must satisfy the input-domain
validation. Observed means available at this event, not an intervention. No
future label or action is an input. No new time discretization is introduced.

The unchanged native chronological TAC scan produces fast/slow/evidence,
associative memory and source sufficient statistics. The new heads have no
feedback into that scan and add no persistent state. Source statistics are
inclusive of the current event; a reconstruction target must have been hidden
from the input to make its evaluation a prediction rather than copying.

The parent decoder builds `phi:[B,102]` from source mean13, source mass13,
evidence mean62, current normalized pollutant ages6 and memory read8. Ages
occupy zero-based columns88..93. Let `x=cat(phi,currentObserved6):[B,108]`.
For each role r (current12, width12, median6), retain its own affine parent L_r
and define `z_r=L_r(phi)+W2_r*tanh(W1_r*x+b1_r)+b2_r`, with independent
108→12→output tensors. Entire W2 and b2 initialize to zero. Availability and
age condition learned corrections; there is no fixed availability gate or
new residual clamp. Parent heads and backbone tensors copy without discarding
any parent learned parameter. Hidden residual parameters have zero gradient
at initialization because W2=0; output parameters can learn on the first step.
This is a structural derivative observation, not a proved Lean/PyTorch AD result.

The current role supplies inherited location `anchor+2*tanh(z_current[:6])`
and variance `0.0025+softplus(z_current[6:])^2+latentVariance`. The inherited
missing-current response, variance propagation and FIT-derived age floor then
supply base location/scale. Width role supplies separate factors
`exp(log(4)*tanh(z_width))`; median role supplies `0.5*tanh(z_median)`.
Final scales retain the FIT variance floor and missing-channel0.8 base-scale
floor. Censored split-lognormal quantile/CDF/NLL laws are unchanged. Outputs
are location/leftScale/rightScale `[B,6]`, inherited state and diagnostic trace.
Parent parameter count100349 plus4314 active residual parameters=104663.
No dummy parameters, attention, Transformer or replacement SSM is introduced.

## Exact and numerical evidence boundaries

`PNDS/ResidualRoleFFN.lean` gives typed residual/context/role definitions and
proves zero-output residual and raw-head identities, simultaneous three-role
identity, current-role independence,104663 arithmetic and a coordinate bound
`abs(residual_i) ≤ abs(b2_i)+sum_j abs(W2_ij)` under the explicitly stated
hidden-activation bound. It does not claim a fixed learned raw-correction cap.
The final-forecast theorem is deterministic decoder congruence **conditional**
on identical inherited decoder/state. It is not a Lean definition/refinement
proof of the entire source decoder. The theorem assumptions are visible in the
declaration; no axiom, sorry or admit is added.

`foundation/v1/residual_role_reference.py` is byte-identical to the pinned TAC
`me109_graph_reference.py`. It rebuilds residual and retained affine heads
with direct functional tensor equations and reuses the existing independent
ME102/101 chronological reference for the native scan. It never calls the new
model/residual or retained head forward methods. Numerical tests deliberately
use nonzero residual output weights and nonzero parent width/median weights,
compare all three forecast tensors and four returned state payloads, and check
all45 parameter gradients plus value, age and calendar gradients. Mixed masks,
missing current events and all-missing histories run in FP32/64. Fixed tolerances
are rtol2e-5/atol2e-6 and rtol1e-9/atol1e-10 respectively. Masked-value invariance
is checked within the inherited normalization domain. These are numerical
correspondence checks, not formal IEEE or automatic-differentiation refinement.

## Analysis and OPEN obligations

Fixed-dimensional state does not establish sufficient information for arbitrary
histories or survival of useful information; long-history interference is OPEN.
Residual heads leave inherited scan stability/forgetting assumptions unchanged;
there is no claim of whole-network contraction or parameter-training stability.
Roles constrain output ownership, not shared-ancestor gradients. Current belief
parameters can affect both location and uncertainty downstream. Width parameters
do not enter the median; the inherited uncertainty training view removes the
direct median-correction tangent without changing its primal value. Neither
role isolation nor a projected first-order dot implies finite-step loss descent.

Inherited outer transforms imply bounded innovations and scale multipliers in
exact arithmetic, with positive inherited variance and protected floors; they
do not imply coverage, normalization refinement, causal identifiability,
calibration, decision utility or generalization. State/input/source units follow
the parent contract; statistical age-based uncertainty is not proven physical
irregular-time dynamics. Full neural Lean graph, probability-law implementation
refinement, AD correctness, IEEE rounding bounds and accumulated numerical error
are OPEN. Source tests cover specified cases, not all real inputs.

Dense extra work per batch event is `3*108*12+12*(12+12+6)=4248` scalar weight
products plus activations/additions. New learned storage is4314 scalars; new
persistent streaming storage is zero. Independent batch entities and post-scan
role branches can run in parallel. No sequential dependency of the inherited
native chronological scan is removed or newly declared legally parallel.

Training objectives, optimizer, authentic-data splits/budgets and held-out gates
belong to the separate TAC preregistration. This no-fit formulation authorizes
no fit, threshold change, favorable-seed selection, extra active family,
automatic merge, deployment or capability promotion. Protected independent
observations, fair comparison, trained replay and useful-memory evidence remain
OPEN. All earlier failed gates and checkpoints retain their original labels.
