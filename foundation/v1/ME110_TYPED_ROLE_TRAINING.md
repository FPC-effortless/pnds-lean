# ME110 typed role training

**Status: exact-real coordinate algebra proved; source parity checked by the
dedicated pinned-source workflow; full AD/IEEE and whole-model proof OPEN. No
training is authorized or represented by this specification.** ME110 keeps
the native ME109 forward model and its 104663 parameters. The original source
is pinned to TAC-OSM commit
[`edb9a3d4f590de386ad985cdc0f8ff5556a25a5d`](https://github.com/FPC-effortless/TAC-OSM/commit/edb9a3d4f590de386ad985cdc0f8ff5556a25a5d),
specifically `src/tac_osm/typed_role_learning.py`; source ownership and all
training outcomes remain in TAC-OSM.

## Coordinate roles

For each of the 45 named parameter tensors, coordinates are flattened in
PyTorch row-major order. Let `U_i=1` denote uncertainty coordinates and
`P_i=1-U_i` point coordinates. Uncertainty coordinates are:

- every scalar of the exactly named `state_noise` parameter;
- every scalar in `width_head.*` and `width_residual.*`;
- rows 6 through 11 of `readout.2.*`, `retrieved_delta.2.*`,
  `current_head.*`, and `current_residual.output.*`.

Every other coordinate is point-role. The pinned source rejects selected
output tensors unless their leading dimension is 12 and rejects models unless
they contain exactly 45 parameter tensors. Source and reference counts are
4410 uncertainty coordinates and 100253 point coordinates.

## Routed gradient and optimizer

Given point and auxiliary losses, the source computes their gradients
separately. Point gradients must be exactly zero on `U`; otherwise routing
raises an error. It masks the auxiliary gradient to `U` and forms the role
vectors

\[
r_P=P\odot g_P,\qquad r_U=U\odot g_U.
\]

Each vector is independently scaled before composition:

\[
\operatorname{clip}(r)=r\min\left(1,\frac{1}{\|r\|_2+10^{-6}}\right),\qquad
g_{route}=\operatorname{clip}(r_P)+\operatorname{clip}(r_U).
\]

Thus masking occurs before norm calculation and clipping. The reference uses
the same epsilon denominator. Fresh coordinatewise AdamW moments use the
source optimizer settings; role masking alone does not model decoupled weight
decay or the guarded optimizer proposal.

For a fixed binary mask, exact-real algebra proves that changing `g_U` leaves
point coordinates of the routed gradient unchanged and changing `g_P` leaves
uncertainty coordinates unchanged. The mask products are disjoint and sum to
one. A decoder that factors through point-masked coordinates is invariant when
only uncertainty coordinates change. These statements do not establish that a
particular neural decoder factors that way, prove actual autograd behavior,
account for finite precision, or imply loss descent. Actual neural mean/state
invariance remains an empirical source probe, and the complete AD/IEEE
refinement remains OPEN.

## Evidence and validation

`PNDS.TypedRoleTraining` contains the exact-real mask and conditional decoder
theorems. `foundation/v1/typed_role_training_reference.py` is the independent
Python equation reference. `tests/test_typed_role_training_source.py` compares
all 104663 flattened coordinate mask entries across the 45 source tensors and
compares routed gradients against the pinned PyTorch implementation. The
workflow `.github/workflows/me110-typed-role-source.yml` checks out the exact
source commit, runs those parity tests, compiles the Lean module, and prints
the theorem axiom dependencies. It performs no training and makes no claim
about independent predictive capability.
