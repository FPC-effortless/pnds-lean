import Mathlib

/-!
LANG10M001 partial real-number mathematics.

This proves scalar recurrence facts used by the upper recurrent blocks of
TAC-OSM PR #658, source commit 5432befbcd36bb8237ae6929433abb4bd2d9003e.

Not a whole-model specification or proof: the actual source also includes
layer-normalized projections, nonlinear gates, an 8×8 associative bank,
two aligned banks, waves, six concept slots, FFN, tokenization and
finite-precision evaluation. All corresponding proof obligations remain OPEN.
-/

namespace PNDS.LANG10M001

/-- A single coordinate of the upper input-conditioned affine state step.
The applied model has alpha = 0.995 * sigmoid(learned_input), a property
not assumed in these algebraic identities. -/
def scalarStep (alpha offer previous : ℝ) : ℝ :=
  alpha * previous + (1 - alpha) * offer

theorem previous_difference (alpha offer x y : ℝ) :
    scalarStep alpha offer x - scalarStep alpha offer y =
      alpha * (x - y) := by
  dsimp [scalarStep]
  ring

theorem zero_retention_writes_offer (offer previous : ℝ) :
    scalarStep 0 offer previous = offer := by
  simp [scalarStep]

theorem full_retention_keeps_previous (offer previous : ℝ) :
    scalarStep 1 offer previous = previous := by
  simp [scalarStep]

/-- This is a scalar open-loop recurrence identity only, not a whole-model
causality or robustness certificate. -/
theorem identical_observation_has_identical_update
    (alpha offer previous : ℝ) :
    scalarStep alpha offer previous =
      alpha * previous + (1 - alpha) * offer := by
  rfl

end PNDS.LANG10M001
