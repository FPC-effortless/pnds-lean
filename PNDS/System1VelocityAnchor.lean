import Mathlib

/-!
# TAC System1 velocity-anchored observational correction — exact-real interface

Source TAC branch: research/system1-velocity-anchored-2026-10-26,
`src/tac_osm/variants/system1_velocity_anchored.py` (S1MOTION002).

This module proves **only** bounded algebraic point-composition from a causal
previous-observation velocity prior and an ASSUMED bounded neural correction.
The `delta` correction bound must be linked to the complete original neural
readout and finite tanh implementation in future refinements. No source-level
PyTorch equivalence, IEEE bit rounding, AD/Jacobian theorem, calibrated adverse
event, physical causal intervention, device SLA or real prediction gain proven.
Original S1MOTION001 empirical gate remains FAIL 0/3; no retraining.
-/

namespace PNDS.System1VelocityAnchor

abbrev Joint14 := Fin 14 → ℝ

/-- The previous and current genuine observations, not future outcomes. -/
noncomputable def velocityPrior (current previous : Joint14) : Joint14 :=
  fun j => 2 * current j - previous j

/-- Source-composition operator, with an externally bounded neural residual. -/
noncomputable def anchoredPrediction
    (current previous correction : Joint14) (gain : ℝ) : Joint14 :=
  fun j => velocityPrior current previous j + gain * correction j

/-- At zero correction gain, exact-real forecasting equals causal velocity. -/
theorem zero_gain_is_velocity
    (current previous correction : Joint14) :
    anchoredPrediction current previous correction 0 =
    velocityPrior current previous := by
  funext j
  simp [anchoredPrediction]

/-- Coordinatewise exact-real bound, conditional on bounded residual input.
This is NOT a theorem that native neural predictions are accurate. -/
theorem deviation_from_velocity_le
    (current previous correction scale : Joint14)
    (gain : ℝ) (hgain : 0 ≤ gain)
    (hcorr : ∀ j, |correction j| ≤ scale j)
    (j : Fin 14) :
    |anchoredPrediction current previous correction gain j -
        velocityPrior current previous j| ≤ gain * scale j := by
  have hdiff :
      anchoredPrediction current previous correction gain j -
        velocityPrior current previous j = gain * correction j := by
    dsimp [anchoredPrediction]
    ring
  rw [hdiff, abs_mul, abs_of_nonneg hgain]
  exact mul_le_mul_of_nonneg_left (hcorr j) hgain

/-- Source-defined cap gain <= .2 yields an explicit worst-case absolute deviation. -/
theorem bounded_gain_deviation
    (current previous correction scale : Joint14)
    (gain : ℝ) (hgain : 0 ≤ gain)
    (hcap : gain ≤ (1 / 5 : ℝ))
    (hscale : ∀ j, 0 ≤ scale j)
    (hcorr : ∀ j, |correction j| ≤ scale j)
    (j : Fin 14) :
    |anchoredPrediction current previous correction gain j -
        velocityPrior current previous j| ≤ (1 / 5 : ℝ) * scale j := by
  calc
    _ ≤ gain * scale j :=
      deviation_from_velocity_le current previous correction scale gain hgain hcorr j
    _ ≤ (1 / 5 : ℝ) * scale j :=
      mul_le_mul_of_nonneg_right hcap (hscale j)

/-- No learned parameter inflation; floating state count increases by 14. -/
def fp32ScalarCount : ℕ := 88 + 8 * 8 + 14

/-- Includes 88D basis sum, 8x8 matrix, and explicit 14D previous observation. -/
theorem fp32_scalar_count_is_166 : fp32ScalarCount = 166 := by decide

/-- 4 bytes per stored FP32 scalar (ignores metadata/weights/runtime buffers). -/
theorem fp32_carried_bytes_is_664 : 4 * fp32ScalarCount = 664 := by decide

end PNDS.System1VelocityAnchor
