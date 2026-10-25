import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
ME104 exact-real scalar kernel for an equal-half Gaussian quantile transform
with a median-preserving asymmetric scale head. This module records only the
scalar parameterization; it does not formalize distribution normalization,
the complete source model, or finite-precision execution.
-/
namespace PNDS.MedianSplitSensor

/-- The median location is inherited from the old model and is independent of
the predicted width. -/
def medianLocation (oldLocation _oldWidth _leftRaw _rightRaw : ℝ) : ℝ := oldLocation

/-- The two positive scales share the old scale and use bounded, asymmetric
log-scale modifiers. -/
noncomputable def leftScale (oldScale leftRaw : ℝ) : ℝ :=
  oldScale * Real.exp (Real.log 4 * Real.tanh leftRaw)

noncomputable def rightScale (oldScale rightRaw : ℝ) : ℝ :=
  oldScale * Real.exp (Real.log 4 * Real.tanh rightRaw)

theorem leftScale_positive (oldScale leftRaw : ℝ) (hbase : 0 < oldScale) :
    0 < leftScale oldScale leftRaw := by
  unfold leftScale
  exact mul_pos hbase (Real.exp_pos _)

theorem rightScale_positive (oldScale rightRaw : ℝ) (hbase : 0 < oldScale) :
    0 < rightScale oldScale rightRaw := by
  unfold rightScale
  exact mul_pos hbase (Real.exp_pos _)

theorem medianLocation_unchanged (oldLocation oldWidth leftRaw rightRaw : ℝ) :
    medianLocation oldLocation oldWidth leftRaw rightRaw = oldLocation := rfl

theorem leftScale_modifier_zero (oldScale : ℝ) :
    leftScale oldScale 0 = oldScale := by
  simp [leftScale]

theorem rightScale_modifier_zero (oldScale : ℝ) :
    rightScale oldScale 0 = oldScale := by
  simp [rightScale]

/-- A generic exponential modifier bounded below by `-log 4` preserves at
least one quarter of a positive base scale. -/
noncomputable def exponentialScale (base modifier : ℝ) : ℝ := base * Real.exp modifier

theorem exponentialScale_lower_bound (base modifier : ℝ)
    (hbase : 0 ≤ base) (hmodifier : -Real.log 4 ≤ modifier) :
    base / 4 ≤ exponentialScale base modifier := by
  unfold exponentialScale
  have hexp : Real.exp (-Real.log 4) ≤ Real.exp modifier :=
    Real.exp_le_exp.mpr hmodifier
  have hexpQuarter : Real.exp (-Real.log 4) = (1 / 4 : ℝ) := by
    rw [← Real.log_inv]
    norm_num [Real.exp_log]
  have hquarter : (1 / 4 : ℝ) ≤ Real.exp modifier := by
    rw [← hexpQuarter]
    exact hexp
  have hmul := mul_le_mul_of_nonneg_left hquarter hbase
  convert hmul using 1
  ring

/-!
OPEN: full-model source correspondence for the 99,731-parameter head, the
equal-half Gaussian probability measure and normalization, quantile/CDF
behavior, censored-lognormal transform, neural-network gradients, IEEE
finite-precision refinement, whole-network invariance, and calibration.
-/

end PNDS.MedianSplitSensor
