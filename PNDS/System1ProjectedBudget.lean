import Mathlib
import PNDS.System1GradientStop

/-!
# S1MOTION005: exact-real signed-gradient projection algebra

Grounding: TAC-OSM S1MOTION005, src/tac_osm/variants/s1motion005_gradient_budget.py.
This formalizes scalar dot-product obligations **assuming** the supplied
squared point norm and raw point/aux dot. Native tensors, vector reductions,
AdamW preconditioning, floating point rounding, and actual source Jacobian
are NOT source-refined by this lemma. Empirical testing remains separate.
-/

namespace PNDS.System1ProjectedBudget

/-- Exact-real dot between point gradient and projected auxiliary gradient.
If the original dot is negative, remove precisely its opposing component. -/
noncomputable def projectedPointAuxDot (pSquared rawDot : ℝ) : ℝ :=
  if rawDot < 0 then rawDot - (rawDot / pSquared) * pSquared else rawDot

/-- The projected auxiliary cannot oppose point gradient assuming that the
squared norm is exact and positive. -/
theorem projectedPointAuxDot_nonneg
    (pSquared rawDot : ℝ) (hp : 0 < pSquared) :
    0 ≤ projectedPointAuxDot pSquared rawDot := by
  by_cases hd : rawDot < 0
  · have hpne : pSquared ≠ 0 := ne_of_gt hp
    simp [projectedPointAuxDot, hd, div_mul_cancel₀ _ hpne]
  · simp only [projectedPointAuxDot, if_neg hd]
    exact le_of_not_gt hd

/-- Ideal SGD first-order point gradient dot the candidate update is
nonpositive. This does NOT establish an AdamW step or nonlinear finite gain. -/
theorem idealSGD_point_direction_nonpositive
    (pSquared projectedDot beta learningRate : ℝ)
    (hp : 0 ≤ pSquared) (hq : 0 ≤ projectedDot)
    (hb : 0 ≤ beta) (hlr : 0 ≤ learningRate) :
    -(learningRate * (pSquared + beta * projectedDot)) ≤ 0 := by
  have hmul : 0 ≤ beta * projectedDot := mul_nonneg hb hq
  have hsum : 0 ≤ pSquared + beta * projectedDot := add_nonneg hp hmul
  have hstep : 0 ≤ learningRate * (pSquared + beta * projectedDot) :=
    mul_nonneg hlr hsum
  linarith

noncomputable def cappedBudget (proposed betaCap : ℝ) : ℝ :=
  min proposed betaCap

theorem cappedBudget_le_cap (proposed betaCap : ℝ) :
    cappedBudget proposed betaCap ≤ betaCap := by
  exact min_le_right _ _

theorem cappedBudget_nonneg
    (proposed betaCap : ℝ) (hp : 0 ≤ proposed) (hc : 0 ≤ betaCap) :
    0 ≤ cappedBudget proposed betaCap := by
  exact le_min hp hc

end PNDS.System1ProjectedBudget
