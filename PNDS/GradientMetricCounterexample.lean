import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum

/-! DIAGGRAD001: optimizer metrics can invalidate a Euclidean direction bound.
Exact two-coordinate arithmetic, not a theorem about trained loss changes or
the full AdamW map. No TAC forward, state, probability law or weights change.
-/
namespace PNDS.GradientMetricCounterexample

/-- p=(1,1), u=(2,-2): auxiliary is orthogonal, so no projection is needed. -/
theorem auxiliary_orthogonal : (1:ℝ)*2 + 1*(-2) = 0 := by norm_num

/-- g=p+u=(3,-1) has a positive point-gradient dot product. -/
theorem combined_euclidean_positive : (1:ℝ)*3 + 1*(-1) > 0 := by norm_num

/-- D=diag(1,100), as a model of unequal group rates, reverses that sign. -/
theorem scaled_direction_reverses : (1:ℝ)*(1*3) + 1*(100*(-1)) < 0 := by norm_num

/-- The update displacement -Dg is locally uphill for p. -/
theorem negative_scaled_update_uphill : (1:ℝ)*(-(1*3)) + 1*(-(100*(-1))) = 97 := by norm_num

end PNDS.GradientMetricCounterexample

#print axioms PNDS.GradientMetricCounterexample.auxiliary_orthogonal
#print axioms PNDS.GradientMetricCounterexample.combined_euclidean_positive
#print axioms PNDS.GradientMetricCounterexample.scaled_direction_reverses
#print axioms PNDS.GradientMetricCounterexample.negative_scaled_update_uphill
