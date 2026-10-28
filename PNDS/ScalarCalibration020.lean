import Mathlib

/-! TEMP020 post-hoc scalar layer ONLY. Frozen TAC logit graph is opaque.
No optimizer convergence, IEEE/AD, provenance or generalization theorem.
Exact TAC implementation source pin is in foundation/v1/temperature020.json.
-/
namespace PNDS.ScalarCalibration020

noncomputable def scaled (a z : ℝ) : ℝ := a * z

/-- Positive inverse temperature preserves every pairwise logit ordering. -/
theorem positive_order (a x y : ℝ) (ha : 0 < a) :
    scaled a x ≤ scaled a y ↔ x ≤ y := by
  exact mul_le_mul_left ha

/-- Thus the entire set of maximizers, including exact ties, is unchanged. -/
theorem maximizers {n : Nat} (z : Fin n → ℝ) (a : ℝ) (ha : 0 < a) (i : Fin n) :
    (∀ j, scaled a (z j) ≤ scaled a (z i)) ↔ (∀ j, z j ≤ z i) := by
  constructor
  · intro h j
    exact (positive_order a (z j) (z i) ha).mp (h j)
  · intro h j
    exact (positive_order a (z j) (z i) ha).mpr (h j)

/-- A class-specific bias has no corresponding mode-preservation guarantee. -/
theorem bias_mode_counterexample : (1 : ℝ) > 0 ∧ 1 + 0 < 0 + 2 := by
  norm_num

noncomputable def law {n : Nat} (z : Fin n → ℝ) (a : ℝ) (i : Fin n) : ℝ :=
  Real.exp (a * z i) / ∑ j, Real.exp (a * z j)

noncomputable def categoricalNLL {n : Nat} (z : Fin n → ℝ) (a : ℝ) (y : Fin n) : ℝ :=
  Real.log (∑ j, Real.exp (a * z j)) - a * z y

/-- This is the identity-candidate guard, not a convergence or heldout theorem. -/
noncomputable def identityGuard (identity candidate : ℝ) : ℝ := min identity candidate

theorem guard_not_worse (identity candidate : ℝ) :
    identityGuard identity candidate ≤ identity := min_le_left _ _

end PNDS.ScalarCalibration020

#print axioms PNDS.ScalarCalibration020.positive_order
#print axioms PNDS.ScalarCalibration020.maximizers
#print axioms PNDS.ScalarCalibration020.bias_mode_counterexample
#print axioms PNDS.ScalarCalibration020.guard_not_worse
