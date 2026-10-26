import PNDS.RoleGuardedTraining
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
ME109 inward role-guard repair, stated over exact real vectors. The proposed
displacement projects onto the half-space `dot g d ≤ -margin`. The rounding
certificate below is scalar and conditional; it does not model IEEE-754,
PyTorch, a trained model, or prove that a runtime error bound holds.
-/
namespace PNDS.InwardRoleGuard

open PNDS.RoleSpecializedFFN

variable {n : ℕ}

/-- Project `d` inward until its inner product with `g` is at most
`-margin`. A nonnegative margin and positive squared norm are required. -/
noncomputable def inwardProjection (g d : Fin n → ℝ) (margin : ℝ) : Fin n → ℝ :=
  sub d (scale (max 0 ((dot g d + margin) / normSq g)) g)

private theorem dot_scale (g : Fin n → ℝ) (a : ℝ) :
    dot g (scale a g) = a * normSq g := by
  unfold dot normSq scale
  calc
    _ = ∑ i, a * (g i * g i) := by
      apply Finset.sum_congr rfl
      intro i hi
      ring
    _ = a * ∑ i, g i * g i := (Finset.mul_sum _ _ _).symm

private theorem dot_sub (g x y : Fin n → ℝ) :
    dot g (sub x y) = dot g x - dot g y := by
  simp [dot, sub, mul_sub, Finset.sum_sub_distrib]

/-- The exact-real inward projection satisfies its half-space constraint. -/
theorem inwardProjection_dot_le (g d : Fin n → ℝ) (margin : ℝ)
    (hnorm : normSq g > 0) :
    dot g (inwardProjection g d margin) ≤ -margin := by
  have hdot : dot g (inwardProjection g d margin) =
      dot g d - max 0 ((dot g d + margin) / normSq g) * normSq g := by
    rw [inwardProjection, dot_sub, dot_scale]
  rw [hdot]
  by_cases hpositive : 0 < (dot g d + margin) / normSq g
  · have hmax : max 0 ((dot g d + margin) / normSq g) =
        (dot g d + margin) / normSq g := max_eq_right (le_of_lt hpositive)
    rw [hmax]
    have hcancel : ((dot g d + margin) / normSq g) * normSq g = dot g d + margin :=
      (div_mul_cancel₀ _ (ne_of_gt hnorm))
    linarith
  · have hratio : (dot g d + margin) / normSq g ≤ 0 := le_of_not_gt hpositive
    have hmax : max 0 ((dot g d + margin) / normSq g) = 0 := max_eq_left hratio
    rw [hmax]
    have hsum' := mul_le_mul_of_nonneg_right hratio (le_of_lt hnorm)
    have hsum : dot g d + margin ≤ 0 := by
      rw [div_mul_cancel₀ _ (ne_of_gt hnorm)] at hsum'
      simpa using hsum'
    linarith

/-- If the ideal scalar dot product is at most `-margin`, and the actual
rounded dot differs from it by no more than `margin`, then the actual dot is
nonpositive. This is a conditional certificate, not an error-bound proof. -/
theorem rounded_dot_nonpos (idealDot actualDot margin : ℝ)
    (hideal : idealDot ≤ -margin) (herror : |actualDot - idealDot| ≤ margin) :
    actualDot ≤ 0 := by
  have hupper : actualDot - idealDot ≤ margin := (abs_le.mp herror).2
  linarith

end PNDS.InwardRoleGuard
