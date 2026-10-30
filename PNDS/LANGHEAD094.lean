import Mathlib

/-!
LANGHEAD094 scoped exact-real formulas only.
Source TAC FPC-effortless/TAC-OSM at acff46cf996cca6f1fefda49e7289060f7ce90e3
research/mathematical-engineering/langhead094/model094.py.

This does NOT assert full GRU/SSM PyTorch computation equivalence, finite-precision
error bounds, whole TAC native fast/slow graph correctness, training capability,
or matched-budget model comparison. Those are OPEN.
-/
namespace PNDS.LANGHEAD094

/-- Scalar coordinate of input-selective diagonal SSM recurrent update. -/
def diagonalStep (alpha previous offer : ℝ) : ℝ :=
  alpha * previous + (1 - alpha) * offer

theorem previous_state_difference (alpha u x y : ℝ) :
    diagonalStep alpha x u - diagonalStep alpha y u =
    alpha * (x - y) := by
  simp [diagonalStep]
  ring

theorem candidate_is_fixed_point (alpha u : ℝ) :
    diagonalStep alpha u u = u := by
  simp [diagonalStep]
  ring

/-- The PyTorch GRU scalar state-update interpolation after r,z,n are computed. -/
def gruBlend (update candidate old : ℝ) : ℝ :=
  (1 - update) * candidate + update * old

theorem gru_update_difference (z n x y : ℝ) :
    gruBlend z n x - gruBlend z n y = z * (x - y) := by
  simp [gruBlend]
  ring

theorem gru_stays_on_candidate (z n : ℝ) :
    gruBlend z n n = n := by
  simp [gruBlend]
  ring

end PNDS.LANGHEAD094
