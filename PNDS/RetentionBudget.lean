import Mathlib

/-!
PNDS v1.7: controlled source-shaped TAC gated delta-retention law.

The original learned sigmoid outputs a,b ∈ [0,1]. No new learned
parameters are added. The input-only retention budget changes effective
write strength from b to b/4 and the raw decay to
  keep=1-(1-a)/50.
The actual right-affine operator used by the continuous TAC model has
  alpha_eff = keep*(1-b/16), beta_eff=b/4.
This is a bounded, parallel-scannable *new candidate*, not the source
DIAGKEY004 recurrence and not evidence of learned long-memory utility.

Under fixed known event gates/keys, the complete affine-memory bounding
proof is conditional on the matrix norm inequality
  newNorm <= alpha_eff*oldNorm + beta_eff.
This norm inequality follows analytically for unit ball keys/values and
beta_eff∈[0,1]; exact original PyTorch floating-point correspondence is
checked separately, not proved here.
-/
namespace PNDS.RetentionBudget

noncomputable def keep (a : ℝ) : ℝ := 1 - (1-a)/50
noncomputable def written (b : ℝ) : ℝ := b/4
noncomputable def alpha (a b : ℝ) : ℝ := keep a * (1-b/16)

theorem keep_range (a : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    49/50 ≤ keep a ∧ keep a ≤ 1 := by
  unfold keep
  constructor <;> linarith

theorem written_range (b : ℝ) (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    0 ≤ written b ∧ written b ≤ 1/4 := by
  unfold written
  constructor <;> linarith

/-- Even the worst admissible 0<=a,b<=1 schedule retains at least 147/160
    of its old matrix when evaluated as a linear state-difference map.
    This is operator attenuation, NOT 147/160 correct factual recall. -/
theorem alpha_lower (a b : ℝ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    147/160 ≤ alpha a b := by
  obtain ⟨hlo,hhi⟩ := keep_range a ha0 ha1
  have hb : 15/16 ≤ 1-b/16 := by linarith
  have h1 : (49/50 : ℝ)*(15/16) ≤ keep a*(15/16) :=
    mul_le_mul_of_nonneg_right hlo (by norm_num)
  have h2 : keep a*(15/16) ≤ keep a*(1-b/16) :=
    mul_le_mul_of_nonneg_left hb (by linarith)
  unfold alpha
  nlinarith

theorem alpha_budget (a b : ℝ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    0 ≤ alpha a b ∧ alpha a b ≤ 1-b/16 := by
  obtain ⟨hk0,hk1⟩ := keep_range a ha0 ha1
  have hq0 : 0 ≤ 1-b/16 := by linarith
  constructor
  · exact mul_nonneg (by linarith) hq0
  · unfold alpha
    simpa using mul_le_mul_of_nonneg_right hk1 hq0

theorem radius_four_next_bounded (a b oldNorm newNorm : ℝ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (h_old : oldNorm ≤ 4)
    (h_next : newNorm ≤ alpha a b * oldNorm + written b) :
    newNorm ≤ 4 := by
  obtain ⟨halpha0,halphaUp⟩ :=
    alpha_budget a b ha0 ha1 hb0 hb1
  have hscaled :=
    mul_le_mul_of_nonneg_left h_old halpha0
  have hbudget : alpha a b * 4 + written b ≤ 4 := by
    have hscale : alpha a b*4 ≤ (1-b/16)*4 :=
      mul_le_mul_of_nonneg_right halphaUp (by norm_num)
    unfold written
    nlinarith
  linarith

end PNDS.RetentionBudget
