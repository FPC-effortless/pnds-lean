import PNDS.LanguageTensorFP

/-!
TAC-LANG-FP-015: the actual 1e-6 key-normalization clamp avoids division
by zero, but does NOT guarantee that tiny input perturbations remain tiny
relative to the original input magnitude. The local zero-norm regime
amplifies absolute error by 1/epsilon. This is a genuine design tradeoff.

This scalar slice of the 8-D Torch F.normalize must not be misconstrued as
a complete vector Jacobian or FP32 kernel proof.
-/
namespace PNDS.LanguageKeySensitivity
noncomputable section
open PNDS.SourceAlignedRoundoff

def scalarKey (eps x : ℝ) : ℝ := x / max |x| eps

/-- Within the epsilon-clamped source branch, the normalized coordinate
is exactly x/eps. No unverified floating-point semantics required. -/
theorem small_key_exact (eps x : ℝ) (hx : |x| ≤ eps) :
    scalarKey eps x = x / eps := by
  simp [scalarKey, max_eq_right hx]

/-- The absolute-error guarantee of a clamped key is enlarged by 1/eps.
For the source eps=1e-6 that factor is 1,000,000, which matters for
memory-address gradients and sensitivity. -/
theorem small_key_error
    (eps x y radius : ℝ) (heps : 0 < eps)
    (hx : |x| ≤ eps) (hy : |y| ≤ eps)
    (h : Close x y radius) :
    Close (scalarKey eps x) (scalarKey eps y) (radius/eps) := by
  rw [small_key_exact eps x hx, small_key_exact eps y hy]
  unfold Close at *
  rw [← sub_div, abs_div, abs_of_pos heps]
  exact div_le_div_of_nonneg_right h heps.le

/-- A HALF-epsilon perturbation of a zero input becomes a one-half change
in a normalized address coordinate, despite the absence of division by zero.
This constructive example is architectural math, not observed data quality. -/
theorem key_amplification_witness :
    |(1/2000000 : ℝ) - 0| = (1/2000000 : ℝ) ∧
    |scalarKey (1/1000000) (1/2000000) -
      scalarKey (1/1000000) 0| = (1/2 : ℝ) := by
  constructor
  · norm_num
  · rw [small_key_exact (1/1000000) (1/2000000)
      (by
        rw [abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/2000000)]
        norm_num), small_key_exact (1/1000000) 0
      (by norm_num)]
    norm_num

end
end PNDS.LanguageKeySensitivity
