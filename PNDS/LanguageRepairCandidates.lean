import PNDS.LanguageKeySensitivity
import PNDS.LanguageCategorical
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
TAC-LANG-REPAIR-016: mathematical candidate solutions, no checkpoint changes.

1. Smooth denominator (square-root regularization) has no zero singularity
   and is at least tau > 0. A scalar normalized coordinate has a
   near-zero absolute-amplification envelope 1/tau.
2. Infinite, exact source ordinal decomposition (era, residue) is
   injective and increments after the known Fourier cycle. The ORIGINAL
   finite eight-component Fourier address did not store the era.

This is a source-compatible proposal at era zero, not a full theorem
that the FLOAT32 transformed learned address remains injective for
arbitrary huge ordinals. The 8D Jacobian norm is separately tested
numerically but not globally Lean-certified in this file.
-/
namespace PNDS.LanguageRepairCandidates
noncomputable section
open PNDS.SourceAlignedRoundoff

def smoothDenom (tau x : ℝ) : ℝ :=
  Real.sqrt (x*x + tau*tau)

def smoothScalarKey (tau x : ℝ) : ℝ :=
  x / smoothDenom tau x

theorem smooth_denom_pos (tau x : ℝ) (htau : 0 < tau) :
    0 < smoothDenom tau x := by
  unfold smoothDenom
  apply Real.sqrt_pos.2
  nlinarith [sq_nonneg x]

theorem smooth_key_zero (tau : ℝ) :
    smoothScalarKey tau 0 = 0 := by
  simp [smoothScalarKey]

/-- The denominator's lower bound avoids the original 1e-6 spike
when choosing tau=1e-2, at the cost of nonunit keys for small inputs. -/
theorem smooth_denom_ge_tau (tau x : ℝ) (htau : 0 < tau) :
    tau ≤ smoothDenom tau x := by
  have hz := smooth_denom_pos tau x htau
  have harg : 0 ≤ x*x+tau*tau := by positivity
  have hsq : (smoothDenom tau x)^2 = x*x+tau*tau := by
    simpa [smoothDenom] using Real.sq_sqrt harg
  nlinarith [mul_self_nonneg x]

theorem smooth_key_amplitude_bound (tau x : ℝ) (htau : 0 < tau) :
    |smoothScalarKey tau x| ≤ |x|/tau := by
  have hden := smooth_denom_pos tau x htau
  have hfloor := smooth_denom_ge_tau tau x htau
  unfold smoothScalarKey
  rw [abs_div, abs_of_pos hden]
  apply (div_le_div_iff₀ hden htau).2
  have h := mul_nonneg (abs_nonneg x) (sub_nonneg.mpr hfloor)
  nlinarith

def ordinalEra (t : ℕ) : ℕ := t / 155701
def ordinalResidue (t : ℕ) : ℕ := t % 155701

theorem ordinal_reconstructed (t : ℕ) :
    ordinalEra t * 155701 + ordinalResidue t = t := by
  unfold ordinalEra ordinalResidue
  omega

theorem era_residue_injective (a b : ℕ)
    (he : ordinalEra a = ordinalEra b)
    (hr : ordinalResidue a = ordinalResidue b) : a = b := by
  have ha := ordinal_reconstructed a
  have hb := ordinal_reconstructed b
  rw [he, hr] at ha
  omega

theorem next_cycle_increments_era (t : ℕ) :
    ordinalEra (t+155701) = ordinalEra t + 1 ∧
    ordinalResidue (t+155701) = ordinalResidue t := by
  unfold ordinalEra ordinalResidue
  omega

theorem first_era_untouched (t : ℕ) (ht : t < 155701) :
    ordinalEra t = 0 ∧ ordinalResidue t = t := by
  unfold ordinalEra ordinalResidue
  omega

/-- Exact arithmetic tag changes at every full cycle. The source address
only sees a rounded finite-dimensional projection of this tag, so
cross-era distinctness in real arithmetic does not prove FP32 uniqueness. -/
def eraFraction (t : ℕ) : ℝ :=
  (ordinalEra t : ℝ)/((ordinalEra t : ℝ)+1)

theorem era_fraction_strict_cycle (t : ℕ) :
    eraFraction t < eraFraction (t+155701) := by
  have hn := (next_cycle_increments_era t).1
  unfold eraFraction
  rw [hn]
  push_cast
  have h1 : (0:ℝ) < (ordinalEra t : ℝ)+1 := by positivity
  have h2 : (0:ℝ) < (ordinalEra t : ℝ)+2 := by positivity
  apply (div_lt_div_iff₀ h1 h2).2
  nlinarith

end
end PNDS.LanguageRepairCandidates
