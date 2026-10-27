import PNDS.SourceAlignedModel
import Mathlib

/-!
TAC-LANG-FORMAL-004: exact-real, source-scoped language propositions.
Categorical emission is a 128-way distribution; the source uses Torch
softmax followed by cross-entropy. Theorems are mathematical, not IEEE/Torch
kernel, checkpoint, empirical performance, or whole-program AD refinements.
-/
namespace PNDS.LanguageCategorical

noncomputable section
open PNDS.SourceAlignedModel

abbrev Scores := Fin 128 → ℝ

def Z (scores : Scores) : ℝ := ∑ j : Fin 128, Real.exp (scores j)

def probability (scores : Scores) (i : Fin 128) : ℝ :=
  Real.exp (scores i) / Z scores

def nll (scores : Scores) (target : Fin 128) : ℝ :=
  -Real.log (probability scores target)

/-- Normalization never divides by zero in the exact-real semantics. -/
theorem Z_pos (scores : Scores) : 0 < Z scores := by
  unfold Z
  exact Finset.sum_pos (fun i _ => Real.exp_pos (scores i)) Finset.univ_nonempty

/-- Every exact-real categorical probability is strictly positive. -/
theorem probability_pos (scores : Scores) (i : Fin 128) :
    0 < probability scores i := by
  exact div_pos (Real.exp_pos _) (Z_pos scores)

/-- The 128 categorical probabilities sum to one in exact real arithmetic. -/
theorem probability_sum_one (scores : Scores) :
    (∑ i : Fin 128, probability scores i) = 1 := by
  calc
    (∑ i : Fin 128, probability scores i) = Z scores / Z scores := by
      simp [probability, Z, Finset.sum_div]
    _ = 1 := div_self (ne_of_gt (Z_pos scores))

/-- This precise theorem is not a statement about calibration. -/
theorem probability_le_one (scores : Scores) (i : Fin 128) :
    probability scores i ≤ 1 := by
  have hp : 0 < Z scores := Z_pos scores
  have hle : Real.exp (scores i) ≤ Z scores := by
    unfold Z
    exact Finset.single_le_sum (fun j _ => (Real.exp_pos (scores j)).le) (Finset.mem_univ i)
  exact (div_le_one hp).2 hle

/-- Negative log-likelihood is nonnegative for a mathematical categorical law. -/
theorem nll_nonnegative (scores : Scores) (i : Fin 128) :
    0 ≤ nll scores i := by
  have hp := probability_pos scores i
  have hle := probability_le_one scores i
  unfold nll
  exact neg_nonneg.mpr (Real.log_nonpos hp.le hle)

/-- The source uses SiLU only INSIDE a sigmoid; SiLU itself is unbounded. -/
def boundedWrite (base modulation route : ℝ) : ℝ :=
  sig (base + silu (2*modulation)) * route

/-- A softmax-router coefficient in [0,1] yields a bounded write strength. -/
theorem bounded_write_range (base modulation route : ℝ)
    (hr0 : 0 ≤ route) (hr1 : route ≤ 1) :
    0 ≤ boundedWrite base modulation route ∧
      boundedWrite base modulation route ≤ 1 := by
  have hs := sigmoid_range (base + silu (2*modulation))
  constructor
  · exact mul_nonneg hs.1.le hr0
  · have hprod : 0 ≤ (1 - sig (base + silu (2*modulation))) * route :=
      mul_nonneg (sub_nonneg.mpr hs.2.le) hr0
    dsimp [boundedWrite]
    nlinarith

/-- The exact scalar component of the 16x8 matrix corrective write. -/
def scalarDelta (m v k α β : ℝ) : ℝ :=
  α*m + β*(v-α*m*k)*k

/-- Exact-source right-affine factorization, input scheduled. -/
theorem scalar_delta_affine (m v k α β : ℝ) :
    scalarDelta m v k α β = m*(α*(1-β*k*k)) + β*v*k := by
  unfold scalarDelta
  ring

/-- Input-independent local difference: no label or future event in the matrix schedule. -/
theorem scalar_delta_difference (m₁ m₂ v k α β : ℝ) :
    scalarDelta m₁ v k α β - scalarDelta m₂ v k α β =
      α*(1-β*k*k)*(m₁-m₂) := by
  unfold scalarDelta
  ring

/-- Alternative bounded fast statistic: exact-real running mean. -/
def meanStep (n : ℕ) (prior incoming : ℝ) : ℝ :=
  prior + (incoming-prior)/((n:ℝ)+1)

/-- A running mean is a reparameterization of prefix sum / number of samples. -/
theorem meanStep_equiv (n : ℕ) (prior incoming : ℝ) :
    meanStep n prior incoming = ((n:ℝ)*prior+incoming)/((n:ℝ)+1) := by
  unfold meanStep
  have hd : ((n:ℝ)+1) ≠ 0 := by positivity
  field_simp
  ring

/-- The known modulo Fourier address is forced to repeat at 155701 steps. -/
theorem modulo_cycle (t : ℕ) :
    (t+155701)%7=t%7 ∧ (t+155701)%13=t%13 ∧
    (t+155701)%29=t%29 ∧ (t+155701)%59=t%59 := by
  omega

end
end PNDS.LanguageCategorical
