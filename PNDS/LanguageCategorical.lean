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

/-- The two-head softmax router shares a total write budget of at most one. -/
def routedWrite (router : Vec 2) (base modulation : Vec 2) (h : Fin 2) : ℝ :=
  sig (base h + silu (2 * modulation h)) * softmax router h

theorem routed_write_budget (router base modulation : Vec 2) :
    0 ≤ routedWrite router base modulation 0 ∧
    0 ≤ routedWrite router base modulation 1 ∧
    routedWrite router base modulation 0 + routedWrite router base modulation 1 ≤ 1 := by
  have r0 := softmax2_range router 0
  have r1 := softmax2_range router 1
  have s0 := sigmoid_range (base 0 + silu (2 * modulation 0))
  have s1 := sigmoid_range (base 1 + silu (2 * modulation 1))
  have h0 : 0 ≤ routedWrite router base modulation 0 := mul_nonneg s0.1.le r0.1
  have h1 : 0 ≤ routedWrite router base modulation 1 := mul_nonneg s1.1.le r1.1
  have b0 : routedWrite router base modulation 0 ≤ softmax router 0 := by
    have h : 0 ≤ (1 - sig (base 0 + silu (2 * modulation 0))) * softmax router 0 :=
      mul_nonneg (sub_nonneg.mpr s0.2.le) r0.1
    dsimp [routedWrite]
    nlinarith
  have b1 : routedWrite router base modulation 1 ≤ softmax router 1 := by
    have h : 0 ≤ (1 - sig (base 1 + silu (2 * modulation 1))) * softmax router 1 :=
      mul_nonneg (sub_nonneg.mpr s1.2.le) r1.1
    dsimp [routedWrite]
    nlinarith
  refine ⟨h0,h1,?_⟩
  calc
    routedWrite router base modulation 0 + routedWrite router base modulation 1
        ≤ softmax router 0 + softmax router 1 := add_le_add b0 b1
    _ = 1 := softmax2_sum router

/-- Exact-real running-mean update preserves the coordinate unit interval. -/
theorem meanStep_unit_box (n : ℕ) (prior incoming : ℝ)
    (hp0 : -1 ≤ prior) (hp1 : prior ≤ 1)
    (hi0 : -1 ≤ incoming) (hi1 : incoming ≤ 1) :
    -1 ≤ meanStep n prior incoming ∧ meanStep n prior incoming ≤ 1 := by
  rw [meanStep_equiv]
  have hn : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
  have hd : 0 < (n : ℝ) + 1 := by positivity
  constructor
  · rw [le_div_iff₀ hd]
    have h : 0 ≤ (n : ℝ) * (prior + 1) :=
      mul_nonneg hn (by linarith)
    nlinarith
  · rw [div_le_iff₀ hd]
    have h : 0 ≤ (n : ℝ) * (1 - prior) :=
      mul_nonneg hn (by linarith)
    nlinarith

/-- Recentring all 128 logits changes the partition function by a positive
    factor. Useful for stable log-sum-exp implementations. -/
theorem partition_shift (scores : Scores) (c : ℝ) :
    Z (fun i => scores i + c) = Z scores * Real.exp c := by
  simp [Z, Real.exp_add, Finset.sum_mul]

/-- Adding any shared finite logit offset leaves the categorical law unchanged.
    This is an exact-real identity; finite-precision kernels need separate tests. -/
theorem probability_shift (scores : Scores) (i : Fin 128) (c : ℝ) :
    probability (fun j => scores j + c) i = probability scores i := by
  have hExp : Real.exp c ≠ 0 := ne_of_gt (Real.exp_pos c)
  have hZ : Z scores ≠ 0 := ne_of_gt (Z_pos scores)
  unfold probability
  rw [partition_shift, Real.exp_add]
  field_simp [hExp,hZ]
  ring

/-- Shift-invariant negative log likelihood under the same exact-real law. -/
theorem nll_shift (scores : Scores) (i : Fin 128) (c : ℝ) :
    nll (fun j => scores j + c) i = nll scores i := by
  simp [nll, probability_shift]

/-- Exact-real log-softmax NLL: logarithm is applied to the positive
    sum-of-exponentials rather than the potentially underflowed rounded
    probability. Torch FP32 correctness is a SEPARATE numerical obligation. -/
theorem nll_equals_log_partition_minus_logit (scores : Scores)
    (i : Fin 128) :
    nll scores i = Real.log (Z scores) - scores i := by
  unfold nll probability
  rw [Real.log_div (ne_of_gt (Real.exp_pos _)) (ne_of_gt (Z_pos scores))]
  rw [Real.log_exp]
  ring

def shiftedStableNLL (scores : Scores) (i : Fin 128) (c : ℝ) : ℝ :=
  Real.log (Z (fun j => scores j - c)) - (scores i - c)

/-- Recentring before log-sum-exp preserves the exact NLL at any real shift.
    A stable implementation commonly chooses c=max(logits). -/
theorem shifted_stable_nll_correct (scores : Scores) (i : Fin 128)
    (c : ℝ) : shiftedStableNLL scores i c = nll scores i := by
  unfold shiftedStableNLL
  calc
    Real.log (Z (fun j => scores j - c)) - (scores i - c) =
      nll (fun j => scores j - c) i :=
        (nll_equals_log_partition_minus_logit (fun j => scores j-c) i).symm
    _ = nll scores i := by
      simpa only [sub_eq_add_neg] using nll_shift scores i (-c)

end
end PNDS.LanguageCategorical
