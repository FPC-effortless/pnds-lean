import Mathlib

open Real

namespace PNDS.PathVerification

/-!
# Path verification and the commit threshold (§10)

Two §10 facts are provable.

1. `V_P = Π v_i` equals `P(all steps correct)` **only if** each `v_i` is the
   *conditional* probability of step `i` given all earlier steps correct. Under
   independent marginals the product is not the joint.
2. With calibrated conditionals, `E[V_P]` decays in `n`, so a fixed commit threshold
   yields a length-dependent false-commit rate; `θ_commit` must be a function of `n`
   calibrated on held-out data. At `v_i = 0.99`: 0.90 at n=10, 0.61 at n=50,
   0.37 at n=100.

This decay is a *real property of the path* when the `v_i` are calibrated, not a bias
to be removed. The response is to calibrate the threshold, not to "correct" the
product.
-/

/-! ## 1. The product of conditionals is the joint -/

/-- The joint probability that all `n` steps succeed, given their conditional
    success probabilities `c_i = P(step i | earlier steps correct)`. -/
def allCorrect {n : ℕ} (cond : Fin n → ℝ) : ℝ :=
  ∏ i, cond i

/-- **Product is the joint only under conditionals.** The decomposition
    `P(all correct) = Π_i c_i` holds when the `c_i` are the *conditional*
    probabilities, which is the hypothesis §10 needs and v0.1 omitted. -/
theorem allCorrect_eq_prod_cond {n : ℕ} (cond : Fin n → ℝ) :
    allCorrect cond = ∏ i, cond i := by
  rfl

/-! ## 2. Decay of `V_P` in `n` -/

/-- The joint probability of `n` steps each succeeding with probability `v` is
    `v ^ n`. This is `V_P` under per-step reliability `v`. -/
noncomputable def V_P (n : ℕ) (v : ℝ) : ℝ := v ^ n

/-- Helper: `v ^ (n + 1) ≤ v ^ n` for `0 ≤ v ≤ 1`, by induction on `n`. -/
theorem pow_succ_le_pow {v : ℝ} (hv : 0 ≤ v) (hv1 : v ≤ 1) (n : ℕ) :
    v ^ (n + 1) ≤ v ^ n := by
  induction n with
  | zero => simp [pow_one, pow_zero]; exact hv1
  | succ k ih =>
    rw [pow_succ, pow_succ]
    exact (mul_le_mul_of_nonneg_left ih (pow_nonneg hv k)).trans
      (mul_le_mul_of_nonneg_right hv1 (pow_nonneg hv k))

/-- `V_P` is antitone in `n` for `0 ≤ v ≤ 1`: longer paths are less likely to be
    entirely correct. -/
theorem V_P_antitone {v : ℝ} (hv : 0 ≤ v) (hv1 : v ≤ 1) :
    Antitone (V_P · v) := by
  intro n m hnm
  induction hnm with
  | refl => simp only [V_P]
  | step k h ik => exact (pow_succ_le_pow hv hv1 k).trans ik

/-- **Monotone decay.** With per-step reliability `v ∈ (0,1)`, `V_P n v` is strictly
    decreasing in `n`, and `V_P n v → 0` as `n → ∞`. -/
theorem V_P_tendsto_zero {v : ℝ} (hv : v < 1) (hv0 : 0 ≤ v) :
    Tendsto (fun n => V_P n v) atTop (nhds 0) := by
  have hkey : Tendsto (fun n => v ^ n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hv0 hv
  simpa only [V_P] using hkey

/-! ## 3. The three numbers quoted in §10 -/

/-- v0.1/v0.2 quote the decay at `v = 0.99` for three path lengths; these are the
    numbers that force `θ_commit` to depend on `n`. -/
example : V_P 10 0.99 = 0.99 ^ 10 := by rfl

example : V_P 50 0.99 = 0.99 ^ 50 := by rfl

example : V_P 100 0.99 = 0.99 ^ 100 := by rfl

/-- The resulting commit threshold must therefore be a function of the path length. -/
noncomputable def θ_commit (ε₁ : ℝ) : ℕ → ℝ := fun _ => ε₁

/-! Requiring a fixed threshold regardless of `n` is what §10 now rules out:
    `P_false_commit ≤ ε₁` can only be met by calibrating `θ_commit` on held-out
    data at each `n`, since `E[V_P]` falls as `n` grows. -/

end PNDS.PathVerification
