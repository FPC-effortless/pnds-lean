import PNDS.LanguageFullGraph
import PNDS.SourceAlignedRoundoff

/-!
TAC-LANG-FP-006: conditional finite-precision refinement of the language
source's 88-scalar fast-basis accumulating sum.

Crucial scope:
* The proof quantifies over EVERY finite chronological event sequence.
* It assumes a local absolute-error certificate for each rounded addition
  actually encountered. It does NOT claim those premises are discharged by
  Torch/IEEE, whose kernels/device/dtype can differ.
* Independent basis-encoder error, approximate recurrence due to parameter
  quantization, nonlinearity/normalization, and all full-model backwards
  bounds remain independent work.
-/
namespace PNDS.LanguageFPContracts
noncomputable section

open PNDS.SourceAlignedRoundoff

/-- Source chronological left fold of exact additions. -/
def exactSum : ℝ → List ℝ → ℝ
  | acc, [] => acc
  | acc, x :: xs => exactSum (acc + x) xs

/-- Explicit rounded source chronological left fold. -/
def roundedSum (round : ℝ → ℝ) : ℝ → List ℝ → ℝ
  | acc, [] => acc
  | acc, x :: xs => roundedSum round (round (acc + x)) xs

/-- A local absolute-error contract on every ACTUAL rounded input.
    This is an event-scoped obligation rather than a fictional global
    uniform absolute rounding bound for arbitrary IEEE magnitudes. -/
def ValidAddTrace (round : ℝ → ℝ) (ε : ℝ) : ℝ → List ℝ → Prop
  | _, [] => True
  | acc, x :: xs =>
    Close (round (acc + x)) (acc + x) ε ∧
    ValidAddTrace round ε (round (acc + x)) xs

theorem rounded_source_sum_error (round : ℝ → ℝ) (ε : ℝ)
    (xs : List ℝ) (approx ideal radius : ℝ)
    (h0 : Close approx ideal radius)
    (hv : ValidAddTrace round ε approx xs) :
    Close (roundedSum round approx xs) (exactSum ideal xs)
      (radius + (xs.length : ℝ) * ε) := by
  induction xs generalizing approx ideal radius with
  | nil =>
    simpa [roundedSum, exactSum] using h0
  | cons x xs ih =>
    have hstep : Close (round (approx + x)) (ideal + x) (radius + ε) := by
      have h := add_error approx x ideal x radius 0
        (round (approx+x)) ε h0 (close_refl x) hv.1
      simpa using h
    have htail := ih (round (approx+x)) (ideal+x)
      (radius+ε) hstep hv.2
    simpa [roundedSum, exactSum, List.length_cons, Nat.cast_add,
      add_mul, add_assoc, add_left_comm, add_comm] using htail

/-- Independent quantization and recurrent-sum errors compose additively.
    Exact upstream feature agreement is NOT silently assumed. -/
theorem sum_with_basis_input_error (approx ideal basisApprox basisIdeal
    εState εBasis εRound rounded : ℝ)
    (hstate : Close approx ideal εState)
    (hbasis : Close basisApprox basisIdeal εBasis)
    (hround : Close rounded (approx + basisApprox) εRound) :
    Close rounded (ideal + basisIdeal)
      (εState + εBasis + εRound) :=
  add_error approx basisApprox ideal basisIdeal εState εBasis
    rounded εRound hstate hbasis hround

/-- Dividing the rounded prefix sum by a known positive integer count
    also divides its absolute error bound; no associativity assumptions. -/
theorem normalized_sum_error (approx ideal radius : ℝ)
    (count : ℕ) (hc : 0 < count)
    (h : Close approx ideal radius) :
    Close (approx / (count : ℝ)) (ideal / (count : ℝ))
      (radius / (count : ℝ)) := by
  have hcpos : 0 < (count : ℝ) := Nat.cast_pos.mpr hc
  unfold Close at *
  rw [← sub_div, abs_div, abs_of_pos hcpos]
  exact div_le_div_of_nonneg_right h hcpos.le

/-- The fixed 88 source-sum coordinates can grow linearly as length grows.
    An exact bound does not imply safe FP32 accumulation or retention. -/
theorem ideal_sum_l1_bound (xs : List ℝ) (budget : ℝ)
    (hbudget : ∀ x ∈ xs, |x| ≤ budget)
    (hb0 : 0 ≤ budget) :
    |exactSum 0 xs| ≤ (xs.length : ℝ) * budget := by
  have aux (ys : List ℝ) (acc : ℝ)
      (h : ∀ x ∈ ys, |x| ≤ budget) :
      |exactSum acc ys| ≤ |acc| + (ys.length : ℝ) * budget := by
    induction ys generalizing acc with
    | nil => simp [exactSum]
    | cons x tail ih =>
      have hx := h x (by simp)
      have ht : ∀ y ∈ tail, |y| ≤ budget :=
        fun y hy => h y (by simp [hy])
      have hr := ih (acc+x) ht
      have ha := abs_add acc x
      calc
        |exactSum acc (x::tail)| =
            |exactSum (acc+x) tail| := rfl
        _ ≤ |acc+x| + (tail.length:ℝ)*budget := hr
        _ ≤ |acc| + (x::tail).length * budget := by
          simp only [List.length_cons, Nat.cast_add, Nat.cast_one]
          nlinarith
  simpa using aux xs 0 hbudget

end
end PNDS.LanguageFPContracts
