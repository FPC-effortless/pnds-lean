import PNDS.LanguageFPContracts
import PNDS.SourceAlignedMemory

/-!
TAC-LANG-FP-014: coordinatewise, finite-reduction and source-specific
memory-read error composition. The source's native and aligned memory reads
are finite 8-term dot products: each backend multiplication and reduction
may have its own explicitly observed or certified rounding error.

All statements quantify over exact real decoded values of finite tensors.
No unproved global PyTorch/GPU/IEEE accuracy axiom is introduced.
-/
namespace PNDS.LanguageTensorFP
noncomputable section
open PNDS.SourceAlignedRoundoff
open PNDS.SourceAlignedMemory

/-- Coordinatewise absolute errors imply a bound for the complete
reduction in any finite dimension, regardless of the sum ordering in
the exact-real mathematical specification. -/
theorem vector_sum_error {n : ℕ} (approx ideal radius : Fin n → ℝ)
    (h : ∀ i, Close (approx i) (ideal i) (radius i)) :
    Close (∑ i : Fin n, approx i) (∑ i : Fin n, ideal i)
      (∑ i : Fin n, radius i) := by
  unfold Close at *
  rw [← Finset.sum_sub_distrib]
  calc
    |∑ i : Fin n, (approx i - ideal i)|
        ≤ ∑ i : Fin n, |approx i - ideal i| :=
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i : Fin n, radius i :=
          Finset.sum_le_sum (fun i _ => h i)

/-- An independently rounded reduction contributes an additional error.
This is the explicit boundary between a mathematical reduction and a
device backend's potentially re-associated tensor implementation. -/
theorem rounded_sum_error {n : ℕ}
    (approx ideal radius : Fin n → ℝ) (rounded reduceError : ℝ)
    (h : ∀ i, Close (approx i) (ideal i) (radius i))
    (hr : Close rounded (∑ i : Fin n, approx i) reduceError) :
    Close rounded (∑ i : Fin n, ideal i)
      ((∑ i : Fin n, radius i) + reduceError) := by
  exact chain_error _ _ _ _ _ (vector_sum_error approx ideal radius h) hr

/-- Forward 8-product native/aligned memory read. Importantly, each
quantized source matrix/key pair can have DIFFERENT errors and each
multiplication has its own local rounding receipt. -/
theorem source_row_read_error
    (approxRow exactRow approxKey exactKey : SourceRow)
    (errRow errKey roundedProducts productRound : SourceRow)
    (roundedRead reductionRound : ℝ)
    (hr : ∀ j : Fin 8, Close (approxRow j) (exactRow j) (errRow j))
    (hk : ∀ j : Fin 8, Close (approxKey j) (exactKey j) (errKey j))
    (hp : ∀ j : Fin 8,
      Close (roundedProducts j) (approxRow j * approxKey j)
        (productRound j))
    (hs : Close roundedRead (∑ j : Fin 8, roundedProducts j)
      reductionRound) :
    Close roundedRead (rowRead exactRow exactKey)
      ((∑ j : Fin 8,
         |approxRow j| * errKey j +
         |approxKey j| * errRow j +
         errRow j * errKey j +
         productRound j) + reductionRound) := by
  have hterm (j : Fin 8) :
      Close (roundedProducts j) (exactRow j * exactKey j)
        (|approxRow j| * errKey j +
         |approxKey j| * errRow j +
         errRow j * errKey j +
         productRound j) :=
    multiply_error (approxRow j) (approxKey j) (exactRow j)
      (exactKey j) (errRow j) (errKey j)
      (roundedProducts j) (productRound j)
      (hr j) (hk j) (hp j)
  simpa only [rowRead] using
    (rounded_sum_error roundedProducts
      (fun j => exactRow j * exactKey j)
      (fun j => |approxRow j| * errKey j +
        |approxKey j| * errRow j +
        errRow j * errKey j + productRound j)
      roundedRead reductionRound hterm hs)

end
end PNDS.LanguageTensorFP
