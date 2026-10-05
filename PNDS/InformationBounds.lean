import Mathlib

namespace PNDS.InformationBounds

/-!
Finite-domain terminal-budget ceiling.

A deterministic evidence channel partitions a candidate population into buckets.
If at most B candidates may be retained from each bucket, then the number of
targets that can possibly be retained is at most q * B for q buckets. Dividing
by the population size M gives the universal qB/M ceiling.
-/

/-- Each evidence bucket contributes at most B retained candidates. -/
theorem capped_bucket_sum_le (q B : ℕ) (n : Fin q → ℕ) :
    (∑ b : Fin q, min B (n b)) ≤ q * B := by
  calc
    (∑ b : Fin q, min B (n b)) ≤ ∑ b : Fin q, B := by
      exact Finset.sum_le_sum (fun b _ => Nat.min_le_left _ _)
    _ = q * B := by simp [Nat.mul_comm]

/-- A fixed B-per-bucket terminal selector cannot retain more than M targets. -/
theorem capped_bucket_sum_le_population
    (q B M : ℕ) (n : Fin q → ℕ) (hsum : (∑ b : Fin q, n b) = M) :
    (∑ b : Fin q, min B (n b)) ≤ M := by
  calc
    (∑ b : Fin q, min B (n b)) ≤ ∑ b : Fin q, n b := by
      exact Finset.sum_le_sum (fun b _ => Nat.min_le_right _ _)
    _ = M := hsum

/-- If the candidate population is M and evidence has q possible buckets,
    B candidates per bucket give the universal success ceiling qB/M, capped at 1. -/
theorem terminal_success_le_min_one
    (q B M : ℕ) (n : Fin q → ℕ)
    (hM : 0 < M) (hsum : (∑ b : Fin q, n b) = M) :
    (((∑ b : Fin q, min B (n b) : ℝ)) / M)
      ≤ min 1 (((q * B : ℕ) : ℝ) / M) := by
  have hMreal : (0 : ℝ) < M := by exact_mod_cast hM
  have hqB : ((∑ b : Fin q, min B (n b) : ℝ) : ℝ) ≤ (q * B : ℕ) := by
    exact_mod_cast capped_bucket_sum_le q B n
  have hMnum :
      ((∑ b : Fin q, min B (n b) : ℝ) : ℝ) ≤ M := by
    exact_mod_cast capped_bucket_sum_le_population q B M n hsum
  have hqBdiv :
      (((∑ b : Fin q, min B (n b) : ℝ)) / M)
        ≤ (((q * B : ℕ) : ℝ) / M) := by
    exact (div_le_div_iff_of_pos_right hMreal).2 hqB
  have hOne :
      (((∑ b : Fin q, min B (n b) : ℝ)) / M) ≤ 1 := by
    exact (div_le_iff₀ hMreal).2 (by simpa using hMnum)
  exact le_min hOne hqBdiv

/-- Budget-capped posterior utility for a finite evidence partition.
    Under a uniform target prior, it is the expected fraction of targets that can
    be retained by a terminal selector with budget B after observing the evidence. -/
noncomputable def budgetCappedUtility
    (q B M : ℕ) (n : Fin q → ℕ) : ℝ :=
    ((∑ b : Fin q, min B (n b) : ℝ) / M)

/-- The budget-capped utility obeys the universal qB/M ceiling.
    The assumptions make the probabilistic interpretation explicit: M is the
    positive candidate population size and n is a complete deterministic partition
    whose bucket sizes sum to M. -/
theorem budgetCappedUtility_le_min_one
    (q B M : ℕ) (n : Fin q → ℕ)
    (hM : 0 < M) (hsum : (∑ b : Fin q, n b) = M) :
    budgetCappedUtility q B M n
      ≤ min 1 (((q * B : ℕ) : ℝ) / M) := by
  simpa [budgetCappedUtility] using
    (terminal_success_le_min_one q B M n hM hsum)
end PNDS.InformationBounds
