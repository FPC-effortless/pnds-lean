import Mathlib
import PNDS.System1MultiTimescale

/-!
# S1TEMP008: exact-real source-indexed per-column memory update

One application/global column + seven source-offer columns; learned gate
gamma ∈ [0,1] updates only the externally selected already-observed slot.
No global hash collisions or future target may authorize the slot.
Same nominal 8×8 matrix = 64 scalar entries, 256 FP32 bytes.

Proof scope: exact-real column isolation and convex update per component.
NOT a proof of actual PyTorch right-affine prefix, IEEE754/autograd source
refinement, timestamp/source identity provenance, observational final loan
outcome prediction, credit decision policy, equal full training FLOPs or
physical safety. No claim that merely changing slots learns relevance.
-/

namespace PNDS.System1IndexedAdmission

abbrev Memory := Fin 8 → Fin 8 → ℝ
abbrev Content := Fin 8 → ℝ
def nextMemory (previous : Memory) (target : Fin 8)
    (gamma : ℝ) (value : Content) : Memory :=
  fun row col => if col=target
    then (1-gamma)*previous row col+gamma*value row
    else previous row col

theorem off_target_exactly_unchanged
    (previous : Memory) (target : Fin 8)
    (gamma : ℝ) (value : Content)
    (row col : Fin 8) (unrelated : col ≠ target) :
    nextMemory previous target gamma value row col =
      previous row col := by
  simp [nextMemory, unrelated]

theorem target_convex_combination
    (previous : Memory) (target : Fin 8)
    (gamma : ℝ) (value : Content)
    (row : Fin 8) :
    nextMemory previous target gamma value row target =
      (1-gamma)*previous row target+gamma*value row := by
  simp [nextMemory]

theorem zero_admission_never_mutates
    (previous : Memory) (target : Fin 8)
    (value : Content) (row col : Fin 8) :
    nextMemory previous target 0 value row col =
      previous row col := by
  by_cases h : col=target
  · subst col
    simp [nextMemory]
  · simp [nextMemory,h]

theorem full_admission_replaces_only_target
    (previous : Memory) (target : Fin 8)
    (value : Content) (row : Fin 8) :
    nextMemory previous target 1 value row target = value row := by
  simp [nextMemory]

theorem gamma_lower_bound (alpha beta : ℝ)
    (a0 : 0 ≤ alpha) (a1 : alpha ≤ 1)
    (b0 : 0 ≤ beta) :
    0 ≤ beta*(1-alpha) := by
  apply mul_nonneg b0
  linarith

theorem gamma_upper_bound (alpha beta : ℝ)
    (a0 : 0 ≤ alpha) (a1 : alpha ≤ 1)
    (b0 : 0 ≤ beta) (b1 : beta ≤ 1) :
    beta*(1-alpha) ≤ 1 := by
  have hq : 0 ≤ 1-alpha := by linarith
  calc
    beta*(1-alpha) ≤ 1*(1-alpha) := by
      exact mul_le_mul_of_nonneg_right b1 hq
    _ = 1-alpha := by ring
    _ ≤ 1 := by linarith

theorem unchanged_matrix_bytes : 8*8*4 = 256 := by decide

theorem unchanged_total_system1_state_bytes :
    (72+16+8*8+14)*4 = 664 := by decide

end PNDS.System1IndexedAdmission
