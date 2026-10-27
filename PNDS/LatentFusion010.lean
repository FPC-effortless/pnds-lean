import Mathlib

/-!
LATFUSE010: exact-real scalar analogue of original DELTA write and bypass.
The source implementation's original per-head value×key memory MATRIX,
normalized vector key, learned gates, FP32 errors, automatic differentiation,
source access control and probability laws are NOT proven by these lemmas.
Original source true-event empirical evidence is retained in TAC-OSM
research/latfuse010-real-event-key-value-attribution-2026-10-27.
-/
namespace PNDS.LatentFusion010

def scalarWrite (alpha beta key value previous : ℝ) : ℝ :=
  alpha * previous + beta * (value - alpha * previous * key) * key

def scalarBypass (previous : ℝ) : ℝ := previous

theorem single_step_write_bypass_difference
    (alpha beta key value previous : ℝ) :
    scalarWrite alpha beta key value previous - scalarBypass previous =
       (alpha - 1) * previous
         + beta * (value - alpha * previous * key) * key := by
  simp only [scalarWrite, scalarBypass]
  ring

theorem common_source_suffix_difference
    (alpha beta key value left right : ℝ) :
    scalarWrite alpha beta key value left
      - scalarWrite alpha beta key value right =
      alpha * (1 - beta * key * key) * (left - right) := by
  simp only [scalarWrite]
  ring

theorem source_write_common_value_cancels
    (alpha beta key previous valueA valueB : ℝ) :
    scalarWrite alpha beta key valueA previous -
      scalarWrite alpha beta key valueB previous =
      beta * (valueA - valueB) * key := by
  simp only [scalarWrite]
  ring

def sourceObservationStateBytes : Nat :=
  4 * (80 + 80 + 2 * 16 * 8 + 16 * 80) + 16

theorem original_state_budget_unchanged :
  sourceObservationStateBytes = 6800 := by decide

end PNDS.LatentFusion010
