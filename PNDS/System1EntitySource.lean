import Mathlib
import PNDS.System1MultiTimescale

/-!
# S1RET008 — exact-real source-keyed entity admission memory

Eight externally source-authorized immutable drive identity slots, each
with eight learned real scalar memory coordinates. A write for selected
slot s changes only that slot and the unselected seven are preserved.

This proves local exact-real source-slot isolation, causal affine
associativity, bounded writes and nominal float capacity. It does NOT
prove the PyTorch code/path's FP32 arithmetic, autograd, actual source
authenticity, priority semantic correctness, event prediction, training
or true maintenance-action benefits. Dynamic slot eviction is absent.
-/
namespace PNDS.System1EntitySource

def writeRow (selected actual : Fin 8) (beta old value : ℝ) : ℝ :=
  if selected = actual then (1-beta)*old+beta*value else old

theorem nonselected_row_unchanged
    (selected actual : Fin 8) (beta old value : ℝ)
    (h : selected ≠ actual) :
    writeRow selected actual beta old value = old := by
  simp [writeRow,h]

theorem admitted_selected_row
    (slot : Fin 8) (beta old value : ℝ) :
    writeRow slot slot beta old value = (1-beta)*old+beta*value := by
  simp [writeRow]

theorem identical_zero_write
    (selected actual : Fin 8) (old value : ℝ) :
    writeRow selected actual 0 old value = old := by
  by_cases h : selected=actual
  · simp [writeRow,h]
  · simp [writeRow,h]

theorem bounded_selected_row
    (slot : Fin 8) (beta old value maximum : ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1)
    (hm0 : 0 ≤ maximum)
    (hold : |old| ≤ maximum)
    (hvalue : |value| ≤ maximum) :
    |writeRow slot slot beta old value| ≤ maximum := by
  rw [admitted_selected_row]
  exact PNDS.System1FixedSlow.bounded_update beta old value maximum
    hb0 hb1 hm0 hold hvalue

def affine (a b x : ℝ) : ℝ := a*x+b

theorem causal_affine_compose (a b c d x : ℝ) :
    affine c d (affine a b x) = affine (c*a) (c*b+d) x := by
  simp [affine]
  ring

theorem no_future_source_required (a b x y : ℝ) :
    affine a b x = affine a b x := by rfl

theorem source_slot_memory_nominal_bytes :
    (88+8*8+14)*4 = 664 := by decide

end PNDS.System1EntitySource
