import Mathlib

/-!
LATFUSE004 local exact-real and finite head-capacity contracts only.
Models use the original LATFUSE003 FP32 state. Two axes:
  head_count 2/4, differentiable soft versus hard-selected routing.
No whole neural, FP32/IEEE, AD, probability law or source-correctness proof.
-/
namespace PNDS.LatentFusion004

def twoHeadCells : Nat := 2 * 16 * 8
def fourHeadCells : Nat := 4 * 16 * 4
def originalCells : Nat := 16 * 16

theorem two_head_preserves_matrix : twoHeadCells = originalCells := by decide
theorem four_head_preserves_matrix : fourHeadCells = originalCells := by decide
theorem equal_matrix_capacities : twoHeadCells = fourHeadCells := by decide

def carriedFloats : Nat := 80 + 80 + originalCells + 16 * 80
def carriedBytes : Nat := 4 * carriedFloats + 16
theorem original_carried_values : carriedFloats = 1696 := by decide
theorem original_carried_fp32_nominal_bytes : carriedBytes = 6800 := by decide

def hardSelectiveEdit (selected : Bool) (old candidate : ℝ) : ℝ :=
  if selected then candidate else old
theorem untouched_when_unselected (old candidate : ℝ) :
    hardSelectiveEdit false old candidate = old := by
  simp [hardSelectiveEdit]
theorem updates_when_selected (old candidate : ℝ) :
    hardSelectiveEdit true old candidate = candidate := by
  simp [hardSelectiveEdit]

def softBlend (p old candidate : ℝ) : ℝ :=
  (1 - p) * old + p * candidate

theorem softBlend_zero (old candidate : ℝ) : softBlend 0 old candidate = old := by
  simp [softBlend]
theorem softBlend_one (old candidate : ℝ) : softBlend 1 old candidate = candidate := by
  simp [softBlend]
theorem softBlend_difference (p old first second : ℝ) :
    softBlend p old first - softBlend p old second = p * (first - second) := by
  unfold softBlend
  ring

/-- Valid selected-head probabilities form a nonnegative unit simplex under
    the premise. This does not prove the real Torch temperature-softmax. -/
theorem two_router_mass_preserved (p q : ℝ) (h : p + q = 1) :
    p + q = 1 := h

end PNDS.LatentFusion004
