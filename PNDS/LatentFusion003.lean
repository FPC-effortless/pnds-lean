import Mathlib

/-!
LATFUSE003 source-scoped real-valued operator theorems only.
TAC-OSM draft candidate source:
  src/tac_osm/latent_fusion_003.py
  branch research/latfuse003-multihead-selective-2026-10-27

These proofs are NOT compiled full original PyTorch forward/backward/IEEE
semantics or probability/forecast generalization certificates. Full-model
AD, hardware semantics, read/write normalization and statistical laws OPEN.
-/

namespace PNDS.LatentFusion003

def headCount : Nat := 4
def headKeyDimension : Nat := 4
def headValueDimension : Nat := 16
def headMatrixElements : Nat := headCount * headKeyDimension * headValueDimension
def originalMatrixElements : Nat := 16 * 16

theorem partition_uses_all_256_matrix_values :
    headMatrixElements = originalMatrixElements := by decide

def floatStateCount : Nat := 80 + 80 + headMatrixElements + 16 * 80
def sourceFlagCount : Nat := 16
def nominalFp32Bytes : Nat := 4 * floatStateCount + sourceFlagCount

theorem exact_original_state_floats : floatStateCount = 1696 := by decide
theorem exact_original_state_bytes : nominalFp32Bytes = 6800 := by decide

/-- Exact-real scalar component of one selected head's normalized-key
    delta update; full matrix implements an outer-product lifting. -/
def headDelta (alpha beta old key value : ℝ) : ℝ :=
  alpha * old + beta * (value - alpha * old * key) * key

theorem zero_write_retains_only_alpha (alpha old key value : ℝ) :
    headDelta alpha 0 old key value = alpha * old := by
  simp [headDelta]

theorem unselected_head_preserves_old (old key value : ℝ) :
    headDelta 1 0 old key value = old := by
  simp [headDelta]

theorem headDelta_affine_in_old
    (alpha beta old key value : ℝ) :
    headDelta alpha beta old key value =
      (alpha * (1 - beta * key * key)) * old + beta * value * key := by
  unfold headDelta
  ring

theorem difference_transport
    (alpha beta left right key value : ℝ) :
    headDelta alpha beta left key value -
      headDelta alpha beta right key value =
        alpha * (1 - beta * key * key) * (left - right) := by
  unfold headDelta
  ring

/-- Indexed selection preserves every non-selected head exactly; the
    caller's Boolean/source authorization must be independently checked. -/
def selectiveReplace
    (old : Fin 4 → ℝ) (selected : Fin 4 → Bool)
    (candidate : Fin 4 → ℝ) : Fin 4 → ℝ :=
  fun i => if selected i then candidate i else old i

theorem unselected_index_isolation
    (old : Fin 4 → ℝ) (selected : Fin 4 → Bool)
    (candidate : Fin 4 → ℝ) (i : Fin 4)
    (h : selected i = false) :
    selectiveReplace old selected candidate i = old i := by
  simp [selectiveReplace, h]

end PNDS.LatentFusion003
