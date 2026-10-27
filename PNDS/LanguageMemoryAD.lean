import PNDS.LanguageFullGraph
import PNDS.SourceAlignedDifferentiation

/-!
TAC-LANG-AD-010: source-specific corrective matrix-write derivative.

The actual two 16x8 language aligned heads and the 8x8 native memory
re-use the kernel-proven source corrective row law. Keys, 16 source
values, both gates and every old-memory coordinate may vary together.

This is exact-real HasDerivAt; full-model parameter-to-loss backprop and
Torch/IEEE refinement remain separate.
-/
namespace PNDS.LanguageMemoryAD
noncomputable section

open PNDS.LanguageFullGraph
open PNDS.SourceAlignedMemory
open PNDS.SourceAlignedDifferentiation

/-- 2 heads by 16 rows by 8 address coordinates; no sidecar. -/
def sourceKernel (old : Fin 2 → M 16 8)
    (keys : Fin 2 → V 8) (value : V 16) (alpha beta : V 2) :
    Fin 2 → M 16 8 :=
  fun h i j => writeRow (old h i) (keys h) (value i)
    (alpha h) (beta h) true j

/-- Every coordinate of the language source's aligned delta update is
definitionally the established corrective write, not a new surrogate. -/
theorem aligned_write_refines_kernel (p : Parameters)
    (old : Fin 2 → M 16 8) (ordinal : ℕ)
    (value : V 16) (alpha beta : V 2) :
    writeAligned p old ordinal value alpha beta =
      sourceKernel old (alignedKey p ordinal) value alpha beta := by
  funext h i j
  rfl

/-- The native 8x8 language corrective matrix obeys the very same
row kernel as the aligned 16x8 banks. -/
def nativeKernel (old : M 8 8) (key value : V 8)
    (alpha beta : ℝ) : M 8 8 :=
  fun i j => writeRow (old i) key (value i) alpha beta true j

theorem native_write_refines_kernel (old : M 8 8) (key value : V 8)
    (alpha beta : ℝ) :
    writeNative old key value alpha beta =
      nativeKernel old key value alpha beta := by
  funext i j
  rfl

/-- Full two-head tensor JVP: gradients through historical source rows,
keys, retained values and BOTH continuous gates simultaneously. -/
theorem source_kernel_coordinate_jvp
    (old : ℝ → Fin 2 → M 16 8) (keys : ℝ → Fin 2 → V 8)
    (value : ℝ → V 16) (alpha beta : ℝ → V 2)
    (dold : Fin 2 → M 16 8) (dkey : Fin 2 → V 8)
    (dvalue : V 16) (dalpha dbeta : V 2)
    (t : ℝ) (h : Fin 2) (i : Fin 16) (j : Fin 8)
    (hm : ∀ k, HasDerivAt (fun z => old z h i k) (dold h i k) t)
    (hk : ∀ k, HasDerivAt (fun z => keys z h k) (dkey h k) t)
    (hv : HasDerivAt (fun z => value z i) (dvalue i) t)
    (ha : HasDerivAt (fun z => alpha z h) (dalpha h) t)
    (hb : HasDerivAt (fun z => beta z h) (dbeta h) t) :
    HasDerivAt
      (fun z => sourceKernel (old z) (keys z) (value z)
        (alpha z) (beta z) h i j)
      (rowJVP (old t h i) (keys t h) (dold h i) (dkey h)
        (value t i) (alpha t h) (beta t h)
        (dvalue i) (dalpha h) (dbeta h) j) t := by
  exact corrective_write_jvp (fun z => old z h i) (fun z => keys z h)
    (dold h i) (dkey h) (fun z => value z i)
    (fun z => alpha z h) (fun z => beta z h)
    (dvalue i) (dalpha h) (dbeta h) t hm hk hv ha hb j

/-- The same complete JVP applies to the native 8x8 language memory. -/
theorem native_kernel_coordinate_jvp
    (old : ℝ → M 8 8) (key value : ℝ → V 8)
    (alpha beta : ℝ → ℝ) (dold : M 8 8)
    (dkey dvalue : V 8) (dalpha dbeta t : ℝ)
    (i j : Fin 8)
    (hm : ∀ k, HasDerivAt (fun z => old z i k) (dold i k) t)
    (hk : ∀ k, HasDerivAt (fun z => key z k) (dkey k) t)
    (hv : HasDerivAt (fun z => value z i) (dvalue i) t)
    (ha : HasDerivAt alpha dalpha t)
    (hb : HasDerivAt beta dbeta t) :
    HasDerivAt
      (fun z => nativeKernel (old z) (key z) (value z)
        (alpha z) (beta z) i j)
      (rowJVP (old t i) (key t) (dold i) dkey
        (value t i) (alpha t) (beta t)
        (dvalue i) dalpha dbeta j) t := by
  exact corrective_write_jvp (fun z => old z i) key
    (dold i) dkey (fun z => value z i) alpha beta
    (dvalue i) dalpha dbeta t hm hk hv ha hb j

end
end PNDS.LanguageMemoryAD
