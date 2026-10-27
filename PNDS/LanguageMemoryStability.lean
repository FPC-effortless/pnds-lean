import PNDS.LanguageMemoryAD
import PNDS.SourceAlignedVerification

/-!
TAC-LANG-STABILITY-011. Exact-real nonexpansiveness of the TWO aligned
16x8 banks for any fixed source input schedule. Includes the trained
SiLU and linear modulator paths. This proves sensitivity to PRIOR MEMORY,
not absolute memory boundedness, retrieval accuracy or FP32 behavior.
-/
namespace PNDS.LanguageMemoryStability
noncomputable section

open PNDS.LanguageFullGraph
open PNDS.SourceAlignedModel
open PNDS.SourceAlignedVerification
open PNDS.SourceAlignedMemory
open PNDS.LanguageMemoryAD

theorem aligned_key_energy (p : Parameters) (ordinal : ℕ) (h : Fin 2) :
    squaredNorm (alignedKey p ordinal h) ≤ 1 := by
  exact normalized_key_energy (linear (p.rotation h) (sourcePhi ordinal))

theorem source_retention_range (p : Parameters) (e : V 48) (h : Fin 2) :
    0 ≤ sig (dense p.sourceRetention e h) ∧
      sig (dense p.sourceRetention e h) ≤ 1 := by
  have hs := sigmoid_range (dense p.sourceRetention e h)
  exact ⟨hs.1.le,hs.2.le⟩

theorem source_beta_range (p : Parameters) (e : V 48) (h : Fin 2) :
    0 ≤ sourceBeta p e h ∧ sourceBeta p e h ≤ 1 := by
  cases hm : p.gateMode with
  | linear =>
    simpa [sourceBeta, hm] using
      (routed_strength_range
        (dense p.sourceStrength e h + linear p.sourceMod e h)
        (dense p.sourceRouter e) h)
  | silu =>
    simpa [sourceBeta, hm] using
      (routed_strength_range
        (dense p.sourceStrength e h + silu (2 * linear p.sourceMod e h))
        (dense p.sourceRouter e) h)

/-- Shared gate/embedding/key/value inputs cannot AMPLIFY any difference
between two previous aligned memory ROWS in the exact-real squared norm. -/
theorem source_aligned_write_nonexpansive (p : Parameters)
    (old other : Fin 2 → M 16 8) (ordinal : ℕ)
    (e : V 48) (value : V 16) (h : Fin 2) (i : Fin 16) :
    squaredNorm
      ((writeAligned p old ordinal value (map sig (dense p.sourceRetention e))
        (sourceBeta p e)) h i -
       (writeAligned p other ordinal value (map sig (dense p.sourceRetention e))
        (sourceBeta p e)) h i) ≤
      squaredNorm (old h i - other h i) := by
  have hk := aligned_key_energy p ordinal h
  have ha := source_retention_range p e h
  have hb := source_beta_range p e h
  have bound := write_nonexpansive (old h i) (other h i)
    (alignedKey p ordinal h) (value i)
    (sig (dense p.sourceRetention e h))
    (sourceBeta p e h) hk ha.1 ha.2 hb.1 hb.2
  simpa only [aligned_write_refines_kernel, sourceKernel, map] using bound

end
end PNDS.LanguageMemoryStability
