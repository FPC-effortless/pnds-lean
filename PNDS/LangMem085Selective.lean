import Mathlib

/-!
LANGMEM085 — PARTIAL exact-real 8x8 addressed-value residual laws.

Native full delta operator for normalized writer k:
  M' = rho M + eta (v-rho M k) kᵀ, 0<=eta<=1.
The key-direction read is a convex combination of retained original read
and current byte code, in exact real arithmetic, when ||k||₂=1. The
source code gates are fixed, not trained; this shows no generic improvement,
no equality in PyTorch FP32, and no historical retrieval guarantee.

The scoped Lean formalization only proves the scalar aligned read law,
error contraction for fixed eta, and discrete selective eta range.
Independently tested Python matrix operator is an engineering reference.
Whole model/real source, collisions, IEEE AD, future label isolation across
the complete native producer and deployed budget are OPEN.
-/
namespace PNDS.LangMem085

def deltaAligned (rho eta prior value : ℝ) : ℝ :=
  rho * prior + eta * (value-rho*prior)

theorem delta_aligned_is_convex (rho eta prior value : ℝ) :
    deltaAligned rho eta prior value =
    (1-eta)*(rho*prior)+eta*value := by
  unfold deltaAligned
  ring

theorem delta_aligned_error (rho eta prior value : ℝ) :
    deltaAligned rho eta prior value - value =
    (1-eta)*(rho*prior-value) := by
  unfold deltaAligned
  ring

noncomputable def selectiveEta (residualNorm : ℝ) : ℝ :=
  if residualNorm <= (3:ℝ)/4 then 1 else 1/4

theorem selective_eta_bounded (z : ℝ) :
    0 <= selectiveEta z ∧ selectiveEta z <= 1 := by
  unfold selectiveEta
  split_ifs <;> constructor <;> norm_num

theorem full_delta_writes_latest (rho old value : ℝ) :
    deltaAligned rho 1 old value = value := by
  unfold deltaAligned
  ring

theorem quarter_delta_residual (rho old value : ℝ) :
    deltaAligned rho (1/4) old value - value =
    (3/4)*(rho*old-value) := by
  unfold deltaAligned
  ring

def frozenMemoryBankBytes : Nat := 8*8*4
def originalFrozenProducerBytes : Nat := 4160
def combinedTensorBytes : Nat := frozenMemoryBankBytes+originalFrozenProducerBytes
theorem registered_bank_bytes : frozenMemoryBankBytes=256 := by decide
theorem registered_candidate_tensors : combinedTensorBytes=4416 := by decide

end PNDS.LangMem085
