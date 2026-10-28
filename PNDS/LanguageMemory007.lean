import Mathlib
import PNDS.LanguageWaveGraphRepair

/-!
TAC-LANG-MEM-500K-007: *scoped exact-real* dual-timescale gate and
source-owner preconditions, pinned to TAC-OSM GRAPH007.
Not a full PyTorch/IEEE/autograd equivalence or recall proof.
-/
namespace PNDS.LanguageMemory007
noncomputable section

def slowRetention (raw : ℝ) : ℝ := (995/1000 : ℝ)+(4/1000 : ℝ)*raw
def slowErasure (raw : ℝ) : ℝ := (2/1000 : ℝ)*raw
def coupledWrite (alpha raw : ℝ) : ℝ := (1-alpha)*raw

theorem slow_retention_range (raw : ℝ)
    (h0 : 0 ≤ raw) (h1 : raw ≤ 1) :
    (995/1000 : ℝ) ≤ slowRetention raw ∧
    slowRetention raw ≤ (999/1000 : ℝ) := by
  unfold slowRetention
  constructor <;> linarith

theorem slow_erasure_range (raw : ℝ)
    (h0 : 0 ≤ raw) (h1 : raw ≤ 1) :
    0 ≤ slowErasure raw ∧ slowErasure raw ≤ (2/1000 : ℝ) := by
  unfold slowErasure
  constructor <;> linarith

theorem slow_write_nonnegative
    (alpha raw : ℝ) (ha0 : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hr0 : 0 ≤ raw) : 0 ≤ coupledWrite alpha raw := by
  unfold coupledWrite
  exact mul_nonneg (sub_nonneg.mpr ha1) hr0

theorem slow_write_bounded
    (alpha raw : ℝ) (ha0 : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hr0 : 0 ≤ raw) (hr1 : raw ≤ 1) :
    coupledWrite alpha raw ≤ 1-alpha := by
  unfold coupledWrite
  exact mul_le_of_le_one_right (sub_nonneg.mpr ha1) hr1

def authorizedCarry (current stored : String) : Prop :=
  current ≠ "" ∧ current = stored

theorem wrong_owner_not_authorized (current stored : String)
    (hwrong : current ≠ stored) :
    ¬ authorizedCarry current stored := by
  intro h
  exact hwrong h.2

end
end PNDS.LanguageMemory007
