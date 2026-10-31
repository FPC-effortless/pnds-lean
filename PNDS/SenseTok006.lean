import Mathlib

/-!
SENSE-TOK-006 preparation operators only. The selected code q is an input to
these declarations; no theorem here establishes PyTorch argmax selection,
floating-point equivalence, gradients, whole-model behavior or learned skill.
See the prospective study record for the immutable original TAC source.
-/
namespace PNDS.SenseTok006

def selectedReadout {k d : Nat} (values : Fin k → Fin d → ℝ)
    (q : Fin k) : Fin d → ℝ := values q

def fixedKeyWrongValue {k d : Nat} (values : Fin k → Fin d → ℝ)
    (permutation : Equiv.Perm (Fin k)) (q : Fin k) : Fin d → ℝ :=
  values (permutation q)

/-- Assumes the selected ID was relabelled inversely; it does not prove that
    a tie-breaking argmax will satisfy that assumption. -/
theorem joint_relabelling_readout {k d : Nat} (values : Fin k → Fin d → ℝ)
    (permutation : Equiv.Perm (Fin k)) (q : Fin k) :
    selectedReadout (fun i => values (permutation i)) (permutation.symm q) =
      selectedReadout values q := by
  simp [selectedReadout]

def realizedRegret (byteLoss tokenLoss : ℝ) (useByte : Bool) : ℝ :=
  (if useByte then byteLoss else tokenLoss) - min byteLoss tokenLoss

theorem realizedRegret_nonnegative (byteLoss tokenLoss : ℝ) (useByte : Bool) :
    0 ≤ realizedRegret byteLoss tokenLoss useByte := by
  cases useByte
  · change 0 ≤ tokenLoss - min byteLoss tokenLoss
    exact sub_nonneg.mpr (min_le_right byteLoss tokenLoss)
  · change 0 ≤ byteLoss - min byteLoss tokenLoss
    exact sub_nonneg.mpr (min_le_left byteLoss tokenLoss)

def denseByteParameterCount : Nat :=
  257 * 32 + 3 * 118 * 32 + 3 * 118 * 118 + 6 * 118 +
    256 * (118 + 32) + 256

theorem denseByteParameterCount_value : denseByteParameterCount = 100688 := by
  decide

end PNDS.SenseTok006

#print axioms PNDS.SenseTok006.joint_relabelling_readout
#print axioms PNDS.SenseTok006.realizedRegret_nonnegative
#print axioms PNDS.SenseTok006.denseByteParameterCount_value
