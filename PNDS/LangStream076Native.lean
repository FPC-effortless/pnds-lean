import Mathlib

/-!
LANGSTREAM076: PARTIAL generic native k×k affine carry and mask semantics.
The actual TAC source performs N_{t+1} = N_t P_t + Q_t if write admitted,
otherwise N_{t+1}=N_t. This theorem is only about abstract real matrices
and a Boolean write flag, not the particular learned write/erase gates,
aligned 16×k bank, finite IEEE PyTorch kernels, concept/wave/upper memory,
gradient equivalence, hardware, source labels or useful fact persistence.
-/
namespace PNDS.LangStream076

abbrev Native (k : Nat) := Matrix (Fin k) (Fin k) ℝ

def maskedWrite (k : Nat) (admit : Bool)
    (old p q : Native k) : Native k :=
  if admit then old * p + q else old

theorem no_write_preserves (k : Nat) (old p q : Native k) :
    maskedWrite k false old p q = old := by
  simp [maskedWrite]

theorem admitted_step (k : Nat) (old p q : Native k) :
    maskedWrite k true old p q = old * p + q := by
  simp [maskedWrite]

theorem independent_of_denied_write (k : Nat)
    (old p₁ q₁ p₂ q₂ : Native k) :
    maskedWrite k false old p₁ q₁ = maskedWrite k false old p₂ q₂ := by
  simp [maskedWrite]

def persistentNativeFloats (k : Nat) : Nat := k*k
def persistentAlignedFloats (k : Nat) : Nat := 2*16*k
def nineFieldTotalFloats (k : Nat) : Nat :=
  976 + (k*k - 64) + 33*(k-8)

theorem state_m8 : nineFieldTotalFloats 8 = 976 := by decide
theorem state_m12 : nineFieldTotalFloats 12 = 1188 := by decide
theorem state_m16 : nineFieldTotalFloats 16 = 1432 := by decide

end PNDS.LangStream076
