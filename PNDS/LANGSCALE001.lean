import PNDS.LANGCAP103

/-! LANGSCALE001 specializes the complete existing dimension-generic exact-real
TAC graph. No update law changes, IEEE theorem or trained capability is claimed.
Source/parameter layout correspondence is a separately pinned Python obligation.
-/
namespace PNDS.LANGSCALE001
open PNDS.LANGCAP103

abbrev SmallHeadToStateParams := TacParams 256 24 6 12 8 106
abbrev SmallHeadToStateState := TacState 256 24 6 12 8
abbrev SmallMemoryScaledParams := TacParams 232 32 8 16 8 152
abbrev SmallMemoryScaledState := TacState 232 32 8 16 8
abbrev MediumHeadToStateParams := TacParams 992 32 8 16 8 116
abbrev MediumHeadToStateState := TacState 992 32 8 16 8
abbrev MediumMemoryScaledParams := TacParams 856 64 16 32 8 571
abbrev MediumMemoryScaledState := TacState 856 64 16 32 8

noncomputable def smallHeadToStateStep (p : SmallHeadToStateParams)
    (s : SmallHeadToStateState) (token : Fin 256) := tacStep p s token
noncomputable def smallMemoryScaledStep (p : SmallMemoryScaledParams)
    (s : SmallMemoryScaledState) (token : Fin 256) := tacStep p s token
noncomputable def mediumHeadToStateStep (p : MediumHeadToStateParams)
    (s : MediumHeadToStateState) (token : Fin 256) := tacStep p s token
noncomputable def mediumMemoryScaledStep (p : MediumMemoryScaledParams)
    (s : MediumMemoryScaledState) (token : Fin 256) := tacStep p s token

/-- Context, two previous keys, two matrices and two mass vectors. -/
def nonlexicalFloats (c k v layers : Nat) : Nat := layers * (c + 4*k + 2*v*k)

theorem small_head_to_state_count : nonlexicalFloats 24 6 12 8 = 1536 := by decide
theorem small_memory_scaled_count : nonlexicalFloats 32 8 16 8 = 2560 := by decide
theorem medium_head_to_state_count : nonlexicalFloats 32 8 16 8 = 2560 := by decide
theorem medium_memory_scaled_count : nonlexicalFloats 64 16 32 8 = 9216 := by decide

#print axioms small_head_to_state_count
#print axioms small_memory_scaled_count
#print axioms medium_head_to_state_count
#print axioms medium_memory_scaled_count
end PNDS.LANGSCALE001
