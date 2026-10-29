import Mathlib

/-!
LANGDECODE063 shared-upper operator contract (PARTIAL ONLY).
Exact-real referential-transparency theorem for the duplicate deterministic
upper projection. This is not a proof of PyTorch IEEE arithmetic, AD, the
full TAC-OSM graph or measured speed. Read uses the PREWRITE state.
Source: FPC-effortless/TAC-OSM PR #602,
commit d0178f69cfdccd706e26e460a85eb59b3547d828.
-/

namespace PNDS.LangDecode063

abbrev UpperInput := Fin 280 → ℝ
abbrev UpperState := Fin 48 → ℝ

/-- Source-order upper read and observation, recomputing pure project twice. -/
def unfused
    (project : UpperInput → UpperState)
    (read : UpperInput → UpperState → UpperState → UpperInput)
    (observe : UpperState → UpperState → UpperState)
    (z : UpperInput) (prior : UpperState) : UpperInput × UpperState :=
  (read z prior (project z), observe prior (project z))

/-- Share a single pure input projection. Both operations read the same prewrite state. -/
def fused
    (project : UpperInput → UpperState)
    (read : UpperInput → UpperState → UpperState → UpperInput)
    (observe : UpperState → UpperState → UpperState)
    (z : UpperInput) (prior : UpperState) : UpperInput × UpperState :=
  let u := project z
  (read z prior u, observe prior u)

/-- Under referential transparency, sharing does not change the exact result. -/
theorem fused_eq_unfused
    (project : UpperInput → UpperState)
    (read : UpperInput → UpperState → UpperState → UpperInput)
    (observe : UpperState → UpperState → UpperState)
    (z : UpperInput) (prior : UpperState) :
    fused project read observe z prior =
    unfused project read observe z prior := by
  rfl

end PNDS.LangDecode063
