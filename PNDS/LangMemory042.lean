import Mathlib

/-! Scoped fixed position-feature controls only, not whole model/IEEE/AD. -/
namespace PNDS.LangMemory042
abbrev Features := Fin 7 → ℝ
noncomputable section

def capLog (c : ℝ) (x : Features) : Features :=
  fun i => if i=0 then min (x i) c else x i

def capAll (c : ℝ) (atCap x : Features) : Features :=
  if c<x 0 then atCap else x

theorem log_identity_below (c : ℝ) (x : Features) (h : x 0≤c) : capLog c x=x := by
  funext i
  by_cases hi : i=0
  · subst i; simp [capLog,min_eq_left h]
  · simp [capLog,hi]

theorem other_coordinates_unchanged (c : ℝ) (x : Features) (i : Fin 7) (hi : i≠0) :
    capLog c x i=x i := by simp [capLog,hi]

theorem all_identity_below (c : ℝ) (k x : Features) (h : x 0≤c) : capAll c k x=x := by
  simp [capAll,not_lt.mpr h]

theorem all_idempotent (c : ℝ) (k x : Features) (hk : k 0=c) :
    capAll c k (capAll c k x)=capAll c k x := by
  by_cases hx : c<x 0
  · simp [capAll,hx,hk]
  · simp [capAll,hx]

theorem transformed_prefix_eq (f : Features → Features) (x y : Nat → Features) (t : Nat)
    (h : ∀ i, i≤t → x i=y i) : ∀ i, i≤t → f (x i)=f (y i) := by
  intro i hi
  rw [h i hi]

end
end PNDS.LangMemory042
#print axioms PNDS.LangMemory042.log_identity_below
#print axioms PNDS.LangMemory042.other_coordinates_unchanged
#print axioms PNDS.LangMemory042.all_identity_below
#print axioms PNDS.LangMemory042.all_idempotent
#print axioms PNDS.LangMemory042.transformed_prefix_eq
