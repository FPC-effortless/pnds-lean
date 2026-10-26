import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Tactic.Ring

/-!
ME110 typed role training contract. This is an exact-real coordinate-mask
specification. The ME110 model version is native ME109 with 104663 parameters;
the ME109 source and its historical results remain unchanged. Source-PIN root
and correspondence to the later TAC-OSM implementation are OPEN. This module
does not prove full automatic-differentiation behavior, IEEE behavior, or loss
descent.
-/
namespace PNDS.TypedRoleTraining

variable {n : ℕ}

abbrev Vec (n : ℕ) := Fin n → ℝ

/-- Point role mask; values are required to be exactly zero or one. -/
def P (p : Vec n) (i : Fin n) : ℝ := p i

/-- Complementary uncertainty role mask. -/
def U (p : Vec n) (i : Fin n) : ℝ := 1 - P p i

def routedGrad (p gPoint gUnc : Vec n) : Vec n :=
  fun i => P p i * gPoint i + U p i * gUnc i

theorem point_masked_coordinate (p gPoint gUnc : Vec n) (i : Fin n)
    (h : P p i = 1) : routedGrad p gPoint gUnc i = gPoint i := by
  simp [routedGrad, U, h]

theorem uncertainty_masked_coordinate (p gPoint gUnc : Vec n) (i : Fin n)
    (h : P p i = 0) : routedGrad p gPoint gUnc i = gUnc i := by
  simp [routedGrad, U, h]

theorem masks_disjoint (p : Vec n) (i : Fin n)
    (h : P p i = 0 ∨ P p i = 1) : P p i * U p i = 0 := by
  rcases h with h | h <;> simp [U, h]

theorem masks_sum_to_one (p : Vec n) (i : Fin n) :
    P p i + U p i = 1 := by
  simp [U]

/-- If a decoder factors through point-masked coordinates, changing only
uncertainty coordinates leaves its output unchanged. -/
theorem decoder_congruent_under_uncertainty_update
    (p x y : Vec n) (decode : Vec n → α)
    (hmask : ∀ i, P p i = 1 → x i = y i)
    (hdepends : ∀ a b : Vec n,
      (∀ i, P p i = 1 → a i = b i) → decode a = decode b) :
    decode x = decode y := by
  apply hdepends x y
  exact hmask

/-
Optimizer contract: with coordinatewise AdamW moments and a fixed role mask,
masked coordinates receive no gradient contribution from the other role.
Independent per-role clipping must occur before routing if clipping is used;
clipping the already combined vector can couple roles. Actual optimizer state,
decoupled decay, source tensor mapping, and full AD remain OPEN.
-/

end PNDS.TypedRoleTraining
