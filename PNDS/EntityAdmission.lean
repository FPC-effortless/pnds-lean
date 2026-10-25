import Mathlib

/-!
Typed, exact-real *identity-isolation* property of a fixed-size TAC entity
memory bank (optional). This proves that a local admitted update cannot
modify a distinct slot. It does NOT prove the source's runtime admission
decision correct, nor does it make its data-dependent eviction decisions
parallel-time associative, nor validate entity identifiers in real data.
-/
namespace PNDS.EntityAdmission

def valueWrite {n : ℕ} (previous : Fin n → ℝ)
    (selected : Fin n) (oldIdentityMatches : Bool)
    (beta : ℝ) (observed : ℝ) : Fin n → ℝ :=
  fun j => if j=selected then
    (1-beta)*(if oldIdentityMatches then previous j else 0)+beta*observed
    else previous j

theorem unrelated_slot_unchanged {n : ℕ} (prev : Fin n → ℝ)
    (selected other : Fin n) (sameEntity : Bool) (beta observed : ℝ)
    (hdiff : other ≠ selected) :
    valueWrite prev selected sameEntity beta observed other = prev other := by
  simp [valueWrite, hdiff]

theorem replacement_does_not_copy_evicted_content {n : ℕ}
    (prev : Fin n → ℝ) (chosen : Fin n) (beta observed : ℝ) :
    valueWrite prev chosen false beta observed chosen = beta*observed := by
  simp [valueWrite]

theorem existing_slot_interpolation {n : ℕ}
    (prev : Fin n → ℝ) (chosen : Fin n) (beta observed : ℝ) :
    valueWrite prev chosen true beta observed chosen =
      (1-beta)*prev chosen+beta*observed := by
  simp [valueWrite]

theorem read_only_unchanged {n : ℕ} (prev : Fin n → ℝ) :
    (fun j : Fin n => prev j) = prev := by
  funext j
  rfl

theorem capacity_full_no_new_slot
    (slots : ℕ) (occupied : Fin slots → Bool)
    (hfull : ∀ i, occupied i = true) :
    ¬ (∃ i, occupied i = false) := by
  intro h
  rcases h with ⟨i,hi⟩
  rw [hfull i] at hi
  cases hi

end PNDS.EntityAdmission
