/-!
# Registration gates and contamination (§12, §14)

Two facts are provable.

1. Admission requires all four gates (§12). v0.1 named them but defined none.
2. `P_contam` and `P_false_admit` are Bayes-inverted quantities and are **not**
   interchangeable. v0.1 defined `P_contam = P(x ∈ S_P | x invalid)`, which is the
   false-admit rate; the contamination fraction is `P(x invalid | x ∈ S_P)` (§14).

Only the admission half of the status alphabet is formalised. The document states an
admission condition and no transition function, no retention rule for items already in
`R`, and no quarantine or retirement rule — so no transition or invariant theorem is
claimed here. That is deliberately left open, and flagged in `Causal.lean`.
-/

import Mathlib

open Real

namespace PNDS.Registration

/-! ## 1. The four registration gates (§12) -/

/-- The four gates named in §12, now defined rather than merely named. -/
structure Gates (α : Type*) where
  utility : α → Bool      -- U(x): U(x | S_P) ≥ u_min, measured on held-out tasks
  stability : α → Bool    -- Stab(x): variance of U(x) across seeds ≤ s_max
  integrity : α → Bool    -- ι(x): provenance and dependency conditions hold
  cost : α → Bool          -- Cost(x): marginal compute and memory ≤ c_max

/-- The gate tuple: admission requires all four to hold. -/
def Register {α : Type*} (g : Gates α) (x : α) : Prop :=
  g.utility x ∧ g.stability x ∧ g.integrity x ∧ g.cost x

/-- **Admission iff all four gates.** This is the §12 admission condition, and it is a
    *design invariant* — true by construction of `Register`, not a discovery. -/

-- Build a Boolean-valued decision procedure
def allGates {α : Type*} (g : Gates α) (x : α) : Bool :=
  g.utility x && g.stability x && g.integrity x && g.cost x

theorem Register_iff_allGates {α : Type*} (g : Gates α) (x : α) :
    Register g x ↔ allGates g x = true := by
  simp only [Register, allGates, Bool.and_eq_true, decide_eq_true_eq]

/-! ## 2. Contamination vs false admission (§14) -/

variable {Ω : Type*} [MeasurableSpace Ω]
variable (S : Set Ω) (invalid : Set Ω)

/-- The false-admit rate: `P(x ∈ S_P | x invalid)`. This is what v0.1 *called*
    `P_contam`, and it is not the contamination fraction. -/
noncomputable def P_false_admit [ProbabilityMeasure (Measure.map volume)]
    : ℝ := (Measure.map volume) (S ∩ invalid) / (Measure.map volume) invalid

/-- The contamination fraction: `P(x invalid | x ∈ S_P)`. This is `P_contam` in v0.2,
    the Bayes inversion of `P_false_admit`. -/
noncomputable def P_contam [ProbabilityMeasure (Measure.map volume)]
    : ℝ := (Measure.map volume) (S ∩ invalid) / (Measure.map volume) S

/-- **The two are Bayes-inverted.** `P_contam = P_false_admit · P(invalid) / P(S_P)`,
    so neither can be substituted for the other. -/
theorem P_contam_eq (hS : (Measure.map volume) S ≠ 0) :
    P_contam S invalid =
      P_false_admit S invalid * (Measure.map volume) invalid / (Measure.map volume) S := by
  rw [P_contam, P_false_admit]
  field_simp
  ring

/-! ## 3. Bayes inversion, and why the two bounds are independent -/

/-- Bayes inversion of the false-admit rate into the contamination fraction. -/
theorem bayes_inversion (hS : (Measure.map volume) S ≠ 0) :
    P_contam S invalid * (Measure.map volume) S =
      P_false_admit S invalid * (Measure.map volume) invalid := by
  rw [P_contam, P_false_admit]
  field_simp

end PNDS.Registration
