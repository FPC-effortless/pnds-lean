import Mathlib

open Real

namespace PNDS.Registration

/-!
# Registration gates and contamination (§12, §14)

Two facts are provable.

1. Admission requires all four gates (§12). v0.1 named them but defined none.
2. `P_contam` and `P_false_admit` are Bayes-inverted quantities and are **not**
   interchangeable. v0.1 defined `P_contam = P(x ∈ S_P | x invalid)`, which is the
   false-admit rate; the contamination fraction is `P(x invalid | x ∈ S_P)` (§14).

Only the admission half of the status alphabet is formalised. The document states an
admission condition and no transition function, no retention rule for items already in
`R`, and no quarantine or retirement rule, so no transition or invariant theorem is
claimed here. That is deliberately left open, and flagged in `Causal.lean`.
-/

/-! ## 1. The four registration gates (§12) -/

/-- The four gates named in §12, now defined rather than merely named. -/
structure Gates (α : Type*) where
  utility : α → Bool      -- U(x): U(x | S_P) >= u_min, measured on held-out tasks
  stability : α → Bool    -- Stab(x): variance of U(x) across seeds <= s_max
  integrity : α → Bool    -- iota(x): provenance and dependency conditions hold
  cost : α → Bool          -- Cost(x): marginal compute and memory <= c_max

/-- The gate tuple: admission requires all four to hold. -/
def Register {α : Type*} (g : Gates α) (x : α) : Prop :=
  g.utility x ∧ g.stability x ∧ g.integrity x ∧ g.cost x

/-- A Boolean-valued decision procedure for the four gates. -/
def allGates {α : Type*} (g : Gates α) (x : α) : Bool :=
  g.utility x && g.stability x && g.integrity x && g.cost x

/-- **Admission iff all four gates.** This is the §12 admission condition, and it is a
    *design invariant*, true by construction of `Register`, not a discovery. -/
theorem Register_iff_allGates {α : Type*} (g : Gates α) (x : α) :
    Register g x ↔ allGates g x = true := by
  simp only [Register, allGates, Bool.and_eq_true, decide_eq_true_eq]
  -- `Bool.and` nests left: ((u && s) && i) && c, whereas `∧` in `Register` nests
  -- right: u ∧ (s ∧ (i ∧ c)). Reassociate.
  rw [and_assoc, and_assoc]

/-! ## 2. Contamination vs false admission (§14)

The two quantities are stated over an abstract measure space, so that the Bayes
inversion is visible without committing to a particular probability model.

`Measure Ω` takes values in `ℝ≥0∞` (`ENNReal`), so the identities below are stated
directly in `ℝ≥0∞` rather than casting to `ℝ`; this keeps the hypotheses honest —
a division is only meaningful when the denominator is nonzero and finite. -/

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ] (S invalid : Set Ω)

/-- The false-admit rate: `μ(S ∩ invalid) / μ(invalid)`. This is what v0.1 *called*
    `P_contam`, and it is not the contamination fraction. -/
noncomputable def P_false_admit : ℝ≥0∞ := μ (S ∩ invalid) / μ invalid

/-- The contamination fraction: `μ(S ∩ invalid) / μ(S)`. This is `P_contam` in v0.2,
    the Bayes inversion of `P_false_admit`. -/
noncomputable def P_contam : ℝ≥0∞ := μ (S ∩ invalid) / μ S

/-- **The two are Bayes-inverted.** `P_contam * μ(S) = P_false_admit * μ(invalid)`,
    so neither can be substituted for the other. The measures are required to be
    finite and nonzero, which in `ℝ≥0∞` is a real restriction (division by `∞` or
    by `0` is degenerate). -/
theorem bayes_inversion (hS : μ S ≠ 0) (hSfin : μ S ≠ ∞) (hI : μ invalid ≠ 0)
    (hIfin : μ invalid ≠ ∞) :
    P_contam μ S invalid * μ S = P_false_admit μ S invalid * μ invalid := by
  rw [P_contam, P_false_admit]
  -- In `ℝ≥0∞`, `(a / b) * b = a` requires `b ≠ 0` and `b ≠ ∞`. Both sides of the
  -- goal therefore collapse to the same numerator `μ (S ∩ invalid)`.
  rw [ENNReal.div_mul_cancel hS hSfin, ENNReal.div_mul_cancel hI hIfin]

end PNDS.Registration
