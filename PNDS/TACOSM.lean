import Mathlib

/-!
# TAC-OSM verified world-memory transition contracts
An explicit *new* transition design, not a theorem about PNDS v0.2's unspecified
status automaton and not a claim that neural verification is infallible.
These facts concern trusted, externally authenticated Boolean inputs.
Authenticity of real-world receipts is outside this model.
-/

namespace PNDS.TACOSM

structure Admission where
  authenticated : Bool
  independentContext : Bool
  utility : Bool
  stability : Bool
  integrity : Bool
  cost : Bool
  deriving Repr, DecidableEq

def eligible (a : Admission) : Bool :=
  a.authenticated && a.independentContext && a.utility &&
  a.stability && a.integrity && a.cost

inductive Tier where
  | provisional
  | persistent
  | quarantined
  deriving Repr, DecidableEq

def transition (tier : Tier) (a : Admission) : Tier :=
  match tier with
  | .provisional => if eligible a then .persistent else .provisional
  | .persistent => .persistent
  | .quarantined => .quarantined

theorem transition_provisional_iff_eligible (a : Admission) :
    transition .provisional a = .persistent ↔ eligible a = true := by
  cases h : eligible a <;> simp [transition, h]

theorem unauthenticated_cannot_promote (a : Admission)
    (h : a.authenticated = false) :
    transition .provisional a = .provisional := by
  simp [transition, eligible, h]

theorem repeated_context_cannot_promote (a : Admission)
    (h : a.independentContext = false) :
    transition .provisional a = .provisional := by
  simp [transition, eligible, h, Bool.and_false]

theorem no_utility_cannot_promote (a : Admission)
    (h : a.utility = false) :
    transition .provisional a = .provisional := by
  simp [transition, eligible, h]

theorem quarantine_never_promotes (a : Admission) :
    transition .quarantined a = .quarantined := by
  rfl

theorem persistent_remains_persistent (a : Admission) :
    transition .persistent a = .persistent := by
  rfl

theorem convex_coordinate_bounded (h p g : ℝ)
    (hh : -1 ≤ h ∧ h ≤ 1)
    (hp : -1 ≤ p ∧ p ≤ 1)
    (hg : 0 ≤ g ∧ g ≤ 1) :
    -1 ≤ (1 - g) * h + g * p ∧ (1 - g) * h + g * p ≤ 1 := by
  have low_h : 0 ≤ (1 - g) * (h + 1) := by
    apply mul_nonneg <;> linarith
  have low_p : 0 ≤ g * (p + 1) := by
    apply mul_nonneg <;> linarith
  have high_h : 0 ≤ (1 - g) * (1 - h) := by
    apply mul_nonneg <;> linarith
  have high_p : 0 ≤ g * (1 - p) := by
    apply mul_nonneg <;> linarith
  constructor <;> nlinarith [low_h, low_p, high_h, high_p]

theorem fixed_target_refinement_contract (h target alpha : ℝ)
    (ha : 0 ≤ alpha ∧ alpha ≤ 1) :
    |(1 - alpha) * h + alpha * target - target| ≤ |h - target| := by
  have heq : (1 - alpha) * h + alpha * target - target =
      (1 - alpha) * (h - target) := by ring
  rw [heq, abs_mul]
  have hpos : 0 ≤ 1 - alpha := by linarith
  rw [abs_of_nonneg hpos]
  nlinarith [mul_nonneg ha.1 (abs_nonneg (h - target))]

end PNDS.TACOSM
