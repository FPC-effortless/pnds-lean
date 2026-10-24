import Mathlib

namespace PNDS.Representation

/-!
# Dynamic representation validity

This file formalises only the logical core of safe persistent abstraction.

A representation phi : S -> Z induces the equivalence relation

    s ~phi t  <->  phi s = phi t.

The key condition is action congruence:

    phi s = phi t
      -> phi (transition s a) = phi (transition t a).

Under that condition, equivalence is preserved through every finite action
sequence. If task-relevant outcomes are also invariant under the same relation,
then two abstractly equivalent states remain indistinguishable to the
registered downstream computation.

No complexity or empirical scaling claim is made here.
-/

/-- Equality in representation space induces the abstraction relation. -/
def InducedEqv {S Z : Type} (phi : S -> Z) (s t : S) : Prop :=
  phi s = phi t

/-- Roll a concrete transition system through a finite action sequence. -/
def rollout {S A : Type} (transition : S -> A -> S) (s : S) : List A -> S
  | [] => s
  | a :: actions => rollout transition (transition s a) actions

/-- One-step action congruence of a representation.

    If two states have the same representation, applying the same relevant
    action must leave them represented equivalently.
-/
def ActionCongruent {S A Z : Type}
    (phi : S -> Z) (transition : S -> A -> S) : Prop :=
  ∀ ⦃s t : S⦄, phi s = phi t -> ∀ a : A,
    phi (transition s a) = phi (transition t a)

/-- A representation satisfying action congruence preserves equivalence through
    every finite action sequence. -/
theorem rollout_representation_eq
    {S A Z : Type}
    (phi : S -> Z)
    (transition : S -> A -> S)
    (hcong : ActionCongruent phi transition)
    {s t : S}
    (heq : phi s = phi t)
    (actions : List A) :
    phi (rollout transition s actions) =
      phi (rollout transition t actions) := by
  induction actions generalizing s t with
  | nil =>
      simpa [rollout] using heq
  | cons a actions ih =>
      have hnext :
          phi (transition s a) = phi (transition t a) :=
        hcong heq a
      simpa [rollout] using ih hnext actions

/-- An outcome invariant under representation equivalence stays invariant after
    any finite rollout, when the same final action is applied.

    This is the formal bridge from geometric/state equivalence to
    task-relevant behavioural equivalence.
-/
theorem outcome_preserved_after_rollout
    {S A Z O : Type}
    (phi : S -> Z)
    (transition : S -> A -> S)
    (outcome : S -> A -> O)
    (hcong : ActionCongruent phi transition)
    (hout :
      ∀ ⦃s t : S⦄, phi s = phi t -> ∀ a : A,
        outcome s a = outcome t a)
    {s t : S}
    (heq : phi s = phi t)
    (actions : List A)
    (a : A) :
    outcome (rollout transition s actions) a =
      outcome (rollout transition t actions) a := by
  apply hout
  exact rollout_representation_eq phi transition hcong heq actions

/-- A dynamic-validity certificate bundles the two conditions used by the PLM
    research program: representation/action congruence and outcome invariance.
-/
structure DynamicValidityCertificate
    {S A Z O : Type}
    (phi : S -> Z)
    (transition : S -> A -> S)
    (outcome : S -> A -> O) : Prop where
  action_congruent : ActionCongruent phi transition
  outcome_invariant :
    ∀ ⦃s t : S⦄, phi s = phi t -> ∀ a : A,
      outcome s a = outcome t a

/-- A dynamic-validity certificate implies behavioural equivalence after any
    finite action sequence followed by a common task-relevant action. -/
theorem certificate_implies_future_outcome_equivalence
    {S A Z O : Type}
    {phi : S -> Z}
    {transition : S -> A -> S}
    {outcome : S -> A -> O}
    (cert : DynamicValidityCertificate phi transition outcome)
    {s t : S}
    (heq : phi s = phi t)
    (actions : List A)
    (a : A) :
    outcome (rollout transition s actions) a =
      outcome (rollout transition t actions) a := by
  exact outcome_preserved_after_rollout
    phi transition outcome
    cert.action_congruent
    cert.outcome_invariant
    heq actions a

end PNDS.Representation
