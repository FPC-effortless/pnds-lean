import Mathlib

open Real

namespace PNDS.Causal

/-!
# Causal execution (§9): the unspecified SCM

This file records a **specification gap**, not a theorem.

§9 writes

    Δ_ij^causal = P(Y|do(g_i)) − P(Y|do(g_j))
    Δ_ij^state  = P(Y|do(s_i)) − P(Y|do(s_j))

and §14 writes `Δ_T = P(Y|do(S)) − P(Y|do(T(S)))`.

A `do(·)` operator applied to an internal node of the substrate presupposes an
**explicit structural causal model** of the substrate: a set of variables, a DAG, and
structural equations. The document supplies none, so the interventional target is not
well defined. `Δ_ij^causal` and `Δ_ij^state` are therefore syntax without semantics.

The postulate is stated as an axiom below so that the gap is visible to
`#print axioms` rather than absorbed silently. It is imported by the root file so that
the audit cannot overlook it.
-/

/-- An SCM over the substrate. The document does not supply this. -/
structure SCM (α : Type*) where
  vars : Set α
  parents : α → Set α
  structural : α → α

/-- The postulated (not constructed) interventional operator. -/
axiom do_op {α : Type*} (scm : SCM α) (x : α) : α

/-- The postulated (not constructed) causal contrast. -/
axiom causalContrast {α : Type*} (scm : SCM α) (x y : α) : ℝ

/-- The §9 warning, stated but not proved: `I(S;Y) > 0` does not imply
    `P(Y|do(S=s₁)) ≠ P(Y|do(S=s₂))`. Not proved here because no SCM is given. -/
axiom mutualInfo_nonzero_does_not_imply_interventional_difference
  {α : Type*} (scm : SCM α) : True

end PNDS.Causal
