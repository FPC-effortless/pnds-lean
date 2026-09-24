/-!
# Causal execution (§9): the unspecified SCM

This file exists to record a **specification gap**, not to prove a theorem.

§9 writes

    Δ_ij^causal = P(Y|do(g_i)) − P(Y|do(g_j))
    Δ_ij^state  = P(Y|do(s_i)) − P(Y|do(s_j))

and §14 writes `Δ_T = P(Y|do(S)) − P(Y|do(T(S)))`.

A `do(·)` operator applied to an internal node of the substrate presupposes an
**explicit structural causal model** of the substrate: a set of variables, a DAG, and
structural equations. The document supplies none, so the interventional target is not
well defined. `Δ_ij^causal` and `Δ_ij^state` are therefore syntax without semantics.

The postulate is stated below as an axiom so that the gap is visible to
`#print axioms` rather than absorbed silently.
-/

import Mathlib

namespace PNDS.Causal

/-- An SCM over the substrate. The document does not supply this. -/
structure SCM (α : Type*) where
  vars : Set α
  parents : α → Set α
  structural : α → α

/-- The postulated (not constructed) interventional operator. -/
@[axiom] constant do_op {α : Type*} (scm : SCM α) (x : α) : α

/-- The postulated (not constructed) causal contrast. -/
@[axiom] constant causalContrast {α : Type*} (scm : SCM α) (x y : α) : ℝ

/-- The `I(S;Y) > 0` does not imply `P(Y|do(S=s₁)) ≠ P(Y|do(S=s₂))` warning of §9
    is stated, but its truth value is not established here because no SCM is given. -/

end PNDS.Causal
