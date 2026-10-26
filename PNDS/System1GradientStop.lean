import Mathlib
import PNDS.System1Head
import PNDS.System1VelocityAnchor

/-!
# S1MOTION004: exact-real forward/tangent semantics for isolated auxiliary heads

Source TAC branch:
research/s1motion004-point-gradient-isolation-2026-10-26
src/tac_osm/variants/s1motion004_gradient_routes.py

The original 198->12->15 observational auxiliary head receives evidence
computed by a separate primary predictor. Isolation is an OPERATIONAL
automatic-differentiation primitive: primal identity, tangent ZERO.
It is *not* the ordinary mathematical derivative of the real identity
function. We explicitly model a primal/tangent pair rather than claiming that
the derivative of x->x is zero.

This module proves exact-real typed structural obligations, not refinement
to a concrete PyTorch autodiff tape, full neural source, AdamW parameter
updates, IEEE floating point, proper risk calibration, causal command effects,
or real-world skill. The original 2026-10-26 S1MOTION004 full predictive gate
remains FAILED (only 1/3 seed passes all gates).
-/

namespace PNDS.System1GradientStop

/-- An exact-real forward value with a derivative/tangent for n parameters. -/
structure Jet (n : ℕ) where
  primal : ℝ
  tangent : Fin n → ℝ

/-- Operational stop-gradient: same forward value, zero backward sensitivity.
This intentionally differs from ordinary differentiation of identity. -/
def detach {n : ℕ} (x : Jet n) : Jet n :=
  ⟨x.primal, fun _ => 0⟩

theorem detach_preserves_primal {n : ℕ} (x : Jet n) :
    (detach x).primal = x.primal := rfl

theorem detach_clears_tangent {n : ℕ} (x : Jet n)
    (p : Fin n) : (detach x).tangent p = 0 := rfl

/-- A typed affine map. Differentiation here concerns ONLY its input evidence,
not the weights/bias parameters themselves. -/
noncomputable def denseJet {n i o : ℕ}
    (w : Fin o → Fin i → ℝ) (b : Fin o → ℝ)
    (x : Fin i → Jet n) : Fin o → Jet n :=
  fun j =>
    ⟨(∑ k, w j k * (x k).primal) + b j,
     fun p => ∑ k, w j k * (x k).tangent p⟩

/-- A generic unary neural nonlinearity, with a supplied local tangent rule.
We do NOT assert this tangent rule is PyTorch's actual SiLU derivative. -/
noncomputable def unaryJet {n : ℕ} (f df : ℝ → ℝ) (x : Jet n) :
    Jet n :=
  ⟨f x.primal, fun p => df x.primal * x.tangent p⟩

def detachEvidence {n : ℕ} (x : Fin 198 → Jet n) :
    Fin 198 → Jet n :=
  fun k => detach (x k)

/-- Original forward-valued head with input evidence sensitivities. -/
noncomputable def sharedAuxRaw {n : ℕ}
    (w1 : Fin 12 → Fin 198 → ℝ) (b1 : Fin 12 → ℝ)
    (w2 : Fin 15 → Fin 12 → ℝ) (b2 : Fin 15 → ℝ)
    (dsilu : ℝ → ℝ) (evidence : Fin 198 → Jet n) :
    Fin 15 → Jet n :=
  denseJet w2 b2 (fun j =>
    unaryJet PNDS.System1Head.silu dsilu
      (denseJet w1 b1 evidence j))

/-- Same head, with operationally detached *evidence*. -/
noncomputable def isolatedAuxRaw {n : ℕ}
    (w1 : Fin 12 → Fin 198 → ℝ) (b1 : Fin 12 → ℝ)
    (w2 : Fin 15 → Fin 12 → ℝ) (b2 : Fin 15 → ℝ)
    (dsilu : ℝ → ℝ) (evidence : Fin 198 → Jet n) :
    Fin 15 → Jet n :=
  sharedAuxRaw w1 b1 w2 b2 dsilu (detachEvidence evidence)

/-- Input-forward values of the isolated head stay identical to shared mode
for the SAME weights, observations, and exact-real arithmetic. -/
theorem isolation_same_auxiliary_forward {n : ℕ}
    (w1 : Fin 12 → Fin 198 → ℝ) (b1 : Fin 12 → ℝ)
    (w2 : Fin 15 → Fin 12 → ℝ) (b2 : Fin 15 → ℝ)
    (dsilu : ℝ → ℝ) (x : Fin 198 → Jet n) (j : Fin 15) :
    (isolatedAuxRaw w1 b1 w2 b2 dsilu x j).primal =
      (sharedAuxRaw w1 b1 w2 b2 dsilu x j).primal := by
  simp [isolatedAuxRaw, sharedAuxRaw, denseJet, unaryJet,
        detachEvidence, detach]

/-- Every backbone/evidence tangent is zeroed after the auxiliary detached
input, independent of the outer dense weights or supplied local unary rule.
This does NOT say the auxiliary head's OWN trainable weight gradients vanish. -/
theorem isolation_zero_auxiliary_backbone_tangent {n : ℕ}
    (w1 : Fin 12 → Fin 198 → ℝ) (b1 : Fin 12 → ℝ)
    (w2 : Fin 15 → Fin 12 → ℝ) (b2 : Fin 15 → ℝ)
    (dsilu : ℝ → ℝ) (x : Fin 198 → Jet n)
    (j : Fin 15) (p : Fin n) :
    (isolatedAuxRaw w1 b1 w2 b2 dsilu x j).tangent p = 0 := by
  simp [isolatedAuxRaw, sharedAuxRaw, denseJet, unaryJet,
        detachEvidence, detach]

/-- A primary point layer remains UNDETACHED in the candidate source.
Its evidence input sensitivities are the normal affine weighted sum. -/
theorem primary_linear_tangent_intact {n : ℕ}
    (w : Fin 14 → Fin 198 → ℝ) (b : Fin 14 → ℝ)
    (x : Fin 198 → Jet n) (j : Fin 14) (p : Fin n) :
    (denseJet w b x j).tangent p =
      ∑ k, w j k * (x k).tangent p := rfl

end PNDS.System1GradientStop
