import Mathlib

/-! LANGMATH095 exact-real operator obligations, not whole-model certification.

Chronological composition uses row states: apply the left pair first, then
the right pair. Matrix/tensor correspondence, numerical error, automatic
differentiation, the full CONTEXT026 graph and candidate readouts remain OPEN.
The norm theorem explicitly assumes the matrix one-step norm inequality;
it does not pretend to prove that inequality from scalar algebra.
-/
namespace PNDS.LANGMATH095
noncomputable section

abbrev Affine := ℝ × ℝ

def compose (a b : Affine) : Affine :=
  (a.1 * b.1, a.2 * b.1 + b.2)

def identity : Affine := (1, 0)

def applyAffine (s : ℝ) (a : Affine) : ℝ := s * a.1 + a.2

theorem compose_assoc (a b c : Affine) :
    compose (compose a b) c = compose a (compose b c) := by
  apply Prod.ext <;> dsimp [compose] <;> ring

theorem compose_identity_left (a : Affine) : compose identity a = a := by
  apply Prod.ext <;> simp [compose, identity]

theorem compose_identity_right (a : Affine) : compose a identity = a := by
  apply Prod.ext <;> simp [compose, identity]

theorem apply_composed (s : ℝ) (a b : Affine) :
    applyAffine s (compose a b) = applyAffine (applyAffine s a) b := by
  dsimp [applyAffine, compose]
  ring

def admitted (write : Bool) (a : Affine) : Affine :=
  if write then a else identity

theorem nonwrite_preserves (s : ℝ) (a : Affine) :
    applyAffine s (admitted false a) = s := by
  simp [admitted, identity, applyAffine]

def retention (h alpha offer : ℝ) : ℝ :=
  alpha * h + (1 - alpha) * offer

theorem retention_difference (h g alpha offer : ℝ) :
    retention h alpha offer - retention g alpha offer = alpha * (h - g) := by
  unfold retention
  ring

theorem convex_retention (h alpha offer : ℝ)
    (ha : 0 ≤ alpha ∧ alpha ≤ 1)
    (hh : -1 ≤ h ∧ h ≤ 1) (hv : -1 ≤ offer ∧ offer ≤ 1) :
    -1 ≤ retention h alpha offer ∧ retention h alpha offer ≤ 1 := by
  have h1 := mul_nonneg ha.1 (show 0 ≤ h + 1 by linarith [hh.1])
  have h2 := mul_nonneg (show 0 ≤ 1 - alpha by linarith [ha.2])
    (show 0 ≤ offer + 1 by linarith [hv.1])
  have h3 := mul_nonneg ha.1 (show 0 ≤ 1 - h by linarith [hh.2])
  have h4 := mul_nonneg (show 0 ≤ 1 - alpha by linarith [ha.2])
    (show 0 ≤ 1 - offer by linarith [hv.2])
  unfold retention
  constructor <;> nlinarith

/-- Scalar fixed-key equilibrium equation; no checkpoint instability claim. -/
theorem fixed_key_equilibrium (m alpha erase write value : ℝ)
    (heq : (1 - alpha * (1 - erase)) * m = write * value) :
    alpha * (1 - erase) * m + write * value = m := by
  nlinarith [heq]

/-- Invariant envelope conditional on the actual matrix-step norm inequality.

For the proposed aligned step, the unformalized matrix derivation uses
`‖k‖₂ ≤ 1`, `0 ≤ erase ≤ 1`, and `‖v‖₂ ≤ valueBound` to obtain hstep.
Unlike a fixed-key denominator condition, this envelope applies to arbitrary
key schedules once that one-step inequality has been established.
-/
theorem bounded_write_envelope
    (nextNorm previousNorm alpha write valueBound capacity : ℝ)
    (ha : 0 ≤ alpha)
    (hv : 0 ≤ valueBound)
    (hprevious : previousNorm ≤ capacity * valueBound)
    (hwrite : write ≤ capacity * (1 - alpha))
    (hstep : nextNorm ≤ alpha * previousNorm + write * valueBound) :
    nextNorm ≤ capacity * valueBound := by
  have h1 := mul_le_mul_of_nonneg_left hprevious ha
  have h2 := mul_le_mul_of_nonneg_right hwrite hv
  nlinarith

end
end PNDS.LANGMATH095

#print axioms PNDS.LANGMATH095.compose_assoc
#print axioms PNDS.LANGMATH095.compose_identity_left
#print axioms PNDS.LANGMATH095.compose_identity_right
#print axioms PNDS.LANGMATH095.apply_composed
#print axioms PNDS.LANGMATH095.nonwrite_preserves
#print axioms PNDS.LANGMATH095.retention_difference
#print axioms PNDS.LANGMATH095.convex_retention
#print axioms PNDS.LANGMATH095.fixed_key_equilibrium
#print axioms PNDS.LANGMATH095.bounded_write_envelope
