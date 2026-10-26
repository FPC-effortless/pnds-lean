import Init

/-!
ME111 exact scalar contract, deliberately standalone (no Mathlib dependency).
`RealOps` exposes the ordered-real operations/laws needed by these proofs;
these are explicit structure parameters, NOT new axioms or a constructed ℝ.
Instantiation by Mathlib ℝ and its exp/sqrt/tanh, analytic width bounds, full
neural refinement, AD and IEEE error remain OPEN. Theorems are conditional on
these primitive ordered-real laws. Python/source parity is separate evidence.
-/
namespace PNDS.CoherentConditionalBelief

-- Use explicit unary transcendental operations, with ordinary scalar laws.
structure RealOps (S : Type) where
  zero : S
  epsilon : S
  logFour : S
  add : S → S → S
  mul : S → S → S
  sub : S → S → S
  sqrt : S → S
  exp : S → S
  tanh : S → S
  positive : S → Prop
  nonnegative : S → Prop
  le : S → S → Prop
  epsilon_pos : positive epsilon
  positive_nonnegative : ∀ x, positive x → nonnegative x
  square_nonnegative : ∀ x, nonnegative (mul x x)
  mul_nonnegative : ∀ x y, nonnegative x → nonnegative y → nonnegative (mul x y)
  add_positive_nonnegative : ∀ x y, positive x → nonnegative y → positive (add x y)
  sqrt_positive : ∀ x, positive x → positive (sqrt x)
  exp_positive : ∀ x, positive (exp x)
  mul_positive : ∀ x y, positive x → positive y → positive (mul x y)
  sub_nonnegative_le : ∀ x y, nonnegative y → le (sub x y) x
  le_add_nonnegative : ∀ x y, nonnegative y → le x (add x y)

def variance {S : Type} (o : RealOps S) (vf response vc missing : S) : S :=
  o.add (o.add o.epsilon vf) (o.mul (o.mul (o.mul response response) vc) missing)

def width {S : Type} (o : RealOps S) (raw : S) : S :=
  o.exp (o.mul o.logFour (o.tanh raw))

def sigma {S : Type} (o : RealOps S) (vf response vc missing raw : S) : S :=
  o.mul (o.sqrt (variance o vf response vc missing)) (width o raw)

theorem variance_positive {S : Type} (o : RealOps S) (vf response vc missing : S)
    (hvf : o.positive vf) (hvc : o.positive vc) (hm : o.nonnegative missing) :
    o.positive (variance o vf response vc missing) := by
  apply o.add_positive_nonnegative
  · exact o.add_positive_nonnegative _ _ o.epsilon_pos (o.positive_nonnegative _ hvf)
  · exact o.mul_nonnegative _ _
      (o.mul_nonnegative _ _ (o.square_nonnegative response) (o.positive_nonnegative _ hvc)) hm

theorem sigma_positive {S : Type} (o : RealOps S) (vf response vc missing raw : S)
    (hvf : o.positive vf) (hvc : o.positive vc) (hm : o.nonnegative missing) :
    o.positive (sigma o vf response vc missing raw) := by
  exact o.mul_positive _ _ (o.sqrt_positive _ (variance_positive o _ _ _ _ hvf hvc hm))
    (o.exp_positive _)

/-- Positive left/right scales give ordered intervals for nonnegative multipliers. -/
theorem quantile_order {S : Type} (o : RealOps S) (mean left right kl kr : S)
    (hl : o.positive left) (hr : o.positive right)
    (hkl : o.nonnegative kl) (hkr : o.nonnegative kr) :
    o.le (o.sub mean (o.mul kl left)) mean ∧ o.le mean (o.add mean (o.mul kr right)) := by
  exact ⟨o.sub_nonnegative_le _ _ (o.mul_nonnegative _ _ hkl (o.positive_nonnegative _ hl)),
    o.le_add_nonnegative _ _ (o.mul_nonnegative _ _ hkr (o.positive_nonnegative _ hr))⟩

/-- The three roles are structural booleans, so the R subset P premise is explicit. -/
def routed {S : Type} (add : S → S → S) (p r : Bool) (gp gr gu : S) : S :=
  if p then (if r then add gp gr else gp) else gu

theorem mean_supervised {S : Type} (add : S → S → S) (gp gr gu : S) :
    routed add true true gp gr gu = add gp gr := rfl

theorem other_point {S : Type} (add : S → S → S) (gp gr gu : S) :
    routed add true false gp gr gu = gp := rfl

theorem uncertainty_only {S : Type} (add : S → S → S) (gp gr gu : S) :
    routed add false false gp gr gu = gu := rfl

/-- Invalid masks R=true,P=false are excluded as a caller obligation. -/
def validMask (p r : Bool) : Prop := r = true → p = true

theorem mean_role_is_point (p r : Bool) (h : validMask p r) (hr : r = true) :
    p = true := h hr

end PNDS.CoherentConditionalBelief
