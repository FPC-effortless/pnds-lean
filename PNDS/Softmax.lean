import Mathlib

open Real

namespace PNDS.Softmax

/-!
# Softmax selection-margin bound (§5)

The document states that if every relevant item outscores every distractor by at least
`m` and all distractors share one logit, then holding distractor mass at `ε` requires

    m ≥ τ · ln((n (1 - ε)) / (r ε))          (§5; worst case r = 1 gives ln(n/ε))

with `n = |I_N| - r` the distractor count and `r` the relevant count.

This is why the `C(R ∪ I_N) → C(R)` claim of §18 must be stated *conditionally*:
at fixed `m` and `τ` the required margin grows as `ln |I_N|`, so it is the index, not
the router, that must exclude distractors as the history grows.

To keep the argument in the language of ordered fields, the exponential is reified as
an abstract strictly increasing additive-to-multiplicative homomorphism. This avoids
depending on continuity or transcendental identities, and makes the whole bound a
consequence of ordered-field algebra.
-/

/-- A strictly increasing additive-to-multiplicative homomorphism. -/
structure Exp (F : Type*) [LinearOrderedField F] where
  app : F → F
  app_one : app 0 = 1
  app_add (x y : F) : app (x + y) = app x * app y
  app_strictMono : StrictMono app

variable {F : Type*} [LinearOrderedField F] (E : Exp F)

/-- Mass on the zero-score (distractor) side of the softmax with `r` items at score
    `s` and `n` items at score `0`. -/
noncomputable def distractorShare (r n s : F) : F :=
  n / (r * E.app s + n)

/-- `Real.exp` is the canonical instance. -/
noncomputable def expExp : Exp ℝ where
  app := Real.exp
  app_one := Real.exp_zero
  app_add := Real.exp_add
  app_strictMono := Real.exp_strictMono

@[simp] theorem expExp_app (x : ℝ) : expExp.app x = Real.exp x := rfl

/-! ## 1. Positivity -/

/-- `app x * app (-x) = 1`, i.e. `app (-x)` is a right inverse of `app x`. -/
theorem app_mul_app_neg (x : F) : E.app x * E.app (-x) = 1 := by
  have h1 : E.app (x + -x) = E.app x * E.app (-x) := E.app_add x (-x)
  have h2 : x + -x = 0 := by ring
  rw [h2, E.app_one] at h1
  exact h1.symm

/-- `app` is strictly positive. -/
theorem app_pos (x : F) : 0 < E.app x := by
  -- `app x * app (-x) = 1`. One of `x`, `-x` is nonnegative; monotonicity
  -- then makes the corresponding factor `≥ app 0 = 1 > 0`, and the other factor is
  -- `1 / (that positive quantity)`, hence positive.
  have hmono : Monotone E.app := E.app_strictMono.monotone
  have hzero : E.app 0 = 1 := E.app_one
  have hpos0 : 0 < E.app 0 := by rw [hzero]; exact one_pos
  have hprod : E.app x * E.app (-x) = 1 := app_mul_app_neg E x
  rcases le_or_lt 0 x with hx | hx
  · -- `x ≥ 0`: monotonicity gives `app x ≥ app 0 = 1 > 0`.
    exact lt_of_lt_of_le hpos0 (hmono hx)
  · -- `x < 0`, so `-x > 0`: monotonicity gives `app(-x) ≥ app 0 = 1 > 0`.
    -- `app x * app(-x) = 1` with `app(-x) > 0` gives `app x = 1 / app(-x) > 0`.
    have hbx : 0 < E.app (-x) := lt_of_lt_of_le hpos0 (hmono (by linarith))
    -- From `app x * app(-x) = 1` and `app(-x) > 0`, ordered-field algebra gives
    -- `app x = 1 / app(-x)`, which is positive.
    have hkey : E.app x * E.app (-x) = 1 := hprod
    have hne : E.app (-x) ≠ 0 := hbx.ne'
    -- `app(-x)⁻¹ * app(-x) = 1` and `app x * app(-x) = 1`, so cancelling the
    -- common nonzero right factor `app(-x)` gives `app x = app(-x)⁻¹`.
    have hsymm : E.app x = (E.app (-x))⁻¹ :=
      mul_right_cancel₀ hne (hkey.trans (inv_mul_cancel₀ hne).symm)
    calc E.app x = 1 / E.app (-x) := by
        rw [div_eq_mul_inv, one_mul]
        exact hsymm
      _ > 0 := one_div_pos.2 hbx

theorem app_ne_zero (x : F) : E.app x ≠ 0 := ne_of_gt (app_pos E x)

/-! ## 2. The bound

The bound is stated in its algebraic form. Solving `n / (r · E(s) + n) ≤ ε`
under positivity gives `r · ε · E(s) ≥ n · (1 - ε)`, which for `E = exp` and
`s = m/τ` is exactly `m ≥ τ · ln(n (1-ε) / (r ε))`. -/

variable {r n ε s : F}

/-- **The selection-margin bound.** If `r > 0` relevant items each score `s = m/τ`
    above `n > 0` equal-logit distractors, then the distractor share is at most `ε`
    whenever `r * ε * E.app s ≥ n * (1 - ε)`. -/
theorem distractorShare_le (hr : 0 < r) (hn : 0 < n) (hε : 0 < ε) (hεε : ε < 1)
    (hbound : r * ε * E.app s ≥ n * (1 - ε)) :
    distractorShare E r n s ≤ ε := by
  -- `E.app s > 0` and `r, n > 0`, so the denominator is strictly positive.
  have hdenom : 0 < r * E.app s + n := by
    have hE : 0 < E.app s := app_pos E s
    positivity
  rw [distractorShare]
  -- Division sits on the *left* of `≤`, so this is `div_le_iff₀`, not `le_div_iff₀`.
  rw [div_le_iff₀ hdenom]
  -- Goal: n ≤ ε * (r * E.app s + n)
  -- `linarith` treats products as opaque atoms, so we line the goal up with
  -- `hbound` by hand and let `ring` do the polynomial bookkeeping.
  calc n = n * (1 - ε) + ε * n := by ring
    _ ≤ r * ε * E.app s + ε * n := add_le_add_right hbound (ε * n)
    _ = ε * (r * E.app s + n) := by ring

/-- **Worst case `r = 1`.** The operational form: holding leakage at `ε` requires
    `E.app s ≥ n * (1 - ε) / ε`. -/
theorem distractorShare_le_one (hn : 0 < n) (hε : 0 < ε) (hεε : ε < 1)
    (hbound : E.app s ≥ n * (1 - ε) / ε) :
    distractorShare E 1 n s ≤ ε := by
  apply distractorShare_le (E := E)
  · norm_num
  · exact hn
  · exact hε
  · exact hεε
  · -- `r = 1`, so `r * ε * E.app s = ε * E.app s ≥ ε * (n * (1-ε) / ε) = n * (1-ε)`
    -- `ε > 0` is needed to cancel it; `field_simp` alone cannot guess that, so
    -- discharge it as a side goal.
    have hE : 0 < E.app s := app_pos E s
    rw [one_mul]
    -- `ε * E.app s ≥ n * (1 - ε)`
    calc ε * E.app s ≥ ε * (n * (1 - ε) / ε) := mul_le_mul_of_nonneg_left hbound hε.le
      _ = n * (1 - ε) := by
        field_simp [hε.ne']

end PNDS.Softmax
