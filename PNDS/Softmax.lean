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

/-- `app` is strictly positive. -/
theorem app_pos (x : F) : 0 < E.app x := by
  have h := E.app_strictMono
  have hzero : E.app 0 = 1 := E.app_one
  have hpos : 0 < E.app 0 := by rw [hzero]; exact one_pos
  exact lt_of_lt_of_le hpos (h.monotone x)

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
  have hdenom : 0 < r * E.app s + n := by positivity
  rw [distractorShare, le_div_iff₀ hdenom]
  -- Goal: n ≤ ε * (r * E.app s + n)
  linarith

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
  · simpa using hbound

end PNDS.Softmax
