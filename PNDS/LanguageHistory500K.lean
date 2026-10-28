import Mathlib

/-!
TAC-LANG-LONG-500K-003 — exact-real OLD-STATE RETENTION LIMITS.
New 503,514-parameter history-supervised source was trained locally on
authentic text (five seeds), but failed baseline and reset controls.  This
small proof explains why bounded/scanable memory is not sufficient for
retrieval; it is NOT a proof of actual Torch/IEEE gradients or learned skill.
-/
namespace PNDS.LanguageHistory500K

noncomputable section

/-- A historical address t-lag can be computed from public ordinal t,
and does not require supplying the historical target value. -/
def priorOrdinal (t lag : ℕ) : ℕ := t-lag

theorem known_past_ordinal_correct (t lag : ℕ) (h : lag ≤ t) :
    priorOrdinal t lag + lag = t := by
  exact Nat.sub_add_cancel h

/-- The write's Q term cancels when comparing two distinct carried states
under the SAME input-only affine schedule. -/
theorem old_state_difference_one_step (s₁ s₂ p q : ℝ) :
    (s₁*p+q)-(s₂*p+q)=(s₁-s₂)*p := by
  ring

/-- A decay/erase schedule with |p|≤1 cannot magnify differences between
carried states, but can still erase their distinguishing history. -/
theorem old_state_difference_nonexpansive (s₁ s₂ p q : ℝ)
    (hp : |p| ≤ 1) :
    |(s₁*p+q)-(s₂*p+q)| ≤ |s₁-s₂| := by
  rw [old_state_difference_one_step, abs_mul]
  have h : 0 ≤ |s₁-s₂| := abs_nonneg _
  nlinarith

/-- A valid bounded corrective write can overwrite an earlier scalar value
completely at a colliding address; stable memory does not entail content
survival or source-specific recall. -/
def scalarCorrective (old value alpha beta : ℝ) : ℝ :=
  alpha*old + beta*(value-alpha*old)

theorem full_corrective_write_erases_old (old value : ℝ) :
    scalarCorrective old value 1 1 = value := by
  simp [scalarCorrective]

/-- Even an arbitrarily separated pair of prior states can be mapped to
the exact same new value under the legal full-write rule. -/
theorem old_state_collision (old₁ old₂ value : ℝ) :
    scalarCorrective old₁ value 1 1 = scalarCorrective old₂ value 1 1 := by
  simp [scalarCorrective]

/- OPEN: A complete [128] categorical historical-reconstruction law at lags
64 and 192; source-identifiable query/mask bounds; 16x8 matrix operator
norm; old-state-to-logit sensitivity; differentiable Torch backward,
FP32 scan reassociation, actual learned checkpoint replay, calibration,
and independent source-conditioned factual retrieval. -/

end
end PNDS.LanguageHistory500K
