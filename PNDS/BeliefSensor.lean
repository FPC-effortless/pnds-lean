import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import PNDS.SourceAlignedTrace

/-!
ME101 scalar exact-real kernel for missing-channel latent uncertainty.

This module inherits the immutable ME100 source-sum/mass trace contract. It
formalizes the proposed diagonal marginal variance only; it is not a proof of
the learned network, arbitrary covariance positivity, or floating-point
refinement.
-/
namespace PNDS.BeliefSensor

/-- Per-channel latent current variance. `observed` denotes an observed current
cell; otherwise the variance grows with source age at fixed confidence. -/
noncomputable def latentVariance (softplus : ℝ → ℝ) (observed : Bool)
    (ageHours confidence b0 b1 b2 : ℝ) : ℝ :=
  if observed then 0 else
    (softplus b0 + softplus b1 * (ageHours / 24)) *
      (1 + softplus b2 * (1 - confidence))

theorem latentVariance_observed_zero (softplus : ℝ → ℝ)
    (ageHours confidence b0 b1 b2 : ℝ) :
    latentVariance softplus true ageHours confidence b0 b1 b2 = 0 := by
  simp [latentVariance]

theorem latentVariance_nonneg (softplus : ℝ → ℝ) (ageHours confidence b0 b1 b2 : ℝ)
    (hage : 0 ≤ ageHours) (hconfidence : confidence ≤ 1)
    (hs0 : 0 ≤ softplus b0) (hs1 : 0 ≤ softplus b1) (hs2 : 0 ≤ softplus b2) :
    0 ≤ latentVariance softplus false ageHours confidence b0 b1 b2 := by
  unfold latentVariance
  have hday : 0 ≤ ageHours / 24 := by positivity
  have hscale : 0 ≤ 1 + softplus b2 * (1 - confidence) := by
    have hc : 0 ≤ 1 - confidence := by linarith
    have hp := mul_nonneg hs2 hc
    linarith
  have hbase : 0 ≤ softplus b0 + softplus b1 * (ageHours / 24) := by
    exact add_nonneg hs0 (mul_nonneg hs1 hday)
  exact mul_nonneg hbase hscale

theorem latentVariance_monotone_age (softplus : ℝ → ℝ)
    (confidence b0 b1 b2 age1 age2 : ℝ)
    (hconfidence : confidence ≤ 1) (hage12 : age1 ≤ age2)
    (hs1 : 0 ≤ softplus b1) (hs2 : 0 ≤ softplus b2) :
    latentVariance softplus false age1 confidence b0 b1 b2 ≤
      latentVariance softplus false age2 confidence b0 b1 b2 := by
  unfold latentVariance
  have hc : 0 ≤ 1 - confidence := by linarith
  have hscale : 0 ≤ 1 + softplus b2 * (1 - confidence) := by
    have hp := mul_nonneg hs2 hc
    linarith
  have hdays : age1 / 24 ≤ age2 / 24 := by
    exact div_le_div_of_nonneg_right hage12 (by norm_num)
  have hbase :
      softplus b0 + softplus b1 * (age1 / 24) ≤
      softplus b0 + softplus b1 * (age2 / 24) := by
    exact add_le_add_left (mul_le_mul_of_nonneg_left hdays hs1) _
  exact mul_le_mul_of_nonneg_right hbase hscale

/-- Exact marginalized scalar variance for an affine emission with independent
future noise. `jacobian` is unconstrained because only its square matters. -/
noncomputable def marginalizedVariance (futureScale jacobian latentVar : ℝ) : ℝ :=
  futureScale ^ 2 + jacobian ^ 2 * latentVar

theorem marginalizedVariance_positive (futureScale jacobian latentVar : ℝ)
    (hscale : (1 / 20 : ℝ) ≤ futureScale) (hlatent : 0 ≤ latentVar) :
    0 < marginalizedVariance futureScale jacobian latentVar := by
  unfold marginalizedVariance
  have hsq : 0 ≤ jacobian ^ 2 := sq_nonneg jacobian
  have hnoise : 0 < futureScale ^ 2 := by nlinarith
  have hlatentTerm : 0 ≤ jacobian ^ 2 * latentVar := mul_nonneg hsq hlatent
  linarith

/-- A variance read is a pure query over the inherited source trace. -/
def readTrace (trace : ℝ × ℝ) (_latentVar : ℝ) : ℝ × ℝ := trace

theorem readTrace_preserves_trace (trace : ℝ × ℝ) (latentVar : ℝ) :
    readTrace trace latentVar = trace := rfl

end PNDS.BeliefSensor
