import Mathlib

/-!
GP-UNC-001: exact-real scalar Gaussian conditioning and precision-addition
facts under explicit positive-variance / covariance premises.
This is NOT a proof of a GP model trained on TAC-OSM embeddings, the
multivariate Cholesky implementation, float roundoff, automatic differentiation,
online calibration, neural graph equivalence, or real-world superiority.
-/

namespace PNDS.GPUncertainty

/-- Marginal scalar variance after conditioning one Gaussian coordinate. -/
noncomputable def conditionedVariance (prior cross observedVariance : ℝ) : ℝ :=
  prior - cross ^ 2 / observedVariance

/-- Conditioning cannot increase marginal variance for positive observed variance. -/
theorem conditioned_le_prior (prior cross observedVariance : ℝ)
    (hPositive : 0 < observedVariance) :
    conditionedVariance prior cross observedVariance ≤ prior := by
  have h : 0 ≤ cross ^ 2 / observedVariance :=
    div_nonneg (sq_nonneg cross) (le_of_lt hPositive)
  dsimp [conditionedVariance]
  linarith

/-- Nonnegative conditional variance requires the stated covariance inequality. -/
theorem conditioned_nonneg (prior cross observedVariance : ℝ)
    (hPositive : 0 < observedVariance)
    (hCovariance : cross ^ 2 ≤ prior * observedVariance) :
    0 ≤ conditionedVariance prior cross observedVariance := by
  have h := (div_le_iff₀ hPositive).2 hCovariance
  dsimp [conditionedVariance]
  linarith

/-- No covariance connection means no posterior variance reduction. -/
theorem disconnected_prior_unchanged (prior observedVariance : ℝ) :
    conditionedVariance prior 0 observedVariance = prior := by
  simp [conditionedVariance]

/-- Online projected-GP diagonal precision update, not the entire matrix theorem. -/
noncomputable def precisionIncrement (old feature noiseVariance : ℝ) : ℝ :=
  old + feature ^ 2 / noiseVariance

/-- A scalar Bayesian precision increment is nonnegative if noise variance is positive. -/
theorem precision_nondecreasing (old feature noiseVariance : ℝ)
    (hPositive : 0 < noiseVariance) :
    old ≤ precisionIncrement old feature noiseVariance := by
  have h : 0 ≤ feature ^ 2 / noiseVariance :=
    div_nonneg (sq_nonneg feature) (le_of_lt hPositive)
  dsimp [precisionIncrement]
  linarith

/-- The zero projected feature carries no information increment. -/
theorem zero_feature_no_information (old noiseVariance : ℝ) :
    precisionIncrement old 0 noiseVariance = old := by
  simp [precisionIncrement]

end PNDS.GPUncertainty
