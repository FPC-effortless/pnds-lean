import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith

/-! ME111 exact-real scalar law and restricted role algebra. Full neural/AD,
IEEE refinement, calibration and split-distribution variance identities OPEN.
No empirical age floor or lower clipping of the width factor occurs here. -/
namespace PNDS.CoherentConditionalBeliefReal

noncomputable def variance (vf response vc missing : ℝ) : ℝ :=
  (1 / 1000000 : ℝ) + vf + response ^ 2 * vc * missing

noncomputable def width (raw : ℝ) : ℝ := Real.exp (Real.log 4 * Real.tanh raw)

noncomputable def sigma (vf response vc missing raw : ℝ) : ℝ :=
  Real.sqrt (variance vf response vc missing) * width raw

theorem variance_positive (vf response vc missing : ℝ)
    (hvf : 0 < vf) (hvc : 0 < vc) (hm : 0 ≤ missing) :
    0 < variance vf response vc missing := by
  unfold variance
  positivity

theorem sigma_positive (vf response vc missing raw : ℝ)
    (hvf : 0 < vf) (hvc : 0 < vc) (hm : 0 ≤ missing) :
    0 < sigma vf response vc missing raw := by
  exact mul_pos (Real.sqrt_pos.2 (variance_positive _ _ _ _ hvf hvc hm)) (Real.exp_pos _)

theorem observed_variance (vf response vc : ℝ) :
    variance vf response vc 0 = (1 / 1000000 : ℝ) + vf := by
  simp [variance]

theorem quantile_order (mean left right kl kr : ℝ)
    (hl : 0 < left) (hr : 0 < right) (hkl : 0 ≤ kl) (hkr : 0 ≤ kr) :
    mean - kl * left ≤ mean ∧ mean ≤ mean + kr * right := by
  constructor
  · linarith [mul_nonneg hkl (le_of_lt hl)]
  · linarith [mul_nonneg hkr (le_of_lt hr)]

def routed (p r : Bool) (gp gr gu : ℝ) : ℝ :=
  if p then (if r then gp + (3 / 10 : ℝ) * gr else gp) else gu

theorem mean_supervised (gp gr gu : ℝ) :
    routed true true gp gr gu = gp + (3 / 10 : ℝ) * gr := rfl

theorem other_point (gp gr gu : ℝ) : routed true false gp gr gu = gp := rfl

theorem uncertainty_only (gp gr gu : ℝ) : routed false false gp gr gu = gu := rfl

end PNDS.CoherentConditionalBeliefReal
