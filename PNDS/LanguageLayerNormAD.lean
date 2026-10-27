import PNDS.LanguageLayerNorm
import PNDS.LanguageActivationAD
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
TAC-LANG-AD-013: compositional derivatives for the exact 248-feature
LayerNorm. Positive epsilon is a necessary analytic precondition. This
concerns the real formula, NOT the exact rounding of torch.layer_norm or its
backward kernel.
-/
namespace PNDS.LanguageLayerNormAD
noncomputable section
open PNDS.LanguageLayerNorm PNDS.LanguageFullGraph
open scoped Topology

/-- The scalar core of a normalized feature with independently differentiable
input, mean and variance, for any positive stabilizing epsilon. -/
theorem normalized_feature_hasDerivAt
    (input mean variance : ℝ → ℝ) (din dmean dvar t scale shift eps : ℝ)
    (hi : HasDerivAt input din t)
    (hm : HasDerivAt mean dmean t)
    (hv : HasDerivAt variance dvar t)
    (hvariance : 0 ≤ variance t) (heps : 0 < eps) :
    HasDerivAt
      (fun z => scale * ((input z - mean z) /
          Real.sqrt (variance z + eps)) + shift)
      (scale * (((din - dmean) * Real.sqrt (variance t + eps) -
        (input t - mean t) *
          (dvar / (2*Real.sqrt (variance t + eps)))) /
            (Real.sqrt (variance t + eps))^2)) t := by
  have harg : variance t + eps ≠ 0 := by linarith
  have hden := ((hv.add_const eps).sqrt harg)
  have hdpos : 0 < Real.sqrt (variance t + eps) := by
    exact Real.sqrt_pos.2 (by linarith)
  have hnum := hi.sub hm
  have hdiv := hnum.div hden (ne_of_gt hdpos)
  have hout := (hdiv.const_mul scale).add_const shift
  convert hout using 1 <;> ring

/-- All 248 features contribute to the mean derivative; no detached
mean, missing bias, or substitution of batch normalization is allowed. -/
theorem mean248_hasDerivAt (path : ℝ → V 248) (velocity : V 248)
    (t : ℝ)
    (h : ∀ i, HasDerivAt (fun z => path z i) (velocity i) t) :
    HasDerivAt (fun z => mean248 (path z)) (mean248 velocity) t := by
  have hs := HasDerivAt.sum (u := Finset.univ) (fun i _ => h i)
  have hscaled := hs.const_mul (1/248 : ℝ)
  convert hscaled using 1 <;> simp [mean248, div_eq_mul_inv]

end
end PNDS.LanguageLayerNormAD
