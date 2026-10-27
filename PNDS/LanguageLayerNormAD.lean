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
  convert hscaled using 1
  · funext z
    change (∑ i : Fin 248, path z i) / (248:ℝ) =
      (1/(248:ℝ)) * (∑ i : Fin 248, path z i)
    ring
  · change (∑ i : Fin 248, velocity i) / (248:ℝ) =
      (1/(248:ℝ)) * (∑ i : Fin 248, velocity i)
    ring

/-- This is the exact mean-centered 248-scalar LayerNorm variance JVP:
    all input-coordinate perturbations and mean coupling are retained. -/
def variance248Slope (x dx : V 248) : ℝ :=
  (∑ i : Fin 248, 2 * (x i - mean248 x) *
      (dx i - mean248 dx)) / 248

theorem variance248_hasDerivAt
    (path : ℝ → V 248) (velocity : V 248) (t : ℝ)
    (h : ∀ i, HasDerivAt (fun z => path z i) (velocity i) t) :
    HasDerivAt (fun z => variance248 (path z))
      (variance248Slope (path t) velocity) t := by
  have hm := mean248_hasDerivAt path velocity t h
  have hc (i : Fin 248) :
      HasDerivAt (fun z => path z i - mean248 (path z))
        (velocity i - mean248 velocity) t :=
    (h i).sub hm
  have hsq (i : Fin 248) :
      HasDerivAt (fun z => (path z i - mean248 (path z))^2)
        (2 * (path t i - mean248 (path t)) *
          (velocity i - mean248 velocity)) t := by
    have hmultiplication := (hc i).mul (hc i)
    convert hmultiplication using 1
    · funext z
      ring
    · ring
  have hs := HasDerivAt.sum (u := Finset.univ) (fun i _ => hsq i)
  have scaled := hs.const_mul (1 / 248 : ℝ)
  convert scaled using 1
  · funext z
    change (∑ i : Fin 248, (path z i - mean248 (path z))^2) / (248:ℝ) =
      (1/(248:ℝ)) * (∑ i : Fin 248, (path z i - mean248 (path z))^2)
    ring
  · change (∑ i : Fin 248, 2 * (path t i - mean248 (path t)) *
          (velocity i - mean248 velocity)) / (248:ℝ) =
      (1/(248:ℝ)) * (∑ i : Fin 248, 2 * (path t i - mean248 (path t)) *
          (velocity i - mean248 velocity))
    ring

/-- The FULL source LayerNorm output-channel JVP now explicitly incorporates
    input, all 248 mean channels, all 248 variance channels, scale, epsilon
    and affine shift. No detached normalization statistics are permitted. -/
theorem norm248_coordinate_hasDerivAt
    (p : Parameters) (path : ℝ → V 248) (velocity : V 248)
    (t : ℝ) (i : Fin 248)
    (h : ∀ j, HasDerivAt (fun z => path z j) (velocity j) t)
    (heps : 0 < p.normEps) :
    HasDerivAt (fun z => norm248 p (path z) i)
      (p.normScale i * (((velocity i - mean248 velocity) *
        Real.sqrt (variance248 (path t) + p.normEps) -
       (path t i - mean248 (path t)) *
         (variance248Slope (path t) velocity /
           (2*Real.sqrt (variance248 (path t) + p.normEps)))) /
       (Real.sqrt (variance248 (path t) + p.normEps))^2)) t := by
  have hm := mean248_hasDerivAt path velocity t h
  have hv := variance248_hasDerivAt path velocity t h
  have hvar := variance248_nonnegative (path t)
  have core := normalized_feature_hasDerivAt
    (fun z => path z i)
    (fun z => mean248 (path z))
    (fun z => variance248 (path z))
    (velocity i) (mean248 velocity)
    (variance248Slope (path t) velocity)
    t (p.normScale i) (p.normShift i) p.normEps
    (h i) hm hv hvar heps
  have hfunction :
      (fun z => norm248 p (path z) i) =
      (fun z => p.normScale i * ((path z i - mean248 (path z)) /
        Real.sqrt (variance248 (path z)+p.normEps)) + p.normShift i) := by
    funext z
    change ((path z i - mean248 (path z)) /
      Real.sqrt (variance248 (path z)+p.normEps)) * p.normScale i +
        p.normShift i =
      p.normScale i * ((path z i - mean248 (path z)) /
        Real.sqrt (variance248 (path z)+p.normEps)) + p.normShift i
    ring
  rw [hfunction]
  exact core

end
end PNDS.LanguageLayerNormAD
