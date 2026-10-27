import PNDS.LanguageFullGraph
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
TAC-LANG-AD-012: exact-real backward primitives actually used by the
source's two-head SiLU modulated write gate and residual head.

Theorems concern smooth real scalar operations; they neither prove Torch
floating kernels nor a full architecture-to-Lean AD refinement.
-/
namespace PNDS.LanguageActivationAD
noncomputable section
open PNDS.SourceAlignedModel
open Filter Set
open scoped Topology

def sigmoidSlope (x : ℝ) : ℝ :=
  Real.exp (-x) / (1 + Real.exp (-x))^2

/-- Stable sigmoid has a classical derivative at every finite real input. -/
theorem sigmoid_hasDerivAt (x : ℝ) :
    HasDerivAt sig (sigmoidSlope x) x := by
  have h₁ : HasDerivAt (fun z : ℝ => Real.exp (-z))
      (-Real.exp (-x)) x := by
    convert ((hasDerivAt_id x).neg.exp) using 1 <;> simp only [id_eq] <;> ring
  have h₂ : HasDerivAt (fun z : ℝ => 1 + Real.exp (-z))
      (-Real.exp (-x)) x := by
    convert ((hasDerivAt_const x (1 : ℝ)).add h₁) using 1 <;> ring
  have hd : (1 + Real.exp (-x)) ≠ 0 := by
    have hp := Real.exp_pos (-x)
    linarith
  have h₃ := h₂.inv hd
  convert h₃ using 1 <;> simp [sig, sigmoidSlope, one_div, id_eq] <;> ring

def siluSlope (x : ℝ) : ℝ := sig x + x * sigmoidSlope x

theorem silu_hasDerivAt (x : ℝ) :
    HasDerivAt silu (siluSlope x) x := by
  have h := (hasDerivAt_id x).mul (sigmoid_hasDerivAt x)
  convert h using 1 <;> simp [silu, siluSlope, id_eq] <;> ring

/-- The exact source SiLU modulation path applies SiLU to 2x before
    bounded sigmoid; the factor 2 in the chain rule is material. -/
theorem scaled_silu_hasDerivAt (x : ℝ) :
    HasDerivAt (fun z : ℝ => silu (2*z))
      (2 * siluSlope (2*x)) x := by
  have h : HasDerivAt (fun z : ℝ => (2:ℝ)*z) 2 x := by
    convert (hasDerivAt_id x).const_mul 2 using 1 <;> ring
  convert (silu_hasDerivAt (2*x)).comp x h using 1 <;> ring

/-- Product of learned SiLU source modifier and normal sigmoid derivative.
    This is exactly the slope of the un-routed scalar source write gate. -/
theorem gated_silu_modulation_hasDerivAt (base : ℝ) (x : ℝ) :
    HasDerivAt (fun z : ℝ => sig (base + silu (2*z)))
      (sigmoidSlope (base + silu (2*x)) * (2*siluSlope (2*x))) x := by
  have hinner : HasDerivAt (fun z : ℝ => base + silu (2*z))
      (2*siluSlope (2*x)) x := by
    convert (hasDerivAt_const x base).add (scaled_silu_hasDerivAt x) using 1 <;> ring
  exact (sigmoid_hasDerivAt (base+silu (2*x))).comp x hinner

end
end PNDS.LanguageActivationAD
