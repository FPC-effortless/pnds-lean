import PNDS.LanguageFullGraph
import Mathlib

/-!
TAC-LANG-NORM-007. A total exact-real Float-to-Torch refinement REQUIRES
positive LayerNorm epsilon, not merely a real-valued learned graph.
These proofs only concern denominator well-definedness over exact reals;
Torch's epsilon, reductions, sqrt and rsqrt must still be independently
certified for FP32/FP64 kernels and backpropagation.
-/
namespace PNDS.LanguageLayerNorm

noncomputable section
open PNDS.LanguageFullGraph

def mean248 (x : V 248) : ℝ := (∑ i : Fin 248, x i)/248
def variance248 (x : V 248) : ℝ :=
  (∑ i : Fin 248, (x i - mean248 x)^2)/248

theorem variance248_nonnegative (x : V 248) : 0 ≤ variance248 x := by
  unfold variance248
  apply div_nonneg
  · exact Finset.sum_nonneg (fun i _ => sq_nonneg _)
  · norm_num

theorem layernorm_denominator_positive (x : V 248) (eps : ℝ)
    (heps : 0 < eps) :
    0 < Real.sqrt (variance248 x + eps) := by
  apply Real.sqrt_pos.2
  have hv := variance248_nonnegative x
  linarith

/-- Per the actual Torch default, 1e-5 ensures the real-valued
    square-root argument is strictly positive for all model activations. -/
theorem deployed_epsilon_denominator_positive (x : V 248) :
    0 < Real.sqrt (variance248 x + (1/100000 : ℝ)) := by
  exact layernorm_denominator_positive x (1/100000) (by norm_num)

end
end PNDS.LanguageLayerNorm
