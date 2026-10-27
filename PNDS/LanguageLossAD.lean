import PNDS.LanguageCategorical
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
TAC-LANG-AD-009: exact-real derivative of the complete 128-way categorical
negative-log-likelihood along ALL 128 logit directions simultaneously.

The result provides the true final decoder cotangent, including the tied
historical decoder. It does NOT yet close backpropagation through every
language layer, PyTorch autograd, or FP32.
-/
namespace PNDS.LanguageLossAD
noncomputable section

open PNDS.LanguageCategorical
open scoped Topology

def logitVelocity (scores velocity : Scores) : ℝ :=
  ∑ j : Fin 128, probability scores j * velocity j

/-- Any differentiable path through 128 logits has a differentiable partition
function. Finite sum/chain rule is kernel checked, with no learned assumptions. -/
theorem partition_hasDerivAt (path : Fin 128 → ℝ → ℝ)
    (velocity : Scores) (t : ℝ)
    (h : ∀ i, HasDerivAt (path i) (velocity i) t) :
    HasDerivAt (fun z => Z (fun i => path i z))
      (∑ i : Fin 128, Real.exp (path i t) * velocity i) t := by
  simpa only [Z] using
    (HasDerivAt.sum (u := Finset.univ) (fun i _ => (h i).exp))

/-- Exact mathematical gradient through a stable logarithmic normalizer.
The positive partition provides the required log-domain side condition. -/
theorem log_partition_hasDerivAt (path : Fin 128 → ℝ → ℝ)
    (velocity : Scores) (t : ℝ)
    (h : ∀ i, HasDerivAt (path i) (velocity i) t) :
    HasDerivAt (fun z => Real.log (Z (fun i => path i z)))
      ((∑ i : Fin 128, Real.exp (path i t) * velocity i) /
        Z (fun i => path i t)) t := by
  exact (partition_hasDerivAt path velocity t h).log
    (ne_of_gt (Z_pos (fun i => path i t)))

theorem logitVelocity_as_fraction (scores velocity : Scores) :
    logitVelocity scores velocity =
      (∑ i : Fin 128, Real.exp (scores i) * velocity i) / Z scores := by
  unfold logitVelocity probability
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The full 128-class NLL has the exact reverse-mode terminal covector
softmax(scores) - one_hot(target). This holds along arbitrary curves of
all logits, not just perturbations of the single correct-class logit. -/
theorem nll_hasDerivAt (path : Fin 128 → ℝ → ℝ)
    (velocity : Scores) (target : Fin 128) (t : ℝ)
    (h : ∀ i, HasDerivAt (path i) (velocity i) t) :
    HasDerivAt (fun z => nll (fun i => path i z) target)
      (logitVelocity (fun i => path i t) velocity - velocity target) t := by
  have hraw := (log_partition_hasDerivAt path velocity t h).sub (h target)
  have heq : (fun z => nll (fun i => path i z) target) =
      (fun z => Real.log (Z (fun i => path i z)) - path target z) := by
    funext z
    exact nll_equals_log_partition_minus_logit (fun i => path i z) target
  rw [heq, logitVelocity_as_fraction]
  exact hraw

end
end PNDS.LanguageLossAD
