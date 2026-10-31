import Mathlib

/-! LANGCONTROL098: fixed-coefficient prefix-supervision contrast.

Only exact-real loss algebra and explicitly assumed scalar differentiation are
proved here. This does not certify IEEE/PyTorch AD, the full inherited model,
resource parity, or learned capability. Counts and masks must be validated by
the executable source before these total real-valued definitions are used.
-/
namespace PNDS.LANGCONTROL098
noncomputable section
open scoped BigOperators

/-- For episode i in domain d, future = (1/2)/n_d/|Y_i| and
prefix = (1/2)/n_d/|P_i|. Padding has no direct loss coefficient. -/
def coefficient (weight domainCases futureTargets prefixTargets lam : ℝ)
    (valid query : Bool) : ℝ :=
  if valid then
    if query then weight / domainCases / futureTargets
    else lam * (weight / domainCases / prefixTargets)
  else 0

/-- Lambda changes prefix supervision, never the direct future coefficient. -/
theorem future_coefficient_fixed (w n y p a b : ℝ) :
    coefficient w n y p a true true = coefficient w n y p b true true := by
  rfl

theorem future_coefficient_value (w n y p lam : ℝ) :
    coefficient w n y p lam true true = w / n / y := by
  rfl

theorem prefix_coefficient_zero (w n y p : ℝ) :
    coefficient w n y p 0 true false = 0 := by
  simp [coefficient]

theorem prefix_coefficient_one (w n y p : ℝ) :
    coefficient w n y p 1 true false = w / n / p := by
  simp [coefficient]

theorem padding_coefficient_zero (w n y p lam : ℝ) (query : Bool) :
    coefficient w n y p lam false query = 0 := by
  rfl

/-- Source accepts only positive counts. Positivity is conditional on those
actual-data premises; a zero denominator is not silently a valid training case. -/
theorem future_coefficient_positive (w n y p lam : ℝ)
    (hw : 0 < w) (hn : 0 < n) (hy : 0 < y) :
    0 < coefficient w n y p lam true true := by
  simp only [coefficient, ite_true]
  exact div_pos (div_pos hw hn) hy

def weightedObjective {length : ℕ} (weights : Fin length → ℝ)
    (losses : Fin length → ℝ) : ℝ := ∑ t, weights t * losses t

/-- The complete contrast is the weighted prefix contribution. There is no
normalization by (1+lambda) or by surviving coefficient mass. -/
def controlledObjective (futureMean prefixMean lam : ℝ) : ℝ :=
  futureMean + lam * prefixMean

theorem query_only (futureMean prefixMean : ℝ) :
    controlledObjective futureMean prefixMean 0 = futureMean := by
  simp [controlledObjective]

theorem prefix_added (futureMean prefixMean : ℝ) :
    controlledObjective futureMean prefixMean 1 = futureMean + prefixMean := by
  simp [controlledObjective]

theorem objective_contrast (futureMean prefixMean a b : ℝ) :
    controlledObjective futureMean prefixMean a -
      controlledObjective futureMean prefixMean b = (a-b)*prefixMean := by
  unfold controlledObjective
  ring

/-- A loss contribution with zero weight cannot directly influence the sum.
This says nothing about indirect state gradients from later future targets. -/
theorem zero_weight_loss_irrelevant {length : ℕ}
    (weights losses : Fin length → ℝ) (index : Fin length) (value : ℝ)
    (zero : weights index = 0) :
    weightedObjective weights (Function.update losses index value) =
      weightedObjective weights losses := by
  unfold weightedObjective
  apply Finset.sum_congr rfl
  intro i _
  by_cases same : i = index
  · subst i
    simp [zero]
  · simp [Function.update, same, Ne.symm same]

/-- Fixed explicit coefficients do not imply fixed AdamW updates or clipping
frequency. This is the scalar derivative before any optimizer transformation. -/
theorem controlled_derivative (future prefix : ℝ → ℝ)
    (theta futureGrad prefixGrad lam : ℝ)
    (hFuture : HasDerivAt future futureGrad theta)
    (hPrefix : HasDerivAt prefix prefixGrad theta) :
    HasDerivAt (fun t => controlledObjective (future t) (prefix t) lam)
      (futureGrad + lam * prefixGrad) theta := by
  simpa [controlledObjective] using hFuture.add (hPrefix.const_mul lam)

end
end PNDS.LANGCONTROL098

#print axioms PNDS.LANGCONTROL098.future_coefficient_fixed
#print axioms PNDS.LANGCONTROL098.future_coefficient_value
#print axioms PNDS.LANGCONTROL098.prefix_coefficient_zero
#print axioms PNDS.LANGCONTROL098.prefix_coefficient_one
#print axioms PNDS.LANGCONTROL098.padding_coefficient_zero
#print axioms PNDS.LANGCONTROL098.future_coefficient_positive
#print axioms PNDS.LANGCONTROL098.query_only
#print axioms PNDS.LANGCONTROL098.prefix_added
#print axioms PNDS.LANGCONTROL098.objective_contrast
#print axioms PNDS.LANGCONTROL098.zero_weight_loss_irrelevant
#print axioms PNDS.LANGCONTROL098.controlled_derivative
