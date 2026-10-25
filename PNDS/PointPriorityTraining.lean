import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import PNDS.SensorGraph105

/-!
ME106: point-priority training contract layered over the unchanged ME105 graph.
The projection results below concern exact finite real vectors only. They do
not establish optimizer-step descent or a refinement to PyTorch autograd.
-/
namespace PNDS.PointPriorityTraining

open PNDS.SensorGraph105

abbrev Grad := Fin trainableParameterCount → ℝ

structure PointPrioritySensorWorld where
  parameters : Parameters

/-- The ME105 typed forward graph is called verbatim; this wrapper adds only
training roles and has no inference-side graph change. -/
noncomputable def forwardWorld {T : ℕ} (world : PointPrioritySensorWorld)
    (o : Observation T) (last : Fin T) (hlast : ∀ i : Fin T, i ≤ last) : Forecast :=
  SensorGraph105.forward world.parameters o last hlast

theorem forwardWorld_eq_inherited {T : ℕ} (world : PointPrioritySensorWorld)
    (o : Observation T) (last : Fin T) (hlast : ∀ i : Fin T, i ≤ last) :
    forwardWorld world o last hlast =
      SensorGraph105.forward world.parameters o last hlast := rfl

structure LossTerms where
  pointWeight : ℝ
  normalizedRawMedianMAE : ℝ
  properNLL : ℝ
  locationMSE : ℝ
  maskedCurrentReconNLL : ℝ
  properIntervalScore : ℝ

noncomputable def uncertaintyObjective (L : LossTerms) : ℝ :=
  L.properNLL + (1/10:ℝ)*L.locationMSE +
    (3/10:ℝ)*L.maskedCurrentReconNLL + (1/4:ℝ)*L.properIntervalScore

noncomputable def pointObjective (L : LossTerms) : ℝ :=
  L.pointWeight * L.normalizedRawMedianMAE

noncomputable def totalObjective (L : LossTerms) : ℝ :=
  pointObjective L + uncertaintyObjective L

/- The point role is the weighted point-priority normalized raw median MAE.
Its normalization denominator, masking convention, and real source evaluator
are intentionally implementation/source obligations, not defined anew here. -/

def dot (x y : Grad) : ℝ := ∑ i, x i * y i
def normSq (x : Grad) : ℝ := dot x x
def sub (x y : Grad) : Grad := fun i => x i - y i
def add (x y : Grad) : Grad := fun i => x i + y i
def scale (a : ℝ) (x : Grad) : Grad := fun i => a * x i

noncomputable def projectedUncertainty (gP gU : Grad) : Grad :=
  if dot gP gU < 0 ∧ normSq gP > 0 then
    sub gU (scale (dot gP gU / normSq gP) gP)
  else gU

noncomputable def totalGradient (gP gU : Grad) : Grad := add gP (projectedUncertainty gP gU)

theorem dot_sub_projection_zero (x y : Grad) (h : normSq x ≠ 0) :
    dot x (sub y (scale (dot x y / normSq x) x)) = 0 := by
  simp only [dot, sub, scale]
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  have hs : (∑ i, x i * (dot x y / normSq x * x i)) =
      (dot x y / normSq x) * normSq x := by
    calc
      _ = ∑ i, (dot x y / normSq x) * (x i * x i) := by
        apply Finset.sum_congr rfl
        intro i hi
        ring
      _ = (dot x y / normSq x) * normSq x := by
        rw [← Finset.mul_sum]
        rfl
  have hs' : (∑ i, x i * ((∑ j, x j * y j) / normSq x * x i)) =
      ((∑ j, x j * y j) / normSq x) * normSq x := by
    simpa [dot] using hs
  rw [hs']
  unfold normSq dot at h ⊢
  field_simp

theorem normSq_nonneg (x : Grad) : normSq x ≥ 0 := by
  unfold normSq dot
  apply Finset.sum_nonneg
  intro i hi
  exact mul_self_nonneg _

theorem projected_dot_nonneg (gP gU : Grad) (hn : normSq gP > 0) :
    0 ≤ dot gP (projectedUncertainty gP gU) := by
  by_cases hd : dot gP gU < 0
  · have hzero := dot_sub_projection_zero gP gU (ne_of_gt hn)
    simp [projectedUncertainty, hd, hn, hzero]
  · simp [projectedUncertainty, hd]
    exact le_of_not_gt hd

theorem total_dot_lower_bound (gP gU : Grad) (hn : normSq gP > 0) :
    normSq gP ≤ dot gP (totalGradient gP gU) := by
  have hp := projected_dot_nonneg gP gU hn
  simp only [totalGradient, add, dot, normSq]
  simp_rw [mul_add]
  rw [Finset.sum_add_distrib]
  have hp' : 0 ≤ ∑ i, gP i * projectedUncertainty gP gU i := hp
  linarith [hp']

theorem zero_priority_gradient (gU : Grad) :
    projectedUncertainty (fun _ => 0) gU = gU := by
  simp [projectedUncertainty, dot, normSq]

theorem nonconflicting_gradient_unchanged (gP gU : Grad)
    (h : 0 ≤ dot gP gU) : projectedUncertainty gP gU = gU := by
  simp [projectedUncertainty, not_lt_of_ge h]

theorem zero_norm_gradient_unchanged (gP gU : Grad)
    (h : normSq gP = 0) : projectedUncertainty gP gU = gU := by
  simp [projectedUncertainty, h]

theorem width_coordinate_unchanged (gP gU : Grad) (i : Fin trainableParameterCount)
    (hi : gP i = 0) : projectedUncertainty gP gU i = gU i := by
  by_cases h : dot gP gU < 0 ∧ normSq gP > 0
  · simp [projectedUncertainty, h, sub, scale, hi]
  · simp [projectedUncertainty, h]

/-- Snapshot correction models the value of a frozen model snapshot. -/
def uncertaintyLocation (mu medianCorrection snapshotCorrection : ℝ) : ℝ :=
  mu - (medianCorrection - snapshotCorrection)

theorem uncertainty_location_snapshot_identity (mu correction : ℝ) :
    uncertaintyLocation mu correction correction = mu := by
  simp [uncertaintyLocation]

/-- Custom-AD tangent of a detached value. The primal value is the snapshot;
its tangent is zero by the declared stop-gradient rule. -/
def stopGradientTangent (_tangent : ℝ) : ℝ := 0

/-- The tangent of `correction - correction.detach()` is `dc`, since the
detached branch contributes zero tangent. -/
def detachedDifferenceTangent (correctionTangent : ℝ) : ℝ :=
  correctionTangent - stopGradientTangent correctionTangent

theorem detached_difference_tangent (tangent : ℝ) :
    detachedDifferenceTangent tangent = tangent := by
  simp [detachedDifferenceTangent, stopGradientTangent]

/-- Tangent of the uncertainty expression, given the tangent of `mu` and the
live correction. Cancellation is compositional: `mu` must include `dc`. -/
def uncertaintyLocationTangent (muTangent correctionTangent : ℝ) : ℝ :=
  muTangent - detachedDifferenceTangent correctionTangent

theorem uncertainty_location_tangent_cancels
    (baseTangent correctionTangent : ℝ) :
    uncertaintyLocationTangent (baseTangent + correctionTangent)
      correctionTangent = baseTangent := by
  simp [uncertaintyLocationTangent, detached_difference_tangent]

/- The gradient losses P, U and source graph execution are provided by the
implementing training system. Their source correspondence is OPEN in the
evidence record; this module formalizes only the finite-vector update rule. -/

theorem inherited_parameter_count : trainableParameterCount = 100349 :=
  PNDS.SensorGraph105.trainableParameterCount_exact

end PNDS.PointPriorityTraining
