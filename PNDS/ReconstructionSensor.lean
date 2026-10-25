import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import PNDS.BeliefSensor

/-!
ME102 exact-real scalar contract for the ReconstructionSensor current-value
head and its missing-current contribution to future uncertainty.

This file formalizes the stated scalar equations under explicit sign and
missingness assumptions. It does not formalize the 94-to-12 neural network,
the fitted source/evidence encoder, Gaussian-NLL training, finite precision,
or any empirical coverage property.
-/
namespace PNDS.ReconstructionSensor

/-- `a + 2*t` is the current reconstruction head in standardized log units;
`t` represents the `tanh` output of the learned head. -/
def currentMean (anchor tanhOutput : ℝ) : ℝ := anchor + 2 * tanhOutput

/-- The learned nonnegative head scale squared, plus inherited latent-state
variance. The constant is the fixed variance floor `0.0025`. -/
noncomputable def currentVariance (softplusOutput latentStateVariance : ℝ) : ℝ :=
  (1 / 400 : ℝ) + softplusOutput ^ 2 + latentStateVariance

/-- Missingness is represented by a real mask so the zero and one cases can be
stated directly. In the implementation this is a Boolean mask cast to float. -/
noncomputable def futureMean (oldMean response anchor reconstructedMean missing : ℝ) : ℝ :=
  oldMean + response * (reconstructedMean - anchor) * missing

/-- Missing-channel future variance is floored by the measured 24-hour
FIT-difference rate times source age. `futureNoise` is the inherited old
future standard deviation; all quantities are in standardized log units. -/
noncomputable def futureVariance (futureNoise response currentVar missing floorRate ageHours : ℝ) : ℝ :=
  max (futureNoise ^ 2 + response ^ 2 * currentVar * missing)
    (missing * floorRate * ageHours / 24)

theorem currentVariance_nonneg (softplusOutput latentStateVariance : ℝ)
    (hlatent : 0 ≤ latentStateVariance) :
    0 ≤ currentVariance softplusOutput latentStateVariance := by
  unfold currentVariance
  positivity

theorem currentVariance_at_least_fixed_floor (softplusOutput latentStateVariance : ℝ)
    (hlatent : 0 ≤ latentStateVariance) :
    (1 / 400 : ℝ) ≤ currentVariance softplusOutput latentStateVariance := by
  unfold currentVariance
  nlinarith [sq_nonneg softplusOutput]

theorem futureVariance_at_least_floor (futureNoise response currentVar missing floorRate ageHours : ℝ) :
    missing * floorRate * ageHours / 24 ≤
      futureVariance futureNoise response currentVar missing floorRate ageHours := by
  unfold futureVariance
  exact le_max_right _ _

theorem futureVariance_at_least_inherited_noise
    (futureNoise response currentVar missing floorRate ageHours : ℝ)
    (hresponse : 0 ≤ response ^ 2 * currentVar * missing) :
    futureNoise ^ 2 ≤ futureVariance futureNoise response currentVar missing floorRate ageHours := by
  unfold futureVariance
  have hfirst : futureNoise ^ 2 ≤ futureNoise ^ 2 + response ^ 2 * currentVar * missing := by
    linarith
  exact le_trans hfirst (le_max_left _ _)

theorem futureVariance_positive_of_nonzero_noise
    (futureNoise response currentVar missing floorRate ageHours : ℝ)
    (hresponse : 0 ≤ response ^ 2 * currentVar * missing)
    (hnoise : futureNoise ≠ 0) :
    0 < futureVariance futureNoise response currentVar missing floorRate ageHours := by
  have hsquare : 0 < futureNoise ^ 2 := sq_pos_of_ne_zero hnoise
  have hbound := futureVariance_at_least_inherited_noise futureNoise response currentVar
    missing floorRate ageHours hresponse
  exact lt_of_lt_of_le hsquare hbound

theorem futureMean_unchanged_when_observed
    (oldMean response anchor reconstructedMean : ℝ) :
    futureMean oldMean response anchor reconstructedMean 0 = oldMean := by
  unfold futureMean
  ring

theorem futureVariance_unchanged_when_observed
    (futureNoise response currentVar floorRate ageHours : ℝ) :
    futureVariance futureNoise response currentVar 0 floorRate ageHours = futureNoise ^ 2 := by
  unfold futureVariance
  simp [futureVariance, max_eq_left (sq_nonneg futureNoise)]

theorem ageFloor_monotone (missing floorRate age1 age2 : ℝ)
    (hmissing : 0 ≤ missing) (hrate : 0 ≤ floorRate) (hage : age1 ≤ age2) :
    missing * floorRate * age1 / 24 ≤ missing * floorRate * age2 / 24 := by
  have hcoef : 0 ≤ missing * floorRate / 24 := by positivity
  have hmul := mul_le_mul_of_nonneg_left hage hcoef
  calc
    missing * floorRate * age1 / 24 = (missing * floorRate / 24) * age1 := by ring
    _ ≤ (missing * floorRate / 24) * age2 := hmul
    _ = missing * floorRate * age2 / 24 := by ring

theorem futureVariance_monotone_age
    (futureNoise response currentVar missing floorRate age1 age2 : ℝ)
    (hmissing : 0 ≤ missing) (hrate : 0 ≤ floorRate) (hage : age1 ≤ age2) :
    futureVariance futureNoise response currentVar missing floorRate age1 ≤
      futureVariance futureNoise response currentVar missing floorRate age2 := by
  unfold futureVariance
  exact max_le_max (le_refl _) (ageFloor_monotone missing floorRate age1 age2
    hmissing hrate hage)

/-- A sensor read is an identity query over the inherited source/memory trace,
as in ME101. This theorem says the current-value adapter alone does not write
to that trace. -/
def readTrace (trace : ℝ × ℝ) (_currentMean _currentVariance : ℝ) : ℝ × ℝ := trace

theorem readTrace_preserves_trace (trace : ℝ × ℝ) (currentMean currentVariance : ℝ) :
    readTrace trace currentMean currentVariance = trace := rfl

/-!
OPEN obligations: source/code correspondence for the 94-input layout and
12-output Linear head; Gaussian-NLL objective and artificial-block-mask
training path; FIT floor-rate construction from authentic training records;
full-model forward/backward equivalence; finite-precision behavior; calibration
and coverage; and source/memory invariants for the complete PyTorch graph.
-/

end PNDS.ReconstructionSensor
