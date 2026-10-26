import PNDS.RoleSpecializedFFN
import Mathlib

/-!
# ME108: censored split-normal as a probability-measure pushforward

This module builds a genuine probability measure from any *supplied* probability
measure on standardized residuals. In particular, a bona fide standard normal
probability measure yields a normalized censored, asymmetric raw-value law.

This is more than an unproved NormalLaw record: it establishes total mass one
and zero probability below the source's hard zero censoring boundary.

OPEN: connect the chosen standard normal measure to torch.special.ndtr/log_ndtr,
prove that its CDF matches the source's exact side-dependent rawCDF expression,
quantile inversion, atom at zero formula, Jacobian and complete IEEE computation.
Those links are NOT postulated or asserted by the theorems below.
-/

namespace PNDS.CensoredMeasureRefinement
open PNDS.RoleSpecializedFFN
open PNDS.SensorGraph105
open MeasureTheory

/-- Source ME108 split half-lines, with the left side selected below zero
standardized-normal residual. -/
noncomputable def splitAffine (f : Forecast) (c : Fin 6) (u : ℝ) : ℝ :=
  if u < 0 then f.location c + f.leftScale c * u
  else f.location c + f.rightScale c * u

/-- Actual raw-unit exp/log1p inverse and physical zero censoring. -/
noncomputable def censorRaw (p : Parameters) (z : ℝ) (c : Fin 6) : ℝ :=
  max 0 (Real.exp (p.targetMean c + p.targetStd c * z) - 1)

/-- One standardized base draw maps to the recorded nonnegative sensor value. -/
noncomputable def physicalDraw (p : Parameters) (f : Forecast) (c : Fin 6)
    (u : ℝ) : ℝ :=
  censorRaw p (splitAffine f c u) c

theorem splitAffine_measurable (f : Forecast) (c : Fin 6) :
    Measurable (splitAffine f c) := by
  unfold splitAffine
  exact Measurable.ite (measurableSet_lt measurable_id measurable_const)
    (by fun_prop) (by fun_prop)

theorem censorRaw_measurable (p : Parameters) (c : Fin 6) :
    Measurable (fun z : ℝ => censorRaw p z c) := by
  unfold censorRaw
  fun_prop

theorem physicalDraw_measurable (p : Parameters) (f : Forecast) (c : Fin 6) :
    Measurable (physicalDraw p f c) := by
  exact (censorRaw_measurable p c).comp (splitAffine_measurable f c)

/-- Censoring guarantees actual draw nonnegativity independently of the
pre-censoring source residual. -/
theorem physicalDraw_nonnegative (p : Parameters) (f : Forecast)
    (c : Fin 6) (u : ℝ) : 0 ≤ physicalDraw p f c u := by
  unfold physicalDraw censorRaw
  exact le_max_left _ _

/-- A mathematical probability law induced by the full source typed Forecast.
This is a measure transformation, not a claim about torch.ndtr bit patterns. -/
noncomputable def rawLaw (μ : Measure ℝ) (p : Parameters)
    (f : Forecast) (c : Fin 6) : Measure ℝ :=
  Measure.map (physicalDraw p f c) μ

/-- The pushforward of a probability law has total probability one:
this formalizes nontrivial normalization for censoring and asymmetric scales. -/
theorem rawLaw_total_mass (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (p : Parameters) (f : Forecast) (c : Fin 6) :
    rawLaw μ p f c Set.univ = 1 := by
  simp [rawLaw, Measure.map_apply (physicalDraw_measurable p f c)]

/-- The induced law assigns no mass to values strictly below physical zero. -/
theorem rawLaw_no_negative_mass (μ : Measure ℝ) (p : Parameters)
    (f : Forecast) (c : Fin 6) :
    rawLaw μ p f c (Set.Iio (0 : ℝ)) = 0 := by
  rw [rawLaw, Measure.map_apply (physicalDraw_measurable p f c) measurableSet_Iio]
  have hpre : (physicalDraw p f c) ⁻¹' Set.Iio (0 : ℝ) = ∅ := by
    ext u
    have hn := physicalDraw_nonnegative p f c u
    simp [Set.mem_preimage, Set.mem_Iio, not_lt.mpr hn]
  simp [hpre]


/-- Non-conditional normalization for the actual mathematically defined
standard normal law N(0,1): Mathlib supplies its genuine probability measure,
rather than assuming an arbitrary caller-defined cdf or mass axiom. -/
theorem standard_gaussian_rawLaw_total_mass (p : Parameters)
    (f : Forecast) (c : Fin 6) :
    rawLaw (ProbabilityTheory.gaussianReal 0 1) p f c Set.univ = 1 := by
  exact rawLaw_total_mass (ProbabilityTheory.gaussianReal 0 1) p f c

/-- The corresponding actual standard-normal pushforward has zero probability
on negative physical sensor values despite asymmetric widths. -/
theorem standard_gaussian_rawLaw_no_negative_mass (p : Parameters)
    (f : Forecast) (c : Fin 6) :
    rawLaw (ProbabilityTheory.gaussianReal 0 1) p f c
      (Set.Iio (0 : ℝ)) = 0 :=
  rawLaw_no_negative_mass _ p f c

end PNDS.CensoredMeasureRefinement
