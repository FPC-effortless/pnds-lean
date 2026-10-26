import PNDS.CensoredMeasureRefinement
import Mathlib

/-!
ME108 *source-shaped* actual raw physical CDF refinement to the underlying
latent-measure event, under explicitly positive target-standard-deviation and
valid nonnegative physical threshold assumptions.

This proves the exp/log1p/censor threshold transformation for ANY input
probability measure, and the special zero-censoring atom. It is a step toward
linking rawCDF and torch.special.ndtr, not the final Gaussian CDF/kernel proof.
-/

namespace PNDS.CensoredCDFRefinement
open PNDS.CensoredMeasureRefinement
open PNDS.SensorGraph105
open MeasureTheory

/-- Exactly the source standardization of a physical y>0 threshold. -/
noncomputable def rawThreshold (p : Parameters) (c : Fin 6) (y : ℝ) : ℝ :=
  (Real.log (1 + y) - p.targetMean c) / p.targetStd c

/-- Censored exponential observation is at most y precisely when the latent
logarithmic state is at most the explicit log1p standardized threshold. -/
theorem physicalDraw_le_iff_latent_le (p : Parameters) (f : Forecast)
    (c : Fin 6) (u y : ℝ)
    (hstd : 0 < p.targetStd c) (hy : 0 ≤ y) :
    physicalDraw p f c u ≤ y ↔
      splitAffine f c u ≤ rawThreshold p c y := by
  have hpos : 0 < 1 + y := by linarith
  have hmax :
      max 0 (Real.exp (p.targetMean c +
        p.targetStd c * splitAffine f c u) - 1) ≤ y ↔
      Real.exp (p.targetMean c +
        p.targetStd c * splitAffine f c u) ≤ 1 + y := by
    constructor
    · intro h
      have hr := le_trans
        (le_max_right (0 : ℝ)
          (Real.exp (p.targetMean c +
            p.targetStd c * splitAffine f c u) - 1)) h
      linarith
    · intro h
      apply max_le (by linarith) (by linarith)
  have hexp :
      Real.exp (p.targetMean c +
        p.targetStd c * splitAffine f c u) ≤ 1 + y ↔
        p.targetMean c + p.targetStd c * splitAffine f c u ≤
          Real.log (1 + y) := by
    constructor
    · intro h
      have h' :
          Real.exp (p.targetMean c +
            p.targetStd c * splitAffine f c u) ≤
              Real.exp (Real.log (1 + y)) := by
        simpa only [Real.exp_log hpos] using h
      exact Real.exp_le_exp.mp h'
    · intro h
      have h' := Real.exp_le_exp.mpr h
      simpa only [Real.exp_log hpos] using h'
  have hlinear :
      p.targetMean c + p.targetStd c * splitAffine f c u ≤
        Real.log (1 + y) ↔
      splitAffine f c u ≤ rawThreshold p c y := by
    unfold rawThreshold
    rw [le_div_iff₀ hstd]
    constructor <;> intro h <;> nlinarith
  unfold physicalDraw censorRaw
  exact hmax.trans (hexp.trans hlinear)

/-- A first genuinely semantic probability law equality: the ME108 source
raw-value distribution's CDF equals its latent Gaussian-base probability
of falling under the exact logged physical threshold. -/
theorem rawLaw_cdf_eq_latent_event (μ : Measure ℝ)
    (p : Parameters) (f : Forecast) (c : Fin 6) (y : ℝ)
    (hstd : 0 < p.targetStd c) (hy : 0 ≤ y) :
    rawLaw μ p f c (Set.Iic y) =
      μ {u : ℝ | splitAffine f c u ≤ rawThreshold p c y} := by
  rw [rawLaw, Measure.map_apply (physicalDraw_measurable p f c)
    measurableSet_Iic]
  congr 1
  ext u
  simp only [Set.mem_preimage, Set.mem_Iic, Set.mem_setOf_eq]
  exact physicalDraw_le_iff_latent_le p f c u y hstd hy

/-- The zero-censoring point mass is exactly the CDF at the origin because
physical outputs never take negative values. This includes the model's
mixed discrete-continuous distribution, not just its density. -/
theorem rawLaw_zero_atom_eq_cdf (μ : Measure ℝ)
    (p : Parameters) (f : Forecast) (c : Fin 6) :
    rawLaw μ p f c {0} = rawLaw μ p f c (Set.Iic 0) := by
  have hsets :
      ((physicalDraw p f c) ⁻¹' ({0} : Set ℝ)) =
      ((physicalDraw p f c) ⁻¹' Set.Iic 0) := by
    ext u
    have hn := physicalDraw_nonnegative p f c u
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_Iic]
    exact ⟨fun h => le_of_eq h, fun h => le_antisymm h hn⟩
  simp only [rawLaw]
  rw [Measure.map_apply (physicalDraw_measurable p f c)
        (MeasurableSet.singleton 0),
      Measure.map_apply (physicalDraw_measurable p f c)
        measurableSet_Iic]
  rw [hsets]

/-- The atom at 0 matches a concrete latent event with standardized threshold
equal to -source_mean/source_standard_deviation. This is the exact special
boundary the source handles using log_ndtr, without yet equating its bits to
Mathlib's normal CDF. -/
theorem rawLaw_zero_atom_eq_latent_event (μ : Measure ℝ)
    (p : Parameters) (f : Forecast) (c : Fin 6)
    (hstd : 0 < p.targetStd c) :
    rawLaw μ p f c {0} =
      μ {u : ℝ | splitAffine f c u ≤
        -p.targetMean c / p.targetStd c} := by
  rw [rawLaw_zero_atom_eq_cdf]
  rw [rawLaw_cdf_eq_latent_event μ p f c 0 hstd (le_refl 0)]
  simp only [rawThreshold, add_zero, Real.log_one, zero_sub]

end PNDS.CensoredCDFRefinement
