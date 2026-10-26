import Mathlib

/-!
# TAC System 1 neural observational head — exact-real local contracts

Companion to TAC-OSM research/system1-observational-parallel-2026-10-26
source revision 650cda65f98131c8d9d4438fae4b4b62581d4bb9
(src/tac_osm/variants/system1_parallel.py).

The trained source graph consists of the pre-existing parallel/stream world
encoder, fast/slow representation, read-before-write 8x8 associative memory
and a new 198 -> 12 -> 15 head. This file types the NEW head over exact reals.
It does not prove complete PyTorch source equivalence, IEEE floating point,
automatic differentiation, calibrated risk, causal action effects or skill.

In particular, the original 14 predicted world dimensions are externally
supplied as part of the 198-vector. They are NOT derived or verified here.
-/

namespace PNDS.System1Head

/-- Mathematically typed dense layer, weights [out,in] and bias [out]. -/
noncomputable def dense {i o : ℕ}
    (w : Fin o → Fin i → ℝ) (b : Fin o → ℝ)
    (x : Fin i → ℝ) : Fin o → ℝ :=
  fun j => (∑ k, w j k * x k) + b j

/-- Risk score before empirical calibration. -/
noncomputable def sigmoid (z : ℝ) : ℝ :=
  1 / (1 + Real.exp (-z))

/-- Exact-real SiLU, implemented by the standard sigmoid. -/
noncomputable def silu (z : ℝ) : ℝ := z * sigmoid z

/-- Exact-real softplus, epsilon added only for positive scale. -/
noncomputable def scale (epsilon z : ℝ) : ℝ :=
  Real.log (1 + Real.exp z) + epsilon

/-- The exact 198 -> 12 -> 15 neural head, with no free shape variables. -/
noncomputable def rawHead
    (w1 : Fin 12 → Fin 198 → ℝ) (b1 : Fin 12 → ℝ)
    (w2 : Fin 15 → Fin 12 → ℝ) (b2 : Fin 15 → ℝ)
    (evidence : Fin 198 → ℝ) : Fin 15 → ℝ :=
  dense w2 b2 (fun j => silu (dense w1 b1 evidence j))

/-- The uncalibrated score is strictly positive in exact real arithmetic. -/
theorem sigmoid_pos (z : ℝ) : 0 < sigmoid z := by
  unfold sigmoid
  have h : 0 < 1 + Real.exp (-z) := by
    have := Real.exp_pos (-z)
    linarith
  exact one_div_pos.mpr h

/-- Exact-real sigmoid lies strictly below one. FP saturation is excluded. -/
theorem sigmoid_lt_one (z : ℝ) : sigmoid z < 1 := by
  unfold sigmoid
  have hp : 0 < Real.exp (-z) := Real.exp_pos _
  have hd : 0 < 1 + Real.exp (-z) := by linarith
  apply (div_lt_iff₀ hd).2
  nlinarith

/-- Positive epsilon structurally enforces positive exact-real scale. -/
theorem scale_pos (epsilon z : ℝ) (hepsilon : 0 < epsilon) :
    0 < scale epsilon z := by
  unfold scale
  have hp : 0 < Real.exp z := Real.exp_pos _
  have harg : 1 ≤ 1 + Real.exp z := by linarith
  have hlog : 0 ≤ Real.log (1 + Real.exp z) := Real.log_nonneg harg
  linarith

/-- Every output of the source-shaped head is an exact-real bounded score. -/
theorem raw_head_risk_range
    (w1 : Fin 12 → Fin 198 → ℝ) (b1 : Fin 12 → ℝ)
    (w2 : Fin 15 → Fin 12 → ℝ) (b2 : Fin 15 → ℝ)
    (x : Fin 198 → ℝ) :
    0 < sigmoid (rawHead w1 b1 w2 b2 x 0) ∧
    sigmoid (rawHead w1 b1 w2 b2 x 0) < 1 := by
  exact ⟨sigmoid_pos _, sigmoid_lt_one _⟩

end PNDS.System1Head
