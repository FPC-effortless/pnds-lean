import Mathlib
import PNDS.System1ProjectedBudget

/-!
# S1TEMP006: exact-real persistent 16D slow-state contract.

TAC source: src/tac_osm/variants/s1temp006_persistent_slow.py.
SLOW alpha = 1/16; decay 15/16. The 88D fast accumulator is split into
72D additive + 16D genuine persistent EMA. The 64D associative matrix and
14D previous observation remain unchanged. The nominal FP32 floating
state remains 166 floats = 664 bytes.

This does not prove Python torch implementation correspondence, IEEE
finite-precision or backprop equivalence, learnt long-term performance,
causal action candidates, probability calibration or physical safety.
-/

namespace PNDS.System1FixedSlow

def slowUpdate (a prior basis : ℝ) : ℝ :=
  (1-a)*prior+a*basis

def slowRate : ℝ := 1/16

theorem rate_in_unit_interval : 0 < slowRate ∧ slowRate < 1 := by
  constructor <;> norm_num [slowRate]

theorem slow_zero_zero : slowUpdate slowRate 0 0 = 0 := by
  norm_num [slowUpdate, slowRate]

theorem next_input_effect (a prior x y : ℝ) :
    slowUpdate a prior y - slowUpdate a prior x = a*(y-x) := by
  simp [slowUpdate]
  ring

theorem swap_two_step_difference (a s x y : ℝ) :
    slowUpdate a (slowUpdate a s x) y -
      slowUpdate a (slowUpdate a s y) x = a^2*(y-x) := by
  simp [slowUpdate]
  ring

theorem bounded_update
    (a prior basis maximum : ℝ)
    (ha : 0 ≤ a) (hA : a ≤ 1) (hmax : 0 ≤ maximum)
    (hprior : |prior| ≤ maximum) (hbasis : |basis| ≤ maximum) :
    |slowUpdate a prior basis| ≤ maximum := by
  have hq : 0 ≤ 1-a := by linarith
  have hpl : -maximum ≤ prior := (abs_le.mp hprior).1
  have hpu : prior ≤ maximum := (abs_le.mp hprior).2
  have hbl : -maximum ≤ basis := (abs_le.mp hbasis).1
  have hbu : basis ≤ maximum := (abs_le.mp hbasis).2
  apply abs_le.mpr
  constructor
  · calc
      -maximum = (1-a)*(-maximum) + a*(-maximum) := by ring
      _ ≤ (1-a)*prior + a*basis := by
        exact add_le_add
          (mul_le_mul_of_nonneg_left hpl hq)
          (mul_le_mul_of_nonneg_left hbl ha)
      _ = slowUpdate a prior basis := rfl
  · calc
      slowUpdate a prior basis = (1-a)*prior+a*basis := rfl
      _ ≤ (1-a)*maximum + a*maximum := by
        exact add_le_add
          (mul_le_mul_of_nonneg_left hpu hq)
          (mul_le_mul_of_nonneg_left hbu ha)
      _ = maximum := by ring

theorem total_float_state_equality : 72+16+64+14 = 88+64+14 := by decide

theorem same_nominal_FP32_bytes :
    (72+16+64+14)*4 = 664 := by decide

end PNDS.System1FixedSlow
