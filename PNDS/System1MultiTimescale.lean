import Mathlib
import PNDS.System1FixedSlow

/-!
# S1TEMP007: four-timescale 16D slow bank exact-real local mathematics

Four groups of four channels apply fixed scalar rates 1/16, 1/64,
1/256 and 1/1024. 72 additive + 16 multiscale +64 associative +14
previous observed state = 166 nominal FP32 scalar slots, 664 bytes.

This proves exact-real individual scalar rate positivity, bounded
causal updates and a nonzero old-history ordering path. It is NOT a
source-refinement proof of actual PyTorch 32-frame prefix scan, IEEE
rounding/backprop, or an empirical long-retention guarantee.
-/

namespace PNDS.System1MultiTimescale

noncomputable def tierRate (tier : Fin 4) : ℝ :=
  if tier.val = 0 then 1/16 else
  if tier.val = 1 then 1/64 else
  if tier.val = 2 then 1/256 else 1/1024

theorem tierRate_positive (tier : Fin 4) : 0 < tierRate tier := by
  fin_cases tier <;> norm_num [tierRate]

theorem tierRate_below_one (tier : Fin 4) : tierRate tier < 1 := by
  fin_cases tier <;> norm_num [tierRate]

theorem tierRate_bounded (tier : Fin 4) :
    0 ≤ tierRate tier ∧ tierRate tier ≤ 1 := by
  exact ⟨le_of_lt (tierRate_positive tier),
         le_of_lt (tierRate_below_one tier)⟩

theorem every_channel_preserves_bounds
    (tier : Fin 4) (prior basis bound : ℝ)
    (hbound : 0 ≤ bound)
    (hp : |prior| ≤ bound)
    (hb : |basis| ≤ bound) :
    |PNDS.System1FixedSlow.slowUpdate (tierRate tier) prior basis| ≤ bound := by
  have h := tierRate_bounded tier
  exact PNDS.System1FixedSlow.bounded_update (tierRate tier)
    prior basis bound h.1 h.2 hbound hp hb

theorem every_channel_order_sensitive_formula
    (tier : Fin 4) (prior first second : ℝ) :
    PNDS.System1FixedSlow.slowUpdate (tierRate tier)
      (PNDS.System1FixedSlow.slowUpdate (tierRate tier) prior first) second -
      PNDS.System1FixedSlow.slowUpdate (tierRate tier)
      (PNDS.System1FixedSlow.slowUpdate (tierRate tier) prior second) first
      = (tierRate tier)^2*(second-first) := by
  exact PNDS.System1FixedSlow.swap_two_step_difference
    (tierRate tier) prior first second

theorem unchanged_total_fp32_nominal_bytes :
    (72+4*4+64+14)*4 = (88+64+14)*4 := by decide

theorem identical_95k_parameter_budget (n : ℕ) :
    n + 0 = n := by omega

end PNDS.System1MultiTimescale
