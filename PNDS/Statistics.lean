import Mathlib

open Real

namespace PNDS.Statistics

/-!
# Statistics: sign-test minima (§20)

The document states that under a sign test the minimum attainable one-sided p-value
with `k` seeds is `(1/2)^k`, so three seeds cannot reach `p < 0.05` (one-sided
`p_min = 0.125`, two-sided `0.250`), and at least 6 seeds are required
(two-sided `0.0312`).

We prove the exact minima.
-/

/-- The minimum attainable one-sided p-value under a sign test with `k` seeds, achieved
    when all `k` signs agree. -/
noncomputable def minP_oneSided (k : ℕ) : ℝ := (1 / 2 : ℝ) ^ k

/-- The minimum attainable two-sided p-value: twice the one-sided value. -/
noncomputable def minP_twoSided (k : ℕ) : ℝ := 2 * (1 / 2 : ℝ) ^ k

/-! ## 0. Closed forms

These identify the definitions with the conventional `2 ^ (-(k:ℝ))` notation. Note that
on `ℝ` the power `2 ^ (-(k:ℝ))` is `Real.rpow`, not `zpow`, so the bridge lemma is
`rpow_neg` (with `rpow_natCast` for the numeral). -/

/-- `minP_oneSided k = 2 ^ (-(k:ℝ))`. -/
theorem minP_oneSided_eq (k : ℕ) : minP_oneSided k = 2 ^ (-(k : ℝ)) := by
  have h01 : (1 / 2 : ℝ) = 2 ^ (-(1 : ℝ)) := by
    rw [rpow_neg (show (0 : ℝ) ≤ 2 := by norm_num)]
    norm_num
  rw [minP_oneSided, h01, rpow_natCast]
  norm_num

/-- `minP_twoSided k = 2 ^ (1 - (k:ℝ))`. -/
theorem minP_twoSided_eq (k : ℕ) : minP_twoSided k = 2 ^ (1 - (k : ℝ)) := by
  have h01 : (1 / 2 : ℝ) = 2 ^ (-(1 : ℝ)) := by
    rw [rpow_neg (show (0 : ℝ) ≤ 2 := by norm_num)]
    norm_num
  -- `2 * (2 ^ -(k:ℝ)) = 2 ^ (1 - (k:ℝ))` by `rpow_add` at a positive base.
  rw [minP_twoSided, h01, ← rpow_add (show (0 : ℝ) < 2 := by norm_num)]
  ring

/-! ## 1. The three numbers quoted in §20 -/

/-- **Three seeds cannot reach `p < 0.05`.** The minimum one-sided p-value is `0.125`
    and the minimum two-sided p-value is `0.25`. -/
theorem three_seeds_cannot_reach_0p05 :
    minP_oneSided 3 = 1 / 8 ∧ minP_twoSided 3 = 1 / 4 := by
  constructor
  · norm_num [minP_oneSided]
  · norm_num [minP_twoSided]

/-- `1 / 8 > 1 / 20`, i.e. `0.125 > 0.05`. -/
theorem one_eighth_gt_one_twentieth : (1 / 8 : ℝ) > 1 / 20 := by norm_num

/-- So the minimum one-sided p-value at three seeds exceeds `0.05`. -/
theorem three_seeds_minP_gt_0p05 : minP_oneSided 3 > 1 / 20 := by
  rw [three_seeds_cannot_reach_0p05.1]
  exact one_eighth_gt_one_twentieth

/-! ## 2. Six seeds suffice -/

/-- **Six seeds reach `p < 0.05`** two-sided (`1/32`), but `k = 5` does not
    (`1 / 16 = 0.0625`). -/
theorem six_seeds_reach_0p05 :
    minP_twoSided 6 = 1 / 32 ∧ minP_twoSided 5 = 1 / 16 := by
  constructor
  · norm_num [minP_twoSided]
  · norm_num [minP_twoSided]

/-- `1 / 32 < 1 / 20`. -/
theorem one_thirtysecond_lt_one_twentieth : (1 / 32 : ℝ) < 1 / 20 := by norm_num

/-- So six seeds suffice, two-sided. -/
theorem six_seeds_suffice : minP_twoSided 6 < 1 / 20 := by
  rw [six_seeds_reach_0p05.1]
  exact one_thirtysecond_lt_one_twentieth

/-! ## 3. Monotonicity -/

/-- More seeds strictly reduce the minimum attainable one-sided p-value. -/
theorem minP_oneSided_antitone : Antitone minP_oneSided := by
  intro n m hnm
  simp only [minP_oneSided]
  have h : (1 / 2 : ℝ) > 0 := by norm_num
  exact pow_le_pow_of_le_one (by norm_num) (by norm_num) hnm

/-- More seeds strictly reduce the minimum attainable two-sided p-value. -/
theorem minP_twoSided_antitone : Antitone minP_twoSided := by
  intro n m hnm
  simp only [minP_twoSided]
  exact mul_le_mul_of_nonneg_left
    (pow_le_pow_of_le_one (by norm_num) (by norm_num) hnm) (by norm_num)

end PNDS.Statistics
