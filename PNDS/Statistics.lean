import Mathlib

open Real

namespace PNDS.Statistics

/-! ## 1. The minimum attainable p-value -/

/-- The minimum attainable one-sided p-value under a sign test with `k` seeds, achieved
    when all `k` signs agree. -/
def minP_oneSided (k : ℕ) : ℝ := (1 / 2 : ℝ) ^ k

/-- The minimum attainable two-sided p-value: twice the one-sided value. -/
def minP_twoSided (k : ℕ) : ℝ := 2 * (1 / 2 : ℝ) ^ k

/-- `minP_oneSided k = 2 ^ (-k)`. -/
theorem minP_oneSided_eq (k : ℕ) : minP_oneSided k = 2 ^ (-(k : ℝ)) := by
  rw [minP_oneSided]
  field_simp
  ring

/-- `minP_twoSided k = 2 ^ (1 - k)`. -/
theorem minP_twoSided_eq (k : ℕ) : minP_twoSided k = 2 ^ (1 - (k : ℝ)) := by
  rw [minP_twoSided, minP_oneSided_eq]
  field_simp
  ring

/-! ## 2. The three numbers quoted in §20 -/

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

/-! ## 3. Six seeds suffice -/

/-- **Six seeds reach `p < 0.05`** two-sided (`0.03125`), but `k = 5` does not
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

/-! ## 4. Monotonicity -/

/-- More seeds strictly reduce the minimum attainable p-value. -/
theorem minP_antitone : Antitone minP_oneSided := by
  intro n m hnm
  simp only [minP_oneSided]
  apply div_le_div₀
  · norm_num
  · norm_num
  · exact one_le_pow_of_one_le (by norm_num) n
  · exact pow_le_pow_of_le_one (by norm_num) (by norm_num) hnm

end PNDS.Statistics

/-!
# Statistics: sign-test minima (§20)

The document states that under a sign test the minimum attainable one-sided p-value
with `k` seeds is `(1/2)^k`, so three seeds cannot reach `p < 0.05` (one-sided
`p_min = 0.125`, two-sided `0.250`), and at least 6 seeds are required
(two-sided `0.0312`).

We prove the exact minima.
-/
