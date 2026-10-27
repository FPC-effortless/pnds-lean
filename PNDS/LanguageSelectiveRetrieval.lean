import PNDS.LanguageRepairCandidates
import PNDS.LanguageCategorical

/-!
TAC-LANG-REPAIR-018: formally scoped selective historical query.

The original 100680-parameter language model was trained with the lag64
historical reconstruction objective. A full delay beyond this horizon
returns a POSITIVE, normalized FIT-only categorical prior and marks the
model's historical memory untrusted. No learned parameter, future label,
model result, or DEV loss participates in the routing decision.

This proves the mathematical failover and optional mixture mass; it does
NOT prove the fallback improves independent real language accuracy,
nor that lag64 predictions are reliably calibrated.
-/
namespace PNDS.LanguageSelectiveRetrieval
noncomputable section
open PNDS.LanguageCategorical

def trainedHorizon : ℕ := 64

def useHistorical (lag : ℕ) : Bool :=
  decide (0 < lag ∧ lag ≤ trainedHorizon)

def selectiveProb (lag : ℕ) (source fit : Scores) : Scores :=
  if useHistorical lag then source else fit

theorem source_within_horizon (lag : ℕ)
    (hpos : 0 < lag) (htrained : lag ≤ trainedHorizon)
    (source fit : Scores) :
    selectiveProb lag source fit = source := by
  simp [selectiveProb, useHistorical, hpos, htrained]

theorem fallback_outside_horizon (lag : ℕ)
    (huntrained : trainedHorizon < lag) (source fit : Scores) :
    selectiveProb lag source fit = fit := by
  simp [selectiveProb, useHistorical, Nat.not_le.mpr huntrained]

theorem fallback_is_strictly_positive (lag : ℕ)
    (huntrained : trainedHorizon < lag) (source fit : Scores)
    (hfit : ∀ i, 0 < fit i) (i : Fin 128) :
    0 < selectiveProb lag source fit i := by
  rw [fallback_outside_horizon lag huntrained]
  exact hfit i

theorem fallback_normalization (lag : ℕ)
    (huntrained : trainedHorizon < lag) (source fit : Scores)
    (hfit : (∑ i : Fin 128, fit i) = 1) :
    (∑ i : Fin 128, selectiveProb lag source fit i) = 1 := by
  rw [fallback_outside_horizon lag huntrained]
  exact hfit

/-- Any optional convex mixture has a baseline probability floor. Here the
    source weight is fixed independently of the observed target.
    The deployed out-of-horizon policy uses weight ZERO instead. -/
def convexHistory (weight : ℝ) (source fit : Scores) : Scores :=
  fun i => weight*source i + (1-weight)*fit i

theorem convex_history_baseline_floor
    (weight : ℝ) (source fit : Scores)
    (hw : 0 ≤ weight) (hs : ∀ i, 0 ≤ source i) (i : Fin 128) :
    (1-weight)*fit i ≤ convexHistory weight source fit i := by
  unfold convexHistory
  have hx := mul_nonneg hw (hs i)
  linarith

theorem convex_history_normalization
    (weight : ℝ) (source fit : Scores)
    (hs : (∑ i : Fin 128, source i) = 1)
    (hf : (∑ i : Fin 128, fit i) = 1) :
    (∑ i : Fin 128, convexHistory weight source fit i) = 1 := by
  unfold convexHistory
  calc
    (∑ i : Fin 128, (weight*source i + (1-weight)*fit i)) =
      weight*(∑ i : Fin 128, source i) +
        (1-weight)*(∑ i : Fin 128, fit i) := by
          rw [← Finset.mul_sum, ← Finset.mul_sum, ← Finset.sum_add_distrib]
    _ = 1 := by rw [hs,hf]; ring

end
end PNDS.LanguageSelectiveRetrieval
