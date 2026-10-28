import PNDS.LanguageRetrieval009
import Mathlib

/-!
GRAPH010: source-causal approximate-context read, separate from GRAPH009
exact 3-character count cache. These are SCOPED real-valued contracts,
NOT a proof of PyTorch topK, learned semantic identity, authentication,
FP32, or external benchmark superiority.

For current t (observed x_0..x_t, predicting x_(t+1)):
a candidate with observed continuation at s is eligible only for s<t
and t-s <= capacity. Exact-hit GRAPH009 overrides approximate matching.
-/
namespace PNDS.LanguageQuery010
noncomputable section

open PNDS.LanguageRetrieval009

def eligible (current continuation capacity : ℕ) : Prop :=
  continuation < current ∧ current - continuation ≤ capacity

theorem candidate_is_strictly_past (current continuation capacity : ℕ)
    (h : eligible current continuation capacity) :
    continuation < current := h.1

theorem candidate_within_retention (current continuation capacity : ℕ)
    (h : eligible current continuation capacity) :
    current-continuation ≤ capacity := h.2

/-- For a given candidate's context end at j, its observed NEXT byte
is at j+1; it must have been observed strictly before current t. -/
theorem source_continuation_causes_no_target_leak
    (current contextEnd : ℕ)
    (h : contextEnd+1 < current) :
    contextEnd+1 ≠ current+1 := by omega

def selectiveQuery (exactHit approxHit : Bool)
    (base exactPosterior approximatePosterior weight : ℝ) : ℝ :=
  if exactHit then exactPosterior
  else if approxHit then fusion base approximatePosterior weight
  else base

theorem exact_match_has_priority (approxHit : Bool)
    (base exactPosterior approximatePosterior weight : ℝ) :
    selectiveQuery true approxHit base exactPosterior approximatePosterior weight =
      exactPosterior := by
  simp [selectiveQuery]

theorem no_match_abstains (base exactPosterior approximatePosterior weight : ℝ) :
    selectiveQuery false false base exactPosterior approximatePosterior weight = base := by
  simp [selectiveQuery]

theorem approximate_only_on_exact_miss
    (base exactPosterior approximatePosterior weight : ℝ) :
    selectiveQuery false true base exactPosterior approximatePosterior weight =
      fusion base approximatePosterior weight := by
  simp [selectiveQuery]

theorem selective_query_normalized
    (exactHit approxHit : Bool)
    (baseMass exactMass approximateMass weight : ℝ)
    (hb : baseMass=1) (he : exactMass=1) (ha : approximateMass=1) :
    selectiveQuery exactHit approxHit baseMass exactMass approximateMass weight=1 := by
  cases exactHit <;> cases approxHit <;>
    simp [selectiveQuery, hb, he, ha, fusion]

/-- Weights are nonnegative and sum to one. The count posterior with
n = sum selected weights is exactly normalized (given a normalized prior);
proof reuses GRAPH009's posterior mass law rather than duplicating it. -/
theorem learned_count_posterior_normalization (weightSum tau priorMass : ℝ)
    (hn : 0 ≤ weightSum) (ht : 0 < tau) (hp : priorMass=1) :
    (weightSum + tau*priorMass)/(weightSum+tau)=1 :=
  posterior_total_mass weightSum tau priorMass hn ht hp

/-- A smooth prior gives positive probability to every label even when
learned topK happens to retrieve none of that label. -/
theorem learned_posterior_strict_positive
    (weightedCount weightSum tau prior : ℝ)
    (hc : 0 ≤ weightedCount) (hn : 0 ≤ weightSum)
    (ht : 0 < tau) (hp : 0 < prior) :
    0 < posterior weightedCount weightSum tau prior := by
  unfold posterior
  have hd : 0 < weightSum+tau := posterior_denom_pos weightSum tau hn ht
  have hnum : 0 < weightedCount+tau*prior :=
    add_pos_of_nonneg_of_pos hc (mul_pos ht hp)
  exact div_pos hnum hd

end
end PNDS.LanguageQuery010
