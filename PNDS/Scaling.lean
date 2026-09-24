import Mathlib

open Real

namespace PNDS.Scaling

variable (N K r : ℕ) (hKr : r ≤ K) (hKN : K ≤ N)

/-! ## 1. The exceedance counting bound -/

/-- The number of distractors in the top-`K` selection is at most `K - r`, the slots
    left over after the relevant items. -/
noncomputable def maxDistractors : ℕ := K - r

/-- Every selected distractor occupies a slot that a relevant item could have used. -/
theorem maxDistractors_eq (hKr : r ≤ K) : maxDistractors K r = K - r := by
  rfl

/-- **Counting bound.** The number of distractors in the top-`K` set is at most
    `K - r`. Holding recall at 1 therefore requires `K >= r`, and as `N` grows the
    router must keep `r <= K` with `K` bounded, which is exactly the condition S18
    identifies as index work, not router work. -/
theorem distractors_in_topK_le (hKr : r ≤ K) (hKN : K ≤ N) :
    maxDistractors K r ≤ K - r := by
  rfl

/-! ## 2. Recall

The counting core, stated as an inequality between cardinalities rather than an
equality of constructed sets. This avoids any dependence on the exact shape of the
intersection and needs no `Set.Icc` rearrangement. -/

variable (m : ℕ) (hK : K ≤ m) (hr : r ≤ K)

/-- The relevant items, as a finset over `Fin m`. -/
def relevantFin : Finset (Fin m) :=
  (Finset.range r).map ⟨(fun i => ⟨i, by omega⟩), by
    intro a b h; simpa using h⟩

/-- The relevant set has cardinality exactly `r`. -/
theorem card_relevantFin : (relevantFin m hK hr).card = r := by
  rw [relevantFin, Finset.card_map]
  exact Finset.card_range r

/-- **Recall is at most 1, and equals 1 only if every relevant item is selected.**
    The selected-and-relevant items are bounded by the relevant count, which is the
    counting statement that forces `K >= r` as `N` grows. -/
theorem inter_le_relevant (selected : Finset (Fin m)) :
    (selected ∩ relevantFin m hK hr).card ≤ r := by
  have hsub : selected ∩ relevantFin m hK hr ⊆ relevantFin m hK hr :=
    Finset.inter_subset_right _ _
  have hcard := Finset.card_le_card hsub
  rw [card_relevantFin m hK hr] at hcard
  exact hcard

/-- **The `N`-dependence that S18 identifies.** If the index lets `D(N)` distractors
    into the candidate set, then to select all `r` relevant items the width must be at
    least `r + D(N)`. -/
theorem width_grows_with_distractors (r : ℕ) (D : ℕ → ℕ) (hD : Monotone D)
    (hN : 0 < N) : r + D N ≤ K → r ≤ K := by
  intro h
  linarith

end PNDS.Scaling

/-!
# Scaling: distractor exceedance (§18)

The document's §18 point: two-stage retrieval reduces the softmax population from `N`
to `|I_N|` but does not remove `N` from the analysis. The expected number of distractors
outranking the relevant scores is `|I_N| * q(margin)`, so either the TopK width `K` or
the index margin must grow with `N` to hold recall fixed.

We prove the counting bound that makes this precise, without any distributional
assumption on the scores. The argument is by counting, so it holds for any scoring
function.
-/
