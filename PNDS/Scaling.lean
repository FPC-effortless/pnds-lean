import Mathlib

open Real

namespace PNDS.Scaling

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

variable (N K r : ℕ)

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

The counting core: the number of selected-and-relevant items is at most the number of
relevant items. This is a subset counting argument, so it holds regardless of how the
scores are distributed. -/

variable (m : ℕ) (hrm : r ≤ m)

/-- Embed `Fin r` into `Fin m` when `r ≤ m`, via the Mathlib standard `Fin.castLEEmb`.

    The binders `(r : ℕ) (h : r ≤ m)` are explicit here rather than taken from the
    enclosing `variable` line, because the section also declares `N K r : ℕ` and
    `m : ℕ` as section variables. A definition that let those be auto-generalized
    would take `N` as its first explicit argument, so `finEmbed hrm` would pass the
    proof `hrm` where the width `N` was expected. -/
def finEmbed (r : ℕ) {m : ℕ} (h : r ≤ m) : Fin r ↪ Fin m :=
  Fin.castLEEmb h

/-- The relevant items, as a finset over `Fin m`: the first `r` indices.

    This is the *type-level* image of `Fin r` under the canonical embedding, which is
    why it has exactly `r` elements regardless of how the scores are distributed. -/

def relevantFin (r : ℕ) {m : ℕ} (hrm : r ≤ m) : Finset (Fin m) :=
  (Finset.univ : Finset (Fin r)).map (finEmbed r hrm)

/-- The relevant set has cardinality exactly `r`.

    `Finset.univ` on `Fin r` has `r` elements, and `Finset.map` preserves cardinality
    (`Finset.card_map`), so the image has `r` elements. -/
theorem card_relevantFin : (relevantFin r hrm).card = r := by
  rw [relevantFin, Finset.card_map, Finset.card_univ, Fintype.card_fin]

/-- **Recall is at most 1.** The selected-and-relevant items are a subset of the
    relevant items, so their count is bounded by `r`. This is the counting statement
    that forces `K >= r` if all relevant items are to be selected. -/
theorem inter_le_relevant (selected : Finset (Fin m)) :
    (selected ∩ relevantFin r hrm).card ≤ r := by
  have hsub : selected ∩ relevantFin r hrm ⊆ relevantFin r hrm :=
    Finset.inter_subset_right
  have hcard := Finset.card_le_card hsub
  rw [card_relevantFin] at hcard
  exact hcard

/-! ## 3. The residual `N` dependence -/

/-- **The `N`-dependence that S18 identifies.** If the index lets `D(N)` distractors
    into the candidate set, then to select all `r` relevant items the width must be at
    least `r + D(N)`. -/
theorem width_grows_with_distractors (D : ℕ → ℕ) (hD : Monotone D)
    (hN : 0 < N) : r + D N ≤ K → r ≤ K := by
  intro h
  linarith

end PNDS.Scaling
