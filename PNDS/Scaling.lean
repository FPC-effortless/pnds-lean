import Mathlib

open Real

namespace PNDS.Scaling

/-!
# Scaling: distractor exceedance (§18)

The §18 claim, in the document's own words:

  `H_N = R ∪ I_N`,  `|R| = r`,  `|I_N| → ∞`
  Desired: `C(R ∪ I_N) → C(R)`
  ... either the TopK width `K` or the index margin must grow with `N`
  to hold recall fixed.

This module proves the **combinatorial core** of that statement by counting, with no
distributional assumption on the scores whatsoever.

**What the counting argument gives.** Let `selected` be the top-`K` set (`#selected = K`),
`relevant` the set of relevant items (`#relevant = r`), and recall
`ρ = #(selected ∩ relevant)`. Then

    #(selected \ relevant) = K - ρ

This is an *identity*, not a bound, and it is the honest statement. It immediately
implies the bound the document is reaching for:

  * **recall-1 form** (`ρ = r`, i.e. every relevant item is retrieved): distractors in
    the top-`K` set number exactly `K - r`. So holding recall at 1 forces `K ≥ r`, and
    the distractor count grows linearly in `K` at fixed `r`.
  * **recall-`ρ` form** (general): the distractor count is `K - ρ`, hence `≤ K - r`
    whenever `ρ ≥ r`, i.e. whenever recall is at least 1. Because `ρ ≤ r` always holds
    (`recall_le`), `ρ ≥ r` is exactly `ρ = r`.

The **hypothesis that makes `K - r` a genuine bound** is thus an explicit lower bound on
recall. Nothing in the counting argument supplies that hypothesis: it is a property of
the *scoring function*, not of cardinalities. Stated as a theorem it must appear as an
assumption. See `distractors_in_topK_eq_recall`, `distractors_in_topK_le_of_recall`,
`recall_one_iff`, `recall_le`.

**What is deliberately NOT here.** The document's §18 is a *scaling* claim about `N → ∞`,
and §5 is the margin condition that would have to supply the recall hypothesis
dynamically. This module proves no limit, no asymptotic, and no connection to `N`
whatsoever, because no hypothesis in this module mentions `N` or the scores. Connecting
the §5 margin inequality `r * ε * E(s) ≥ n * (1 - ε)` to the recall hypothesis below is a
probabilistic statement (distractor exceedance is a random event), and the document gives
no SCM for the scores; the §5 bound bounds *softmax mass*, not *top-`K` membership*. That
gap is recorded explicitly by `distractors_in_topK_ge_of_missed` below: once a relevant
item is missed, the distractor count *exceeds* `K - r`, so the bound fails outright
without the recall hypothesis. The counting identity `distractors_in_topK_eq_recall` is
the strongest thing provable from cardinality alone.

The empirical C5 hypothesis (see the project README, "Empirical status") is *not*
established by anything in this file, and must never be presented as if it were.
-/

variable {N K r : ℕ}

/-! ## 1. The candidate pool and the relevant items

The universe is `Fin m` for some `m`; the relevant items are the first `r` indices.
Nothing depends on which particular `r`-subset is chosen — the counting argument below
works for any relevant set of cardinality `r` — so the canonical choice is made once
here.

The binders `m` and `hrm` are declared per-theorem below rather than as section
variables. Declaring `variable (m : ℕ)` would insert `m` as the first *explicit*
parameter of every declaration in the section, which changes every call site: a call
written as `recall_le selected` would then have to pass a `Finset (Fin m)` where the
width `m : ℕ` was expected. Keeping them on the declarations keeps the signatures
stable and the error messages localized. -/

/-- Embed `Fin r` into `Fin m` when `r ≤ m`, via the Mathlib standard `Fin.castLEEmb`. -/
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
theorem card_relevantFin {m : ℕ} (hrm : r ≤ m) : (relevantFin r hrm).card = r := by
  rw [relevantFin, Finset.card_map, Finset.card_univ, Fintype.card_fin]

/-! ## 2. Recall

Recall `ρ` is the number of *selected-and-relevant* items. -/

/-- **Recall is at most 1.** The selected-and-relevant items are a subset of the
    relevant items, so their count is bounded by `r`. This is the counting statement
    that forces `K ≥ r` if all relevant items are to be selected. -/
theorem recall_le {m : ℕ} (hrm : r ≤ m) (selected : Finset (Fin m)) :
    (selected ∩ relevantFin r hrm).card ≤ r := by
  have hsub : selected ∩ relevantFin r hrm ⊆ relevantFin r hrm :=
    Finset.inter_subset_right
  have hcard := Finset.card_le_card hsub
  rw [card_relevantFin hrm] at hcard
  exact hcard

/-! ## 3. The §18 counting identity

This is the module's main theorem. It says: the number of distractors (non-relevant
items) inside the top-`K` selection is `K - ρ`, where `ρ` is the recall. It is an
identity, not a bound, and it needs no assumption beyond `#selected = K`. -/

/-- **Distractor-count identity (§18 counting core).** A top-`K` selection of width `K`
    contains exactly `K - ρ` distractors, where `ρ = #(selected ∩ relevant)` is the
    recall. Equivalently, holding recall at `ρ` while adding one more distractor to the
    candidate pool costs one unit of width. This holds for *any* scoring function, with
    no distributional assumption: it is pure counting. -/
theorem distractors_in_topK_eq_recall {m : ℕ} (hrm : r ≤ m)
    (selected : Finset (Fin m)) (hcard : selected.card = K) :
    (selected \ relevantFin r hrm).card = K - (selected ∩ relevantFin r hrm).card := by
  -- Reduce to the complement of the intersection: `s \ t = s \ (s ∩ t)`.
  rw [← Finset.sdiff_inter_self_left]
  -- Complement cardinality, `#(s \ t) = #s - #(s ∩ t)`, valid since `s ∩ t ⊆ s`.
  have hsub : selected ∩ relevantFin r hrm ⊆ selected := Finset.inter_subset_left
  rw [Finset.card_sdiff hsub, hcard]

/-- **Recall-1 distractor count (§18).** A top-`K` selection that contains all `r`
    relevant items has exactly `K - r` distractors. This is the form in which §18 is
    usually quoted: holding recall at 1 requires `K ≥ r`, and the number of distractors
    that can enter the selected set grows linearly with `K` at fixed `r`. Pure counting;
    no distributional assumption. -/
theorem distractors_in_topK_eq {m : ℕ} (hrm : r ≤ m)
    (selected : Finset (Fin m)) (hcard : selected.card = K)
    (hrel : relevantFin r hrm ⊆ selected) :
    (selected \ relevantFin r hrm).card = K - r := by
  -- Recall 1 means every relevant item is selected, so the intersection is exactly
  -- `relevantFin` (`inter_eq_right`), whose cardinality is `r` (`card_relevantFin`).
  have hinter : selected ∩ relevantFin r hrm = relevantFin r hrm :=
    Finset.inter_eq_right.mpr hrel
  rw [distractors_in_topK_eq_recall hrm selected hcard, hinter, card_relevantFin hrm]

/-- **The `K - r` distractor bound.** The number of distractors in a top-`K` selection
    is at most `K - r` **provided recall is at least 1** — that is, provided every
    relevant item is retrieved. This is exactly the §18 statement: the distractor
    content of the selected set is bounded by the width minus the relevant count, and
    the hypothesis that makes the bound hold is a recall hypothesis, not a consequence
    of cardinality. -/
theorem distractors_in_topK_le_of_recall {m : ℕ} (hrm : r ≤ m)
    (selected : Finset (Fin m)) (hcard : selected.card = K)
    (hrec : r ≤ (selected ∩ relevantFin r hrm).card) :
    (selected \ relevantFin r hrm).card ≤ K - r := by
  rw [distractors_in_topK_eq_recall hrm selected hcard]
  -- `recall_le` gives `ρ ≤ r`, and the hypothesis gives `r ≤ ρ`, so `ρ = r`.
  have hρ := recall_le hrm selected
  have hkey : (selected ∩ relevantFin r hrm).card = r := Nat.le_antisymm hρ hrec
  rw [hkey]

/-! ## 4. Recall is exactly 1 iff every relevant item is selected

`distractors_in_topK_le_of_recall` needs `r ≤ ρ`, and `recall_le` gives `ρ ≤ r`, so the
usable case is `ρ = r`. The next theorem characterizes it: recall is 1 exactly when the
relevant set is contained in the selection. -/

/-- Recall is `1` (every relevant item selected) exactly when the intersection has
    cardinality `r`. Since `ρ ≤ r` always (`recall_le`), this is the unique way to
    reach recall 1. -/
theorem recall_one_iff {m : ℕ} (hrm : r ≤ m) (selected : Finset (Fin m)) :
    (selected ∩ relevantFin r hrm).card = r ↔ relevantFin r hrm ⊆ selected := by
  constructor
  · -- `ρ = r` with `inter ⊆ relevant` forces `inter = relevant`, hence `relevant ⊆ selected`.
    intro heq
    have hsub : selected ∩ relevantFin r hrm ⊆ relevantFin r hrm := Finset.inter_subset_right
    -- Equal cardinality plus a subset gives equality of finsets.
    have hcard_eq : selected ∩ relevantFin r hrm = relevantFin r hrm := by
      apply Finset.eq_of_subset_of_card_le hsub
      rw [heq, card_relevantFin hrm]
    -- `(s ∩ relevant) ⊆ selected` together with `(s ∩ relevant) = relevant`.
    have hint : selected ∩ relevantFin r hrm ⊆ selected := Finset.inter_subset_left
    rw [hcard_eq] at hint
    exact hint
  · intro hsub
    have hinter : selected ∩ relevantFin r hrm = relevantFin r hrm :=
      Finset.inter_eq_right.mpr hsub
    rw [hinter, card_relevantFin hrm]

/-! ## 5. The control: the distractor bound is not a cardinality consequence

The theorem above needs a recall hypothesis. Without one the bound is not merely
unprovable — it is *false*. The counting identity says the distractor count is `K - ρ`,
and `ρ` can be strictly below `r` whenever the scoring function ranks a distractor above a
relevant item. This is the point the document makes qualitatively ("either the TopK width
`K` or the index margin must grow with `N` to hold recall fixed"): the width is forced to
grow *because* recall is not guaranteed by counting alone. -/

/-- **The distractor bound fails when recall is below 1.** If the recall `ρ` is strictly
    below `r`, the number of distractors in the top-`K` selection is strictly greater than
    `K - r`. So the bound `|D ∩ TopK| ≤ K - r` is a consequence of the recall hypothesis,
    not of the width and the relevant count alone.

    The assumption `r ≤ K` is needed: without it `K - r = 0` and the counting identity
    can leave the distractor count at `0` too (all `K < r` slots filled with relevant
    items), so `K - r < distractors` would read `0 < 0`. -/
theorem distractors_in_topK_ge_of_missed {m : ℕ} (hrm : r ≤ m)
    (selected : Finset (Fin m)) (hcard : selected.card = K) (hKr : r ≤ K)
    (hmiss : (selected ∩ relevantFin r hrm).card < r) :
    K - r < (selected \ relevantFin r hrm).card := by
  rw [distractors_in_topK_eq_recall hrm selected hcard]
  -- `recall_le` gives `ρ ≤ r`; with `ρ < r` and `r ≤ K` the counting identity yields
  -- `K - ρ > K - r`.
  have hρ := recall_le hrm selected
  omega

/-- **Tightness: the bound is attained.** When recall is exactly 1 the distractor count
    equals `K - r`, so the recall hypothesis of `distractors_in_topK_le_of_recall` cannot
    be weakened. The assumption `r ≤ K` is needed for the forward direction: without it
    both `#(selected \ relevant)` and `K - r` are `0` while the relevant set is not
    contained in `selected`, so the iff would fail. -/
theorem distractors_in_topK_eq_iff_recall_one {m : ℕ} (hrm : r ≤ m)
    (selected : Finset (Fin m)) (hcard : selected.card = K) (hKr : r ≤ K) :
    (selected \ relevantFin r hrm).card = K - r ↔ relevantFin r hrm ⊆ selected := by
  rw [distractors_in_topK_eq_recall hrm selected hcard]
  constructor
  · -- Forward: `K - ρ = K - r` with `ρ ≤ r` and `r ≤ K` forces `ρ = r`.
    intro heq
    have hρ := recall_le hrm selected
    have hρ_eq : (selected ∩ relevantFin r hrm).card = r := by
      by_contra hne
      have hlt : (selected ∩ relevantFin r hrm).card < r := lt_of_le_of_ne hρ hne
      -- `ρ < r` and `r ≤ K` give `K - ρ > K - r`, contradicting `heq`.
      have : K - (selected ∩ relevantFin r hrm).card > K - r := by omega
      linarith
    exact (recall_one_iff hrm selected).1 hρ_eq
  · -- Backward: recall 1 substitutes `ρ = r` into the counting identity.
    intro hrel
    have hρ_eq : (selected ∩ relevantFin r hrm).card = r :=
      (recall_one_iff hrm selected).2 hrel
    rw [hρ_eq]

end PNDS.Scaling
