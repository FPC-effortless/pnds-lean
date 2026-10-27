import PNDS.LanguageMultiLagObjective
import Mathlib.Tactic

/-!
MLAG-003: RAW real-gradient conflict projection contract. Let s = <g_main,
g_history> and q = <g_main,g_main> > 0 on the selected source encoder.
Projection of historical gradients by (min(s,0)/q)*g_main changes this
overlap to max(s,0). Thus the first-order raw descent overlap with the
main objective is not worsened by a conflicting historical component.

NOT a theorem about AdamW, clipping, actual GPU floating arithmetic, the
complete transformer/TAC neural graph or expected language quality.
-/
namespace PNDS.LanguageGradientGuard
noncomputable section

def guardedOverlap (score mainNormSq : ℝ) : ℝ :=
  if score < 0 then score - (score / mainNormSq) * mainNormSq else score

theorem conflict_overlap_becomes_zero
    (score mainNormSq : ℝ) (hn : 0 < mainNormSq) (hs : score < 0) :
    guardedOverlap score mainNormSq = 0 := by
  simp only [guardedOverlap, if_pos hs]
  have hne : mainNormSq ≠ 0 := ne_of_gt hn
  have cancel : (score / mainNormSq) * mainNormSq = score := by
    field_simp
  rw [cancel]
  ring

theorem nonconflicting_overlap_is_unchanged
    (score mainNormSq : ℝ) (hs : 0 ≤ score) :
    guardedOverlap score mainNormSq = score := by
  simp [guardedOverlap, not_lt.mpr hs]

theorem guarded_overlap_nonnegative
    (score mainNormSq : ℝ) (hn : 0 < mainNormSq) :
    0 ≤ guardedOverlap score mainNormSq := by
  by_cases hs : score < 0
  · rw [conflict_overlap_becomes_zero score mainNormSq hn hs]
  · rw [nonconflicting_overlap_is_unchanged score mainNormSq (le_of_not_gt hs)]
    exact le_of_not_gt hs

/-- The inner product of g_main with (g_main + guarded g_history)
is at least its own squared norm; this is a RAW gradient condition,
not a promise that the optimizer improves the next-token objective. -/
theorem guarded_total_alignment
    (score mainNormSq : ℝ) (hn : 0 < mainNormSq) :
    mainNormSq ≤ mainNormSq + guardedOverlap score mainNormSq := by
  have hs := guarded_overlap_nonnegative score mainNormSq hn
  linarith

end
end PNDS.LanguageGradientGuard
