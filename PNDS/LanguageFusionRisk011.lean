import PNDS.LanguageQuery010
import Mathlib

/-!
GRAPH011: exact-real risk analysis of a source-probability mixture.
A frequently activated approximate query is NOT automatically useful:
if it gives the observed true label lower probability than the base,
every strictly positive fusion weight worsens its true-label NLL.

This is an algebraic statement, not a source-inference oracle. A policy
cannot access the future true label to select its weight; GRAPH011 freezes
its weight 0.1 using earlier FIT and checks new unscored source files.
No PyTorch IEEE/AD or complete learner correctness theorem is asserted.
-/
namespace PNDS.LanguageFusionRisk011
noncomputable section
open PNDS.LanguageRetrieval009

theorem fusion_true_label_delta (base retrieved weight : ℝ) :
    fusion base retrieved weight - base =
      weight*(retrieved-base) := by
  unfold fusion
  ring

theorem wrong_continuation_fusion_harms
    (base retrieved weight : ℝ)
    (hw : 0 < weight) (hr : retrieved < base) :
    fusion base retrieved weight < base := by
  have h : weight*(retrieved-base) < 0 :=
    mul_neg_of_pos_of_neg hw (sub_neg.mpr hr)
  rw [← fusion_true_label_delta] at h
  linarith

theorem correct_continuation_fusion_helps
    (base retrieved weight : ℝ)
    (hw : 0 < weight) (hr : base < retrieved) :
    base < fusion base retrieved weight := by
  have h : 0 < weight*(retrieved-base) :=
    mul_pos hw (sub_pos.mpr hr)
  rw [← fusion_true_label_delta] at h
  linarith

/-- The absolute probability movement is linear in the fixed nonnegative
fusion weight. It is NOT a guarantee of the sign of a true-target gain. -/
theorem fusion_deviation_bound
    (base retrieved weight : ℝ) (hw : 0 ≤ weight) :
    |fusion base retrieved weight - base| =
      weight*|retrieved-base| := by
  rw [fusion_true_label_delta, abs_mul, abs_of_nonneg hw]

theorem wrong_continuation_increases_nll
    (base retrieved weight : ℝ)
    (hb : 0 < base) (hrpos : 0 < retrieved)
    (hw : 0 < weight) (hw1 : weight < 1)
    (hwrong : retrieved < base) :
    -Real.log base < -Real.log (fusion base retrieved weight) := by
  have hcoeff : 0 < 1-weight := by linarith
  have hmix : 0 < fusion base retrieved weight := by
    unfold fusion
    exact add_pos (mul_pos hcoeff hb) (mul_pos hw hrpos)
  have hdelta := wrong_continuation_fusion_harms base retrieved weight hw hwrong
  have hlog := Real.log_lt_log hmix hdelta
  linarith

end
end PNDS.LanguageFusionRisk011
