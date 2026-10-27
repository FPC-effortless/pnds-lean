import PNDS.SourceAlignedVerification
import Mathlib.Algebra.Order.Group.MinMax
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
Explicit absolute-error contracts. These include underflow through an additive
term and do not assume associativity of rounded arithmetic. Every primitive
backend accuracy assumption appears as a premise, rather than an axiom or a
test silently promoted to a universal fact.
-/
namespace PNDS.SourceAlignedRoundoff

noncomputable section

def Close (approx ideal radius : ℝ) : Prop := |approx-ideal| ≤ radius

theorem close_refl (x : ℝ) : Close x x 0 := by simp [Close]

theorem add_error (a b x y ea eb rounded roundingError : ℝ)
    (ha : Close a x ea) (hb : Close b y eb)
    (hr : Close rounded (a+b) roundingError) :
    Close rounded (x+y) (ea+eb+roundingError) := by
  have htriangle := abs_add (a-x) (b-y)
  have hlocal := abs_add (rounded-(a+b)) ((a-x)+(b-y))
  have heq : rounded-(x+y) = (rounded-(a+b))+((a-x)+(b-y)) := by ring
  unfold Close at *
  rw [heq]
  linarith

theorem neg_error (a x ea rounded roundingError : ℝ)
    (ha : Close a x ea) (hr : Close rounded (-a) roundingError) :
    Close rounded (-x) (ea+roundingError) := by
  have ht := abs_add (rounded+a) (-(a-x))
  have heq : rounded-(-x) = (rounded+a)+(-(a-x)) := by ring
  unfold Close at *
  rw [heq]
  simp only [sub_neg_eq_add] at hr
  simp only [abs_neg] at ht
  linarith

theorem multiply_error (a b x y ea eb rounded roundingError : ℝ)
    (ha : Close a x ea) (hb : Close b y eb)
    (hr : Close rounded (a*b) roundingError) :
    Close rounded (x*y) (|a| * eb+|b| * ea+ea*eb+roundingError) := by
  unfold Close at *
  have hea : 0 ≤ ea := (abs_nonneg _).trans ha
  have heb : 0 ≤ eb := (abs_nonneg _).trans hb
  have h1 := mul_le_mul_of_nonneg_left hb (abs_nonneg a)
  have h2 := mul_le_mul_of_nonneg_left ha (abs_nonneg b)
  have h3 : |a-x| * |b-y| ≤ ea*eb := mul_le_mul ha hb (abs_nonneg _) hea
  have ht1 := abs_add (a*(b-y)) (b*(a-x))
  have ht2 := abs_sub (a*(b-y)+b*(a-x)) ((a-x)*(b-y))
  have ht3 := abs_add (rounded-a*b) (a*b-x*y)
  have heq : a*b-x*y = a*(b-y)+b*(a-x)-(a-x)*(b-y) := by ring
  rw [heq] at ht3
  simp only [abs_mul] at ht1 ht2
  have hrw : rounded-x*y = (rounded-a*b)+(a*b-x*y) := by ring
  rw [hrw, heq]
  linarith

/-- Clamp/max is forward-stable even at ties; the derivative need not be. -/
theorem max_error (a b x y ea eb rounded roundingError : ℝ)
    (ha : Close a x ea) (hb : Close b y eb)
    (hr : Close rounded (max a b) roundingError) :
    Close rounded (max x y) (max ea eb+roundingError) := by
  unfold Close at *
  have hm := (abs_max_sub_max_le_max a b x y).trans (max_le_max ha hb)
  have ht := abs_add (rounded-max a b) (max a b-max x y)
  have heq : rounded-max x y = (rounded-max a b)+(max a b-max x y) := by ring
  rw [heq]
  linarith

theorem chain_error (a x ea rounded roundingError : ℝ)
    (ha : Close a x ea) (hr : Close rounded a roundingError) :
    Close rounded x (ea+roundingError) := by
  unfold Close at *
  have ht := abs_add (rounded-a) (a-x)
  have heq : rounded-x = (rounded-a)+(a-x) := by ring
  rw [heq]
  linarith

theorem inverse_error (a x ea rounded roundingError margin : ℝ)
    (ha : Close a x ea) (hm : 0 < margin)
    (hax : margin ≤ |a|) (hxx : margin ≤ |x|)
    (hr : Close rounded a⁻¹ roundingError) :
    Close rounded x⁻¹ (ea/(margin*margin)+roundingError) := by
  have han : a ≠ 0 := abs_pos.1 (hm.trans_le hax)
  have hxn : x ≠ 0 := abs_pos.1 (hm.trans_le hxx)
  have habs : |a⁻¹-x⁻¹| = |a-x|/(|a|*|x|) := by
    have heq : a⁻¹-x⁻¹ = (x-a)/(a*x) := by field_simp; ring
    rw [heq,abs_div,abs_mul,abs_sub_comm]
  have he : 0 ≤ ea := (abs_nonneg _).trans ha
  have hden : margin*margin ≤ |a|*|x| :=
    mul_le_mul hax hxx (le_of_lt hm) ((le_of_lt hm).trans hax)
  have hbound : Close a⁻¹ x⁻¹ (ea/(margin*margin)) := by
    unfold Close
    rw [habs]
    apply (div_le_div_iff₀ (mul_pos (abs_pos.2 han) (abs_pos.2 hxn)) (mul_pos hm hm)).2
    have h1 := mul_le_mul_of_nonneg_right ha (le_of_lt (mul_pos hm hm))
    have h2 := mul_le_mul_of_nonneg_left hden he
    exact h1.trans h2
  exact chain_error _ _ _ _ _ hbound hr

theorem sqrt_error (a x ea rounded roundingError : ℝ)
    (ha0 : 0 ≤ a) (hx0 : 0 ≤ x) (ha : Close a x ea)
    (hr : Close rounded (Real.sqrt a) roundingError) :
    Close rounded (Real.sqrt x) (Real.sqrt ea+roundingError) := by
  have hpa := Real.sqrt_nonneg a
  have hpx := Real.sqrt_nonneg x
  have hsa := Real.sq_sqrt ha0
  have hsx := Real.sq_sqrt hx0
  have hs : |Real.sqrt a-Real.sqrt x|^2 ≤ |a-x| := by
    rcases le_total (Real.sqrt a) (Real.sqrt x) with h | h
    · rw [abs_of_nonpos (sub_nonpos.2 h),abs_of_nonpos (show a-x ≤ 0 by nlinarith)]
      nlinarith
    · rw [abs_of_nonneg (sub_nonneg.2 h),abs_of_nonneg (show 0 ≤ a-x by nlinarith)]
      nlinarith
  have hbound : Close (Real.sqrt a) (Real.sqrt x) (Real.sqrt ea) := by
    have hea : 0 ≤ ea := (abs_nonneg _).trans ha
    have he := Real.sq_sqrt hea
    have hp := Real.sqrt_nonneg ea
    have hn := abs_nonneg (Real.sqrt a-Real.sqrt x)
    unfold Close at *
    nlinarith
  exact chain_error _ _ _ _ _ hbound hr

theorem sin_difference_bound (a x : ℝ) : |Real.sin a-Real.sin x| ≤ |a-x| := by
  rw [Real.sin_sub_sin,abs_mul,abs_mul,abs_of_nonneg (by norm_num : (0:ℝ) ≤ 2)]
  have hs := Real.abs_sin_le_abs (x := (a-x)/2)
  have hc := Real.abs_cos_le_one ((a+x)/2)
  have ht := mul_le_mul_of_nonneg_left hc (mul_nonneg (by norm_num) (abs_nonneg _))
  have hhalf : 2*|(a-x)/2| = |a-x| := by rw [abs_div]; norm_num
  nlinarith

theorem cos_difference_bound (a x : ℝ) : |Real.cos a-Real.cos x| ≤ |a-x| := by
  rw [Real.cos_sub_cos,abs_mul,abs_mul]
  have hneg : |(-2:ℝ)| = 2 := by norm_num
  rw [hneg]
  have hs := Real.abs_sin_le_abs (x := (a-x)/2)
  have hc := Real.abs_sin_le_one ((a+x)/2)
  have ht := mul_le_mul_of_nonneg_right hc (abs_nonneg (Real.sin ((a-x)/2)))
  have hhalf : 2*|(a-x)/2| = |a-x| := by rw [abs_div]; norm_num
  nlinarith

theorem sin_error (a x ea rounded roundingError : ℝ) (ha : Close a x ea)
    (hr : Close rounded (Real.sin a) roundingError) :
    Close rounded (Real.sin x) (ea+roundingError) :=
  chain_error _ _ _ _ _ ((sin_difference_bound a x).trans ha) hr

theorem cos_error (a x ea rounded roundingError : ℝ) (ha : Close a x ea)
    (hr : Close rounded (Real.cos a) roundingError) :
    Close rounded (Real.cos x) (ea+roundingError) :=
  chain_error _ _ _ _ _ ((cos_difference_bound a x).trans ha) hr

theorem exp_error (a x ea rounded roundingError : ℝ) (ha : Close a x ea) (he1 : ea ≤ 1)
    (hr : Close rounded (Real.exp a) roundingError) :
    Close rounded (Real.exp x) (2*Real.exp a*ea+roundingError) := by
  have hx : |x-a| ≤ 1 := by rw [abs_sub_comm]; exact ha.trans he1
  have he := Real.abs_exp_sub_one_le hx
  have heq : Real.exp x-Real.exp a = Real.exp a*(Real.exp (x-a)-1) := by
    rw [Real.exp_sub]
    field_simp [Real.exp_ne_zero]
    ring
  have hbound : Close (Real.exp a) (Real.exp x) (2*Real.exp a*ea) := by
    unfold Close
    rw [abs_sub_comm,heq,abs_mul,abs_of_pos (Real.exp_pos a)]
    have hmul := mul_le_mul_of_nonneg_left he (le_of_lt (Real.exp_pos a))
    have ha' : |x-a| ≤ ea := by rw [abs_sub_comm]; exact ha
    nlinarith [Real.exp_pos a]
  exact chain_error _ _ _ _ _ hbound hr

theorem log_error (a x ea rounded roundingError margin : ℝ)
    (ha : Close a x ea) (hm : 0 < margin) (ham : margin ≤ a) (hxm : margin ≤ x)
    (hr : Close rounded (Real.log a) roundingError) :
    Close rounded (Real.log x) (ea/margin+roundingError) := by
  have hap : 0 < a := hm.trans_le ham
  have hxp : 0 < x := hm.trans_le hxm
  have upper (a x : ℝ) (hap : 0 < a) (hxp : 0 < x) (hxm : margin ≤ x)
      (hclose : |a-x| ≤ ea) : Real.log a-Real.log x ≤ ea/margin := by
    have hl := Real.log_le_sub_one_of_pos (div_pos hap hxp)
    rw [Real.log_div (ne_of_gt hap) (ne_of_gt hxp)] at hl
    have heq : a/x-1 = (a-x)/x := by field_simp; ring
    rw [heq] at hl
    apply hl.trans
    apply (div_le_div_iff₀ hxp hm).2
    have hdiff := le_abs_self (a-x)
    have hea := (abs_nonneg (a-x)).trans hclose
    nlinarith [mul_nonneg hea (sub_nonneg.2 hxm)]
  have hu := upper a x hap hxp hxm ha
  have hl := upper x a hxp hap ham (by rw [abs_sub_comm]; exact ha)
  have hbound : Close (Real.log a) (Real.log x) (ea/margin) := by
    exact abs_le.2 ⟨by linarith,hu⟩
  exact chain_error _ _ _ _ _ hbound hr

/-- IEEE-style local contract, with explicit relative and absolute terms.
An additive eta is essential for subnormals/underflow. No overflow or NaN
has a real interpretation, so finite decoded operands/results are required. -/
def RoundingContract (domain : Set ℝ) (round : ℝ → ℝ) (u eta : ℝ) : Prop :=
  ∀ x ∈ domain, |round x-x| ≤ u*|x|+eta

def roundedSum (round : ℝ → ℝ) (initial : ℝ) : List ℝ → ℝ
  | [] => initial
  | x::xs => roundedSum round (round (initial+x)) xs

def ValidSum (domain : Set ℝ) (round : ℝ → ℝ) (initial : ℝ) : List ℝ → Prop
  | [] => True
  | x::xs => initial+x ∈ domain ∧ ValidSum domain round (round (initial+x)) xs

def sumBudget (u eta : ℝ) (ideal previous : ℝ) : List ℝ → ℝ
  | [] => previous
  | x::xs => sumBudget u eta (ideal+x) ((1+u)*previous+u*|ideal+x|+eta) xs

/-- Rounded accumulation with explicit ideal partial-sum magnitudes,
initial error and underflow allowance. Overflow exclusion is a checked domain
premise; no global fixed absolute tolerance is asserted. -/
theorem rounded_sum_error (domain : Set ℝ) (round : ℝ → ℝ) (u eta : ℝ)
    (hu : 0 ≤ u) (hc : RoundingContract domain round u eta)
    (approx ideal error : ℝ) (hstart : Close approx ideal error) (xs : List ℝ)
    (hvalid : ValidSum domain round approx xs) :
    Close (roundedSum round approx xs) (ideal+xs.sum)
      (sumBudget u eta ideal error xs) := by
  induction xs generalizing approx ideal error with
  | nil => simpa [roundedSum, sumBudget, Close] using hstart
  | cons x xs ih =>
    have hr := hc (approx+x) hvalid.1
    have ht := abs_add (round (approx+x)-(approx+x)) (approx-ideal)
    have hid : round (approx+x)-(ideal+x) =
        (round (approx+x)-(approx+x))+(approx-ideal) := by ring
    have hmag : |approx+x| ≤ |approx-ideal|+|ideal+x| := by
      have hid' : approx+x = (approx-ideal)+(ideal+x) := by ring
      rw [hid']
      exact abs_add _ _
    have hbound : Close (round (approx+x)) (ideal+x)
        ((1+u)*error+u*|ideal+x|+eta) := by
      unfold Close at *
      rw [hid]
      nlinarith
    have hnext := ih (round (approx+x)) (ideal+x)
      ((1+u)*error+u*|ideal+x|+eta) hbound hvalid.2
    simpa only [roundedSum, sumBudget, List.sum_cons, add_assoc] using hnext

end
end PNDS.SourceAlignedRoundoff
