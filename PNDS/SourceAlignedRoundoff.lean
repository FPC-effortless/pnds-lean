import PNDS.SourceAlignedVerification
import Mathlib.Algebra.Order.Group.MinMax

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
