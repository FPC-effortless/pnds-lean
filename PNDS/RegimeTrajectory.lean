import PNDS.CoherentTrajectory
import Mathlib.Tactic.Linarith

/- Conditional CDF laws, not a formalization of the full learned measure. -/
namespace PNDS.RegimeTrajectory

noncomputable def mix (p a b : ℝ) : ℝ := (1-p)*a+p*b

theorem mix_range (p a b : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    0 ≤ mix p a b ∧ mix p a b ≤ 1 := by
  unfold mix
  have h0 : 0 ≤ 1-p := by linarith
  constructor
  · exact add_nonneg (mul_nonneg h0 ha0) (mul_nonneg hp0 hb0)
  · have h1 := mul_le_mul_of_nonneg_left ha1 h0
    have h2 := mul_le_mul_of_nonneg_left hb1 hp0
    nlinarith

theorem mix_monotone (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (f g : ℝ → ℝ) (hf : Monotone f) (hg : Monotone g) :
    Monotone (fun x => mix p (f x) (g x)) := by
  intro x y hxy
  unfold mix
  exact add_le_add (mul_le_mul_of_nonneg_left (hf hxy) (by linarith))
    (mul_le_mul_of_nonneg_left (hg hxy) hp0)

/-- Law of total covariance for equal-location components: scale inflation factor. -/
theorem variance_not_smaller (p k q : ℝ)
    (hp : 0 ≤ p) (hk : 1 ≤ k) (hq : 0 ≤ q) :
    q ≤ ((1-p)+p*k^2)*q := by
  have hk2 : 1 ≤ k^2 := by nlinarith [sq_nonneg (k-1)]
  have h := mul_nonneg hp (sub_nonneg.mpr hk2)
  have hf : 1 ≤ (1-p)+p*k^2 := by nlinarith
  simpa using mul_le_mul_of_nonneg_right hf hq

end PNDS.RegimeTrajectory
