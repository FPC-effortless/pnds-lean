import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
/- Exact-real source-trace kernel only; not a proof of the neural network. -/
namespace PNDS.SourceAlignedTrace

def update (s c beta x : ℝ) : ℝ × ℝ :=
  ((1-beta)*s+beta*x, (1-beta)*c+beta)

theorem no_observation (s c x : ℝ) : update s c 0 x = (s,c) := by
  simp [update]

theorem bounded (s c beta x : ℝ) (hs : |s| ≤ c) (hc : c ≤ 1)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1) (hx : |x| ≤ 1) :
    |(update s c beta x).1| ≤ (update s c beta x).2 ∧
    (update s c beta x).2 ≤ 1 := by
  dsimp [update]
  have hn : 0 ≤ 1-beta := sub_nonneg.mpr hb1
  constructor
  · calc
      |(1-beta)*s+beta*x| ≤ |(1-beta)*s|+|beta*x| := abs_add _ _
      _ = (1-beta)*|s|+beta*|x| := by rw [abs_mul,abs_mul,abs_of_nonneg hn,abs_of_nonneg hb0]
      _ ≤ (1-beta)*c+beta*1 := add_le_add (mul_le_mul_of_nonneg_left hs hn) (mul_le_mul_of_nonneg_left hx hb0)
      _ = (1-beta)*c+beta := by ring
  · nlinarith [mul_nonneg hn (sub_nonneg.mpr hc)]

-- A scalar affine action on each of the sum/mass coordinates.
def compose (p q : ℝ × ℝ) : ℝ × ℝ := (q.1*p.1,q.1*p.2+q.2)
theorem composition_associative (p q r : ℝ × ℝ) :
    compose (compose p q) r = compose p (compose q r) := by
  apply Prod.ext <;> dsimp [compose] <;> ring

def applyAffine (p : ℝ × ℝ) (s : ℝ) : ℝ := p.1*s+p.2
theorem composition_action (p q : ℝ × ℝ) (s : ℝ) :
    applyAffine (compose p q) s = applyAffine q (applyAffine p s) := by
  dsimp [applyAffine,compose]; ring
end PNDS.SourceAlignedTrace
