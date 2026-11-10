/- Scalar restoration domain only. Not PyTorch/IEEE/source authentication. -/
import Mathlib

namespace StateIngress

def Admissible (n d epsilon : ℝ) : Prop :=
  0 ≤ d ∧ |n| ≤ d ∧ 0 < epsilon

theorem denominator_positive (n d epsilon : ℝ) (h : Admissible n d epsilon) :
    0 < d + epsilon := by
  rcases h with ⟨hd, _, he⟩
  linarith

theorem numerator_bounded_by_denominator (n d epsilon : ℝ)
    (h : Admissible n d epsilon) : |n| ≤ d + epsilon := by
  rcases h with ⟨_, hn, he⟩
  linarith

theorem normalized_read_bound (n d epsilon : ℝ) (h : Admissible n d epsilon) :
    |n / (d + epsilon)| ≤ 1 := by
  have hp := denominator_positive n d epsilon h
  have hn := numerator_bounded_by_denominator n d epsilon h
  rw [abs_div, abs_of_pos hp]
  apply (div_le_iff₀ hp).2
  simpa using hn

end StateIngress
