import PNDS.RoleSpecializedFFN
import Mathlib
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
# ME108: source-linked exact-real neural/AD and limited finite-error refinements

This follows the actual 102 -> 16 -> {12,12,6} torch RoleFFN graph.
The pointwise derivative of tanh is an EXPLICIT supplied factor d; these
theorems establish reverse/forward adjoint algebra, not PyTorch autograd or IEEE
refinement. The full-input masked scan, normal-law integral/quantile inversion,
censored atom and all IEEE kernels remain OPEN proof obligations.
No opaque admits, sorry, or unproved global correctness theorems.
-/

namespace PNDS.NeuralRefinement
open PNDS.RoleSpecializedFFN
open PNDS.SensorGraph105

variable {inp out : ℕ}

/-- Linear-layer tangent in the input direction, for a fixed weight matrix. -/
def linearInputJVP (p : Linear out inp) (dx : Fin inp → ℝ) : Fin out → ℝ :=
  fun i => ∑ j, p.weight i j * dx j

/-- Linear-layer reverse-mode input cotangent, for a supplied output cotangent. -/
def linearInputVJP (p : Linear out inp) (dy : Fin out → ℝ) : Fin inp → ℝ :=
  fun j => ∑ i, dy i * p.weight i j

/-- Exact-real adjoint identity: JVP and VJP are transposes.
This algebraic identity is valid for the actual source layer shape. -/
theorem linear_input_vjp_correct (p : Linear out inp)
    (dx : Fin inp → ℝ) (dy : Fin out → ℝ) :
    (∑ i, dy i * linearInputJVP p dx i) =
      ∑ j, linearInputVJP p dy j * dx j := by
  calc
    (∑ i, dy i * linearInputJVP p dx i)
        = ∑ i, ∑ j, dy i * (p.weight i j * dx j) := by
            simp only [linearInputJVP, Finset.mul_sum]
    _ = ∑ j, ∑ i, dy i * (p.weight i j * dx j) := by
          rw [Finset.sum_comm]
    _ = ∑ j, linearInputVJP p dy j * dx j := by
          apply Finset.sum_congr rfl
          intro j _
          simp only [linearInputVJP, Finset.sum_mul]
          apply Finset.sum_congr rfl
          intro i _
          ring

/-- Weight/bias perturbation in a source-style Linear(out, inp). -/
def linearParameterJVP (x : Fin inp → ℝ)
    (dw : Fin out → Fin inp → ℝ) (db : Fin out → ℝ) : Fin out → ℝ :=
  fun i => db i + ∑ j, dw i j * x j

/-- Reverse-mode weight and bias gradients are the standard outer products. -/
theorem linear_parameter_vjp_correct (x : Fin inp → ℝ)
    (dw : Fin out → Fin inp → ℝ) (db dy : Fin out → ℝ) :
    (∑ i, dy i * linearParameterJVP x dw db i) =
      (∑ i, dy i * db i) + ∑ i, ∑ j, (dy i * x j) * dw i j := by
  simp only [linearParameterJVP, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Exact-real scalar derivative used by all 3 ME108 tanh FFN hidden layers.
This discharges the local activation derivative rather than merely supplying
it as an unverified heuristic. IEEE torch.tanh remains a separate problem. -/
theorem tanh_derivative_real (x : ℝ) :
    HasDerivAt Real.tanh (1 - (Real.tanh x)^2) x := by
  have hc : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
  have hd : HasDerivAt (fun y : ℝ => Real.sinh y / Real.cosh y)
      ((Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) /
        (Real.cosh x)^2) x :=
    (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hc
  have hf : (fun y : ℝ => Real.sinh y / Real.cosh y) = Real.tanh := by
    funext y
    exact (Real.tanh_eq_sinh_div_cosh y).symm
  rw [hf] at hd
  have hvalue :
      (Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) /
        (Real.cosh x)^2 = 1 - (Real.tanh x)^2 := by
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp [hc]
    simpa only [pow_two] using Real.cosh_sq_sub_sinh_sq x
  rw [← hvalue]
  exact hd

/-- JVP of the complete source-shaped two-layer FFN in its feature input.
The supplied diagonal D is intended to be 1 - tanh(hidden)^2 at that input.
Proving the transcendental tanh derivative and the source implementation's
automatic differentiation requires a separate refinement theorem. -/
def ffnInputJVP (f : TanhFFN out) (d : Fin 16 → ℝ)
    (dx : Fin 102 → ℝ) : Fin out → ℝ :=
  linearInputJVP f.output (fun h => d h * linearInputJVP f.hidden dx h)

/-- Proposed reverse-mode input cotangent for the same fixed diagonal D. -/
def ffnInputVJP (f : TanhFFN out) (d : Fin 16 → ℝ)
    (dy : Fin out → ℝ) : Fin 102 → ℝ :=
  linearInputVJP f.hidden
    (fun h => d h * linearInputVJP f.output dy h)

/-- Nontrivial full two-affine-layer adjoint (source shapes 102 -> 16 -> out).
This proves algebraic consistency of backprop for any supplied diagonal,
including the correct tanh derivative once its source law is established. -/
theorem ffn_input_vjp_correct (f : TanhFFN out)
    (d : Fin 16 → ℝ) (dx : Fin 102 → ℝ) (dy : Fin out → ℝ) :
    (∑ i, dy i * ffnInputJVP f d dx i) =
      ∑ j, ffnInputVJP f d dy j * dx j := by
  calc
    (∑ i, dy i * ffnInputJVP f d dx i)
        = ∑ h, linearInputVJP f.output dy h *
              (d h * linearInputJVP f.hidden dx h) := by
            exact linear_input_vjp_correct f.output
              (fun h => d h * linearInputJVP f.hidden dx h) dy
    _ = ∑ h, (d h * linearInputVJP f.output dy h) *
                 linearInputJVP f.hidden dx h := by
            apply Finset.sum_congr rfl
            intro h _
            ring
    _ = ∑ j, ffnInputVJP f d dy j * dx j := by
            exact linear_input_vjp_correct f.hidden dx
              (fun h => d h * linearInputVJP f.output dy h)

/-- The actual three output dimensions of ME108 instantiate the same theorem,
not three separately postulated gradient procedures. -/
theorem role_reconstruction_input_adjoint (f : TanhFFN 12)
    (d : Fin 16 → ℝ) (dx : Fin 102 → ℝ) (dy : Fin 12 → ℝ) :
    (∑ i, dy i * ffnInputJVP f d dx i) =
      ∑ j, ffnInputVJP f d dy j * dx j :=
  ffn_input_vjp_correct f d dx dy

theorem role_width_input_adjoint (f : TanhFFN 12)
    (d : Fin 16 → ℝ) (dx : Fin 102 → ℝ) (dy : Fin 12 → ℝ) :
    (∑ i, dy i * ffnInputJVP f d dx i) =
      ∑ j, ffnInputVJP f d dy j * dx j :=
  ffn_input_vjp_correct f d dx dy

theorem role_median_input_adjoint (f : TanhFFN 6)
    (d : Fin 16 → ℝ) (dx : Fin 102 → ℝ) (dy : Fin 6 → ℝ) :
    (∑ i, dy i * ffnInputJVP f d dx i) =
      ∑ j, ffnInputVJP f d dy j * dx j :=
  ffn_input_vjp_correct f d dx dy

/-- An explicitly conditional two-operation finite-arithmetic error budget.
Here xhat, productHat, sumHat denote *actual* rounded intermediates,
provided the caller establishes their independent error bounds.
This is not an IEEE-754 formalization or an overflow/underflow guarantee. -/
theorem two_stage_affine_roundoff
    (a b x xhat productHat sumHat epsInput epsMul epsAdd : ℝ)
    (hx : |xhat - x| ≤ epsInput)
    (hm : |productHat - a * xhat| ≤ epsMul)
    (ha : |sumHat - (productHat + b)| ≤ epsAdd)
    (_he0 : 0 ≤ epsInput) (_hm0 : 0 ≤ epsMul) (_ha0 : 0 ≤ epsAdd) :
    |sumHat - (a * x + b)| ≤ epsAdd + epsMul + |a| * epsInput := by
  have hid : sumHat - (a * x + b) =
      (sumHat - (productHat + b)) +
      ((productHat - a * xhat) + a * (xhat - x)) := by ring
  rw [hid]
  have hfirst := abs_add_le (sumHat - (productHat + b))
    ((productHat - a * xhat) + a * (xhat - x))
  have hsecond := abs_add_le (productHat - a * xhat) (a * (xhat - x))
  rw [abs_mul] at hsecond
  have hmul : |a| * |xhat - x| ≤ |a| * epsInput :=
    mul_le_mul_of_nonneg_left hx (abs_nonneg _)
  linarith

/-- Probability-law consistency at the split normal's location: both sides
join at one-half when the supplied standard Gaussian CDF has Phi(0)=1/2.
No Gaussian integral or inverse-CDF correctness is assumed as a theorem. -/
theorem split_cdf_at_location (law : NormalLaw) (f : Forecast)
    (c : Fin 6) (h0 : law.cdf 0 = (1 / 2 : ℝ)) :
    splitCDF law f c (f.location c) = 1 / 2 := by
  simp [splitCDF, sideScale, h0]

/-- Censoring at the physical nonnegative boundary gives zero CDF below zero.
No claim that the registered arbitrary NormalLaw is a bona fide probability
measure is made by this elementary boundary theorem. -/
theorem raw_cdf_below_zero (law : NormalLaw) (p : Parameters)
    (f : Forecast) (c : Fin 6) (y : ℝ) (h : y < 0) :
    rawCDF law p f c y = 0 := by
  simp [rawCDF, h]

end PNDS.NeuralRefinement
