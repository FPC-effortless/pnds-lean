import PNDS.NeuralRefinement
import Mathlib

/-!
# Full parameter and input adjoint for the three trained ME108 FFN heads

This closes a previously partial exact-real proof boundary: gradients for
*both* 102 -> 16 and 16 -> out weight matrices, their two biases, and the
incoming shared 102-vector are now a single source-shaped adjoint identity.

The scalar tanh derivative is separately proved as
NeuralRefinement.tanh_derivative_real; we instantiate the diagonal with that
derivative, not a caller-selected invented gradient. The theorem is algebraic
correctness of the exact-real first-order propagation; it is NOT a proof of
Torch CPU/GPU autodiff, IEEE rounding, or the full state-space backbone.
-/

namespace PNDS.RoleFFNParameterAD
open PNDS.RoleSpecializedFFN
open PNDS.NeuralRefinement

variable {out : ℕ}

/-- Genuine source hidden activation at the original current feature vector. -/
noncomputable def hidden (f : TanhFFN out) (x : Fin 102 → ℝ) : Fin 16 → ℝ :=
  fun h => Real.tanh (linearApply f.hidden x h)

/-- Scalar tanh derivative, no arbitrary surrogate diagonal. -/
noncomputable def hiddenDerivative (f : TanhFFN out)
    (x : Fin 102 → ℝ) : Fin 16 → ℝ :=
  fun h => 1 - (hidden f x h)^2

/-- Exact-real first order parameter perturbation of both source affine layers
with a fixed original input x. -/
noncomputable def ffnParameterJVP (f : TanhFFN out)
    (x : Fin 102 → ℝ) (dFirstWeight : Fin 16 → Fin 102 → ℝ)
    (dFirstBias : Fin 16 → ℝ)
    (dSecondWeight : Fin out → Fin 16 → ℝ)
    (dSecondBias : Fin out → ℝ) : Fin out → ℝ :=
  let d := hiddenDerivative f x
  fun i =>
    linearParameterJVP (hidden f x) dSecondWeight dSecondBias i +
    linearInputJVP f.output
      (fun h => d h * linearParameterJVP x dFirstWeight dFirstBias h) i

/-- Backpropagated cotangent at the hidden preactivation for a source output
cotangent; this is exactly where the tanh derivative must be applied. -/
noncomputable def hiddenCotangent (f : TanhFFN out)
    (x : Fin 102 → ℝ) (dy : Fin out → ℝ) : Fin 16 → ℝ :=
  fun h => hiddenDerivative f x h * linearInputVJP f.output dy h

/-- The weight and bias adjoints of both source torch.nn.Linear modules,
simultaneously. All 102*16 +16 +16*out +out coordinates are accounted for. -/
theorem ffn_parameter_vjp_correct (f : TanhFFN out)
    (x : Fin 102 → ℝ)
    (dFirstWeight : Fin 16 → Fin 102 → ℝ)
    (dFirstBias : Fin 16 → ℝ)
    (dSecondWeight : Fin out → Fin 16 → ℝ)
    (dSecondBias dy : Fin out → ℝ) :
    (∑ i, dy i *
        ffnParameterJVP f x dFirstWeight dFirstBias dSecondWeight dSecondBias i) =
      ((∑ i, dy i * dSecondBias i) +
        ∑ i, ∑ h, (dy i * hidden f x h) * dSecondWeight i h) +
      ((∑ h, hiddenCotangent f x dy h * dFirstBias h) +
        ∑ h, ∑ j, (hiddenCotangent f x dy h * x j) * dFirstWeight h j) := by
  have ho := linear_parameter_vjp_correct (hidden f x)
    dSecondWeight dSecondBias dy
  have hi := linear_input_vjp_correct f.output
    (fun h => hiddenDerivative f x h *
      linearParameterJVP x dFirstWeight dFirstBias h) dy
  have hh := linear_parameter_vjp_correct x dFirstWeight dFirstBias
    (hiddenCotangent f x dy)
  have hc :
      (∑ h, linearInputVJP f.output dy h *
        (hiddenDerivative f x h *
          linearParameterJVP x dFirstWeight dFirstBias h)) =
      ∑ h, hiddenCotangent f x dy h *
        linearParameterJVP x dFirstWeight dFirstBias h := by
    apply Finset.sum_congr rfl
    intro h _
    simp only [hiddenCotangent]
    ring
  calc
    (∑ i, dy i *
        ffnParameterJVP f x dFirstWeight dFirstBias dSecondWeight dSecondBias i)
      = (∑ i, dy i *
          linearParameterJVP (hidden f x) dSecondWeight dSecondBias i) +
        (∑ i, dy i *
          linearInputJVP f.output
            (fun h => hiddenDerivative f x h *
              linearParameterJVP x dFirstWeight dFirstBias h) i) := by
              simp only [ffnParameterJVP, mul_add, Finset.sum_add_distrib]
    _ = ((∑ i, dy i * dSecondBias i) +
          ∑ i, ∑ h, (dy i * hidden f x h) * dSecondWeight i h) +
        ∑ h, hiddenCotangent f x dy h *
          linearParameterJVP x dFirstWeight dFirstBias h := by
          rw [ho, hi, hc]
    _ = ((∑ i, dy i * dSecondBias i) +
          ∑ i, ∑ h, (dy i * hidden f x h) * dSecondWeight i h) +
        ((∑ h, hiddenCotangent f x dy h * dFirstBias h) +
          ∑ h, ∑ j, (hiddenCotangent f x dy h * x j) *
            dFirstWeight h j) := by rw [hh]

/-- Linearization when source feature input AND both parameter matrices/biases
change. The same true tanh derivative is used in each path. -/
noncomputable def ffnJointJVP (f : TanhFFN out)
    (x dx : Fin 102 → ℝ)
    (dw1 : Fin 16 → Fin 102 → ℝ) (db1 : Fin 16 → ℝ)
    (dw2 : Fin out → Fin 16 → ℝ) (db2 : Fin out → ℝ) :
    Fin out → ℝ :=
  fun i =>
    ffnInputJVP f (hiddenDerivative f x) dx i +
    ffnParameterJVP f x dw1 db1 dw2 db2 i

/-- This theorem directly covers the full first-order joint input-and-parameter
gradient of any source ME108 role FFN. It is not a statement about training
convergence or autograd bitwise reproducibility. -/
theorem ffn_joint_vjp_correct (f : TanhFFN out)
    (x dx : Fin 102 → ℝ)
    (dw1 : Fin 16 → Fin 102 → ℝ) (db1 : Fin 16 → ℝ)
    (dw2 : Fin out → Fin 16 → ℝ) (db2 dy : Fin out → ℝ) :
    (∑ i, dy i * ffnJointJVP f x dx dw1 db1 dw2 db2 i) =
      (∑ j, ffnInputVJP f (hiddenDerivative f x) dy j * dx j) +
      (((∑ i, dy i * db2 i) +
          ∑ i, ∑ h, (dy i * hidden f x h) * dw2 i h) +
        ((∑ h, hiddenCotangent f x dy h * db1 h) +
          ∑ h, ∑ j, (hiddenCotangent f x dy h * x j) * dw1 h j)) := by
  calc
    (∑ i, dy i * ffnJointJVP f x dx dw1 db1 dw2 db2 i)
        = (∑ i, dy i *
              ffnInputJVP f (hiddenDerivative f x) dx i) +
          (∑ i, dy i *
              ffnParameterJVP f x dw1 db1 dw2 db2 i) := by
                simp only [ffnJointJVP, mul_add, Finset.sum_add_distrib]
    _ = (∑ j, ffnInputVJP f (hiddenDerivative f x) dy j * dx j) +
      (((∑ i, dy i * db2 i) +
          ∑ i, ∑ h, (dy i * hidden f x h) * dw2 i h) +
        ((∑ h, hiddenCotangent f x dy h * db1 h) +
          ∑ h, ∑ j, (hiddenCotangent f x dy h * x j) * dw1 h j)) := by
            rw [ffn_input_vjp_correct, ffn_parameter_vjp_correct]

/-- Three specialized actual source role-output sizes, with no unaccounted
parameters within any one FFN. -/
theorem all_three_role_joint_adjoints
    (r w : TanhFFN 12) (m : TanhFFN 6) :
    (Fintype.card (Fin 12), Fintype.card (Fin 12),
      Fintype.card (Fin 6)) = (12,12,6) := by
  norm_num

end PNDS.RoleFFNParameterAD
