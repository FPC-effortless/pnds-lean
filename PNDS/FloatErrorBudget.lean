import PNDS.RoleFFNParameterAD
import Mathlib

/-!
# ME108 source-shaped deterministic finite-error budgets

These are exact-real implications from explicit local input and kernel error
bounds. Source operator dimensions are the true trained ME108 FFN widths
(102,16,12/6), with a general finite-sum theorem for every output neuron.

The theorems are useful *only after* an IEEE model certifies the local
assumptions for a concrete BLAS/tanh/exp kernel. They are not a claim about
Torch float32, fast math, special values, GPU, or measured calibrated errors.
-/

namespace PNDS.FloatErrorBudget
open PNDS.RoleSpecializedFFN
open PNDS.RoleFFNParameterAD

variable {out inp : ℕ}

/-- An individual actual approximate affine coordinate has a certified
output error bounded by kernel arithmetic error plus the weighted input
representation error. Includes every one of the finite input dimensions. -/
theorem affine_coordinate_error
    (p : Linear out inp) (i : Fin out)
    (x xHat : Fin inp → ℝ) (computed : ℝ)
    (inputError : Fin inp → ℝ) (kernelError : ℝ)
    (hinput : ∀ j, |xHat j - x j| ≤ inputError j)
    (hkernel : |computed -
      (p.bias i + ∑ j, p.weight i j * xHat j)| ≤ kernelError) :
    |computed - linearApply p x i| ≤
      kernelError + ∑ j, |p.weight i j| * inputError j := by
  have hsum :
      (∑ j, p.weight i j * xHat j) -
        (∑ j, p.weight i j * x j) =
      ∑ j, p.weight i j * (xHat j - x j) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  have hdecomp :
      computed - (p.bias i + ∑ j, p.weight i j * x j) =
        (computed - (p.bias i + ∑ j, p.weight i j * xHat j)) +
          ∑ j, p.weight i j * (xHat j - x j) := by
    rw [← hsum]
    ring
  have htriangle :=
    abs_add_le
      (computed - (p.bias i + ∑ j, p.weight i j * xHat j))
      (∑ j, p.weight i j * (xHat j - x j))
  have hsumAbs :
      |∑ j, p.weight i j * (xHat j - x j)| ≤
        ∑ j, |p.weight i j * (xHat j - x j)| := by
    exact Finset.abs_sum_le_sum_abs _ _
  have hweighted :
      (∑ j, |p.weight i j * (xHat j - x j)|) ≤
        ∑ j, |p.weight i j| * inputError j := by
    apply Finset.sum_le_sum
    intro j _
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (hinput j) (abs_nonneg _)
  change |computed - (p.bias i + ∑ j, p.weight i j * x j)| ≤ _
  rw [hdecomp]
  linarith

/-- Every ME108 hidden preactivation coordinate has this propagated input and
GEMM error budget with EXACT 102 inputs. -/
theorem first_layer_102_error (f : TanhFFN out)
    (h : Fin 16) (x xHat : Fin 102 → ℝ)
    (preHat : ℝ) (inputError : Fin 102 → ℝ) (localError : ℝ)
    (hx : ∀ j, |xHat j - x j| ≤ inputError j)
    (hl : |preHat -
      (f.hidden.bias h + ∑ j, f.hidden.weight h j * xHat j)| ≤ localError) :
    |preHat - linearApply f.hidden x h| ≤
      localError + ∑ j, |f.hidden.weight h j| * inputError j :=
  affine_coordinate_error f.hidden h x xHat preHat inputError localError hx hl

/-- A second-stage source head (16 -> out) error, with explicitly supplied
*post-activation* error bounds. No implicit activation Lipschitz assumption. -/
theorem output_layer_16_error (f : TanhFFN out)
    (i : Fin out) (x : Fin 102 → ℝ)
    (activationHat : Fin 16 → ℝ)
    (outputHat : ℝ)
    (activationError : Fin 16 → ℝ) (outputKernelError : ℝ)
    (ha : ∀ h,
      |activationHat h - PNDS.RoleFFNParameterAD.hidden f x h| ≤ activationError h)
    (ho : |outputHat -
      (f.output.bias i + ∑ h, f.output.weight i h * activationHat h)| ≤
        outputKernelError) :
    |outputHat - ffnApply f x i| ≤ outputKernelError +
      ∑ h, |f.output.weight i h| * activationError h := by
  have hb := affine_coordinate_error f.output i (PNDS.RoleFFNParameterAD.hidden f x)
    activationHat outputHat activationError outputKernelError ha ho
  simpa only [ffnApply, hidden] using hb

/-- ME108 full two-layer bounded error composition once the actual tanh
evaluation/error budget is certified separately by an IEEE backend.
The first-layer 102D errors are propagated through 16 hidden activations
and all 12 or 6 output rows: no dimension is ignored. -/
theorem full_role_ffn_error (f : TanhFFN out)
    (i : Fin out) (x xHat : Fin 102 → ℝ)
    (preHat activationHat : Fin 16 → ℝ) (outputHat : ℝ)
    (inputError : Fin 102 → ℝ) (firstError activationError : Fin 16 → ℝ)
    (outputError : ℝ)
    (hx : ∀ j, |xHat j - x j| ≤ inputError j)
    (hfirst : ∀ h, |preHat h -
      (f.hidden.bias h + ∑ j, f.hidden.weight h j * xHat j)| ≤ firstError h)
    (hactivationEval : ∀ h, |activationHat h - Real.tanh (preHat h)| ≤
      activationError h)
    (hLipschitz : ∀ h, |Real.tanh (preHat h) - PNDS.RoleFFNParameterAD.hidden f x h| ≤
      |preHat h - linearApply f.hidden x h|)
    (hout : |outputHat -
      (f.output.bias i + ∑ h, f.output.weight i h * activationHat h)| ≤
        outputError) :
    |outputHat - ffnApply f x i| ≤ outputError +
      ∑ h, |f.output.weight i h| *
        (activationError h + firstError h +
          ∑ j, |f.hidden.weight h j| * inputError j) := by
  have ha : ∀ h, |activationHat h - PNDS.RoleFFNParameterAD.hidden f x h| ≤
      activationError h + firstError h +
        ∑ j, |f.hidden.weight h j| * inputError j := by
    intro h
    have hpre := first_layer_102_error f h x xHat (preHat h)
      inputError (firstError h) hx (hfirst h)
    have hsum :
        activationHat h - PNDS.RoleFFNParameterAD.hidden f x h =
          (activationHat h - Real.tanh (preHat h)) +
          (Real.tanh (preHat h) - PNDS.RoleFFNParameterAD.hidden f x h) := by ring
    rw [hsum]
    have ht := abs_add_le (activationHat h - Real.tanh (preHat h))
      (Real.tanh (preHat h) - PNDS.RoleFFNParameterAD.hidden f x h)
    linarith [hactivationEval h, hLipschitz h]
  exact output_layer_16_error f i x activationHat outputHat
    (fun h => activationError h + firstError h +
      ∑ j, |f.hidden.weight h j| * inputError j)
    outputError ha hout

end PNDS.FloatErrorBudget
