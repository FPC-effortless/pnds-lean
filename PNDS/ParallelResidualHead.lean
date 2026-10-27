import Mathlib

/-!
A deliberately NARROW exact-real mathematical contract for the optional
post-scan residual FFN on the TAC parallel 152-float state core.

The neural FFN is summarized by its bounded tanh output; sigmoid gate and
positive TRAIN-defined scale are explicit assumptions. Theorems do NOT prove
the learned FFN, its PyTorch autograd, floating-point kernels, calibration,
the whole recurrence, or any predictive improvement.
-/

namespace PNDS.ParallelResidualHead

/-- One scalar of a post-scan residual readout. -/
def prediction (base scale gate innovation : ℝ) : ℝ :=
  base + (scale * gate) * innovation

/-- A zero output gate leaves the pretrained core prediction unchanged. -/
theorem zero_gate (base scale innovation : ℝ) :
    prediction base scale 0 innovation = base := by
  simp [prediction]

/-- A zero residual output leaves the pretrained core prediction unchanged. -/
theorem zero_innovation (base scale gate : ℝ) :
    prediction base scale gate 0 = base := by
  simp [prediction]

/-- The post-scan residual is additive and cannot, at this interface,
    depend on or change any existing recurrent matrix coordinate. -/
def appendReadout {n : ℕ} (state : Matrix (Fin n) (Fin n) ℝ)
    (base scale gate innovation : ℝ) :
    Matrix (Fin n) (Fin n) ℝ × ℝ :=
  (state, prediction base scale gate innovation)

theorem postscan_state_unchanged {n : ℕ}
    (state : Matrix (Fin n) (Fin n) ℝ)
    (base scale gate innovation : ℝ) :
    (appendReadout state base scale gate innovation).1 = state := by
  rfl

theorem additive_difference (base scale gate innovation : ℝ) :
    prediction base scale gate innovation - base =
      (scale * gate) * innovation := by
  dsimp [prediction]
  ring

/-- If gate is in [0,1], tanh output in [-1,1] and scale is nonnegative,
    post-scan correction stays within the TRAIN-declared scale. This says
    nothing about whether that correction improves a forecast. -/
theorem bounded_prediction (base scale gate innovation : ℝ)
    (hscale : 0 ≤ scale) (hgate0 : 0 ≤ gate) (hgate1 : gate ≤ 1)
    (hinnov0 : -1 ≤ innovation) (hinnov1 : innovation ≤ 1) :
    base - scale ≤ prediction base scale gate innovation ∧
      prediction base scale gate innovation ≤ base + scale := by
  have hsg : 0 ≤ scale * gate := mul_nonneg hscale hgate0
  have hsgUpper : scale * gate ≤ scale := by
    nlinarith [mul_le_mul_of_nonneg_left hgate1 hscale]
  have hl : -(scale * gate) ≤ (scale * gate) * innovation := by
    nlinarith [mul_le_mul_of_nonneg_left hinnov0 hsg]
  have hu : (scale * gate) * innovation ≤ scale * gate := by
    nlinarith [mul_le_mul_of_nonneg_left hinnov1 hsg]
  constructor <;> dsimp [prediction] <;> nlinarith

end PNDS.ParallelResidualHead
