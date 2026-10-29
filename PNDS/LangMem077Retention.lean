import Mathlib

/-!
LANGMEM077 partial native learned retention law:
source `N_next=N_old P+Q`, with P = alpha(I-beta*w*wᵀ),
alpha=.995 sigmoid(gate0) and beta=sigmoid(gate1).

These scoped Lean theorems establish scalar affine same-input
cancellation and bounded scalar retention for alpha>=0, beta∈[0,1].
The full k×k spectral/operator-norm theorem for unit w, actual PyTorch
float32 implementation, gradient/source correspondence and existence
of useful old facts across other memories are OPEN, not proved here.
-/
namespace PNDS.LangMem077

def scalarNativeStep (old p q : ℝ) : ℝ := old*p+q

theorem same_source_input_difference (a b p q : ℝ) :
    scalarNativeStep a p q - scalarNativeStep b p q = (a-b)*p := by
  dsimp [scalarNativeStep]
  ring

def nativeScalarGate (alpha beta : ℝ) : ℝ := alpha * (1-beta)

theorem directional_gate_nonnegative (alpha beta : ℝ)
    (ha : 0 <= alpha) (hb : beta <= 1) :
    0 <= nativeScalarGate alpha beta := by
  unfold nativeScalarGate
  exact mul_nonneg ha (sub_nonneg.mpr hb)

theorem directional_gate_no_greater_than_retention (alpha beta : ℝ)
    (ha : 0 <= alpha) (hb : 0 <= beta) :
    nativeScalarGate alpha beta <= alpha := by
  unfold nativeScalarGate
  calc
    alpha * (1-beta) <= alpha * 1 :=
      mul_le_mul_of_nonneg_left (by linarith) ha
    _ = alpha := by ring

theorem product_of_retention_not_greater_than_first (alpha gamma : ℝ)
    (ha : 0 <= alpha) (hg : gamma <= 1) :
    alpha*gamma <= alpha := by
  calc
    alpha*gamma <= alpha*1 := mul_le_mul_of_nonneg_left hg ha
    _ = alpha := by ring

end PNDS.LangMem077
