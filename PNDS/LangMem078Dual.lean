import Mathlib
/-!
LANGMEM078: scoped two-timescale source-native operator mathematics.

The original fast associative source is unchanged. A second slow k=8
bank uses scalar rho = exp(-softplus(log_rate) * DeltaByte), with
DeltaByte=1 here, beta_s=.05 sigmoid(erase logit) and write epsilon_s=
.01 sigmoid(write logit). The exact-real scalar ring-one-direction
model is S_next = rho(1-beta_s*w²)S + epsilon_s*v*w;
masked write must be the identity. This file proves only pure Real
scalar bounds and integer parameter/state budget, NOT the complete
trained PyTorch k8 matrix/gradients, source IEEE/AD or serial/scan parity.
-/
namespace PNDS.LangMem078

def slowRetention (rate : ℝ) : ℝ := Real.exp (-rate)

theorem slowRetention_pos (rate : ℝ) :
    0 < slowRetention rate := by
  unfold slowRetention
  exact Real.exp_pos _

theorem slowRetention_lt_one (rate : ℝ) (h : 0 < rate) :
    slowRetention rate < 1 := by
  unfold slowRetention
  have hn : -rate < 0 := by linarith
  have he := Real.exp_lt_exp.mpr hn
  simpa only [Real.exp_zero] using he

def slowScalarP (rho beta wSq : ℝ) : ℝ := rho*(1-beta*wSq)

def slowScalarStep (old rho beta wSq q : ℝ) : ℝ :=
  old * slowScalarP rho beta wSq + q

theorem slow_scalar_same_input_difference
    (a b rho beta wSq q : ℝ) :
    slowScalarStep a rho beta wSq q - slowScalarStep b rho beta wSq q =
    (a-b)*slowScalarP rho beta wSq := by
  dsimp [slowScalarStep]
  ring

theorem slow_scalar_directional_bound
    (rho beta wSq : ℝ)
    (hr : 0 <= rho) (hb : 0 <= beta) (hbp : beta <= 1)
    (hw : 0 <= wSq) (hwp : wSq <= 1) :
    0 <= slowScalarP rho beta wSq ∧
    slowScalarP rho beta wSq <= rho := by
  unfold slowScalarP
  have prod_nonneg : 0 <= beta*wSq := mul_nonneg hb hw
  have prod_le_one : beta*wSq <= 1 := by
    calc
      beta*wSq <= 1*wSq := mul_le_mul_of_nonneg_right hbp hw
      _ = wSq := by ring
      _ <= 1 := hwp
  constructor
  · exact mul_nonneg hr (by linarith)
  · calc
      rho*(1-beta*wSq) <= rho*1 :=
        mul_le_mul_of_nonneg_left (by linarith) hr
      _ = rho := by ring

def admitted (write : Bool) (old rho beta wSq q : ℝ) : ℝ :=
  if write then slowScalarStep old rho beta wSq q else old

theorem mask_preserves (old rho beta wSq q : ℝ) :
    admitted false old rho beta wSq q = old := by
  simp [admitted]

def dualParameters : Nat := 499623+4
def addedSlowStateBytes : Nat := 4*8*8
theorem registered_parameter_count : dualParameters = 499627 := by decide
theorem registered_added_state_bytes : addedSlowStateBytes = 256 := by decide
theorem registered_full_stream_bytes_from_legacy : 3904+addedSlowStateBytes = 4160 := by decide

end PNDS.LangMem078
