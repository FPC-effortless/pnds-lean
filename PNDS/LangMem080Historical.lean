import Mathlib

/-!
LANGMEM080 partial exact-real single historical write attribution.

For source-conditioned native affine update N_next = N*P+Q, one
individually observed source write has initial Q_a and later contribution
C_(t+1)=C_t*P_(t+1), because newer Q contributions are additive.
At a fixed query step, replacing N by N-C (with all other source states
unchanged) subtracts exactly C*P from the next native matrix in exact
real algebra. This is an instantaneous contribution, NOT replay of
history without the source event, nor a theorem of byte/fact recovery.

This Lean module proves the scalar source-affine identity only. Python
reference tests the 8x8 matrix analogue. Whole trained PyTorch/IEEE/AD,
write masks, source semantic address/labels, source read-to-logit and
independently qualified capability are explicitly OPEN.
-/
namespace PNDS.LangMem080

def originalSourceStep (n p q : ℝ) : ℝ := n*p+q
def sourceTagStep (c p : ℝ) : ℝ := c*p

theorem one_source_write_additive (n c p q : ℝ) :
    originalSourceStep (n+c) p q =
    originalSourceStep n p q + sourceTagStep c p := by
  unfold originalSourceStep sourceTagStep
  ring

theorem same_input_event_subtraction (n c p q : ℝ) :
    originalSourceStep n p q - originalSourceStep (n-c) p q =
    sourceTagStep c p := by
  unfold originalSourceStep sourceTagStep
  ring

theorem sham_subtract_zero (n p q : ℝ) :
    originalSourceStep (n-0) p q = originalSourceStep n p q := by
  ring

def registeredAnchors : List Nat := [32,64,96,128,192]
def registeredDelays : List Nat := [1,8,32,64,128,192]
def eligible (anchor delay : Nat) : Bool := anchor + delay < 320

theorem query_lag_positive : (registeredDelays.all fun d => decide (0 < d)) = true := by
  decide

theorem example_early_event_eligible : eligible 32 192 = true := by decide
theorem example_latest_event_ineligible : eligible 192 192 = false := by decide

end PNDS.LangMem080
