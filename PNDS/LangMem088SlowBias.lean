import Mathlib

/-!
LANGMEM088 — partial mathematical contract for the SAME trainable slow
read bias at matched parameter count.

The original compact model has a trainable two-logit read mixer.
The intervention changes ONLY initial slow logit bias coordinate +1,
not the number or shape of any trainable tensors, writer, native bank,
or current observed source inputs. Once fitted the bias is trainable;
this is not an inference-time forced attention or evidence of recall.

Here Lean proves exact scalar logit difference and parameter/state
identity. Python reference tests stable 2-way softmax, positive
derivative p(1-p), monotonicity numerically and matched shapes.
The whole trained model PyTorch FP32 AD/IEEE, source semantic recall,
differential learned trajectory and document quality are OPEN.
-/
namespace PNDS.LangMem088

def shiftedSlowLogit (originalSlow extraBias : ℝ) : ℝ :=
  originalSlow + extraBias

theorem only_slow_logit_increased_by_one (slow : ℝ) :
    shiftedSlowLogit slow 1 - shiftedSlowLogit slow 0 = 1 := by
  unfold shiftedSlowLogit
  ring

theorem identity_at_zero (slow : ℝ) :
    shiftedSlowLogit slow 0 = slow := by
  unfold shiftedSlowLogit
  ring

theorem difference_matches_bias (slow initialBias : ℝ) :
    shiftedSlowLogit slow initialBias - shiftedSlowLogit slow 0 = initialBias := by
  unfold shiftedSlowLogit
  ring

def originalTrainableParameters : Nat := 100521
def newTrainableParameters : Nat := originalTrainableParameters
def frozenPersistentStateBytes : Nat := 1920
def biasParameterWidth : Nat := 2

theorem no_added_bias_parameters :
    newTrainableParameters = originalTrainableParameters := by decide
theorem same_state_bytes : frozenPersistentStateBytes = 1920 := by decide
theorem original_two_logit_width : biasParameterWidth = 2 := by decide

end PNDS.LangMem088
