import Mathlib

/-!
LANGMEM083 — scoped exact-source-index and frozen-probe parameter contract.

The original TAC source has 499627 learned model weights, held frozen.
The diagnostic auxiliary head is 20→32→256 (2 dense affine layers),
9,120 newly fitted learned weights/biases. A copy of the original
supplied writer key and its 8-dimensional compressed source readout,
plus one-hot age of four categories, constitute exactly 20 features.

A historical byte at source clock a=t-age is an already-observed target,
but MUST NOT be supplied to the auxiliary feature producer. The model
query feature must only depend on original bytes observed through t.
This file proves integer dimensional bookkeeping and list-prefix
future-byte exclusion only; actual PyTorch source feature derivation,
IEEE forward/backward/AD, learned logits, state preservation, and
historical semantic recall are OPEN.
-/
namespace PNDS.LangMem083

def frozenBaseParameters : Nat := 499627
def probeInputWidth : Nat := 8+8+4
def probeHiddenWidth : Nat := 32
def probeOutputClasses : Nat := 256
def trainedProbeParameters : Nat :=
    probeInputWidth*probeHiddenWidth+probeHiddenWidth+
    probeHiddenWidth*probeOutputClasses+probeOutputClasses
def deployedTotalParameters : Nat := frozenBaseParameters+trainedProbeParameters
def additionalProbeFP32WeightBytes : Nat := 4*trainedProbeParameters

theorem input_width_registered : probeInputWidth = 20 := by decide
theorem probe_exact_fitted_weight_count : trainedProbeParameters = 9120 := by decide
theorem frozen_full_model_plus_probe : deployedTotalParameters = 508747 := by decide
theorem additional_FP32_bytes : additionalProbeFP32WeightBytes = 36480 := by decide

def readWidth : Nat := 8
def writerWidth : Nat := 8
def ageOnehotWidth : Nat := 4
theorem three_feature_components :
    readWidth + writerWidth + ageOnehotWidth = probeInputWidth := by decide

theorem future_target_not_in_observed_prefix (prefix : List Nat) (future : Nat) :
    (prefix ++ [future]).take prefix.length = prefix := by
  simp

theorem same_prefix_for_any_future_target (prefix : List Nat) (a b : Nat) :
    (prefix ++ [a]).take prefix.length = (prefix ++ [b]).take prefix.length := by
  simp

def permittedAges : List Nat := [8,32,128,256]
theorem all_registered_age_delays_positive :
    (permittedAges.all fun age => decide (0<age)) = true := by decide

end PNDS.LangMem083
