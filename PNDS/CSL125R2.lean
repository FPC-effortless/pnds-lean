/-
  CSL-125-R2 independent mathematical record: exact-real selective admission.
  LOCAL SOURCE CANDIDATE ONLY until Lean compilation and source correspondence.
  Not a full neural model semantics/gradient/information-flow proof.
-/
import Mathlib
namespace CSL125R2

def beta (rate gate : ℝ) : ℝ := Real.exp (-(rate * gate))
def numStep (rate gate address value numerator : ℝ) : ℝ :=
  beta rate gate * numerator + (1-beta rate gate)*address*value
def massStep (rate gate address mass : ℝ) : ℝ :=
  beta rate gate * mass + (1-beta rate gate)*address

theorem no_write_holds_numerator (rate address value numerator : ℝ) :
    numStep rate 0 address value numerator = numerator := by
  simp [numStep,beta]

theorem no_write_holds_mass (rate address mass : ℝ) :
    massStep rate 0 address mass = mass := by
  simp [massStep,beta]

theorem full_write_matches_decay (rate address value numerator : ℝ) :
    numStep rate 1 address value numerator =
      Real.exp (-rate)*numerator + (1-Real.exp (-rate))*address*value := by
  simp [numStep,beta]

theorem step_upper (rate gate address value numerator mass : ℝ)
    (hBetaNonneg : 0 ≤ beta rate gate)
    (hBetaLeOne : beta rate gate ≤ 1)
    (hAddress : 0 ≤ address) (hValue : value ≤ 1)
    (hState : numerator ≤ mass) :
    numStep rate gate address value numerator ≤ massStep rate gate address mass := by
  unfold numStep massStep
  have hLeft := mul_le_mul_of_nonneg_left hState hBetaNonneg
  have hCoeff : 0 ≤ (1-beta rate gate)*address :=
    mul_nonneg (sub_nonneg.mpr hBetaLeOne) hAddress
  have hRight : ((1-beta rate gate)*address)*value ≤ (1-beta rate gate)*address := by
    nlinarith [mul_nonneg hCoeff (sub_nonneg.mpr hValue)]
  linarith

theorem step_lower (rate gate address value numerator mass : ℝ)
    (hBetaNonneg : 0 ≤ beta rate gate)
    (hBetaLeOne : beta rate gate ≤ 1)
    (hAddress : 0 ≤ address) (hValue : -1 ≤ value)
    (hState : -mass ≤ numerator) :
    -massStep rate gate address mass ≤ numStep rate gate address value numerator := by
  unfold numStep massStep
  have hLeft := mul_le_mul_of_nonneg_left hState hBetaNonneg
  have hCoeff : 0 ≤ (1-beta rate gate)*address :=
    mul_nonneg (sub_nonneg.mpr hBetaLeOne) hAddress
  have hRight : -((1-beta rate gate)*address) ≤ ((1-beta rate gate)*address)*value := by
    nlinarith [mul_nonneg hCoeff (sub_nonneg.mpr hValue)]
  linarith

theorem model_parameter_ceiling : 123822574 ≤ 125000000 := by decide
end CSL125R2
