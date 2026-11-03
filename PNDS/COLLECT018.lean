import Mathlib
import PNDS.COLLECT016

/- Source-specific partial exact-real formulation. Full source/IEEE/VJP proof
   remains OPEN. No claim of argmax differentiability or successful retrieval.
-/
noncomputable section
namespace COLLECT018
abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (n m : Nat) := Fin n → Fin m → ℝ
def affine {n m : Nat} (w : Mat n m) (x : Vec m) (b : Vec n) : Vec n :=
  fun i => (∑ j, w i j * x j) + b i
def nativeContext (rms : Vec 16 → Vec 16) (projection : Vec 392 → Vec 16)
    (scale : ℝ) (x : Vec 16) (nativeBundleWithZeroPhase : Vec 392) : Vec 16 :=
  rms (fun i => x i + scale * projection nativeBundleWithZeroPhase i)
structure SeparatedPhaseInputs where
  valueInput : Vec 16
  addressInput : Vec 16
def separated (valueInput addressInput : Vec 16) : SeparatedPhaseInputs :=
  ⟨valueInput, addressInput⟩
def gruStep (sigmoid tanh : ℝ → ℝ) (inputAffine hiddenAffine : Vec 375)
    (previous : Vec 125) : Vec 125 := fun i =>
  let r := sigmoid (inputAffine ⟨i.val, by omega⟩ + hiddenAffine ⟨i.val, by omega⟩)
  let z := sigmoid (inputAffine ⟨125+i.val, by omega⟩ + hiddenAffine ⟨125+i.val, by omega⟩)
  let n := tanh (inputAffine ⟨250+i.val, by omega⟩ + r * hiddenAffine ⟨250+i.val, by omega⟩)
  (1-z)*n + z*previous i
def jointLoss (natural task answer balance : ℝ) : ℝ :=
  natural + task/4 + answer + balance/100
def firstLoss (natural task answer balance firstNLL : ℝ) : ℝ :=
  jointLoss natural task answer balance + 3*firstNLL
theorem value_path_independent (v a b : Vec 16) :
    (separated v a).valueInput = (separated v b).valueInput := rfl
theorem first_loss_increment (n t a b f : ℝ) :
    firstLoss n t a b f - jointLoss n t a b = 3*f := by
  simp only [firstLoss]
  ring
theorem first_loss_not_smaller (n t a b f : ℝ) (nonnegative : 0 ≤ f) :
    jointLoss n t a b ≤ firstLoss n t a b f := by
  unfold firstLoss
  nlinarith
structure TypedSequenceGraph (observation state : Type) where
  initial : state
  transition : state → observation → state
  logits : state → observation → Vec 256
def carry {O S : Type} (g : TypedSequenceGraph O S) (xs : List O) : S :=
  xs.foldl g.transition g.initial
end COLLECT018
