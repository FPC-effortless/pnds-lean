import Mathlib

/-! Full exact-real extra recurrence-only block, not composed parent/IEEE/AD.
Inputs are source-legal base features; source-to-feature equivalence is OPEN. -/
namespace PNDS.LangLayer038
abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (m n : Nat) := Fin m → Fin n → ℝ
noncomputable section

def linear {n m : Nat} (w : Mat m n) (b : Vec m) (x : Vec n) : Vec m :=
  fun i => (∑ j, w i j*x j)+b i

def sigmoid (x : ℝ) : ℝ := 1/(1+Real.exp (-x))

def join (u h : Vec 59) : Vec 118 := fun i =>
  if hh : i.val<59 then u ⟨i.val,hh⟩ else h ⟨i.val-59,by have := i.isLt; omega⟩

structure Weights where
  normW : Vec 280
  normB : Vec 280
  inputW : Mat 59 280
  inputB : Vec 59
  forgetW : Mat 59 59
  forgetB : Vec 59
  offerW : Mat 59 59
  offerB : Vec 59
  readW : Mat 59 118
  readB : Vec 59
  outW : Mat 280 59
  outB : Vec 280
  gain : ℝ

def normalized (w : Weights) (z : Vec 280) : Vec 280 :=
  let mean := (∑ i, z i)/280
  let variance := (∑ i, (z i-mean)^2)/280
  fun i => w.normW i*(z i-mean)/Real.sqrt (variance+1/100000)+w.normB i

def projected (w : Weights) (z : Vec 280) : Vec 59 :=
  fun i => Real.tanh (linear w.inputW w.inputB (normalized w z) i)

def affine (h a v : ℝ) : ℝ := a*h+(1-a)*v

def observe (w : Weights) (h : Vec 59) (z : Vec 280) : Vec 59 :=
  let u := projected w z
  fun i => affine (h i) ((199/200 : ℝ)*sigmoid (linear w.forgetW w.forgetB u i))
    (Real.tanh (linear w.offerW w.offerB u i))

def read (w : Weights) (h : Vec 59) (z : Vec 280) : Vec 280 :=
  let u := projected w z
  let gate : Vec 59 := fun i => sigmoid (linear w.readW w.readB (join u h) i)
  let combined : Vec 59 := fun i => Real.tanh (u i+gate i*h i)
  fun i => z i+sigmoid w.gain*linear w.outW w.outB combined i

def step (w : Weights) (h : Vec 59) (z : Vec 280) : Vec 59 × Vec 280 :=
  (observe w h z,read w h z)

def run (w : Weights) (z : Nat → Vec 280) : Nat → Vec 59
  | 0 => 0
  | t+1 => observe w (run w z t) (z t)

def outputAt (w : Weights) (z : Nat → Vec 280) (t : Nat) : Vec 280 :=
  read w (run w z t) (z t)

theorem sigmoid_bounds (x : ℝ) : 0≤sigmoid x ∧ sigmoid x≤1 := by
  have he := Real.exp_pos (-x)
  have hd : 0<1+Real.exp (-x) := by linarith
  constructor
  · exact div_nonneg (by norm_num) (le_of_lt hd)
  · unfold sigmoid
    apply (div_le_iff₀ hd).2
    linarith

/-- Conditional coordinate bound; not end-to-end state utility or stability. -/
theorem convex_invariant (h a v : ℝ) (ha : 0≤a ∧ a≤1)
    (hh : -1≤h ∧ h≤1) (hv : -1≤v ∧ v≤1) :
    -1≤affine h a v ∧ affine h a v≤1 := by
  have h1 := mul_nonneg ha.1 (show 0≤h+1 by linarith [hh.1])
  have h2 := mul_nonneg (show 0≤1-a by linarith [ha.2]) (show 0≤v+1 by linarith [hv.1])
  have h3 := mul_nonneg ha.1 (show 0≤1-h by linarith [hh.2])
  have h4 := mul_nonneg (show 0≤1-a by linarith [ha.2]) (show 0≤1-v by linarith [hv.2])
  unfold affine
  constructor <;> nlinarith

theorem same_schedule_difference (h g a v : ℝ) :
    affine h a v-affine g a v=a*(h-g) := by unfold affine; ring

theorem first_read_uses_zero_prior (w : Weights) (z : Nat → Vec 280) :
    outputAt w z 0=read w 0 (z 0) := by rfl

theorem prefix_state_eq (w : Weights) (x y : Nat → Vec 280) (n : Nat) :
    (∀ i, i<n → x i=y i) → run w x n=run w y n := by
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hx
    have hs := ih (fun i hi => hx i (Nat.lt_trans hi (Nat.lt_succ_self n)))
    simp only [run]
    rw [hs,hx n (Nat.lt_succ_self n)]

theorem output_prefix_causal (w : Weights) (x y : Nat → Vec 280) (t : Nat)
    (hx : ∀ i, i≤t → x i=y i) : outputAt w x t=outputAt w y t := by
  have hs := prefix_state_eq w x y t (fun i hi => hx i (Nat.le_of_lt hi))
  unfold outputAt
  rw [hs,hx t (Nat.le_refl t)]

end
end PNDS.LangLayer038
#print axioms PNDS.LangLayer038.sigmoid_bounds
#print axioms PNDS.LangLayer038.convex_invariant
#print axioms PNDS.LangLayer038.same_schedule_difference
#print axioms PNDS.LangLayer038.first_read_uses_zero_prior
#print axioms PNDS.LangLayer038.prefix_state_eq
#print axioms PNDS.LangLayer038.output_prefix_causal
