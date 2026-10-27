import PNDS.SourceAlignedExecution
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Abs

/-!
Compositional real automatic differentiation for the scalar primitives of
PAR-MEM-010, a simultaneous memory-write JVP, and full-model derivative
preservation by the scan rewrite. Nondifferentiable clamp ties are explicitly
classified; a chosen Torch subgradient is not called a classical derivative.
-/
namespace PNDS.SourceAlignedDifferentiation

noncomputable section
open PNDS.SourceAlignedMemory PNDS.SourceAlignedModel PNDS.SourceAlignedExecution
open Filter Set
open scoped Topology

inductive Expr (n : ℕ) where
  | input : Fin n → Expr n
  | constant : ℝ → Expr n
  | add : Expr n → Expr n → Expr n
  | mul : Expr n → Expr n → Expr n
  | neg : Expr n → Expr n
  | inv : Expr n → Expr n
  | exp : Expr n → Expr n
  | log : Expr n → Expr n
  | sqrt : Expr n → Expr n
  | sin : Expr n → Expr n
  | cos : Expr n → Expr n
  | max : Expr n → Expr n → Expr n

def eval {n : ℕ} : Expr n → (Fin n → ℝ) → ℝ
  | .input i, x => x i
  | .constant c, _ => c
  | .add a b, x => eval a x + eval b x
  | .mul a b, x => eval a x * eval b x
  | .neg a, x => -eval a x
  | .inv a, x => (eval a x)⁻¹
  | .exp a, x => Real.exp (eval a x)
  | .log a, x => Real.log (eval a x)
  | .sqrt a, x => Real.sqrt (eval a x)
  | .sin a, x => Real.sin (eval a x)
  | .cos a, x => Real.cos (eval a x)
  | .max a b, x => max (eval a x) (eval b x)

def tangent {n : ℕ} : Expr n → (Fin n → ℝ) → (Fin n → ℝ) → ℝ
  | .input i, _, dx => dx i
  | .constant _, _, _ => 0
  | .add a b, x, dx => tangent a x dx + tangent b x dx
  | .mul a b, x, dx => tangent a x dx * eval b x + eval a x * tangent b x dx
  | .neg a, x, dx => -tangent a x dx
  | .inv a, x, dx => -tangent a x dx / (eval a x)^2
  | .exp a, x, dx => Real.exp (eval a x) * tangent a x dx
  | .log a, x, dx => tangent a x dx / eval a x
  | .sqrt a, x, dx => tangent a x dx / (2*Real.sqrt (eval a x))
  | .sin a, x, dx => Real.cos (eval a x) * tangent a x dx
  | .cos a, x, dx => -Real.sin (eval a x) * tangent a x dx
  | .max a b, x, dx => if eval a x < eval b x then tangent b x dx else tangent a x dx

def Regular {n : ℕ} : Expr n → (Fin n → ℝ) → Prop
  | .input _, _ => True
  | .constant _, _ => True
  | .add a b, x => Regular a x ∧ Regular b x
  | .mul a b, x => Regular a x ∧ Regular b x
  | .neg a, x => Regular a x
  | .inv a, x => Regular a x ∧ eval a x ≠ 0
  | .exp a, x => Regular a x
  | .log a, x => Regular a x ∧ eval a x ≠ 0
  | .sqrt a, x => Regular a x ∧ eval a x ≠ 0
  | .sin a, x => Regular a x
  | .cos a, x => Regular a x
  | .max a b, x => Regular a x ∧ Regular b x ∧ eval a x ≠ eval b x

/-- Forward AD is the actual derivative for every composition of the
complete scalar vocabulary, at explicitly regular points and any input curve.
The theorem is not limited to linear maps or the associative-memory kernel. -/
theorem forward_ad_correct {n : ℕ} (e : Expr n)
    (curve : Fin n → ℝ → ℝ) (dx : Fin n → ℝ) (t : ℝ)
    (hc : ∀ i, HasDerivAt (curve i) (dx i) t)
    (hr : Regular e (fun i => curve i t)) :
    HasDerivAt (fun z => eval e (fun i => curve i z))
      (tangent e (fun i => curve i t) dx) t := by
  induction e with
  | input i => exact hc i
  | constant c => exact hasDerivAt_const t c
  | add a b iha ihb => exact (iha hr.1).add (ihb hr.2)
  | mul a b iha ihb => exact (iha hr.1).mul (ihb hr.2)
  | neg a ih => exact (ih hr).neg
  | inv a ih => exact (ih hr.1).inv hr.2
  | exp a ih => exact (ih hr).exp
  | log a ih => exact (ih hr.1).log hr.2
  | sqrt a ih => exact (ih hr.1).sqrt hr.2
  | sin a ih => exact (ih hr).sin
  | cos a ih => exact (ih hr).cos
  | max a b iha ihb =>
    have ha := iha hr.1
    have hb := ihb hr.2.1
    by_cases hab : eval a (fun i => curve i t) < eval b (fun i => curve i t)
    · have he := ha.continuousAt.eventually_lt hb.continuousAt hab
      have hd := hb.congr_of_eventuallyEq (he.mono (fun z hz => max_eq_right hz.le))
      simpa only [eval, tangent, if_pos hab] using hd
    · have hba : eval b (fun i => curve i t) < eval a (fun i => curve i t) :=
        lt_of_le_of_ne (le_of_not_gt hab) (Ne.symm hr.2.2)
      have he := hb.continuousAt.eventually_lt ha.continuousAt hba
      have hd := ha.congr_of_eventuallyEq (he.mono (fun z hz => max_eq_left hz.le))
      simpa only [eval, tangent, if_neg hab] using hd

def rowJVP (row key drow dkey : SourceRow) (value alpha beta dvalue dalpha dbeta : ℝ) :
    SourceRow :=
  let old := rowRead row key
  let dold := rowRead drow key + rowRead row dkey
  fun j => dalpha*row j + alpha*drow j +
    (dbeta*(value-alpha*old) + beta*(dvalue-(dalpha*old+alpha*dold)))*key j +
    beta*(value-alpha*old)*dkey j

theorem row_read_derivative (row key : ℝ → SourceRow) (drow dkey : SourceRow) (t : ℝ)
    (hm : ∀ j, HasDerivAt (fun z => row z j) (drow j) t)
    (hk : ∀ j, HasDerivAt (fun z => key z j) (dkey j) t) :
    HasDerivAt (fun z => rowRead (row z) (key z))
      (rowRead drow (key t) + rowRead (row t) dkey) t := by
  have h := HasDerivAt.sum (u := Finset.univ) (fun j _ => (hm j).mul (hk j))
  simpa only [rowRead, Finset.sum_add_distrib] using h

/-- All row coordinates, key coordinates, value and both gates vary
simultaneously. This proves the JVP used to derive the independent VJP. -/
theorem corrective_write_jvp (row key : ℝ → SourceRow) (drow dkey : SourceRow)
    (value alpha beta : ℝ → ℝ) (dvalue dalpha dbeta t : ℝ)
    (hm : ∀ j, HasDerivAt (fun z => row z j) (drow j) t)
    (hk : ∀ j, HasDerivAt (fun z => key z j) (dkey j) t)
    (hv : HasDerivAt value dvalue t) (ha : HasDerivAt alpha dalpha t)
    (hb : HasDerivAt beta dbeta t) (j : Fin 8) :
    HasDerivAt (fun z => writeRow (row z) (key z) (value z) (alpha z) (beta z) true j)
      (rowJVP (row t) (key t) drow dkey (value t) (alpha t) (beta t)
        dvalue dalpha dbeta j) t := by
  have hr := row_read_derivative row key drow dkey t hm hk
  have h := (ha.mul (hm j)).add ((hb.mul (hv.sub (ha.mul hr))).mul (hk j))
  simpa only [rowJVP, writeRow, Bool.true_eq, ↓reduceIte, mul_assoc, add_assoc] using h

theorem rejected_write_jvp (row key : ℝ → SourceRow) (drow : SourceRow)
    (value alpha beta : ℝ → ℝ) (t : ℝ)
    (hm : ∀ j, HasDerivAt (fun z => row z j) (drow j) t) (j : Fin 8) :
    HasDerivAt (fun z => writeRow (row z) (key z) (value z) (alpha z) (beta z) false j)
      (drow j) t := hm j

/-- Nonsmooth ties are a real mathematical obstruction. A clamp cannot
have a classical derivative at zero; this is not an unfinished proof. -/
theorem clamp_not_differentiable_at_zero :
    ¬ DifferentiableAt ℝ (fun x : ℝ => max x 0) 0 := by
  intro h
  have hd := (h.const_mul 2).sub differentiableAt_id
  change DifferentiableAt ℝ (fun x : ℝ => 2*max x 0-x) 0 at hd
  have heq : (fun x : ℝ => 2*max x 0-x) = abs := by
    funext x
    rcases le_total 0 x with hx | hx
    · rw [max_eq_left hx, abs_of_nonneg hx]
      ring
    · rw [max_eq_right hx, abs_of_nonpos hx]
      ring
  rw [heq] at hd
  exact not_differentiableAt_abs_zero hd

theorem clamp_right_derivative :
    HasDerivWithinAt (fun x : ℝ => max x 0) 1 (Ici 0) 0 := by
  apply (hasDerivWithinAt_id (0 : ℝ) (Ici 0)).congr_of_mem
  · intro x hx
    exact max_eq_left hx
  · exact Set.left_mem_Ici

theorem clamp_left_derivative :
    HasDerivWithinAt (fun x : ℝ => max x 0) 0 (Iic 0) 0 := by
  apply (hasDerivWithinAt_const (0 : ℝ) (Iic 0) (0 : ℝ)).congr_of_mem
  · intro x hx
    exact max_eq_right hx
  · exact Set.right_mem_Iic

def parallelPrediction (w : Parameters) (s : State) (past : List Input) (current : Input) :
    Vec 14 :=
  let reconstructed := restore (contextRun w s past)
    (jointApply (((prepass w (eraseMemory s) past).map eventSchedule).foldl
      jointThen jointIdentity) (memories s))
  decode w reconstructed current (prepare w (contextRun w s past) current)

/-- The full neural scan rewrite preserves derivatives for arbitrary
parameter/input/state curves, losses on any output component, and history.
Existence of a classical derivative remains an explicit premise at kinks. -/
theorem full_model_derivative_preserved (w : ℝ → Parameters) (s : ℝ → State)
    (past : ℝ → List Input) (current : ℝ → Input) (j : Fin 14) (derivative t : ℝ)
    (h : HasDerivAt (fun z => (step (w z) (runState (w z) (s z) (past z)) (current z)).1 j)
      derivative t) :
    HasDerivAt (fun z => parallelPrediction (w z) (s z) (past z) (current z) j)
      derivative t := by
  have heq : (fun z => parallelPrediction (w z) (s z) (past z) (current z) j) =
      (fun z => (step (w z) (runState (w z) (s z) (past z)) (current z)).1 j) := by
    funext z
    exact congrFun (full_prediction_after_scan (w z) (s z) (past z) (current z)) j
  rw [heq]
  exact h

end
end PNDS.SourceAlignedDifferentiation
