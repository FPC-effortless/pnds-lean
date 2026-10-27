import PNDS.LanguageCategorical
import PNDS.SourceAlignedModel
import Mathlib

/-!
TAC-LANG-FORMAL-005: complete *exact-real forward graph specification* for
TAC-LANG-GATE-003, with the source's 128-character/48-embedding/88 fast basis/
8x8 native plus 2x16x8 associative state and 75/10 decoder geometry.

This is typed executable mathematics, not yet a Torch-to-Lean implementation
refinement: floating kernels, autograd, loss training, and real-text performance
are separate obligations. Original pretrained weights are NOT specified here.
-/
namespace PNDS.LanguageFullGraph
noncomputable section
open PNDS.SourceAlignedModel

abbrev V (n : ℕ) := Vec n
abbrev M (m n : ℕ) := Mat m n
abbrev Token := Fin 128
abbrev Logits := V 128

inductive PositionMode where
  | legacy | noTime | multiscale | multiscaleRelative
  deriving DecidableEq, Repr

inductive GateMode where
  | linear | silu
  deriving DecidableEq, Repr

structure Parameters where
  embedding : Token → V 48
  timeProj : M 48 7
  keyNative : Dense 48 8
  valueNative : Dense 48 8
  fastMap : Dense 144 176
  slowMap : Dense 184 176
  nativeGates : Dense 224 2
  rotation : Fin 2 → M 8 8
  sourceContent : Dense 48 16
  sourceRouter : Dense 48 2
  sourceRetention : Dense 48 2
  sourceStrength : Dense 48 2
  sourceMod : M 2 48
  histProject : M 48 16
  readRouter : Dense 8 2
  normScale : V 248
  normShift : V 248
  normEps : ℝ
  readIn : Dense 248 75
  readOut : Dense 75 128
  residualIn : Dense 248 10
  residualOut : Dense 10 128
  residualLogit : ℝ
  positionMode : PositionMode
  gateMode : GateMode

structure State where
  sum : V 88
  native : M 8 8
  aligned : Fin 2 → M 16 8
  count : ℕ
  owner : ℕ

def vadd {n : ℕ} (x y : V n) : V n := fun i => x i + y i
def vsub {n : ℕ} (x y : V n) : V n := fun i => x i - y i
def vscale {n : ℕ} (a : ℝ) (x : V n) : V n := fun i => a * x i
def hadamard {n : ℕ} (x y : V n) : V n := fun i => x i * y i
def vmix {n : ℕ} (a b w : V n) : V n := fun i => a i + w i * (b i - a i)
def dot {n : ℕ} (a b : V n) : ℝ := ∑ i, a i * b i
def zeros (n : ℕ) : V n := fun _ => 0
def mread {rows cols : ℕ} (m : M rows cols) (k : V cols) : V rows :=
  fun i => ∑ j, m i j * k j

def sourcePhi (t : ℕ) : V 8 := fun j =>
  let p : ℕ := if j.val < 2 then 7 else
    if j.val < 4 then 13 else if j.val < 6 then 29 else 59
  let phase : ℝ := ((t % p : ℕ) : ℝ) * (2 * Real.pi / (p : ℝ))
  if j.val % 2 = 0 then Real.cos phase else Real.sin phase

def timeFeatures (mode : PositionMode) (t : ℕ) : V 7 := fun j =>
  let f : ℝ := (t : ℝ)
  match mode with
  | .noTime => 0
  | .legacy =>
    if j.val = 0 then Real.log (1+f)
    else if j.val = 1 then Real.sin (f/16)
    else if j.val = 2 then Real.cos (f/16)
    else if j.val = 3 then Real.log 2
    else if j.val = 4 then Real.log (f+2)
    else if j.val = 5 then Real.log 2
    else if t > 0 then 1 else 0
  | .multiscale | .multiscaleRelative =>
    if j.val = 0 then Real.log (1+f)
    else if j.val = 1 then Real.sin (f/16)
    else if j.val = 2 then Real.cos (f/16)
    else if j.val = 3 then Real.sin (f/64)
    else if j.val = 4 then Real.cos (f/64)
    else if j.val = 5 then Real.sin (f/256)
    else Real.cos (f/256)

def alignedKey (p : Parameters) (t : ℕ) (h : Fin 2) : V 8 :=
  normalize (linear (p.rotation h) (sourcePhi t))

def readAligned (p : Parameters) (s : State) (query : ℕ) : V 16 :=
  let phi := sourcePhi query
  let route := softmax (dense p.readRouter phi)
  fun i => ∑ h : Fin 2, route h * mread (s.aligned h) (alignedKey p query h) i

def lagIndex : Fin 4 → ℕ := fun j =>
  if j.val = 0 then 1 else if j.val = 1 then 8 else
  if j.val = 2 then 32 else 128

def relativeRead (p : Parameters) (s : State) (nativeKey : V 8) : V 16 :=
  if p.positionMode = .multiscaleRelative then
    let scores : V 4 := fun j =>
      if lagIndex j ≤ s.count
      then dot nativeKey (normalize (sourcePhi (s.count-lagIndex j)))
      else -10000
    let attention := softmax scores
    fun i => ∑ j : Fin 4,
      attention j * (if lagIndex j ≤ s.count
        then readAligned p s (s.count-lagIndex j) i else 0)
  else if s.count ≥ 1 then readAligned p s (s.count-1)
  else zeros 16

def writeNative (old : M 8 8) (key value : V 8) (a b : ℝ) : M 8 8 :=
  fun i j => a*old i j + b*(value i - a*mread old key i)*key j

def sourceBeta (p : Parameters) (e : V 48) : V 2 :=
  let route := softmax (dense p.sourceRouter e)
  let raw := linear p.sourceMod e
  let modulated := match p.gateMode with
    | .linear => raw
    | .silu => fun h => silu (2 * raw h)
  let strength := dense p.sourceStrength e
  fun h => sig (strength h + modulated h) * route h

def writeAligned (p : Parameters) (old : Fin 2 → M 16 8)
    (ordinal : ℕ) (value : V 16) (retention beta : V 2) :
    Fin 2 → M 16 8 :=
  fun h i j =>
    retention h * old h i j +
    beta h * (value i - retention h *
       mread (old h) (alignedKey p ordinal h) i) * (alignedKey p ordinal h j)

def norm248 (p : Parameters) (x : V 248) : V 248 :=
  let μ := (∑ i : Fin 248, x i)/248
  let variance := (∑ i : Fin 248, (x i-μ)^2)/248
  let denom := Real.sqrt (variance+p.normEps)
  fun i => ((x i-μ)/denom)*p.normScale i+p.normShift i

def emit (p : Parameters) (inputs : V 248) : Logits :=
  let normed := norm248 p inputs
  let primary := dense p.readOut (map silu (dense p.readIn normed))
  let residual := dense p.residualOut (map silu (dense p.residualIn normed))
  vadd primary (vscale (sig p.residualLogit) residual)

/-- The entire source's read-before-write language step, expressed in exact reals.
    The returned state is used *only at the next position*. -/
def step (p : Parameters) (s : State) (token : Token) : Logits × State :=
  let e := map th (vadd (p.embedding token)
    (linear p.timeProj (timeFeatures p.positionMode s.count)))
  let basis := map th (first (dense p.fastMap
    (cat (cat e (zeros 88)) (zeros 8))))
  let context : V 88 := fun i => s.sum i / ((max s.count 1 : ℕ) : ℝ)
  let provisional := dense p.fastMap (cat (cat e context) (zeros 8))
  let fast0 := vmix context (map th (first provisional)) (map sig (second provisional))
  let coarse := dense p.slowMap (cat (cat fast0 context) (zeros 8))
  let slow0 := vmix context (map th (vadd context (map th (first coarse))))
    (vscale (1/4) (map sig (second coarse)))
  let nk := normalize (dense p.keyNative e)
  let nv := normalize (dense p.valueNative e)
  let gates := dense p.nativeGates (cat e (cat fast0 slow0))
  let b := sig (gates 1)
  let a := sig (gates 0) * (1-b/4)
  let priorNative := mread s.native nk
  let afterNative := writeNative s.native nk nv a b
  let fine := dense p.fastMap (cat (cat e fast0) priorNative)
  let fast := vmix fast0 (map th (first fine)) (map sig (second fine))
  let refinement := dense p.slowMap (cat (cat fast slow0) priorNative)
  let slow := vmix slow0 (map th (vadd slow0 (map th (first refinement))))
    (vscale (1/4) (map sig (second refinement)))
  let value := map th (dense p.sourceContent e)
  let α := map sig (dense p.sourceRetention e)
  let β := sourceBeta p e
  let afterAligned := writeAligned p s.aligned s.count value α β
  let past := relativeRead p s nk
  let logits := emit p (cat (cat (cat (cat fast slow) priorNative) e) past)
  let next : State := {
    sum := vadd s.sum basis
    native := afterNative
    aligned := afterAligned
    count := s.count + 1
    owner := s.owner
  }
  (logits, next)

/-- The historical head looks up only the existing prior state.
    The observed historical target is NOT an argument to this function. -/
def historicalLogits (p : Parameters) (s : State) (lag : ℕ) : Logits :=
  let read := readAligned p s (s.count-lag)
  let latent := linear p.histProject read
  fun i => dot (p.embedding i) latent

def advance (p : Parameters) : State → List Token → State
  | s, [] => s
  | s, t :: ts => advance p (step p s t).2 ts

def emissions (p : Parameters) : State → List Token → List Logits
  | _, [] => []
  | s, t :: ts => (step p s t).1 :: emissions p (step p s t).2 ts

theorem step_owner (p : Parameters) (s : State) (x : Token) :
    (step p s x).2.owner = s.owner := by
  simp [step]

theorem step_counter (p : Parameters) (s : State) (x : Token) :
    (step p s x).2.count = s.count+1 := by
  simp [step]

theorem advance_append (p : Parameters) (s : State) (xs ys : List Token) :
    advance p s (xs++ys) = advance p (advance p s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => simpa [advance] using ih (step p s x).2

/-- Complete categorical emission at every source step is normalized. -/
theorem token_probability_sum_one (p : Parameters) (s : State) (x : Token) :
    (∑ k : Fin 128, PNDS.LanguageCategorical.probability (step p s x).1 k) = 1 :=
  PNDS.LanguageCategorical.probability_sum_one _

/-- Future suffixes cannot change a source-prefix logit, all learned maps
    included. This proves mathematical causality, not IEEE or Torch equivalence. -/
theorem prefix_causal (p : Parameters) (s : State) (xs ys : List Token) :
    (emissions p s (xs++ys)).take xs.length = emissions p s xs := by
  induction xs generalizing s with
  | nil => simp [emissions]
  | cons x xs ih =>
    simp [emissions, ih]

/-- The owner is unchanged across arbitrary streams. -/
theorem advance_owner (p : Parameters) (s : State) (xs : List Token) :
    (advance p s xs).owner = s.owner := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih =>
    simpa [advance, step_owner] using ih (step p s x).2

end
end PNDS.LanguageFullGraph
