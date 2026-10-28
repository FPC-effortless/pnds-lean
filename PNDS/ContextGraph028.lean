import Mathlib

/-! Exact-real chronological CONTEXT026 graph, source SHA c88f3f7c... .
No extracted execution/IEEE/AD equivalence is asserted. The inverse below is
Mathlib's totalized inverse; correspondence to torch.linalg.solve additionally
requires nonsingularity, which remains an explicit OPEN obligation. -/
namespace PNDS.ContextGraph028

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (m n : Nat) := Matrix (Fin m) (Fin n) ℝ

noncomputable section

def append {n m : Nat} (x : Vec n) (y : Vec m) : Vec (n+m) := fun i =>
  if h : i.val < n then x ⟨i.val,h⟩ else y ⟨i.val-n,by have := i.isLt; omega⟩

def dot {n : Nat} (x y : Vec n) : ℝ := ∑ i, x i * y i

def linear {n m : Nat} (w : Mat m n) (b : Vec m) (x : Vec n) : Vec m :=
  fun i => dot (w i) x + b i

def unit {n : Nat} (x : Vec n) : Vec n :=
  fun i => x i / max (1/1000000 : ℝ) (Real.sqrt (dot x x))

def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))
def silu (x : ℝ) : ℝ := x * sigmoid x

def softmax {n : Nat} (x : Vec n) : Vec n :=
  fun i => Real.exp (x i) / ∑ j, Real.exp (x j)

def clockFeatures (t : Nat) : Vec 7 :=
  ![Real.log (1+(t : ℝ)), Real.sin ((t : ℝ)/16), Real.cos ((t : ℝ)/16),
    Real.sin ((t : ℝ)/64), Real.cos ((t : ℝ)/64),
    Real.sin ((t : ℝ)/256), Real.cos ((t : ℝ)/256)]

def periods : Fin 4 → Nat := ![7,13,29,59]
def ordinal (t : Nat) : Vec 8 := fun i =>
  let period := periods ⟨i.val/2,by have := i.isLt; omega⟩
  let angle := ((t % period : ℕ) : ℝ) * (2*Real.pi/(period : ℝ))
  if i.val % 2 = 0 then Real.cos angle else Real.sin angle

structure Weights where
  embedding : Mat 256 48
  pos : Mat 48 7
  basisW : Mat 88 48
  basisB : Vec 88
  lags : Mat 3 48
  keyW : Mat 8 48
  keyB : Vec 8
  valueW : Mat 8 48
  valueB : Vec 8
  nativeGateW : Mat 2 48
  nativeGateB : Vec 2
  alignedValueW : Mat 16 48
  alignedValueB : Vec 16
  alignedGateW : Mat 6 48
  alignedGateB : Vec 6
  rotation : Fin 2 → Mat 8 8
  waveDecayW : Mat 8 48
  waveDecayB : Vec 8
  waveInputW : Mat 16 48
  waveInputB : Vec 16
  waveAngle : Vec 8
  conceptValueW : Mat 96 48
  conceptValueB : Vec 96
  conceptGateW : Mat 96 48
  conceptGateB : Vec 96
  fastW : Mat 88 160
  fastB : Vec 88
  slowW : Mat 88 184
  slowB : Vec 88
  evidenceW : Mat 96 256
  evidenceB : Vec 96
  conceptQ : Mat 16 16
  conceptK : Mat 16 16
  normWeight : Vec 280
  normBias : Vec 280
  upW : Mat 501 280
  upB : Vec 501
  gateW : Mat 501 280
  gateB : Vec 501
  outW : Mat 256 501
  outB : Vec 256
  conceptOut : Mat 256 16

structure State where
  clock : Nat
  native : Mat 8 8
  aligned : Fin 2 → Mat 16 8
  wave : Mat 8 2
  concepts : Mat 6 16
  basisSum : Vec 88
  lexicalPast : Mat 3 48
  previousKey : Vec 8

def zeroState : State := ⟨0,0,fun _ => 0,0,0,0,0,0⟩

def unflatten {n m : Nat} (x : Vec (n*m)) : Mat n m :=
  fun i j => x ⟨i.val*m+j.val,by
    have hm : i.val*m+m ≤ n*m := by
      simpa [Nat.succ_mul] using Nat.mul_le_mul_right m (Nat.succ_le_of_lt i.isLt)
    have := j.isLt
    omega⟩

def waveFlat (w : Mat 8 2) : Vec 16 := fun i =>
  w ⟨i.val/2,by have := i.isLt; omega⟩ ⟨i.val%2,by omega⟩

/-- Separate erase/write scalars; native uses equal values, aligned does not. -/
def edit {rows : Nat} (m : Mat rows 8) (k : Vec 8) (v : Vec rows)
    (a erase write : ℝ) (admitted : Bool) : Mat rows 8 := fun i j =>
  if admitted then a*m i j - a*erase*dot (m i) k*k j + write*v i*k j else m i j

def read {rows : Nat} (m : Mat rows 8) (k : Vec 8) : Vec rows :=
  fun i => dot (m i) k

def layerNorm (w b : Vec 280) (x : Vec 280) : Vec 280 :=
  let mean := (∑ i, x i)/280
  let variance := (∑ i, (x i-mean)^2)/280
  fun i => w i*(x i-mean)/Real.sqrt (variance+1/100000) + b i

/-- One observed byte is consumed, then a NEXT-byte logit vector is emitted. -/
def step (w : Weights) (s : State) (token : Fin 256) (admit : Bool) : State × Vec 256 :=
  let t := s.clock
  let lexical : Vec 48 := fun c => Real.tanh (w.embedding token c)
  let pos := linear w.pos 0 (clockFeatures t)
  let e : Vec 48 := fun c => Real.tanh (w.embedding token c+pos c)
  let contextual : Vec 48 := fun c => Real.tanh (lexical c + ∑ lag, Real.tanh (w.lags lag c)*s.lexicalPast lag c)
  let key := unit (linear w.keyW w.keyB contextual)
  let value : Vec 8 := fun i => Real.tanh (linear w.valueW w.valueB e i)
  let ng := linear w.nativeGateW w.nativeGateB e
  let a := (199/200 : ℝ)*sigmoid (ng 0)
  let beta := sigmoid (ng 1)
  let native := edit s.native s.previousKey value a beta beta (admit && decide (0<t))
  let nativeRead := read native key
  let ak : Fin 2 → Vec 8 := fun h => unit (fun j => ∑ i, ordinal t i*w.rotation h i j)
  let pk : Fin 2 → Vec 8 := fun h => unit (fun j => ∑ i, ordinal (t-1) i*w.rotation h i j)
  let alignedRead : Vec 16 := fun i => if t=0 then 0 else (∑ h, read (s.aligned h) (pk h) i)/2
  let ag := unflatten (n:=2) (m:=3) (fun i => sigmoid (linear w.alignedGateW w.alignedGateB e i))
  let av : Vec 16 := fun i => Real.tanh (linear w.alignedValueW w.alignedValueB e i)
  let aligned : Fin 2 → Mat 16 8 := fun h => edit (s.aligned h) (ak h) av (ag h 0) (ag h 1) (ag h 2) admit
  let ctx : Vec 88 := fun i => s.basisSum i / ((max 1 t : ℕ) : ℝ)
  let oldwave := waveFlat s.wave
  let fast : Vec 88 := fun i => Real.tanh (linear w.fastW w.fastB (append (append (append e ctx) oldwave) nativeRead) i)
  let slow : Vec 88 := fun i => Real.tanh (linear w.slowW w.slowB (append (append fast ctx) nativeRead) i)
  let evidence := unflatten (n:=6) (m:=16) (fun i => Real.tanh (linear w.evidenceW w.evidenceB
    (append (append (append (append e fast) slow) alignedRead) oldwave) i))
  let cq : Mat 6 16 := fun i => linear w.conceptQ 0 (evidence i)
  let ck : Mat 6 16 := fun i => linear w.conceptK 0 (evidence i)
  let sm : Mat 6 6 := fun i => softmax (fun j => dot (cq i) (ck j)/4)
  let adjacency : Mat 6 6 := fun i j => if i=j then 0 else (sm i j+sm j i)/2
  let laplacian : Mat 6 6 := fun i j => (if i=j then ∑ k, adjacency i k else 0)-adjacency i j
  let h : Mat 6 6 := fun i j => (if i=j then 5/4 else 0) + laplacian i j/4
  let rhs : Mat 6 16 := fun i j => evidence i j+s.concepts i j/4
  let posterior : Mat 6 16 := h⁻¹ * rhs
  let poolWeights := softmax (fun i => (∑ j, evidence i j)/16)
  let pooled : Vec 16 := fun j => ∑ i, poolWeights i*posterior i j
  let z := layerNorm w.normWeight w.normBias
    (append (append (append (append (append (append fast slow) nativeRead) alignedRead) oldwave) pooled) e)
  let up := linear w.upW w.upB z
  let gate := linear w.gateW w.gateB z
  let output := linear w.outW w.outB (fun i => up i*silu (gate i))
  let logits : Vec 256 := fun i => output i+linear w.conceptOut 0 pooled i
  let wi := unflatten (n:=8) (m:=2) (linear w.waveInputW w.waveInputB e)
  let wave : Mat 8 2 := fun i j =>
    let c := Real.cos (w.waveAngle i)
    let sin := Real.sin (w.waveAngle i)
    let rotated := if j.val=0 then s.wave i 0*c+s.wave i 1*sin else -s.wave i 0*sin+s.wave i 1*c
    (97/100 + 29/1000*sigmoid (linear w.waveDecayW w.waveDecayB e i))*rotated + Real.tanh (wi i j)/40
  let cg := unflatten (n:=6) (m:=16) (fun i => sigmoid (linear w.conceptGateW w.conceptGateB e i)/20)
  let cv := unflatten (n:=6) (m:=16) (fun i => Real.tanh (linear w.conceptValueW w.conceptValueB e i))
  let concepts : Mat 6 16 := fun i j => (1-cg i j)*s.concepts i j+cg i j*cv i j
  let basisSum : Vec 88 := fun i => s.basisSum i+Real.tanh (linear w.basisW w.basisB e i)
  let lexicalPast : Mat 3 48 := fun lag c => if hh : lag.val=0 then lexical c else
    s.lexicalPast ⟨lag.val-1,by have := lag.isLt; omega⟩ c
  (⟨t+1,native,aligned,wave,concepts,basisSum,lexicalPast,key⟩,logits)

def run (w : Weights) (tokens : Nat → Fin 256) (mask : Nat → Bool) : Nat → State
  | 0 => zeroState
  | n+1 => (step w (run w tokens mask n) (tokens n) (mask n)).1

def logitsAt (w : Weights) (tokens : Nat → Fin 256) (mask : Nat → Bool) (t : Nat) : Vec 256 :=
  (step w (run w tokens mask t) (tokens t) (mask t)).2

theorem prefix_state_eq (w : Weights) (x y : Nat → Fin 256) (mx my : Nat → Bool) (n : Nat) :
    (∀ i, i<n → x i=y i) → (∀ i, i<n → mx i=my i) → run w x mx n=run w y my n := by
  induction n with
  | zero => intro _ _; rfl
  | succ n ih =>
    intro hx hm
    have hs := ih (fun i hi => hx i (Nat.lt_trans hi (Nat.lt_succ_self n)))
      (fun i hi => hm i (Nat.lt_trans hi (Nat.lt_succ_self n)))
    simp only [run]
    rw [hs, hx n (Nat.lt_succ_self n), hm n (Nat.lt_succ_self n)]

theorem logits_prefix_causal (w : Weights) (x y : Nat → Fin 256) (mx my : Nat → Bool) (t : Nat)
    (hx : ∀ i, i≤t → x i=y i) (hm : ∀ i, i≤t → mx i=my i) :
    logitsAt w x mx t=logitsAt w y my t := by
  have hs := prefix_state_eq w x y mx my t (fun i hi => hx i (Nat.le_of_lt hi))
    (fun i hi => hm i (Nat.le_of_lt hi))
  unfold logitsAt
  rw [hs, hx t (Nat.le_refl t), hm t (Nat.le_refl t)]

theorem native_nonwrite_identity (w : Weights) (s : State) (token : Fin 256) :
    (step w s token false).1.native=s.native := by rfl

theorem first_native_write_absent (w : Weights) (token : Fin 256) (admit : Bool) :
    (step w zeroState token admit).1.native=0 := by
  cases admit <;> rfl

end
end PNDS.ContextGraph028
#print axioms PNDS.ContextGraph028.prefix_state_eq
#print axioms PNDS.ContextGraph028.logits_prefix_causal
#print axioms PNDS.ContextGraph028.native_nonwrite_identity
#print axioms PNDS.ContextGraph028.first_native_write_absent
