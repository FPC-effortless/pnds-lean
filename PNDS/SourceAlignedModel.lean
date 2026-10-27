import PNDS.SourceAlignedVerification
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Tactic.IntervalCases

/-!
Typed exact-real observation-to-output specification of default PAR-MEM-010.
TAC source: 3b278acc9daf587fd20892e470c857bfc933248e.
All default learned maps are explicit, including the inherited backbone and
post-scan FFN. This is a specification, not a verified Python/Torch compiler.
Units of dt and normalization of observations/actions are caller contracts.
Actions are observed covariates; outputs are point predictions, not laws.
-/
namespace PNDS.SourceAlignedModel

noncomputable section

open PNDS.SourceAlignedMemory

abbrev Vec (n : ℕ) := Fin n → ℝ
abbrev Mat (m n : ℕ) := Matrix (Fin m) (Fin n) ℝ

structure Dense (i o : ℕ) where
  weight : Mat o i
  bias : Vec o

def linear {i o : ℕ} (w : Mat o i) (x : Vec i) : Vec o :=
  fun j => ∑ k, w j k * x k

def dense {i o : ℕ} (w : Dense i o) (x : Vec i) : Vec o :=
  fun j => linear w.weight x j + w.bias j

def cat {m n : ℕ} (x : Vec m) (y : Vec n) : Vec (m+n) := Fin.append x y
def sig (x : ℝ) := 1 / (1 + Real.exp (-x))
def th (x : ℝ) := (Real.exp (2*x)-1)/(Real.exp (2*x)+1)
def silu (x : ℝ) := x * sig x
def map {n : ℕ} (f : ℝ → ℝ) (x : Vec n) : Vec n := fun j => f (x j)
def mix {n : ℕ} (x y gate : Vec n) : Vec n := fun j => x j + gate j*(y j-x j)
def first {n : ℕ} (x : Vec (n+n)) : Vec n := fun j => x (Fin.castAdd n j)
def second {n : ℕ} (x : Vec (n+n)) : Vec n := fun j => x (Fin.natAdd n j)
def normalize {n : ℕ} (x : Vec n) : Vec n :=
  fun j => x j / max (Real.sqrt (∑ k, x k*x k)) (1/1000000)
def softmax {n : ℕ} (x : Vec n) : Vec n :=
  fun j => Real.exp (x j) / ∑ k, Real.exp (x k)

theorem sigmoid_range (x : ℝ) : 0 < sig x ∧ sig x < 1 := by
  have hp := Real.exp_pos (-x)
  constructor
  · exact div_pos (by norm_num) (by linarith)
  · apply (div_lt_one (show 0 < 1+Real.exp (-x) by linarith)).2
    linarith

theorem softmax2_sum (x : Vec 2) : softmax x 0 + softmax x 1 = 1 := by
  have hp : Real.exp (x 0) + Real.exp (x 1) ≠ 0 :=
    ne_of_gt (add_pos (Real.exp_pos _) (Real.exp_pos _))
  simp only [softmax, Fin.sum_univ_two]
  field_simp

theorem softmax2_range (x : Vec 2) (h : Fin 2) :
    0 ≤ softmax x h ∧ softmax x h ≤ 1 := by
  have hp0 := Real.exp_pos (x 0)
  have hp1 := Real.exp_pos (x 1)
  have hd : 0 < Real.exp (x 0) + Real.exp (x 1) := by linarith
  simp only [softmax, Fin.sum_univ_two]
  constructor
  · exact le_of_lt (div_pos (Real.exp_pos _) hd)
  · apply (div_le_one hd).2
    fin_cases h <;> linarith

/-- Epsilon normalization never requires a nonzero key. In exact reals it
produces a key of norm at most one, not necessarily exactly one. -/
theorem normalized_key_energy (x : Vec 8) :
    SourceAlignedVerification.squaredNorm (normalize x) ≤ 1 := by
  let ss := SourceAlignedVerification.squaredNorm x
  have hs : 0 ≤ ss := by
    apply Finset.sum_nonneg
    intro j _
    exact mul_self_nonneg _
  let d := max (Real.sqrt ss) (1/1000000)
  have hd : 0 < d := lt_of_lt_of_le (by norm_num) (le_max_right _ _)
  have hroot : Real.sqrt ss ≤ d := le_max_left _ _
  have hsq := Real.sq_sqrt hs
  have hroot0 := Real.sqrt_nonneg ss
  have heq : SourceAlignedVerification.squaredNorm (normalize x) = ss/(d*d) := by
    change (∑ j : Fin 8, (x j/d)*(x j/d)) = ss/(d*d)
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [heq]
  apply (div_le_one (mul_pos hd hd)).2
  nlinarith

/-- Default geometry only: every active learned tensor has a typed slot. -/
structure Parameters where
  encoder : Dense 28 48
  timeProjection : Mat 48 7
  address : Dense 48 8
  content : Dense 48 8
  fastMap : Dense 144 176
  slowMap : Dense 184 176
  memoryGates : Dense 224 2
  readoutIn : Dense 232 132
  readoutOut : Dense 132 14
  residualIn : Dense 232 12
  residualOut : Dense 12 14
  residualLogit : ℝ
  rotations : Fin 2 → Mat 8 8
  sourceContent : Dense 48 16
  writeRouter : Dense 48 2
  readRouter : Dense 8 2
  retention : Dense 48 2
  writeStrength : Dense 48 2
  historicalContent : Dense 16 14
  predictiveMemory : Mat 14 16
  predictiveLogit : ℝ
  deltaScale : Vec 14

def parameterCount : ℕ :=
  (28*48+48) + (48*7) + 2*(48*8+8) + (144*176+176) +
  (184*176+176) + (224*2+2) + (232*132+132) + (132*14+14) +
  (232*12+12) + (12*14+14) + 1 + (2*8*8) + (48*16+16) +
  3*(48*2+2) + (8*2+2) + (16*14+14) + (16*14) + 1

theorem exact_parameter_count : parameterCount = 98326 := by norm_num [parameterCount]
theorem exact_persistent_float_count : 88+8*8+2+2*16*8 = (410 : ℕ) := by norm_num

structure Input where
  observation : Vec 14
  observedAction : Vec 14
  elapsedDelta : ℝ
  admitted : Bool
  owner : ℕ
  ordinal : ℕ
  query : Option ℕ

structure State where
  fastSum : Vec 88
  memory : Mat 8 8
  elapsedClock : ℝ
  lastWriteClock : ℝ
  writeSeen : Bool
  count : ℕ
  aligned : SourceMatrix
  owner : ℕ
  lastOrdinal : Option ℕ

def initial (owner : ℕ) : State :=
  ⟨0, 0, 0, 0, false, 0, fun _ _ _ => 0, owner, none⟩

def validStep (s : State) (x : Input) : Prop :=
  x.owner = s.owner ∧ 0 ≤ x.elapsedDelta ∧
  (∀ n ∈ s.lastOrdinal, n < x.ordinal) ∧
  (∀ q ∈ x.query, q < x.ordinal)

def periods : Fin 4 → ℕ := ![7,13,29,59]
def ordinalCode (n : ℕ) : Vec 8 := fun j =>
  let p := periods ⟨j.val / 2, by omega⟩
  let phase := ((n % p : ℕ) : ℝ) * (2 * Real.pi / p)
  if j.val % 2 = 0 then Real.cos phase else Real.sin phase

def keys (w : Parameters) (n : ℕ) : Fin 2 → Address := fun h =>
  normalize (fun j => ∑ i, ordinalCode n i * w.rotations h i j)

def memoryRead (w : Parameters) (m : SourceMatrix) (n : ℕ) : Vec 16 :=
  let k := keys w n
  let route := softmax (dense w.readRouter (ordinalCode n))
  fun v => ∑ h, route h * rowRead (m h v) (k h)

def historicalQuery (w : Parameters) (s : State) (n : ℕ) : Vec 14 :=
  dense w.historicalContent (memoryRead w s.aligned n)

def validHistoricalQuery (s : State) (owner n : ℕ) : Prop :=
  owner = s.owner ∧ ∃ last ∈ s.lastOrdinal, n ≤ last

def temporalFeatures (s : State) (x : Input) : Vec 7 :=
  let now := s.elapsedClock + x.elapsedDelta
  ![Real.log (1+(s.count : ℝ)), Real.sin ((s.count : ℝ)/16), Real.cos ((s.count : ℝ)/16),
    Real.log (1+x.elapsedDelta), Real.log (1+now),
    Real.log (1+max (now-s.lastWriteClock) 0), if s.writeSeen then 1 else 0]

/-- Prepared schedule excludes BOTH associative states; the contexts come
from source-prefix clocks, admission history and a source-only basis sum. -/
structure Prepared where
  embedding : Vec 48
  basis : Vec 88
  proposalFast : Vec 88
  proposalSlow : Vec 88
  nativeKey : Vec 8
  nativeValue : Vec 8
  nativeAlpha : ℝ
  nativeBeta : ℝ
  sourceKey : Fin 2 → Address
  sourceValue : Vec 16
  sourceAlpha : Vec 2
  sourceBeta : Vec 2

def prepare (w : Parameters) (s : State) (x : Input) : Prepared := Id.run do
  let e := map th (dense w.encoder (cat x.observation x.observedAction) +
                  linear w.timeProjection (temporalFeatures s x))
  let ctx : Vec 88 := fun j => s.fastSum j / max (s.count : ℝ) 1
  let basis := map th (first (dense w.fastMap (cat (cat e (0 : Vec 88)) (0 : Vec 8))))
  let fg := dense w.fastMap (cat (cat e ctx) (0 : Vec 8))
  let pf := mix ctx (map th (first fg)) (map sig (second fg))
  let sg := dense w.slowMap (cat (cat pf ctx) (0 : Vec 8))
  let ps := mix ctx (map th (ctx + map th (first sg))) (map (fun a => sig a/4) (second sg))
  let gates := map sig (dense w.memoryGates (cat (cat e pf) ps))
  let a := map sig (dense w.retention e)
  let b := map sig (dense w.writeStrength e)
  let route := softmax (dense w.writeRouter e)
  return ⟨e, basis, pf, ps, normalize (dense w.address e),
          normalize (dense w.content e), gates 0 * (1-gates 1/4), gates 1,
          keys w x.ordinal, map th (dense w.sourceContent e), a,
          fun h => b h * route h⟩

/-- Definitional nondependence: changing either associative memory cannot
alter the write schedule. This does not remove the separate prefix sum. -/
theorem prepare_memory_independent (w : Parameters) (s : State) (x : Input)
    (native : Mat 8 8) (aligned : SourceMatrix) :
    prepare w {s with memory := native, aligned := aligned} x = prepare w s x := by
  rfl

/-- Every ordinal collision in the residue code survives the complete
learned key and read-router graph. -/
theorem ordinalCode_periodic (n : ℕ) : ordinalCode (n+155701) = ordinalCode n := by
  funext j
  have hres : (n+155701) % periods ⟨j.val/2, by omega⟩ =
      n % periods ⟨j.val/2, by omega⟩ := by
    have hj : j.val/2 < 4 := by omega
    interval_cases h : j.val/2 <;> simp [periods, h, Nat.add_mod]
  simp only [ordinalCode, hres]

theorem memoryRead_periodic (w : Parameters) (m : SourceMatrix) (n : ℕ) :
    memoryRead w m (n+155701) = memoryRead w m n := by
  simp only [memoryRead, keys, ordinalCode_periodic]

theorem historicalQuery_periodic (w : Parameters) (s : State) (n : ℕ) :
    historicalQuery w s (n+155701) = historicalQuery w s n := by
  simp only [historicalQuery, memoryRead_periodic]

def decode (w : Parameters) (s : State) (x : Input) (p : Prepared) : Vec 14 := Id.run do
  let r := fun v => rowRead (s.memory v) p.nativeKey
  let fg := dense w.fastMap (cat (cat p.embedding p.proposalFast) r)
  let f := mix p.proposalFast (map th (first fg)) (map sig (second fg))
  let sg := dense w.slowMap (cat (cat f p.proposalSlow) r)
  let slow := mix p.proposalSlow (map th (p.proposalSlow + map th (first sg)))
                (map (fun a => sig a/4) (second sg))
  let z := cat (cat (cat f slow) r) p.embedding
  let d := map th (dense w.readoutOut (map silu (dense w.readoutIn z)))
  let post := map th (dense w.residualOut (map silu (dense w.residualIn z)))
  let old := match x.query with
    | some q => memoryRead w s.aligned q
    | none => if x.ordinal = 0 then 0 else memoryRead w s.aligned (x.ordinal-1)
  let extra := map th (linear w.predictiveMemory old)
  return fun j => x.observation j + w.deltaScale j * d j +
    w.deltaScale j * sig w.residualLogit * post j +
    w.deltaScale j * sig w.predictiveLogit * extra j

def advance (s : State) (x : Input) (p : Prepared) : State :=
  {fastSum := s.fastSum + p.basis
   memory := fun v => writeRow (s.memory v) p.nativeKey (p.nativeValue v)
                     p.nativeAlpha p.nativeBeta x.admitted
   elapsedClock := s.elapsedClock + x.elapsedDelta
   lastWriteClock := if x.admitted then s.elapsedClock+x.elapsedDelta else s.lastWriteClock
   writeSeen := s.writeSeen || x.admitted
   count := s.count+1
   aligned := SourceAlignedVerification.writeMatrix s.aligned p.sourceKey p.sourceValue
                p.sourceAlpha p.sourceBeta x.admitted
   owner := s.owner
   lastOrdinal := some x.ordinal}

/-- Current prediction reads the PRE-write memory, then advances the state. -/
def step (w : Parameters) (s : State) (x : Input) : Vec 14 × State :=
  let p := prepare w s x
  (decode w s x p, advance s x p)

def checkedStep (w : Parameters) (s : State) (x : Input) : Option (Vec 14) × State := by
  classical
  exact if validStep s x then (some (step w s x).1, (step w s x).2) else (none,s)

theorem rejected_step_keeps_state (w : Parameters) (s : State) (x : Input)
    (h : ¬ validStep s x) : checkedStep w s x = (none,s) := by
  classical
  simp [checkedStep,h]

theorem accepted_step_matches_graph (w : Parameters) (s : State) (x : Input)
    (h : validStep s x) :
    checkedStep w s x = (some (step w s x).1,(step w s x).2) := by
  classical
  simp [checkedStep,h]

def checkedHistoricalQuery (w : Parameters) (s : State) (owner n : ℕ) :
    Option (Vec 14) × State := by
  classical
  exact (if validHistoricalQuery s owner n then some (historicalQuery w s n) else none,s)

theorem checked_historical_query_preserves_state
    (w : Parameters) (s : State) (owner n : ℕ) :
    (checkedHistoricalQuery w s owner n).2 = s := by rfl

theorem owner_preserved (w : Parameters) (s : State) (x : Input) :
    (step w s x).2.owner = s.owner := by rfl

theorem count_advances (w : Parameters) (s : State) (x : Input) :
    (step w s x).2.count = s.count+1 := by rfl

theorem declined_aligned_memory_unchanged (w : Parameters) (s : State) (x : Input)
    (h : x.admitted = false) : (step w s x).2.aligned = s.aligned := by
  funext a v j
  simp [step, advance, SourceAlignedVerification.writeMatrix, writeRow, h]

theorem declined_native_memory_unchanged (w : Parameters) (s : State) (x : Input)
    (h : x.admitted = false) : (step w s x).2.memory = s.memory := by
  funext v j
  simp [step, advance, writeRow, h]

def runState (w : Parameters) (s : State) (events : List Input) : State :=
  events.foldl (fun old x => (step w old x).2) s

theorem stream_resume (w : Parameters) (s : State) (first second : List Input) :
    runState w s (first++second) = runState w (runState w s first) second := by
  simp [runState, List.foldl_append]

def predictions (w : Parameters) (s : State) : List Input → List (Vec 14)
  | [] => []
  | x::xs => (step w s x).1 :: predictions w (step w s x).2 xs

/-- Future inputs cannot affect any already emitted prediction of the full
exact-real graph. No claim about authenticity of caller inputs follows. -/
theorem causal_prefix (w : Parameters) (s : State) (past future : List Input) :
    (predictions w s (past++future)).take past.length = predictions w s past := by
  induction past generalizing s with
  | nil => simp [predictions]
  | cons x xs ih => simpa [predictions] using ih (step w s x).2

end PNDS.SourceAlignedModel
