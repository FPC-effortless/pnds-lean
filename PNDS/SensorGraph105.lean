import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Lean.Elab.Tactic.Omega

/-!
ME105 full-graph exact-real reference.  This file gives a typed, parameterized
input-to-output definition: learned maps are finite matrices and biases (not
opaque functions); the time fold carries source, evidence, and associative
memory state; the decoder has distinct pre-event and inclusive reads; and the
belief, source floor, median, and split-width heads are evaluated in order.

This is a mathematical real-arithmetic specification.  PyTorch refinement,
IEEE rounding, gradients at nonsmooth boundaries, and empirical calibration
remain OPEN; compiling these definitions does not close those obligations.
-/
namespace PNDS.SensorGraph105

abbrev Vec (n : ℕ) := Fin n → ℝ
abbrev Mat (m n : ℕ) := Fin m → Fin n → ℝ
abbrev V13 := Vec 13
abbrev V6 := Vec 6
abbrev V8 := Vec 8
abbrev V48 := Vec 48
abbrev V62 := Vec 62
abbrev V88 := Vec 88
abbrev V176 := Vec 176
abbrev V12 := Vec 12
abbrev M8 := Mat 8 8

/-- A trainable affine layer represented by all of its real weights/biases. -/
structure Affine (m n : ℕ) where
  weight : Mat m n
  bias : Vec m

noncomputable def affine {m n : ℕ} (L : Affine m n) (x : Vec n) : Vec m :=
  fun i => (∑ j : Fin n, L.weight i j * x j) + L.bias i

noncomputable def sigmoid (x : ℝ) := 1 / (1 + Real.exp (-x))
noncomputable def softplus (x : ℝ) := Real.log (1 + Real.exp x)
noncomputable def silu (x : ℝ) := x * sigmoid x
noncomputable def vmap {n : ℕ} (f : ℝ → ℝ) (x : Vec n) : Vec n := fun i => f (x i)
noncomputable def vadd {n : ℕ} (x y : Vec n) : Vec n := fun i => x i + y i
noncomputable def vsub {n : ℕ} (x y : Vec n) : Vec n := fun i => x i - y i
noncomputable def vmul {n : ℕ} (x y : Vec n) : Vec n := fun i => x i * y i
noncomputable def vscale {n : ℕ} (a : ℝ) (x : Vec n) : Vec n := fun i => a * x i
noncomputable def vmax {n : ℕ} (x y : Vec n) : Vec n := fun i => max (x i) (y i)
noncomputable def vcat {m n : ℕ} (x : Vec m) (y : Vec n) : Vec (m+n) := Fin.append x y

/-- Explicit state retained across chronological sensor events. -/
structure State where
  sourceSum : V13
  sourceMass : V13
  evidenceSum : V62
  eventCount : ℕ
  memory : M8

def zeroState : State :=
  { sourceSum := fun _ => 0, sourceMass := fun _ => 0,
    evidenceSum := fun _ => 0, eventCount := 0, memory := fun _ _ => 0 }

/-- One time-indexed observation.  Masked values are zeroed by the graph. -/
structure Observation (T : ℕ) where
  value : Fin T → V13
  observed : Fin T → Fin 13 → Bool
  age : Fin T → Fin 13 → ℝ
  clock : Fin T → Fin 6 → ℝ
  station : Fin 12
  hour : Fin T → ℝ

/-- Every learned affine map in the source architecture, with source dimensions. -/
structure Parameters where
  encoder : Affine 48 49
  admission : Affine 13 48
  fast : Affine 176 144
  slow : Affine 176 184
  address : Affine 8 48
  content : Affine 8 48
  gates : Affine 2 224
  query : Affine 8 224
  readout1 : Affine 132 232
  readout2 : Affine 12 132
  retrieved1 : Affine 32 8
  retrieved2 : Affine 12 32
  current : Affine 12 102
  width : Affine 12 102
  median : Affine 6 102
  stationEmbedding : Fin 12 → Fin 4 → ℝ
  latentCoeff : Fin 3 → Fin 6 → ℝ
  response : V6
  fitFloorRate : V6
  targetMean : V6
  targetStd : V6

/-- Trainable scalar count for the explicit learned tensors above. FIT floor
rates and target mean/std are buffers and therefore excluded. -/
def trainableParameterCount : ℕ :=
  (48*49+48) + (13*48+13) + (176*144+176) + (176*184+176) +
  (8*48+8) + (8*48+8) + (2*224+2) + (8*224+8) +
  (132*232+132) + (12*132+12) + (32*8+32) + (12*32+12) +
  (12*102+12) + (12*102+12) + (6*102+6) + 12*4 + 3*6 + 6

theorem trainableParameterCount_exact : trainableParameterCount = 100349 := by
  norm_num [trainableParameterCount]

noncomputable def safeInput {T : ℕ} (o : Observation T) (t : Fin T) : V13 :=
  fun c => if o.observed t c then o.value t c else 0

noncomputable def encode {T : ℕ} (p : Parameters) (o : Observation T) (t : Fin T) : V48 :=
  let safe := safeInput o t
  let mask : V13 := fun c => if o.observed t c then 1 else 0
  let ages : V13 := o.age t
  let clocks : V6 := o.clock t
  let station := p.stationEmbedding o.station
  let input := vcat (vcat (vcat (vcat safe mask) ages) clocks) station
  vmap Real.tanh (affine p.encoder input)

/-- Source summaries and evidence means used by both causal proposals and the
current-event feature vector. -/
noncomputable def sourceMean (s : State) (c : Fin 13) :=
  8 * s.sourceSum c / max (s.sourceMass c) (1/1000000 : ℝ)
noncomputable def sourceContextMean (s : State) (c : Fin 13) :=
  s.sourceSum c / max (s.sourceMass c) (1/1000000 : ℝ)
noncomputable def evidenceMean (s : State) (c : Fin 62) :=
  s.evidenceSum c / max (s.eventCount : ℝ) 1

noncomputable def context (s : State) : V88 :=
  vcat (vcat (fun c : Fin 13 => sourceContextMean s c)
    (fun c : Fin 13 => s.sourceMass c))
    (fun c : Fin 62 => evidenceMean s c)

noncomputable def half (x : V176) (i : Fin 88) : ℝ := x ⟨i.val, by omega⟩
noncomputable def secondHalf (x : V176) (i : Fin 88) : ℝ := x ⟨88+i.val, by omega⟩

/-- Normalization with the source epsilon convention. -/
noncomputable def normalize8 (x : V8) : V8 :=
  let norm := Real.sqrt (∑ i : Fin 8, (x i)^2)
  fun i => x i / max norm (1/1000000 : ℝ)

noncomputable def memRead (m : M8) (q : V8) : V8 :=
  fun i => ∑ j : Fin 8, m i j * q j

noncomputable def outerUpdate (s : State) (k v : V8) (alpha beta : ℝ) : M8 :=
  fun i j => alpha * s.memory i j + beta * (v i - alpha * memRead s.memory k i) * k j

/-- Complete one source event through encoder, masked admission, source/evidence
prefixes, proposals and the rank-one associative memory write. -/
noncomputable def eventStep {T : ℕ} (p : Parameters) (o : Observation T)
    (t : Fin T) (s : State) : State × V48 × V88 × V88 × V8 × V8 := by
  let x := encode p o t
  let mask : V13 := fun c => if o.observed t c then 1 else 0
  let adm := vmap sigmoid (affine p.admission x)
  let betaSrc := vmul mask adm
  let src : State :=
    { sourceSum := fun c => (1-betaSrc c)*s.sourceSum c + betaSrc c*(safeInput o t c/8)
      sourceMass := fun c => (1-betaSrc c)*s.sourceMass c + betaSrc c
      evidenceSum := s.evidenceSum
      eventCount := s.eventCount
      memory := s.memory }
  let eligible := decide (∃ c : Fin 13, o.observed t c = true)
  let basisRaw := affine p.fast (vcat (vcat x (fun _ : Fin 88 => 0)) (fun _ : Fin 8 => 0))
  let basis : V62 := fun c => if eligible then Real.tanh (half basisRaw ⟨26+c.val, by omega⟩) else 0
  let updatedPrefix := { src with
    evidenceSum := fun c => s.evidenceSum c + basis c
    eventCount := s.eventCount + (if eligible then 1 else 0) }
  let ctx := context s
  let proposal := affine p.fast (vcat (vcat x ctx) (fun _ : Fin 8 => 0))
  let fastPre : V88 := fun i => (1-sigmoid (secondHalf proposal i))*ctx i + sigmoid (secondHalf proposal i)*Real.tanh (half proposal i)
  let slowRaw := affine p.slow (vcat (vcat fastPre ctx) (fun _ : Fin 8 => 0))
  let slowPre : V88 := fun i => (1-(1/4:ℝ)*sigmoid (secondHalf slowRaw i))*ctx i + (1/4:ℝ)*sigmoid (secondHalf slowRaw i)*Real.tanh (ctx i+Real.tanh (half slowRaw i))
  let k := normalize8 (affine p.address x)
  let v := normalize8 (affine p.content x)
  let gateRaw := affine p.gates (vcat (vcat x fastPre) slowPre)
  let a := sigmoid (gateRaw ⟨0, by omega⟩)
  let b := (1/4:ℝ)*sigmoid (gateRaw ⟨1, by omega⟩)
  let alpha := min (1-(1/50:ℝ)*(1-a)) (1-(1/100000:ℝ)) * (1-b/4)
  let beta := if eligible then b else 0
  let written := if eligible then outerUpdate s k v alpha beta else s.memory
  exact ( { updatedPrefix with memory := written }, x, fastPre, slowPre, k, v )

/-- Chronological fold, preserving the last encoded/proposal/write data for the
decoder. The returned `beforeLast` is precisely M_(T-1). -/
structure Trace where
  finalState : State
  beforeLast : M8
  lastEncoded : V48
  lastFast : V88
  lastSlow : V88
  lastKey : V8
  lastValue : V8

/-- Decoder state vectors from the original pre-event read; the write for the
last event has already been incorporated into `trace.finalState`. -/
noncomputable def decodeRaw (p : Parameters) (trace : Trace) : V12 :=
  let x := trace.lastEncoded
  let preM := trace.beforeLast
  let kp := trace.lastKey
  let qp := affine p.query (vcat (vcat x trace.lastFast) trace.lastSlow)
  let q := normalize8 (vadd kp qp)
  let r := memRead preM q
  let fRaw := affine p.fast (vcat (vcat x trace.lastFast) r)
  let f : V88 := fun i => (1-sigmoid (secondHalf fRaw i))*trace.lastFast i + sigmoid (secondHalf fRaw i)*Real.tanh (half fRaw i)
  let sRaw := affine p.slow (vcat (vcat f trace.lastSlow) r)
  let s : V88 := fun i => (1-(1/4:ℝ)*sigmoid (secondHalf sRaw i))*trace.lastSlow i + (1/4:ℝ)*sigmoid (secondHalf sRaw i)*Real.tanh (trace.lastSlow i+Real.tanh (half sRaw i))
  let joined := vcat (vcat (vcat f s) r) x
  let y := affine p.readout2 (vmap silu (affine p.readout1 joined))
  let z := affine p.retrieved2 (vmap silu (affine p.retrieved1 r))
  vadd y z

/-- Serial chronological scan. At each event the old state memory is saved as
the decoder's pre-event memory, while `eventStep` returns the inclusive write.
`Fin.foldl` visits 0,1,...,T-1 in this order. -/
noncomputable def scanFrom {T : ℕ} (p : Parameters) (o : Observation T)
    (initial : State) : Trace :=
  Fin.foldl T (fun tr t =>
    let next := eventStep p o t tr.finalState
    let payload := next.2
    { finalState := next.1
      beforeLast := tr.finalState.memory
      lastEncoded := payload.1
      lastFast := payload.2.1
      lastSlow := payload.2.2.1
      lastKey := payload.2.2.2.1
      lastValue := payload.2.2.2.2 })
    { finalState := initial, beforeLast := initial.memory,
      lastEncoded := fun _ => 0, lastFast := fun _ => 0, lastSlow := fun _ => 0,
      lastKey := fun _ => 0, lastValue := fun _ => 0 }

noncomputable def scan {T : ℕ} (p : Parameters) (o : Observation T) : Trace :=
  scanFrom p o zeroState

/-! The final belief and protected distribution are direct finite-vector
operations. Probability-law definitions are deferred to the source-linked
ME104 formulation rather than re-declaring an unverified integral here. -/
structure Forecast where
  location : V6
  leftScale : V6
  rightScale : V6
  floorVariance : V6
  features : Vec 102

noncomputable def feature102 {T : ℕ} (o : Observation T)
    (t : Fin T) (s : State) (rb : V8) : Vec 102 :=
  vcat (vcat (vcat (fun c : Fin 13 => sourceMean s c)
      (fun c : Fin 13 => s.sourceMass c))
    (fun c : Fin 62 => evidenceMean s c))
    (vcat (fun c : Fin 6 => o.age t ⟨c.val, by omega⟩) rb)

noncomputable def protect (p : Parameters) (phi : Vec 102)
    (baseLocation baseScale floor : V6)
    (observed : Fin 6 → Bool) : Forecast :=
  let medianRaw := affine p.median phi
  let widthRaw := affine p.width phi
  { location := fun c => baseLocation c + (1/2:ℝ)*Real.tanh (medianRaw c)
    leftScale := fun c => max (Real.sqrt (floor c))
      (max (if observed c then 0 else (4/5:ℝ)*baseScale c)
        (baseScale c*Real.exp (Real.log 4*Real.tanh (widthRaw ⟨c.val, by omega⟩))))
    rightScale := fun c => max (Real.sqrt (floor c))
      (max (if observed c then 0 else (4/5:ℝ)*baseScale c)
        (baseScale c*Real.exp (Real.log 4*Real.tanh (widthRaw ⟨6+c.val, by omega⟩))))
    floorVariance := floor, features := phi }

/-- Distribution interface supplied by the exact standard-normal law. Its
analytic integral/normalization and inverse identities are intentionally OPEN;
the sensor graph's split, censoring, and target transforms are explicit. -/
structure NormalLaw where
  cdf : ℝ → ℝ
  quantile : ℝ → ℝ
  logTwoPi : ℝ

noncomputable def sideScale (f : Forecast) (c : Fin 6) (z : ℝ) : ℝ :=
  if z < f.location c then f.leftScale c else f.rightScale c

noncomputable def splitCDF (law : NormalLaw) (f : Forecast) (c : Fin 6) (z : ℝ) : ℝ :=
  law.cdf ((z-f.location c) / sideScale f c z)

/-- Censored-lognormal quantile expressed using a supplied standard-normal
quantile. The caller's law assumptions include 0<p<1 and PhiInv(.5)=0. -/
noncomputable def rawQuantile (law : NormalLaw) (p : Parameters)
    (f : Forecast) (c : Fin 6) (probability : ℝ) : ℝ :=
  let side := if probability < 1/2 then f.leftScale c else f.rightScale c
  max 0 (Real.exp (p.targetMean c + p.targetStd c *
    (f.location c + side * law.quantile probability)) - 1)

/-- Raw-unit censored CDF, including its atom at zero. -/
noncomputable def rawCDF (law : NormalLaw) (p : Parameters)
    (f : Forecast) (c : Fin 6) (y : ℝ) : ℝ :=
  if y < 0 then 0 else if y = 0 then
    let threshold := -p.targetMean c / p.targetStd c
    law.cdf ((threshold-f.location c) / sideScale f c threshold)
  else
    let z := (Real.log (1+y)-p.targetMean c)/p.targetStd c
    splitCDF law f c z

/-- Elementwise censored negative log likelihood from the source equations. -/
noncomputable def nllElement (law : NormalLaw) (p : Parameters)
    (f : Forecast) (c : Fin 6) (y : ℝ) : ℝ :=
  let threshold := -p.targetMean c / p.targetStd c
  if y = 0 then
    -Real.log (law.cdf ((threshold-f.location c) / sideScale f c threshold))
  else
    let z := (Real.log (1+y)-p.targetMean c)/p.targetStd c
    let side := sideScale f c z
    (1/2:ℝ)*((z-f.location c)/side)^2 + Real.log side +
      (1/2:ℝ)*law.logTwoPi + Real.log (p.targetStd c) + Real.log (1+y)

structure Belief where
  location : V6
  variance : V6

/-- ME101/102 current-channel belief. Ages are converted back from their
log1p(age)/log(257) encoding. The latent term is active only when missing. -/
noncomputable def currentBelief {T : ℕ} (p : Parameters) (o : Observation T)
    (t : Fin T) (s : State) (phi : Vec 102) (anchor : V6) : Belief :=
  let raw := affine p.current phi
  { location := fun c => anchor c + 2*Real.tanh (raw c)
    variance := fun c =>
      let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
      let ageHours := Real.exp (o.age t ⟨c.val, by omega⟩ * Real.log 257) - 1
      let confidence := min 1 (max 0 (s.sourceMass ⟨c.val, by omega⟩))
      let latent :=
        (softplus (p.latentCoeff 0 c) + softplus (p.latentCoeff 1 c)*(ageHours/24)) *
          (1 + softplus (p.latentCoeff 2 c)*(1-confidence))
      (1/400:ℝ) + (softplus (raw ⟨6+c.val, by omega⟩))^2 + missing*latent }

/-- Complete forecast from the serial scan trace. The inherited decoder reads
`beforeLast`; the belief feature read uses inclusive `M_T`. -/
noncomputable def forwardFromTrace {T : ℕ} (p : Parameters) (o : Observation T)
    (t : Fin T) (tr : Trace) : Forecast := by
  let rb := memRead tr.finalState.memory (normalize8 (affine p.address tr.lastEncoded))
  let phi := feature102 o t tr.finalState rb
  let raw := decodeRaw p tr
  let anchor : V6 := fun c =>
    if o.observed t ⟨c.val, by omega⟩ then safeInput o t ⟨c.val, by omega⟩
    else sourceMean tr.finalState ⟨c.val, by omega⟩
  let baseLocation : V6 := fun c => anchor c + 2*Real.tanh (raw c)
  let inheritedScale : V6 := fun c => (1/20:ℝ) + softplus (raw ⟨6+c.val, by omega⟩)
  let belief := currentBelief p o t tr.finalState phi anchor
  let response : V6 := fun c => 1+(1/4:ℝ)*Real.tanh (p.response c)
  let location : V6 := fun c =>
    let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
    baseLocation c + response c*(belief.location c-anchor c)*missing
  let floor : V6 := fun c =>
    let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
    let ageHours := Real.exp (o.age t ⟨c.val, by omega⟩ * Real.log 257) - 1
    p.fitFloorRate c * missing * ageHours/24
  let modelVariance : V6 := fun c =>
    let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
    inheritedScale c^2 + response c^2*belief.variance c*missing
  let parentScale : V6 := fun c => Real.sqrt (max (modelVariance c) (floor c))
  exact protect p phi location parentScale floor (fun c => o.observed t ⟨c.val, by omega⟩)

/-- Input-to-forecast graph: encode and update each event in chronological
order, then apply the inherited decoder and protected current-belief heads. -/
noncomputable def forward {T : ℕ} (p : Parameters) (o : Observation T)
    (last : Fin T) (hlast : ∀ i : Fin T, i ≤ last) : Forecast := by
  have _coversHistory := hlast
  exact forwardFromTrace p o last (scan p o)

noncomputable def forwardFromState {T : ℕ} (p : Parameters) (o : Observation T)
    (last : Fin T) (hlast : ∀ i : Fin T, i ≤ last) (initial : State) : Forecast := by
  have _coversHistory := hlast
  exact forwardFromTrace p o last (scanFrom p o initial)

/- OPEN: `NormalLaw` must satisfy the standard normal CDF/inverse and
`logTwoPi=log(2*pi)` identities.  Its analytic normalization/inverse proofs,
nonnegative FIT floor/input conventions, state owner/time/model-version checks,
history shape/finiteness checks, source refinement, and gradients are not
formalized here. -/

theorem affine_coordinate (m n : ℕ) (L : Affine m n) (x : Vec n) (i : Fin m) :
    affine L x i = (∑ j : Fin n, L.weight i j*x j)+L.bias i := rfl

theorem protected_floor_lower_bound (p : Parameters) (phi : Vec 102)
    (loc scale floor : V6) (obs : Fin 6 → Bool) (c : Fin 6) :
    Real.sqrt (floor c) ≤ (protect p phi loc scale floor obs).leftScale c := by
  simp only [protect]
  exact le_max_left _ _

theorem protected_right_floor_lower_bound (p : Parameters) (phi : Vec 102)
    (loc scale floor : V6) (obs : Fin 6 → Bool) (c : Fin 6) :
    Real.sqrt (floor c) ≤ (protect p phi loc scale floor obs).rightScale c := by
  simp only [protect]
  exact le_max_left _ _

/- OPEN: exact Python/source full-forward and backward correspondence, history
state validation (owner/model/parameter version/time), input shape and finite
checks, torch's normalization/clamp subgradient details, proof that the
parameterized current-belief/floor equations match their source buffers,
half-normal CDF integral measure proof and inverse-CDF identities,
censored zero-atom likelihood, all IEEE/FP32/FP64 refinement, and all gradient
claims. This file specifies exact-real layer equations; it does not prove those
implementation or empirical claims. -/

end PNDS.SensorGraph105
