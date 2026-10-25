import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Data.Real.Sqrt
import Lean.Elab.Tactic.Omega
import PNDS.MedianSplitSensor

/-!
ME105 exact-real contract for a protected median and source-floor-preserving
asymmetric scales.  The declarations below specify typed graph interfaces and
prove scalar consequences.  They do not assert that a particular PyTorch
network refines these interfaces.
-/
namespace PNDS.ProtectedSensor

abbrev Channel := Fin 6
abbrev Feature := Fin 102
abbrev SourceChannel := Fin 13
abbrev EvidenceChannel := Fin 62
abbrev MemorySlot := Fin 8
abbrev Vec (n : ℕ) := Fin n → ℝ
abbrev Matrix8 := MemorySlot → MemorySlot → ℝ

/-- Persistent causal state retained by the inherited ME101/102/104 core.
Source moments, evidence moments, count and the associative 8×8 memory are
separate coordinates because they have different update semantics. -/
structure CoreState where
  sourceSum : Vec 13
  sourceMass : Vec 13
  evidenceSum : Vec 62
  count : ℝ
  memory : Matrix8

/-- One admitted observation event. `safeSource` is already zeroed at missing
coordinates; `basis` is the source-defined 62-vector evidence contribution.
The scalar schedules and memory key/value are explicit inputs to the step. -/
structure CoreEvent where
  safeSource : Vec 13
  beta : Vec 13
  write_strength : ℝ
  basis : Vec 62
  eligible : Bool
  alpha : ℝ
  key : Vec 8
  value : Vec 8

noncomputable def coreStep (s : CoreState) (e : CoreEvent) : CoreState :=
  { sourceSum := fun i => (1 - e.beta i) * s.sourceSum i +
      e.beta i * (e.safeSource i / 8)
    , sourceMass := fun i => (1 - e.beta i) * s.sourceMass i + e.beta i
    , evidenceSum := fun i => s.evidenceSum i +
      (if e.eligible then e.basis i else 0)
    , count := s.count + (if e.eligible then 1 else 0)
    , memory := fun i j => if e.eligible then e.alpha * s.memory i j + e.write_strength *
      (e.value i - e.alpha * (∑ q : MemorySlot, s.memory i q * e.key q)) * e.key j else s.memory i j }

/-- The causal chronological fold. No event at a later position can affect
the state returned for an earlier prefix. -/
noncomputable def serialFold (s : CoreState) : List CoreEvent → CoreState
  | [] => s
  | e :: es => serialFold (coreStep s e) es

theorem serialFold_append (s : CoreState) (xs ys : List CoreEvent) :
    serialFold s (xs ++ ys) = serialFold (serialFold s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, serialFold]; exact ih (coreStep s x)

theorem coreStep_sourceSum (s : CoreState) (e : CoreEvent) (i : SourceChannel) :
    (coreStep s e).sourceSum i =
      (1 - e.beta i) * s.sourceSum i + e.beta i * (e.safeSource i / 8) := rfl

theorem coreStep_sourceMass (s : CoreState) (e : CoreEvent) (i : SourceChannel) :
    (coreStep s e).sourceMass i = (1 - e.beta i) * s.sourceMass i + e.beta i := rfl

theorem coreStep_evidence (s : CoreState) (e : CoreEvent) (i : EvidenceChannel) :
    (coreStep s e).evidenceSum i = s.evidenceSum i +
      (if e.eligible then e.basis i else 0) := rfl

theorem coreStep_count (s : CoreState) (e : CoreEvent) :
    (coreStep s e).count = s.count + (if e.eligible then 1 else 0) := rfl

theorem coreStep_memory (s : CoreState) (e : CoreEvent) (i j : MemorySlot) :
    (coreStep s e).memory i j = if e.eligible then e.alpha * s.memory i j + e.write_strength *
      (e.value i - e.alpha * (∑ q : MemorySlot, s.memory i q * e.key q)) * e.key j else s.memory i j := rfl

noncomputable def safeMean (s : CoreState) (i : SourceChannel) : ℝ :=
  8 * s.sourceSum i / max (s.sourceMass i) (1 / 1000000 : ℝ)

noncomputable def evidenceMean (s : CoreState) (i : EvidenceChannel) : ℝ :=
  s.evidenceSum i / max s.count 1

noncomputable def inclusiveMemoryRead (s : CoreState) (key : Vec 8) (j : MemorySlot) : ℝ :=
  ∑ i : MemorySlot, s.memory j i * key i

/-- The concrete feature-coordinate contract: 13 source means, 13 source
masses, 62 evidence means, six ages and eight inclusive memory reads. -/
noncomputable def featureConcat (s : CoreState) (ages : Vec 6) (key : Vec 8) : Feature → ℝ :=
  fun f => if h0 : f.val < 13 then
      safeMean s ⟨f.val, h0⟩
    else if h1 : f.val < 26 then
      s.sourceMass ⟨f.val - 13, by omega⟩
    else if h2 : f.val < 88 then
      evidenceMean s ⟨f.val - 26, by omega⟩
    else if h3 : f.val < 94 then
      ages ⟨f.val - 88, by omega⟩
    else
      inclusiveMemoryRead s key ⟨f.val - 94, by omega⟩

theorem featureConcat_source (s : CoreState) (ages : Vec 6) (key : Vec 8)
    (i : SourceChannel) :
    featureConcat s ages key ⟨i.val, by omega⟩ = safeMean s i := by
  simp [featureConcat]

theorem featureConcat_sourceMass (s : CoreState) (ages : Vec 6) (key : Vec 8)
    (i : SourceChannel) :
    featureConcat s ages key ⟨13 + i.val, by omega⟩ = s.sourceMass i := by
  simp [featureConcat, show ¬ (13 + i.val < 13) by omega, show 13 + i.val < 26 by omega]

theorem featureConcat_evidence (s : CoreState) (ages : Vec 6) (key : Vec 8)
    (i : EvidenceChannel) :
    featureConcat s ages key ⟨26 + i.val, by omega⟩ = evidenceMean s i := by
  simp [featureConcat, show ¬ (26 + i.val < 13) by omega, show ¬ (26 + i.val < 26) by omega, show 26 + i.val < 88 by omega]

theorem featureConcat_age (s : CoreState) (ages : Vec 6) (key : Vec 8)
    (i : Fin 6) :
    featureConcat s ages key ⟨88 + i.val, by omega⟩ = ages i := by
  simp [featureConcat, show ¬ (88 + i.val < 13) by omega, show ¬ (88 + i.val < 26) by omega, show 88 + i.val < 94 by omega]

theorem featureConcat_memory (s : CoreState) (ages : Vec 6) (key : Vec 8)
    (i : MemorySlot) :
    featureConcat s ages key ⟨94 + i.val, by omega⟩ = inclusiveMemoryRead s key i := by
  simp [featureConcat, show ¬ (94 + i.val < 13) by omega, show ¬ (94 + i.val < 26) by omega, show ¬ (94 + i.val < 88) by omega]

/-- Input observations with explicit missingness and elapsed-time fields. -/
structure Observation (T : ℕ) where
  values : Fin T → Fin 13 → ℝ
  observed : Fin T → Fin 13 → Bool
  ages : Fin T → Fin 13 → ℝ
  clock : Fin T → Fin 6 → ℝ
  owner : ℕ
  hours : Fin T → ℝ

structure InheritedOutput where
  median : Channel → ℝ
  baseScale : Channel → ℝ
  floorVariance : Channel → ℝ
  currentMissing : Channel → Bool
  features : Feature → ℝ

structure ProtectedHeads where
  medianHead : Channel → (Feature → ℝ) → ℝ
  leftWidthHead : Channel → (Feature → ℝ) → ℝ
  rightWidthHead : Channel → (Feature → ℝ) → ℝ

structure Forecast where
  median : Channel → ℝ
  leftScale : Channel → ℝ
  rightScale : Channel → ℝ

noncomputable def protectedScale (baseScale floorVariance : ℝ)
    (currentMissing : Bool) (raw : ℝ) : ℝ :=
  max (Real.sqrt floorVariance)
    (max (if currentMissing then (4 / 5 : ℝ) * baseScale else 0)
      (baseScale * Real.exp (Real.log 4 * Real.tanh raw)))

noncomputable def headForward (inherited : InheritedOutput)
    (heads : ProtectedHeads) : Forecast :=
  { median := fun c => inherited.median c +
      (1 / 2 : ℝ) * Real.tanh (heads.medianHead c inherited.features)
    , leftScale := fun c => protectedScale (inherited.baseScale c)
        (inherited.floorVariance c) (inherited.currentMissing c)
        (heads.leftWidthHead c inherited.features)
    , rightScale := fun c => protectedScale (inherited.baseScale c)
        (inherited.floorVariance c) (inherited.currentMissing c)
        (heads.rightWidthHead c inherited.features) }

theorem protectedScale_floor (base floor : ℝ) (missing : Bool) (raw : ℝ) :
    Real.sqrt floor ≤ protectedScale base floor missing raw := by
  simp only [protectedScale]
  exact le_max_left _ _

theorem protectedScale_widthCandidate (base floor : ℝ) (missing : Bool) (raw : ℝ) :
    base * Real.exp (Real.log 4 * Real.tanh raw) ≤
      protectedScale base floor missing raw := by
  simp only [protectedScale]
  exact le_trans (le_max_right _ _) (le_max_right _ _)

theorem protectedScale_missingGuard (base floor raw : ℝ) :
    (4 / 5 : ℝ) * base ≤ protectedScale base floor true raw := by
  simp only [protectedScale, if_pos rfl]
  exact le_trans (le_max_left _ _) (le_max_right _ _)

theorem protectedScale_positive_of_floor_positive
    (base floor raw : ℝ) (missing : Bool) (hfloor : 0 < floor) :
    0 < protectedScale base floor missing raw := by
  have hsqrt : 0 < Real.sqrt floor := Real.sqrt_pos.2 hfloor
  exact lt_of_lt_of_le hsqrt (protectedScale_floor base floor missing raw)

/-- With a nonnegative variance floor, the scale bound implies the same
variance floor after squaring. -/
theorem protectedScale_sq_floor (base floor : ℝ) (missing : Bool) (raw : ℝ)
    (hfloor : 0 ≤ floor) :
    floor ≤ (protectedScale base floor missing raw) ^ 2 := by
  have hsqrt : 0 ≤ Real.sqrt floor := Real.sqrt_nonneg _
  have hscale := protectedScale_floor base floor missing raw
  have hs : 0 ≤ protectedScale base floor missing raw := le_trans hsqrt hscale
  have hdiff : 0 ≤ protectedScale base floor missing raw - Real.sqrt floor := sub_nonneg.mpr hscale
  nlinarith [Real.sq_sqrt hfloor, sq_nonneg (Real.sqrt floor), sq_nonneg (protectedScale base floor missing raw - Real.sqrt floor)]

theorem protectedScale_positive_of_base_positive
    (base floor raw : ℝ) (missing : Bool) (hbase : 0 < base) :
    0 < protectedScale base floor missing raw := by
  have hfactor : 0 < Real.exp (Real.log 4 * Real.tanh raw) := Real.exp_pos _
  have hcand : 0 < base * Real.exp (Real.log 4 * Real.tanh raw) := mul_pos hbase hfactor
  exact lt_of_lt_of_le hcand (protectedScale_widthCandidate base floor missing raw)

theorem protectedMedian_correction_bound (oldMedian raw : ℝ)
    (htanh : |Real.tanh raw| ≤ 1) :
    |(oldMedian + (1 / 2 : ℝ) * Real.tanh raw) - oldMedian| ≤ 1 / 2 := by
  rw [show (oldMedian + (1 / 2 : ℝ) * Real.tanh raw) - oldMedian =
      (1 / 2 : ℝ) * Real.tanh raw by ring]
  rw [abs_mul]
  have hhalf : |(1 / 2 : ℝ)| = 1 / 2 := by norm_num
  rw [hhalf]
  exact mul_le_of_le_one_right (by norm_num : 0 ≤ (1 / 2 : ℝ)) htanh

theorem zeroMedianHead_preserves_parent (oldMedian raw : ℝ)
    (hraw : raw = 0) :
    oldMedian + (1 / 2 : ℝ) * Real.tanh raw = oldMedian := by
  simp [hraw]

theorem zeroWidthHead_preserves_parent_scale (base floor : ℝ)
    (missing : Bool) :
    protectedScale base floor missing 0 =
      max (Real.sqrt floor) (max (if missing then (4 / 5 : ℝ) * base else 0) base) := by
  simp [protectedScale]

/-- Typed top-level graph boundary. `parentGraph` includes the inherited
encoder, admission/masking, source and evidence summaries, associative memory
scan, inclusive read, and inherited distribution state. Those operations are
deliberately abstract interfaces until each is linked to the exact source. -/
structure Graph (T : ℕ) where
  parentGraph : Observation T → InheritedOutput
  heads : ProtectedHeads

noncomputable def forward {T : ℕ} (graph : Graph T) (input : Observation T) : Forecast :=
  headForward (graph.parentGraph input) graph.heads

theorem forward_median_formula {T : ℕ} (graph : Graph T) (input : Observation T)
    (c : Channel) :
    (forward graph input).median c =
      (graph.parentGraph input).median c + (1 / 2 : ℝ) *
        Real.tanh (graph.heads.medianHead c
          ((graph.parentGraph input).features)) := rfl

theorem forward_left_floor {T : ℕ} (graph : Graph T) (input : Observation T)
    (c : Channel) :
    Real.sqrt ((graph.parentGraph input).floorVariance c) ≤
      (forward graph input).leftScale c := by
  exact protectedScale_floor _ _ _ _

/-!
OPEN: concrete encoder/source-summary/associative-scan/read implementation and
shape correspondence; exact 102→6 and 102→12 linear parameter maps; inherited
state transition, streaming/chunk and serial/parallel equivalence; exact
PyTorch forward/backward/source pin correspondence; equal-half Gaussian
measure normalization and quantile/CDF laws; censoring atom and NLL/Jacobian;
finite-precision refinement; trained calibration, coverage and capability.
The scalar theorems above do not discharge any of these obligations.
-/

end PNDS.ProtectedSensor
