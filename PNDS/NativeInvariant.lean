import Mathlib
import PNDS.NativeMemory

/-!
TAC-OSM native fast/slow/associative **forward boundedness envelope**.

Source contract: TAC-OSM 081e534a074001c6a3b2251b3ff31ad20ec4fca8,
src/tac_osm/aligned_associative_tac.py. Each fast coordinate mixes its
previous state and a tanh-bounded learned candidate, with sigmoid gate.
Each slow coordinate similarly mixes its previous state and a
tanh(slow + tanh(delta))-bounded candidate, with 0.25*sigmoid gate.
On writable events the original code projects the whole 8x8 matrix into
a radius-4 Frobenius ball; on read-only events it preserves that matrix.

The theorem below abstracts the *boundedness of tanh/sigmoid and radius
projection* into EXPLICIT premises. Those premises are not inferred
from an arbitrary learned operator or floating-point library.
Therefore it proves a compositional envelope conditional on correct
implementations of those suboperators. It proves NO contraction,
nonvanishing memory, input/output numerical equivalence, or real skill.

The noncontraction counterexample is source-significant: exact read-only
memory preservation is incompatible with a universal strict contraction
of the entire state in a norm that resolves memory coordinates.
-/

namespace PNDS.NativeInvariant

def InUnitBox (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def InRadiusFour (x : ℝ) : Prop := -4 ≤ x ∧ x ≤ 4

def mixed (old target gate : ℝ) : ℝ :=
  (1 - gate) * old + gate * target

theorem mixed_unit_box (old target gate : ℝ)
    (ho : InUnitBox old) (ht : InUnitBox target)
    (hg0 : 0 ≤ gate) (hg1 : gate ≤ 1) :
    InUnitBox (mixed old target gate) := by
  rcases ho with ⟨hlo, hhi⟩
  rcases ht with ⟨htlo, hthi⟩
  have hremain : 0 ≤ 1 - gate := by linarith
  have l1 : 0 ≤ (1 - gate) * (old + 1) :=
    mul_nonneg hremain (by linarith)
  have l2 : 0 ≤ gate * (target + 1) :=
    mul_nonneg hg0 (by linarith)
  have u1 : 0 ≤ (1 - gate) * (1 - old) :=
    mul_nonneg hremain (by linarith)
  have u2 : 0 ≤ gate * (1 - target) :=
    mul_nonneg hg0 (by linarith)
  constructor
  · dsimp [InUnitBox, mixed]
    nlinarith [l1, l2]
  · dsimp [InUnitBox, mixed]
    nlinarith [u1, u2]

abbrev Register := Fin 88 → ℝ
abbrev Memory := Fin 8 → Fin 8 → ℝ

structure NativeState where
  fast : Register
  slow : Register
  memory : Memory

def ForwardEnvelope (s : NativeState) : Prop :=
  (∀ i, InUnitBox (s.fast i)) ∧
  (∀ i, InUnitBox (s.slow i)) ∧
  (∀ i j, InRadiusFour (s.memory i j))

/-- Native source-shaped abstract forward update. Arguments target and
gate are **already evaluated** from ordered old-read→fast→slow
dependencies. Actual gates/targets must separately satisfy the
postconditions, and projected must be the implemented radius-4 mapping.
-/
def step (s : NativeState)
    (fastTarget fastGate slowTarget slowGate : Register)
    (unprojected : Memory) (writable : Bool)
    (projected : Memory → Memory) : NativeState :=
  { fast := fun i => mixed (s.fast i) (fastTarget i) (fastGate i)
    slow := fun i => mixed (s.slow i) (slowTarget i) (slowGate i)
    memory := if writable then projected unprojected else s.memory }

/-- Complete-register, one-step *conditional* source envelope. The matrix
premise is stronger than per-entry bounds: the actual Frobenius projection
must guarantee all entries lie in [-4,4]; no theorem about its implementation
is smuggled in here. -/
theorem step_preserves_envelope
    (s : NativeState) (fastTarget fastGate slowTarget slowGate : Register)
    (unprojected : Memory) (writable : Bool)
    (projected : Memory → Memory)
    (hs : ForwardEnvelope s)
    (hft : ∀ i, InUnitBox (fastTarget i))
    (hfg : ∀ i, 0 ≤ fastGate i ∧ fastGate i ≤ 1)
    (hst : ∀ i, InUnitBox (slowTarget i))
    (hsg : ∀ i, 0 ≤ slowGate i ∧ slowGate i ≤ 1)
    (hproj : ∀ m i j, InRadiusFour (projected m i j)) :
    ForwardEnvelope
      (step s fastTarget fastGate slowTarget slowGate
        unprojected writable projected) := by
  rcases hs with ⟨hf, hslow, hmem⟩
  constructor
  · intro i
    exact mixed_unit_box (s.fast i) (fastTarget i) (fastGate i)
      (hf i) (hft i) (hfg i).1 (hfg i).2
  constructor
  · intro i
    exact mixed_unit_box (s.slow i) (slowTarget i) (slowGate i)
      (hslow i) (hst i) (hsg i).1 (hsg i).2
  · intro i j
    cases writable with
    | false =>
      simpa [step] using (hmem i j)
    | true =>
      simpa [step] using (hproj unprojected i j)

/-- The structural bound composes indefinitely when every successive
transition satisfies the conditional one-step premises. -/
def iterateSteps (F : NativeState → NativeState) (initial : NativeState) :
    ℕ → NativeState
  | 0 => initial
  | n + 1 => F (iterateSteps F initial n)

theorem repeated_forward_envelope
    (F : NativeState → NativeState) (initial : NativeState)
    (hstep : ∀ s, ForwardEnvelope s → ForwardEnvelope (F s))
    (hinit : ForwardEnvelope initial) (n : ℕ) :
    ForwardEnvelope (iterateSteps F initial n) := by
  induction n with
  | zero => simpa [iterateSteps] using hinit
  | succ n ih =>
    simpa [iterateSteps] using hstep _ ih

/-- Read-only memory carries the *same* coordinate difference, even if
fast/slow components change. Thus the full state is not uniformly strictly
contractive in any product metric that retains this memory difference. -/
theorem no_strict_contraction_of_read_only_memory (rho : ℝ)
    (hrho : rho < 1) :
    ¬ (∀ a b : ℝ,
       |PNDS.NativeMemory.maskedCoordinate false a b -
        PNDS.NativeMemory.maskedCoordinate false b a| ≤ rho * |a - b|) := by
  intro h
  have hle : (1 : ℝ) ≤ rho := by
    simpa [PNDS.NativeMemory.maskedCoordinate] using h 1 0
  linarith


/-- Exact scalar magnitude mapping induced by the original source's radial
    Frobenius clipping formula, provided radius>0 and norm>=0. The tensor
    identity ||U/c||_F=||U||_F/c is a separate implementation obligation. -/
noncomputable def projectedMagnitude (radius originalNorm : ℝ) : ℝ :=
  originalNorm / max 1 (originalNorm / radius)

theorem projectedMagnitude_bounds (radius originalNorm : ℝ)
    (hradius : 0 < radius) (hnorm : 0 ≤ originalNorm) :
    0 ≤ projectedMagnitude radius originalNorm ∧
      projectedMagnitude radius originalNorm ≤ radius := by
  have hd : 0 < max (1 : ℝ) (originalNorm / radius) :=
    lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  constructor
  · exact div_nonneg hnorm (le_of_lt hd)
  · apply (div_le_iff₀ hd).2
    have hquot : originalNorm ≤
        max (1 : ℝ) (originalNorm / radius) * radius :=
      (div_le_iff₀ hradius).1 (le_max_right _ _)
    nlinarith


end PNDS.NativeInvariant
