import Mathlib

/-!
PNDS unified TAC-OSM mathematical foundation v1.
These are exact-real operator facts under explicit assumptions, not a full
PyTorch proof, trained retrieval guarantee or causal/real-world qualification.
The PNDS v0.2 modules remain untouched.
-/

namespace PNDS.UnifiedTACOSM

/-- Scalar gated coordinate. Full native fast/slow recurrence is nonlinear. -/
def gated (old target g : ℝ) : ℝ := (1 - g) * old + g * target

/-- Coordinate boundedness of a gate within [0,1]. -/
theorem gated_interval (old target g : ℝ)
    (hold : -1 ≤ old ∧ old ≤ 1)
    (htarget : -1 ≤ target ∧ target ≤ 1)
    (hg : 0 ≤ g ∧ g ≤ 1) :
    -1 ≤ gated old target g ∧ gated old target g ≤ 1 := by
  have hlowOld : 0 ≤ (1 - g) * (old + 1) := by
    apply mul_nonneg <;> linarith
  have hlowTarget : 0 ≤ g * (target + 1) := by
    apply mul_nonneg <;> linarith
  have hhighOld : 0 ≤ (1 - g) * (1 - old) := by
    apply mul_nonneg <;> linarith
  have hhighTarget : 0 ≤ g * (1 - target) := by
    apply mul_nonneg <;> linarith
  constructor <;> dsimp [gated] <;> nlinarith [hlowOld, hlowTarget, hhighOld, hhighTarget]

/-- Exact target-error recurrence. Not a long-memory retention guarantee. -/
theorem gated_error (old target g : ℝ) :
    target - gated old target g = (1 - g) * (target - old) := by
  dsimp [gated]
  ring

/-- Scalar affine transition, liftable coordinatewise to diagonal filters. -/
structure AffineTransition where
  a : ℝ
  b : ℝ

def runAffine (t : AffineTransition) (state : ℝ) : ℝ :=
  t.a * state + t.b

/-- Compose transitions applying earlier first, then later. -/
def composeAffine (later earlier : AffineTransition) : AffineTransition :=
  ⟨later.a * earlier.a, later.a * earlier.b + later.b⟩

theorem run_compose (later earlier : AffineTransition) (state : ℝ) :
    runAffine (composeAffine later earlier) state =
      runAffine later (runAffine earlier state) := by
  dsimp [runAffine, composeAffine]
  ring

/-- Exact-real associative composition admits a scan for the affine subgraph.
    It does not parallelize arbitrary nonlinear TAC recurrence. -/
theorem compose_assoc (x y z : AffineTransition) :
    composeAffine (composeAffine x y) z =
      composeAffine x (composeAffine y z) := by
  cases x with
  | mk xa xb =>
    cases y with
    | mk ya yb =>
      cases z with
      | mk za zb =>
        dsimp [composeAffine]
        congr 1 <;> ring

/-- Conditional source EOR/activity normalization. Activity mass=1 is assumed. -/
theorem source_end_mass_normalized (survival activityTotal : ℝ)
    (htotal : activityTotal = 1) :
    (1 - survival) + survival * activityTotal = 1 := by
  rw [htotal]
  ring

theorem source_end_mass_nonnegative (survival : ℝ)
    (h : 0 ≤ survival ∧ survival ≤ 1) :
    0 ≤ survival ∧ 0 ≤ 1 - survival := by
  constructor
  · exact h.1
  · linarith [h.2]

/-- Exact zero measured speed suppresses the quadratic residual gate.
    This makes no no-regression promise at small nonzero noisy speeds. -/
def quadraticMotionGate (v c : ℝ) : ℝ := v ^ 2 / (v ^ 2 + c)

theorem quadratic_zero (c : ℝ) :
    quadraticMotionGate 0 c = 0 := by
  simp [quadraticMotionGate]

/-- These are epistemic evidence categories, not model-state transitions. -/
inductive EvidenceTier where
  | mathematicalContract
  | syntheticDiagnostic
  | exposedRealDevelopment
  | independentRealTest
  | measuredOperationalUse
  deriving Repr, DecidableEq

/-- Eligibility for further real-world evaluation claims is NOT itself proof
    of model superiority, real action benefit or commercial deployment. -/
def hasIndependentRealBasis : EvidenceTier → Bool
  | .independentRealTest => true
  | .measuredOperationalUse => true
  | _ => false

theorem synthetic_not_real_qualification :
    hasIndependentRealBasis .syntheticDiagnostic = false := by
  rfl

theorem development_not_independent_qualification :
    hasIndependentRealBasis .exposedRealDevelopment = false := by
  rfl

end PNDS.UnifiedTACOSM
