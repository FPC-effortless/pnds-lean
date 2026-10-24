import Mathlib

/-!
PNDS v1.6: fixed-size, bounded, *independently scheduled* multi-timescale
evidence retention (88 coordinates grouped in four fixed decay bands).

These are exact-real laws for one coordinate. Product-valued full memory
follows coordinatewise. They are NOT a full nonlinear/Torch floating-point
refinement and do NOT establish trained world-model skill.

For external observed basis b ∈ [-1,1], scalar evidence s ∈ [-1,1],
and decay rate g ∈ [0,1]:
    next(s,b)= (1-g)*s + g*b
A new fixed schedule of this form is associative under right-affine pair
composition; hence balanced prefix grouping is legitimate. Streaming keeps
one scalar per coordinate with no entire-event history cache.

Distinctive TAC associative 8x8 value-key memory is a DIFFERENT state
component with its own (PNDS.ParallelMemory) derivation.
-/
namespace PNDS.TimescaleEvidence

def update (gamma state observed : ℝ) : ℝ :=
  (1-gamma)*state + gamma*observed

theorem bound_unit_interval
    (g s x : ℝ)
    (hg0 : 0 ≤ g) (hg1 : g ≤ 1)
    (hs0 : -1 ≤ s) (hs1 : s ≤ 1)
    (hx0 : -1 ≤ x) (hx1 : x ≤ 1) :
    -1 ≤ update g s x ∧ update g s x ≤ 1 := by
  have hkeep : 0 ≤ 1-g := by linarith
  have l1 := mul_le_mul_of_nonneg_left hs0 hkeep
  have l2 := mul_le_mul_of_nonneg_left hx0 hg0
  have u1 := mul_le_mul_of_nonneg_left hs1 hkeep
  have u2 := mul_le_mul_of_nonneg_left hx1 hg0
  constructor <;> dsimp [update] <;> nlinarith

theorem state_difference_linear (g s t x : ℝ) :
    update g s x-update g t x = (1-g)*(s-t) := by
  dsimp [update]
  ring

theorem same_observation_nonexpansive
    (g s t x : ℝ) (hg0 : 0 ≤ g) (hg1 : g ≤ 1) :
    |update g s x - update g t x| = (1-g)*|s-t| := by
  rw [state_difference_linear, abs_mul, abs_of_nonneg (by linarith : 0 ≤ 1-g)]

/-- One coordinate of an independent-gate affine update has zero cache.
    a is retention, b is innovation. Chronological first-then-second
    composition is associatively groupable but not generally reorderable. -/
structure ScalarPair where
  p : ℝ
  q : ℝ

def applyPair (a : ScalarPair) (s : ℝ) : ℝ := s*a.p+a.q
def thenPair (a b : ScalarPair) : ScalarPair :=
  ⟨a.p*b.p, a.q*b.p+b.q⟩

theorem apply_then (a b : ScalarPair) (s : ℝ) :
    applyPair (thenPair a b) s = applyPair b (applyPair a s) := by
  unfold applyPair thenPair
  ring

theorem then_assoc (a b c : ScalarPair) :
    thenPair (thenPair a b) c = thenPair a (thenPair b c) := by
  cases a with
  | mk ap aq =>
    cases b with
    | mk bp bq =>
      cases c with
      | mk cp cq =>
        simp [thenPair]
        constructor <;> ring

def pairOf (gamma observed : ℝ) : ScalarPair :=
  ⟨1-gamma,gamma*observed⟩

theorem update_is_affine (g s x : ℝ) :
    applyPair (pairOf g x) s = update g s x := by
  simp [applyPair,pairOf,update,mul_comm]

/-- Fixed decay exposes exponentially bounded old-state sensitivity under
    repeated observed updates, without making the entire native TAC model
    globally contractive. -/
theorem one_step_perturbation_bound
    (g s t x : ℝ) (hg0 : 0 ≤ g) (hg1 : g ≤ 1) :
    |update g s x-update g t x| ≤ |s-t| := by
  rw [same_observation_nonexpansive g s t x hg0 hg1]
  have hkeep : 0 ≤ 1-g := by linarith
  calc
    (1-g)*|s-t| ≤ 1*|s-t| :=
      mul_le_mul_of_nonneg_right (by linarith) (abs_nonneg _)
    _ = |s-t| := by ring

end PNDS.TimescaleEvidence
