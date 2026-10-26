import PNDS.RoleSpecializedFFN
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.Linarith

/-!
ME108 exact-real training-step contract for the ME107 role-specialized native
TAC decoder. This module specifies a raw-gradient AdamW proposal followed by
the ME107 per-block projection guard. Clipping is an externally supplied
transformation; the module makes no claim about the trained-source gradient,
clipping implementation, finite IEEE arithmetic, or model-level derivatives.
-/
namespace PNDS.RoleGuardedTraining

open PNDS.RoleSpecializedFFN
open PNDS.SensorGraph105

variable {n : ℕ}
abbrev Vec (n : ℕ) := Fin n → ℝ

/-- A clipping operation is supplied by the caller before the optimizer step.
No particular norm, threshold, or implementation is assumed here. -/
def ClippedGradient (clip : Vec n → Vec n) (raw : Vec n) : Vec n := clip raw

/-- AdamW hyperparameters. Positivity and stability conditions are explicit
step assumptions rather than hidden requirements on the data. -/
structure AdamWConfig where
  beta1 : ℝ
  beta2 : ℝ
  learningRate : ℝ
  weightDecay : ℝ
  epsilon : ℝ

/-- First and second moments and the one-based step counter. -/
structure AdamWState (n : ℕ) where
  firstMoment : Vec n
  secondMoment : Vec n
  step : ℕ

/-- Exact-real AdamW moments from the clipped combined gradient. -/
def nextState (cfg : AdamWConfig) (state : AdamWState n) (g : Vec n) : AdamWState n :=
  { firstMoment := fun i => cfg.beta1 * state.firstMoment i + (1 - cfg.beta1) * g i
    secondMoment := fun i => cfg.beta2 * state.secondMoment i + (1 - cfg.beta2) * g i ^ 2
    step := state.step + 1 }

/-- Bias-corrected AdamW proposal, including decoupled weight decay. -/
noncomputable def adamwProposal (cfg : AdamWConfig) (state : AdamWState n)
    (weights g : Vec n) : Vec n := by
  let next := nextState cfg state g
  let t := next.step
  exact fun i => -cfg.learningRate *
    (((next.firstMoment i / (1 - cfg.beta1 ^ t)) /
        (Real.sqrt (next.secondMoment i / (1 - cfg.beta2 ^ t)) + cfg.epsilon)) +
      cfg.weightDecay * weights i)

/-- One guarded step: the gradient drives Adam moments even if the proposed
weight displacement is subsequently projected by the role protection. -/
noncomputable def guardedStep (cfg : AdamWConfig) (state : AdamWState n)
    (weights pointGradient : Vec n) (clippedGradient : Vec n) : AdamWState n × Vec n := by
  let proposal := adamwProposal cfg state weights clippedGradient
  exact (nextState cfg state clippedGradient,
    fun i => weights i + protectedDisplacement pointGradient proposal i)

theorem guardedStep_moments_use_clipped_gradient (cfg : AdamWConfig)
    (state : AdamWState n) (weights pointGradient clippedGradient : Vec n) :
    (guardedStep cfg state weights pointGradient clippedGradient).1 =
      nextState cfg state clippedGradient := by
  simp [guardedStep, adamwProposal]

theorem guardedStep_weights_use_protected_displacement (cfg : AdamWConfig)
    (state : AdamWState n) (weights pointGradient clippedGradient : Vec n) (i : Fin n) :
    (guardedStep cfg state weights pointGradient clippedGradient).2 i =
      weights i + protectedDisplacement pointGradient
        (adamwProposal cfg state weights clippedGradient) i := by
  simp [guardedStep, adamwProposal]

/-- The committed displacement has nonpositive inner product with the point
gradient whenever that gradient has positive squared norm. -/
theorem guardedStep_point_dot_nonpos (cfg : AdamWConfig) (state : AdamWState n)
    (weights pointGradient clippedGradient : Vec n)
    (hn : normSq pointGradient > 0) :
    dot pointGradient
      (fun i => (guardedStep cfg state weights pointGradient clippedGradient).2 i - weights i) ≤ 0 := by
  have h := protected_dot_nonpos pointGradient
    (adamwProposal cfg state weights clippedGradient) hn
  convert h using 1 <;> simp [guardedStep, adamwProposal]

theorem guardedStep_zero_point_coordinate_unchanged (cfg : AdamWConfig)
    (state : AdamWState n) (weights pointGradient clippedGradient : Vec n) (i : Fin n)
    (hi : pointGradient i = 0) :
    (guardedStep cfg state weights pointGradient clippedGradient).2 i =
      weights i + adamwProposal cfg state weights clippedGradient i := by
  rw [guardedStep_weights_use_protected_displacement]
  rw [zero_protected_coordinate_unchanged _ _ i hi]

theorem guardedStep_unopposed_proposal_unchanged (cfg : AdamWConfig)
    (state : AdamWState n) (weights pointGradient clippedGradient : Vec n)
    (hfeasible : dot pointGradient (adamwProposal cfg state weights clippedGradient) ≤ 0) :
    (guardedStep cfg state weights pointGradient clippedGradient).2 =
      fun i => weights i + adamwProposal cfg state weights clippedGradient i := by
  funext i
  rw [guardedStep_weights_use_protected_displacement]
  rw [feasible_displacement_unchanged _ _ hfeasible]

/-- The role-guarded model uses exactly the inherited ME107 forward function.
The optimizer is external to this forward computation. -/
noncomputable def forwardRole := @PNDS.RoleSpecializedFFN.forwardRole

theorem forwardRole_unchanged {T : ℕ} (b : BackboneParameters) (h : RoleHeads)
    (o : Observation T) (last : Fin T) (hlast : ∀ i : Fin T, i ≤ last) :
    forwardRole b h o last hlast =
      PNDS.RoleSpecializedFFN.forwardRole b h o last hlast := rfl

theorem model_parameter_count : roleSpecializedParameterCount = 102713 :=
  roleSpecializedParameterCount_exact

/-- The proposal is produced from the supplied clipped combined gradient and
weights; no derivative through AdamW or the projection is represented. -/
theorem proposal_is_explicit (cfg : AdamWConfig) (state : AdamWState n)
    (weights clippedGradient : Vec n) :
    adamwProposal cfg state weights clippedGradient =
      (fun i => -cfg.learningRate *
        (((nextState cfg state clippedGradient).firstMoment i /
            (1 - cfg.beta1 ^ (nextState cfg state clippedGradient).step)) /
            (Real.sqrt ((nextState cfg state clippedGradient).secondMoment i /
              (1 - cfg.beta2 ^ (nextState cfg state clippedGradient).step)) + cfg.epsilon) +
          cfg.weightDecay * weights i)) := by rfl

end PNDS.RoleGuardedTraining
