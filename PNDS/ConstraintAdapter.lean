import Mathlib

/-!
A *conditional* civilian planar single-integrator safety adapter. This is not
the existing native TAC-OSM world model, not a trajectory safety proof for
second-order robots, and not a certificate for uncertain or unmodelled dynamics.

The halfspace projection is Euclidean, scalar in its formal KKT/optimality proof.
The two-coordinate barrier theorem assumes constant relative velocities and
exact translations over a sample of duration dt. It includes neither motor
dynamics nor uncertain obstacle movement. New formal facts do not certify
the associated Python floating-point implementation or real-world performance.
-/

namespace PNDS.ConstraintAdapter

/-- Exact one-dimensional projection onto a closed feasible half-line. -/
def projectLower (desired lower : ℝ) : ℝ := max desired lower

theorem projection_feasible (desired lower : ℝ) :
    lower ≤ projectLower desired lower := by
  exact le_max_right desired lower

theorem projection_optimal (desired lower candidate : ℝ)
    (hc : lower ≤ candidate) :
    (projectLower desired lower - desired) ^ 2 ≤
      (candidate - desired) ^ 2 := by
  by_cases h : lower ≤ desired
  · rw [show projectLower desired lower = desired by
       simp [projectLower, max_eq_left h]]
    nlinarith [sq_nonneg (candidate - desired)]
  · have hlt : desired < lower := lt_of_not_ge h
    rw [show projectLower desired lower = lower by
      simp [projectLower, max_eq_right (le_of_lt hlt)]]
    have ha : 0 ≤ candidate - lower := by linarith
    have hb : 0 ≤ candidate + lower - 2 * desired := by linarith
    nlinarith [mul_nonneg ha hb]

/-- Dual multiplier for min (u-desired)^2 subject to u>=lower. -/
def dualLower (desired lower : ℝ) : ℝ :=
  2 * (projectLower desired lower - desired)

theorem dual_lower_nonnegative (desired lower : ℝ) :
    0 ≤ dualLower desired lower := by
  unfold dualLower projectLower
  have h := le_max_left desired lower
  linarith

theorem kkt_stationarity (desired lower : ℝ) :
    2 * (projectLower desired lower - desired) - dualLower desired lower = 0 := by
  unfold dualLower
  ring

theorem kkt_complementary (desired lower : ℝ) :
    dualLower desired lower * (projectLower desired lower - lower) = 0 := by
  by_cases h : lower ≤ desired
  · simp [dualLower, projectLower, max_eq_left h]
  · have hlt : desired < lower := lt_of_not_ge h
    simp [dualLower, projectLower, max_eq_right (le_of_lt hlt)]

/-- Planar distance-squared barrier relative to an obstacle at the origin. -/
def planarBarrier (r0 r1 radius : ℝ) : ℝ :=
  r0*r0 + r1*r1 - radius*radius

/-- The barrier under constant relative velocity v for a duration dt. -/
def translatedBarrier (r0 r1 v0 v1 dt radius : ℝ) : ℝ :=
  planarBarrier (r0 + dt*v0) (r1 + dt*v1) radius

/-- Exact sampled update identity for planar single-integrator kinematics.
    The dt² term is nonnegative over reals; no numerical-integration error. -/
theorem planar_sample_identity
    (r0 r1 v0 v1 dt radius gamma : ℝ) :
    translatedBarrier r0 r1 v0 v1 dt radius =
      (1 - gamma*dt) * planarBarrier r0 r1 radius +
      dt * (2*(r0*v0 + r1*v1) + gamma*planarBarrier r0 r1 radius) +
      (dt*v0)^2 + (dt*v1)^2 := by
  unfold translatedBarrier planarBarrier
  ring

/-- Conditional one-sample safety; h>=0 at start, CBF inequality at start,
    and gamma*dt<=1. Motion during the sample has CONSTANT relative velocity.
    It makes no claim about unknown dynamics or optimizer failure. -/
theorem planar_sample_safe
    (r0 r1 v0 v1 dt radius gamma : ℝ)
    (hstart : 0 ≤ planarBarrier r0 r1 radius)
    (hdt : 0 ≤ dt)
    (hperiod : gamma*dt ≤ 1)
    (hcbf : 0 ≤ 2*(r0*v0 + r1*v1) +
      gamma*planarBarrier r0 r1 radius) :
    0 ≤ translatedBarrier r0 r1 v0 v1 dt radius := by
  rw [planar_sample_identity]
  have hfactor : 0 ≤ 1 - gamma*dt := by linarith
  have hret : 0 ≤ (1 - gamma*dt) * planarBarrier r0 r1 radius :=
    mul_nonneg hfactor hstart
  have hcbfscaled : 0 ≤ dt * (2*(r0*v0 + r1*v1) +
      gamma*planarBarrier r0 r1 radius) :=
    mul_nonneg hdt hcbf
  nlinarith [sq_nonneg (dt*v0), sq_nonneg (dt*v1)]

end PNDS.ConstraintAdapter
