import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring

/- Exact-real discrete stable-velocity forecast law, fixed conditioning.
   Does not formalize learned neural maps, IEEE arithmetic or empirical skill. -/
open scoped BigOperators
namespace PNDS.CoherentTrajectory

noncomputable def gain (r : ℝ) : ℕ → ℝ
  | 0 => 0
  | n+1 => gain r n + r^n

theorem gain_add (r : ℝ) (m n : ℕ) :
    gain r (m+n) = gain r m + r^m * gain r n := by
  induction n with
  | zero => simp [gain]
  | succ n ih =>
    rw [Nat.add_succ, gain, ih, gain, pow_add]
    ring

structure Motion where
  position : ℝ
  velocity : ℝ

noncomputable def advance (r terminal : ℝ) (n : ℕ) (s : Motion) : Motion :=
  ⟨s.position + (n:ℝ)*terminal + gain r n*(s.velocity-terminal),
   terminal + r^n*(s.velocity-terminal)⟩

theorem advance_zero (r terminal : ℝ) (s : Motion) :
    advance r terminal 0 s = s := by
  cases s
  simp [advance,gain]

/-- Forecast composition is legal ONLY with the SAME fixed r and terminal. -/
theorem advance_compose (r terminal : ℝ) (m n : ℕ) (s : Motion) :
    advance r terminal n (advance r terminal m s) =
    advance r terminal (m+n) s := by
  dsimp [advance]
  congr 1
  · rw [gain_add, Nat.cast_add]
    ring
  · rw [pow_add]
    ring

/-- Quadratic form of a response-Gram scale, expressing cross-horizon coherence. -/
noncomputable def gramQuadratic {H N : ℕ}
    (z : Fin H → ℝ) (response : Fin H → Fin N → ℝ) : ℝ :=
  ∑ j, (∑ i, z i * response i j)^2

theorem gram_nonnegative {H N : ℕ}
    (z : Fin H → ℝ) (response : Fin H → Fin N → ℝ) :
    0 ≤ gramQuadratic z response := by
  unfold gramQuadratic
  exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)

noncomputable def noiseQuadratic {H N : ℕ} (sx sv : ℝ)
    (z : Fin H → ℝ) (w k : Fin H → Fin N → ℝ) : ℝ :=
  sx^2 * gramQuadratic z w + sv^2 * gramQuadratic z k

theorem noise_nonnegative {H N : ℕ} (sx sv : ℝ)
    (z : Fin H → ℝ) (w k : Fin H → Fin N → ℝ) :
    0 ≤ noiseQuadratic sx sv z w k := by
  exact add_nonneg (mul_nonneg (sq_nonneg sx) (gram_nonnegative z w))
    (mul_nonneg (sq_nonneg sv) (gram_nonnegative z k))

theorem retained_delta_bound (r d : ℝ) (hr : |r| ≤ 1) (n : ℕ) :
    |r^n * d| ≤ |d| := by
  induction n with
  | zero => simp
  | succ n ih =>
    have step : |r * (r^n * d)| ≤ |r^n * d| := by
      rw [abs_mul]
      simpa using mul_le_mul_of_nonneg_right hr (abs_nonneg (r^n*d))
    have eqn : r^(n+1)*d = r*(r^n*d) := by rw [pow_succ]; ring
    rw [eqn]
    exact le_trans step ih

structure Evidence where
  evidenceSum : Fin 88 → ℝ
  count : ℕ
  memory : Fin 8 → Fin 8 → ℝ

structure Observation where
  measured : Fin 14 → ℝ
  laggedAction : Fin 14 → ℝ
  valid : Bool

structure Coefficients where
  initial : Fin 14 → Motion
  terminal : Fin 14 → ℝ
  retention : Fin 14 → ℝ
  positionNoise : Fin 14 → ℝ
  velocityNoise : Fin 14 → ℝ

/-- The concrete learned maps remain an explicit neural refinement obligation. -/
structure WorldGraph where
  condition : Evidence → Observation → Evidence × Coefficients

noncomputable def predict (g : WorldGraph) (s : Evidence) (x : Observation)
    (horizons : List ℕ) : Evidence × List (Fin 14 → Motion) :=
  let c := g.condition s x
  (c.1, horizons.map (fun h i =>
      advance (c.2.retention i) (c.2.terminal i) h (c.2.initial i)))

/-- Imagined future queries never overwrite observed-world evidence. -/
theorem future_queries_preserve_evidence (g : WorldGraph) (s : Evidence)
    (x : Observation) (hs : List ℕ) :
    (predict g s x hs).1 = (g.condition s x).1 := by rfl

end PNDS.CoherentTrajectory
