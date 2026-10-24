import Mathlib
import PNDS.UnifiedTACOSM

/-!
# TAC-OSM native associative write algebra — exact real semantics

Source anchor: TAC-OSM 081e534a074001c6a3b2251b3ff31ad20ec4fca8,
  src/tac_osm/aligned_associative_tac.py, gated_write().
Actual runtime uses B×8×8 stores, normalized addresses, input-dependent and state-dependent
retention/strength, write-role mask and Frobenius projection. Here we prove the
unprojected, one-row/two-key-coordinate algebra only, under explicit premises.
The same algebra is componentwise for an m×n store before projection.

No source-level floating-point, learned-state Jacobian, data, causal, or
long-range reliability assertion follows. Crucially:
* an unrelated write can damage an old address if its dot product is nonzero;
* retention α<1 attenuates an orthogonal old read (unless read-only);
* even a β=1 corrective write need not be exact after radius projection.
-/

namespace PNDS.NativeMemory

def read2 (m0 m1 k0 k1 : ℝ) : ℝ := m0 * k0 + m1 * k1

def unprojected0 (m0 m1 k0 k1 v alpha beta : ℝ) : ℝ :=
  alpha * m0 + beta * (v - alpha * read2 m0 m1 k0 k1) * k0

def unprojected1 (m0 m1 k0 k1 v alpha beta : ℝ) : ℝ :=
  alpha * m1 + beta * (v - alpha * read2 m0 m1 k0 k1) * k1

theorem arbitrary_query_decomposition
    (m0 m1 k0 k1 v alpha beta q0 q1 : ℝ) :
    read2 (unprojected0 m0 m1 k0 k1 v alpha beta)
      (unprojected1 m0 m1 k0 k1 v alpha beta) q0 q1 =
      alpha * read2 m0 m1 q0 q1 +
        beta * (v - alpha * read2 m0 m1 k0 k1) * read2 k0 k1 q0 q1 := by
  dsimp [read2, unprojected0, unprojected1]
  ring

theorem orthogonal_query_retains_only_decay
    (m0 m1 k0 k1 v alpha beta q0 q1 : ℝ)
    (horth : read2 k0 k1 q0 q1 = 0) :
    read2 (unprojected0 m0 m1 k0 k1 v alpha beta)
      (unprojected1 m0 m1 k0 k1 v alpha beta) q0 q1 =
      alpha * read2 m0 m1 q0 q1 := by
  rw [arbitrary_query_decomposition, horth]
  ring

theorem unit_address_self_read
    (m0 m1 k0 k1 v alpha beta : ℝ)
    (hunit : k0 * k0 + k1 * k1 = 1) :
    read2 (unprojected0 m0 m1 k0 k1 v alpha beta)
      (unprojected1 m0 m1 k0 k1 v alpha beta) k0 k1 =
      (1 - beta) * (alpha * read2 m0 m1 k0 k1) + beta * v := by
  rw [arbitrary_query_decomposition]
  rw [show read2 k0 k1 k0 k1 = 1 by
        simpa [read2] using hunit]
  ring

theorem unit_address_residual
    (m0 m1 k0 k1 v alpha beta : ℝ)
    (hunit : k0 * k0 + k1 * k1 = 1) :
    v - read2 (unprojected0 m0 m1 k0 k1 v alpha beta)
      (unprojected1 m0 m1 k0 k1 v alpha beta) k0 k1 =
      (1 - beta) * (v - alpha * read2 m0 m1 k0 k1) := by
  rw [unit_address_self_read _ _ _ _ _ _ _ hunit]
  ring

def maskedCoordinate (writable : Bool) (old newValue : ℝ) : ℝ :=
  if writable then newValue else old

theorem read_only_no_mutation (old newValue : ℝ) :
    maskedCoordinate false old newValue = old := by
  rfl

theorem shared_key_exact_collision_impossible
    (readValue first second : ℝ)
    (hfirst : readValue = first)
    (hsecond : readValue = second) : first = second := by
  calc first = readValue := hfirst.symm
       _ = second := hsecond

noncomputable def positiveRadiusClip (radius x : ℝ) : ℝ :=
  if x > radius then radius else x

theorem projected_exact_write_counterexample :
    read2 (unprojected0 0 0 1 0 2 1 1)
      (unprojected1 0 0 1 0 2 1 1) 1 0 = 2 ∧
    positiveRadiusClip 1
      (read2 (unprojected0 0 0 1 0 2 1 1)
        (unprojected1 0 0 1 0 2 1 1) 1 0) = 1 := by
  constructor <;> norm_num [read2, unprojected0, unprojected1, positiveRadiusClip]

theorem fixed_gate_incremental_sensitivity
    (old1 old2 target gate : ℝ) :
    PNDS.UnifiedTACOSM.gated old1 target gate -
    PNDS.UnifiedTACOSM.gated old2 target gate =
      (1 - gate) * (old1 - old2) := by
  unfold PNDS.UnifiedTACOSM.gated
  ring


/-- Fully dimension-generic row definition; used on each of the eight
    value rows independently. This includes the genuine 8D key dimensions,
    but deliberately excludes nonlinear projection and learned gate gradients. -/
def rowRead {n : ℕ} (m k : Fin n → ℝ) : ℝ :=
  ∑ i, m i * k i

def rowWrite {n : ℕ} (m k : Fin n → ℝ)
    (v alpha beta : ℝ) (i : Fin n) : ℝ :=
  alpha * m i + beta * (v - alpha * rowRead m k) * k i

theorem row_query_decomposition {n : ℕ}
    (m k q : Fin n → ℝ) (v alpha beta : ℝ) :
    rowRead (rowWrite m k v alpha beta) q =
      alpha * rowRead m q +
      beta * (v - alpha * rowRead m k) * rowRead k q := by
  calc
    rowRead (rowWrite m k v alpha beta) q =
        ∑ i : Fin n, (alpha * (m i * q i) +
          (beta * (v - alpha * rowRead m k)) * (k i * q i)) := by
      unfold rowRead
      apply Finset.sum_congr rfl
      intro i _
      dsimp [rowWrite]
      ring
    _ = alpha * rowRead m q +
        beta * (v - alpha * rowRead m k) * rowRead k q := by
      rw [Finset.sum_add_distrib]
      simp only [← Finset.mul_sum, rowRead]

theorem row_unit_key_residual {n : ℕ}
    (m k : Fin n → ℝ) (v alpha beta : ℝ)
    (hunit : rowRead k k = 1) :
    v - rowRead (rowWrite m k v alpha beta) k =
      (1 - beta) * (v - alpha * rowRead m k) := by
  rw [row_query_decomposition, hunit]
  ring

end PNDS.NativeMemory
