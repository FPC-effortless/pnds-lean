import Mathlib

/-!
# Native TAC-OSM conditional parallel affine associative-memory operator

Pinned original source:
TAC-OSM commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8,
src/tac_osm/aligned_associative_tac.py gated_write.

When the schedule (retention alpha, corrective gate beta, normalized learned
key k and value v) is fixed *externally* and radius projection is disabled,
one genuine 8x8 gated-delta update is right-affine:

 M' = M P + Q,
 P = alpha*(I - beta*k*k^T), Q = beta*v*k^T.

It is associative to compose pre-computed pairs IN CHRONOLOGICAL ORDER.
The full native fast→slow→memory recurrence remains sequential because its
gates depend on new states and earlier recalled memory. In addition the
source's global Frobenius projection is nonlinear, so it does NOT commute
with the affine scan. Thus this proves mathematical scan legality for the
conditioned memory subgraph only; no full model scan/speedup/skill follows.

The input-only matrix-vector projections of the neural layer may be
pre-computed over all B×T inputs via affine linearity; the original nonlinear
state→gate transitions cannot be reordered using these theorems.
-/

namespace PNDS.ParallelMemory

/-- A right-affine endomorphism on square real matrices: X ↦ X P + Q. -/
structure AffinePair (n : ℕ) where
  P : Matrix (Fin n) (Fin n) ℝ
  Q : Matrix (Fin n) (Fin n) ℝ

def applyPair {n : ℕ} (a : AffinePair n)
    (m : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ := m * a.P + a.Q

/-- Chronological order: a executes first, b second. -/
def thenPair {n : ℕ} (a b : AffinePair n) : AffinePair n :=
  ⟨a.P * b.P, a.Q * b.P + b.Q⟩

/-- Associative scan requires this exact semantic law. -/
theorem apply_then {n : ℕ} (a b : AffinePair n)
    (m : Matrix (Fin n) (Fin n) ℝ) :
    applyPair (thenPair a b) m = applyPair b (applyPair a m) := by
  simp [applyPair, thenPair, add_mul, mul_assoc, add_assoc]

/-- Grouping a chronological triple does not change its mathematical
    operator, even though swapping order generally does. -/
theorem then_assoc {n : ℕ} (a b c : AffinePair n) :
    thenPair (thenPair a b) c = thenPair a (thenPair b c) := by
  cases a with
  | mk pa qa =>
    cases b with
    | mk pb qb =>
      cases c with
      | mk pc qc =>
        simp [thenPair, mul_assoc, add_mul, add_assoc]

/-- Identity for the full matrix affine transform. -/
def identity (n : ℕ) : AffinePair n :=
  ⟨1, 0⟩

theorem apply_identity {n : ℕ} (m : Matrix (Fin n) (Fin n) ℝ) :
    applyPair (identity n) m = m := by
  simp [applyPair, identity]

theorem then_identity {n : ℕ} (a : AffinePair n) :
    thenPair a (identity n) = a := by
  cases a with
  | mk pa qa =>
    simp [thenPair, identity]

theorem identity_then {n : ℕ} (a : AffinePair n) :
    thenPair (identity n) a = a := by
  cases a with
  | mk pa qa =>
    simp [thenPair, identity]

/-- An ordered list may be partitioned into contiguous chunks and grouped
    without changing the chronological fold. This relies on the associative
    thenPair, NOT on commutation or independence of the full original model. -/
theorem chronological_chunks {n : ℕ}
    (xs ys : List (AffinePair n)) :
    List.foldl thenPair (identity n) (xs ++ ys) =
    List.foldl thenPair
      (List.foldl thenPair (identity n) xs) ys := by
  simp [List.foldl_append]

/-- A zero initial memory returns exactly the right-affine accumulated
    additive offset, as in a prefix scan on source-zero-initialized M. -/
theorem zero_apply {n : ℕ} (a : AffinePair n) :
    applyPair a (0 : Matrix (Fin n) (Fin n) ℝ) = a.Q := by
  simp [applyPair]

/-- Read-only means the identity affine pair and no memory change. -/
theorem read_only_no_mutation {n : ℕ}
    (m : Matrix (Fin n) (Fin n) ℝ) :
    applyPair (identity n) m = m :=
  apply_identity m


/-- A conservative retained-norm budget permits a *projection-free* affine
    memory update while keeping capacity R. If ||M'|| is upper-bounded by
    alpha_eff*||M|| + beta, choosing
        alpha_eff = raw_retention*(1-beta/R)
    keeps the state within R given raw_retention∈[0,1], beta∈[0,1],
    R≥1, and ||M||≤R.

    The norm recurrence is an **explicit premise**. Its derivation from
    the spectral norm of I-beta*k*kᵀ and unit k/v is established analytically
    in the paired report and checked numerically. This exact-real scalar
    theorem is NOT a proof of all original Torch kernels, output quality or
    long-term retrieval stability. -/
theorem bounded_projection_free_budget
    (R beta rawRetention oldNorm newNorm : ℝ)
    (hR : 1 ≤ R)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1)
    (ha0 : 0 ≤ rawRetention) (ha1 : rawRetention ≤ 1)
    (ho0 : 0 ≤ oldNorm) (hoR : oldNorm ≤ R)
    (hn : newNorm ≤
       (rawRetention * (1 - beta / R)) * oldNorm + beta) :
    newNorm ≤ R := by
  have hpos : 0 < R := by linarith
  have hkeep : 0 ≤ 1 - beta / R := by
    have hdiv : beta / R ≤ 1 :=
      (div_le_iff₀ hpos).2 (by linarith)
    linarith
  have heff : 0 ≤ rawRetention * (1 - beta / R) :=
    mul_nonneg ha0 hkeep
  have hmul := mul_le_mul_of_nonneg_left hoR heff
  have hformula :
      (rawRetention * (1 - beta / R)) * R =
        rawRetention * (R - beta) := by
    field_simp
    ring
  have hd : 0 ≤ R - beta := by linarith
  have hgap : 0 ≤ (1 - rawRetention) * (R - beta) :=
    mul_nonneg (by linarith) hd
  rw [hformula] at hmul
  nlinarith

end PNDS.ParallelMemory
