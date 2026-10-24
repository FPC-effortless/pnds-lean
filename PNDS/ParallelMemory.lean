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
  simp [applyPair, thenPair, add_mul, mul_assoc]

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

end PNDS.ParallelMemory
