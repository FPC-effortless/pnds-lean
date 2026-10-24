import Mathlib
import PNDS.ParallelMemory

/-!
Finite-dimensional exact-real source math for one TAC constant-state stream
and for chronological group/scan equivalence. These theorems describe the
external-gate *candidate* and are not the original native recurrent fast/slow
dynamics, not a proof of machine floating-point / autograd, and not predictive
skill. Memory fixed dimension is n×n for arbitrary finite n (source uses n=8).

Parallel-time scan requires every gate schedule to be independent of the
old state used by the scan. Runtime stream processes the resulting transforms
one event at a time and retains ONLY one matrix; parallel batch groups those
same chronological transformations by the associative thenPair law.
-/

namespace PNDS.DualModeMemory

open PNDS.ParallelMemory

/-- Abstract constant-memory stream step. No history cache is an argument. -/
def step {n : ℕ} (state : Matrix (Fin n) (Fin n) ℝ)
    (event : AffinePair n) : Matrix (Fin n) (Fin n) ℝ :=
  applyPair event state

/-- Streaming fold with an arbitrary initial matrix and fixed event schedule. -/
def stream {n : ℕ} (state : Matrix (Fin n) (Fin n) ℝ)
    (events : List (AffinePair n)) : Matrix (Fin n) (Fin n) ℝ :=
  events.foldl step state

/-- Generalized associativity bridge from chronological scan summary to the
    actual event-by-event recurrent stream for a finite event list. -/
theorem scan_matches_stream_from {n : ℕ}
    (events : List (AffinePair n))
    (acc : AffinePair n) (initial : Matrix (Fin n) (Fin n) ℝ) :
    applyPair (events.foldl thenPair acc) initial =
      events.foldl step (applyPair acc initial) := by
  induction events generalizing acc initial with
  | nil => rfl
  | cons event rest ih =>
    simpa only [List.foldl_cons, step, apply_then] using
      ih (thenPair acc event) initial

theorem scan_matches_stream {n : ℕ}
    (events : List (AffinePair n))
    (initial : Matrix (Fin n) (Fin n) ℝ) :
    applyPair (events.foldl thenPair (identity n)) initial =
      stream initial events := by
  simpa only [stream, apply_identity] using
    (scan_matches_stream_from events (identity n) initial)

/-- A resumed step stream has the SAME exact-real result as processing a
    concatenated observed sequence; no previous event history need be kept. -/
theorem stream_resume {n : ℕ}
    (initial : Matrix (Fin n) (Fin n) ℝ)
    (first second : List (AffinePair n)) :
    stream initial (first ++ second) =
      stream (stream initial first) second := by
  simp [stream, List.foldl_append]

/-- Independent read-only events do not change any memory coordinate. -/
theorem read_only_stream {n : ℕ}
    (initial : Matrix (Fin n) (Fin n) ℝ) :
    step initial (identity n) = initial := by
  simp [step, apply_identity]

/-- Source-shaped input-only running sufficient statistic: old running
    sum plus current basis. This is exact-real arithmetic. -/
def advancePrefix (sum basis : ℝ) : ℝ := sum + basis

theorem append_basis_sum (past : List ℝ) (basis : ℝ) :
    (past ++ [basis]).sum = advancePrefix past.sum basis := by
  simp [advancePrefix]

/-- Context uses ONLY a list of prior observations and the fixed pair
    (running sum, count). Neither a future event nor an unbounded cache
    appears in this mathematical interface. -/
noncomputable def context (past : List ℝ) : ℝ :=
  past.sum / max (past.length : ℝ) 1

theorem context_decomposes_through_statistic (past : List ℝ) :
    context past = past.sum / max (past.length : ℝ) 1 := by
  rfl

end PNDS.DualModeMemory
