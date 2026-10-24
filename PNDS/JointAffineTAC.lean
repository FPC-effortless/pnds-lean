import Mathlib
import PNDS.ParallelMemory
import PNDS.TimescaleEvidence

/-!
PNDS v1.6 joint *linearizable subgraph* proof. The candidate contains two
independently scheduled associative states:
  - 88D stable multi-timescale evidence,
  - 8×8 TAC addressed associative memory.
This is a PRODUCT of the exact-real affine semigroups, not a replacement of
TAC memory by a conventional SSM. It proves the joint bounded-state operator
may be grouped chronologically in a parallel scan while streaming one event.

Crucial assumption: each input gate/key/value is computed from earlier
source observations, without dependencies on *this same memory scan's*
current hidden read. The learned read-refinement and output decoder remain
separate nonlinear maps after the scan, not formally refined here.
-/

namespace PNDS.JointAffineTAC

open PNDS

structure JointPair (n : ℕ) where
  evidence : Fin 88 → TimescaleEvidence.ScalarPair
  memory : ParallelMemory.AffinePair n

def applyPair {n : ℕ} (a : JointPair n)
    (s : (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ) :
    (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ :=
  ((fun i => TimescaleEvidence.applyPair (a.evidence i) (s.1 i)),
   ParallelMemory.applyPair a.memory s.2)

/-- Chronologically compose first a and THEN b without reordering time. -/
def thenPair {n : ℕ} (a b : JointPair n) : JointPair n :=
  ⟨(fun i => TimescaleEvidence.thenPair (a.evidence i) (b.evidence i)),
    ParallelMemory.thenPair a.memory b.memory⟩

theorem apply_then {n : ℕ} (a b : JointPair n)
    (s : (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ) :
    applyPair (thenPair a b) s = applyPair b (applyPair a s) := by
  apply Prod.ext
  · funext i
    exact TimescaleEvidence.apply_then (a.evidence i) (b.evidence i) (s.1 i)
  · exact ParallelMemory.apply_then a.memory b.memory s.2

theorem then_assoc {n : ℕ} (a b c : JointPair n) :
    thenPair (thenPair a b) c = thenPair a (thenPair b c) := by
  cases a with
  | mk ae am =>
    cases b with
    | mk be bm =>
      cases c with
      | mk ce cm =>
        have he :
            (fun i => TimescaleEvidence.thenPair
                (TimescaleEvidence.thenPair (ae i) (be i)) (ce i)) =
            (fun i => TimescaleEvidence.thenPair
                (ae i) (TimescaleEvidence.thenPair (be i) (ce i))) := by
          funext i
          exact TimescaleEvidence.then_assoc (ae i) (be i) (ce i)
        have hm := ParallelMemory.then_assoc am bm cm
        simp only [thenPair, he, hm]

def stream {n : ℕ}
    (s : (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ)
    (events : List (JointPair n)) :=
  events.foldl (fun state event => applyPair event state) s

/-- Exact-real equivalence of parallel grouped whole-*state* affine schedule
    and single-event fixed-size streaming, for any finite sequence. -/
theorem fold_matches_stream {n : ℕ}
    (events : List (JointPair n)) (acc : JointPair n)
    (initial : (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ) :
    applyPair (events.foldl thenPair acc) initial =
      events.foldl (fun state event => applyPair event state)
         (applyPair acc initial) := by
  induction events generalizing acc with
  | nil => rfl
  | cons event rest ih =>
    simpa only [List.foldl_cons, apply_then] using
      ih (thenPair acc event)

theorem stream_resume {n : ℕ}
    (initial : (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ)
    (first second : List (JointPair n)) :
    stream initial (first ++ second) =
      stream (stream initial first) second := by
  simp [stream, List.foldl_append]


/-- Full *abstract output* equivalence: ANY pure deterministic nonlinear
    downstream function of the identical scheduled state (tanh, sigmoid,
    MLP, codebook routing) has exactly the same result in real arithmetic.

    This theorem is conditional on a fixed input-derived gate schedule and
    correct tensor realization of the paired algebra; it is NOT a neural
    tensor source-code or floating-point equivalence proof. -/
theorem same_decoder_after_joint_parallel_or_stream
    {n : ℕ} {Output : Type}
    (events : List (JointPair n)) (acc : JointPair n)
    (initial : (Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ)
    (decoder : ((Fin 88 → ℝ) × Matrix (Fin n) (Fin n) ℝ) → Output) :
    decoder (applyPair (events.foldl thenPair acc) initial) =
      decoder (events.foldl (fun state event => applyPair event state)
                       (applyPair acc initial)) := by
  exact congrArg decoder (fold_matches_stream events acc initial)


end PNDS.JointAffineTAC
