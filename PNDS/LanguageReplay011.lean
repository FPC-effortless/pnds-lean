import Mathlib

/-!
GRAPH011: deterministic source replay and checkpoint space/time receipts.

Motivation: the supplied transcript discusses R. Ryan Williams' STOC 2025
theorem TIME[t] ⊆ SPACE[O(sqrt(t log t))] on multitape Turing machines.
This module DOES NOT formalize that theorem or implement Cook-Mertz
catalytic tree evaluation. It proves a much weaker, practical deterministic
checkpoint/recomputation identity on the *available original source*.

Intended source: TAC-OSM/GRAPH011/source_checkpoint_replay_v7.py.
Source archive bytes and checkpoint bytes are BOTH counted; no claim
that old original information can be inferred from a lossy recurrent state.
-/
namespace PNDS.LanguageReplay011
noncomputable section

/-- A deterministic source transition applied to the actual observed events. -/
def run {σ α : Type*} (step : σ → α → σ)
    (initial : σ) (events : List α) : σ :=
  events.foldl step initial

/-- The exact algebraic reason a saved intermediate state can replace a
prefix of computation when the original source suffix remains available. -/
theorem replay_composition {σ α : Type*}
    (step : σ → α → σ) (s : σ) (a b : List α) :
    run step s (a ++ b) = run step (run step s a) b := by
  simp [run, List.foldl_append]

/-- Splitting a genuine source at ANY checkpoint produces the same
exact-real/algebraic final state under deterministic iteration.
This does not assert bitwise PyTorch CPU/GPU equality. -/
theorem replay_at_any_checkpoint {σ α : Type*}
    (step : σ → α → σ) (s : σ) (events : List α) (cut : ℕ) :
    run step s events =
      run step (run step s (events.take cut)) (events.drop cut) := by
  rw [← replay_composition]
  simp

/-- A unit-sized source prefix is enough for a later deterministic replay,
provided the genuine remaining bytes are still accessible. -/
theorem replay_after_single_event {σ α : Type*}
    (step : σ → α → σ) (s : σ) (head : α) (tail : List α) :
    run step s (head :: tail) = run step (step s head) tail := by
  simp [run]

/-- Latest interval boundary strictly BEFORE target token t. -/
def checkpointBefore (t block : ℕ) : ℕ :=
  ((t - 1) / block) * block

theorem checkpoint_before_target
    (t block : ℕ) (ht : 0 < t) (hb : 0 < block) :
    checkpointBefore t block < t := by
  have hdiv := Nat.mod_add_div (t-1) block
  have hrem := Nat.mod_lt (t-1) hb
  unfold checkpointBefore
  omega

/-- The real computation replayed is at most one block (and nonzero). -/
theorem replay_steps_in_block
    (t block : ℕ) (ht : 0 < t) (hb : 0 < block) :
    1 ≤ t - checkpointBefore t block ∧
    t - checkpointBefore t block ≤ block := by
  have hdiv := Nat.mod_add_div (t-1) block
  have hrem := Nat.mod_lt (t-1) hb
  unfold checkpointBefore
  omega

/-- Retaining original input bytes is not optional: saving checkpoints
does not make their independent source storage magically disappear. -/
theorem source_bytes_remain_accounted
    (sourceBytes checkpointBytes : ℕ) :
    sourceBytes ≤ sourceBytes + checkpointBytes := Nat.le_add_right _ _

/-- A source/owner match is a precondition of a replay query. It is
a provenance equality check, NOT user authentication or access control. -/
def sameSource (storedOwner requestedOwner : String)
    (storedHash requestedHash : String) : Bool :=
  decide (storedOwner = requestedOwner ∧ storedHash = requestedHash)

theorem wrong_owner_not_replayable
    (owner requested digest : String) (h : owner ≠ requested) :
    sameSource owner requested digest digest = false := by
  simp [sameSource, h]

theorem wrong_hash_not_replayable
    (owner digest requested : String) (h : digest ≠ requested) :
    sameSource owner owner digest requested = false := by
  simp [sameSource, h]

/-- An overwriting recurrence is not invertible: checkpoint replay works
only because the original input and earlier checkpoint are retained. -/
theorem erasing_transition_loses_old_state {σ α : Type*}
    (replacement old1 old2 : σ) (x : α) :
    run (fun _ _ => replacement) old1 [x] =
    run (fun _ _ => replacement) old2 [x] := by
  simp [run]

end
end PNDS.LanguageReplay011
