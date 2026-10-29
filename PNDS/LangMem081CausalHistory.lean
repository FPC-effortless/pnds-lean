import Mathlib

/-!
LANGMEM081 — PARTIAL causal source-prefix historical-context match contract.

The historical match diagnostic may compare only *previously observed*
source bytes and previously stored write addresses. The byte at t+1 is
an evaluation target and must not enter selection at time t.
Here a is an observed historical event clock satisfying a<t and
the query uses already observed prefix 0..t. With a fixed observed
prefix, attaching any candidate next byte cannot change any result
computed only from the prefix.

This does NOT formalize original trained TAC PyTorch learned keys, cosine
ranking, NLL calibration, actual original PSF source labels, full model
IEEE AD, or any direct historical-byte reconstruction success.
-/
namespace PNDS.LangMem081

def priorWrite (a t : Nat) : Prop := 4 <= a ∧ a < t

def registeredLength (n : Nat) : Bool := n == 2 || n == 3 || n == 4
def registeredHistory (n : Nat) : Bool := n == 32 || n == 128 || n == 256

def sourceObservedContext (observed : List Nat) (exclusiveEnd length : Nat) : List Nat :=
  (observed.take exclusiveEnd).drop (exclusiveEnd-length)

def exactPastContextMatch (observed : List Nat) (a t length : Nat) : Bool :=
  sourceObservedContext observed a length ==
  sourceObservedContext observed (t+1) length

theorem target_future_not_in_prefix (observed : List Nat) (future : Nat) :
    (observed ++ [future]).take observed.length = observed := by
  simp

theorem same_observed_prefix_target_blind
    (observed : List Nat) (futureOne futureTwo a t length : Nat) :
    exactPastContextMatch ((observed ++ [futureOne]).take observed.length)
       a t length =
    exactPastContextMatch ((observed ++ [futureTwo]).take observed.length)
       a t length := by
  simp [target_future_not_in_prefix]

theorem registered_context_two : registeredLength 2 = true := by decide
theorem registered_context_three : registeredLength 3 = true := by decide
theorem registered_context_four : registeredLength 4 = true := by decide
theorem registered_history_32 : registeredHistory 32 = true := by decide
theorem registered_history_128 : registeredHistory 128 = true := by decide
theorem registered_history_256 : registeredHistory 256 = true := by decide

theorem positive_history_lag (a t : Nat) (h : priorWrite a t) : 0 < t - a := by
  rcases h with ⟨_, ht⟩
  omega

end PNDS.LangMem081
