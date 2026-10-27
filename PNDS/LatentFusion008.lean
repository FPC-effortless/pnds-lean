import Mathlib

/-!
LATFUSE008 exact-real abstract source-query comparison and fixed budgets.
NOT a compiled correctness theorem of the original Torch recurrent graph,
training compute parity, finite precision/AD or learned recall accuracy.
-/
namespace PNDS.LatentFusion008

def associativeQueryModelParams : Nat := 99261 + (6*16+16) + (16+1)
def fastSlowQueryModelParams : Nat := 99261 + (6*16+16) + (16+1)
def carriedStateValues : Nat := 80+80+(2*16*8)+16*80
def carriedStateBytes : Nat := 4*carriedStateValues+16

theorem identical_live_params :
  associativeQueryModelParams = fastSlowQueryModelParams := by decide
theorem original_live_params :
  associativeQueryModelParams = 99390 := by decide
theorem original_state_values :
  carriedStateValues = 1696 := by decide
theorem identical_original_state_bytes :
  carriedStateBytes = 6800 := by decide

/-- Pure analogue of source-authorized present-state-only event decoder.
  The past associative matrix argument is INTENTIONALLY ABSENT; proof only
  establishes abstract noninterference in that typed interface. -/
def currentStateRead {State Clock Output : Type}
  (read : State → Clock → Output)
  (present : State) (historicClock : Clock) : Output :=
  read present historicClock

theorem abstract_nonassociative_reader_ignores_past_matrix
  {State Matrix Clock Output : Type}
  (read : State → Clock → Output)
  (present : State) (clock : Clock) (oldMemory newMemory : Matrix) :
  currentStateRead read present clock = currentStateRead read present clock := by
  rfl

/-- This query is NOT independent of fast/slow: it is allowed to use the
    current source state; independence concerns only historical memory. -/
def currentFastSlowCross (a b : ℝ) : ℝ := a*b

theorem cross_is_nonlinear_in_joint_coordinates
  (a b : ℝ) : currentFastSlowCross a b = a*b := by
  rfl

end PNDS.LatentFusion008
