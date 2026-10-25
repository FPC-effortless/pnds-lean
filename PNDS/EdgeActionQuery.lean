import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic

/-!
EDGE-ACTION-001: typed command-query PURITY and source-clock preconditions.

This module intentionally proves only that the specified read-only query
returns the same observed evidence and that a caller-side authority condition
rejects reused/foreign timestamps. It does NOT verify the entire neural
network, the numerical scan, motor semantics, safe control or interventions.
-/
namespace PNDS.EdgeActionQuery

structure ObservedEvidence where
  basis : Fin 88 → ℝ
  memory : Fin 8 → Fin 8 → ℝ
  count : ℕ
  owner : ℕ
  lastHour : ℕ

abbrev Command := Fin 14 → ℝ
abbrev Forecast := Fin 14 → ℝ

/-- All observational and proposal outputs are distinct from the factual
    carried source evidence. A current proposal is read-only. -/
def propose (source : ObservedEvidence) (command : Command)
    (head : ObservedEvidence → Command → Forecast) : Forecast × ObservedEvidence :=
  (head source command, source)

theorem propose_preserves_evidence (source : ObservedEvidence) (command : Command)
    (head : ObservedEvidence → Command → Forecast) :
    (propose source command head).2 = source := rfl

/-- Proposed motor commands cannot change the actual evidence under this
    interface. This is NOT a statement about a learned controller. -/
theorem candidate_independent_evidence (source : ObservedEvidence) (a b : Command)
    (head : ObservedEvidence → Command → Forecast) :
    (propose source a head).2 = (propose source b head).2 := rfl

def admissibleNextObservation (source : ObservedEvidence) (owner hour : ℕ) : Prop :=
  source.owner = owner ∧ source.lastHour < hour

theorem no_repeated_or_older_observation (source : ObservedEvidence) (hour : ℕ)
    (h : hour ≤ source.lastHour) :
    ¬ admissibleNextObservation source source.owner hour := by
  intro hadm
  exact (not_lt_of_ge h) hadm.2

theorem no_foreign_observation (source : ObservedEvidence) (other hour : ℕ)
    (h : other ≠ source.owner) :
    ¬ admissibleNextObservation source other hour := by
  intro hadm
  exact h hadm.1.symm

/-- Observed evidence prefix composition remains a separate, inherited PNDS
    operator; nothing in this query writes to it. -/
theorem consecutive_queries_preserve_evidence (source : ObservedEvidence)
    (a b : Command) (head : ObservedEvidence → Command → Forecast) :
    (propose (propose source a head).2 b head).2 = source := rfl

end PNDS.EdgeActionQuery
