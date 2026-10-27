import PNDS.LanguageLossAD
import PNDS.LanguageSelectiveRetrieval
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
TAC-LANG-MLAG-001: source-exact, *conditional* real-valued math for a
paired 64/96 historical objective, matching the archived original total
auxiliary weight 0.30 and original autoregressive token loss.

No claim of a validated gradient through the full PyTorch computational
graph: this states the analytic composition of scalar differentiable
losses, conditional on each individual source loss' derivative.
-/
namespace PNDS.LanguageMultiLagObjective
noncomputable section
open scoped Topology

def sourceTotal (main history64 : ℝ) : ℝ :=
  main + (3/10 : ℝ) * history64

def pairedTotal (main history64 history96 : ℝ) : ℝ :=
  main + (3/20 : ℝ) * history64 + (3/20 : ℝ) * history96

/-- The sum of the new auxiliary coefficients equals the original
coefficient. This is loss-weight parity, NOT a floating-FLOP guarantee. -/
theorem auxiliary_budget_preserved :
    (3/20 : ℝ) + (3/20 : ℝ) = (3/10 : ℝ) := by norm_num

theorem paired_objective_average (main h64 h96 : ℝ) :
    pairedTotal main h64 h96 =
      main + (3/10 : ℝ) * ((h64+h96)/2) := by
  unfold pairedTotal
  ring

/-- Under ordinary differentiability assumptions, the new objective has
the exact weighted JVP/backward scalar derivative. -/
theorem paired_objective_hasDerivAt
    (main h64 h96 : ℝ → ℝ) (dm d64 d96 t : ℝ)
    (hm : HasDerivAt main dm t)
    (h64 : HasDerivAt h64 d64 t)
    (h96 : HasDerivAt h96 d96 t) :
    HasDerivAt (fun z => pairedTotal (main z) (h64 z) (h96 z))
      (dm + (3/20 : ℝ) * d64 + (3/20 : ℝ) * d96) t := by
  convert ((hm.add (h64.const_mul (3/20 : ℝ))).add
    (h96.const_mul (3/20 : ℝ))) using 1
  · rfl
  · ring

/-- The new objective is a nonnegative weighted sum if all source losses
are nonnegative. This is NOT a guarantee it will optimize well. -/
theorem paired_objective_nonnegative (main h64 h96 : ℝ)
    (hm : 0 ≤ main) (h64 : 0 ≤ h64) (h96 : 0 ≤ h96) :
    0 ≤ pairedTotal main h64 h96 := by
  unfold pairedTotal
  positivity

/-- Both query lags are positive, so queried source ordinal t-lag is
strictly earlier than current t whenever the query is valid. No
future-token label is required for an authorized historical read. -/
theorem paired_historical_queries_are_past (t : ℕ)
    (h96 : 96 ≤ t) :
    t-64 < t ∧ t-96 < t := by
  constructor <;> omega

end
end PNDS.LanguageMultiLagObjective
