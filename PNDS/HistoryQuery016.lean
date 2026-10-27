import Mathlib

/-! HISTORY-QUERY016 revised loss only. Original inference graph is opaque,
source-pinned in the reciprocal manifest. No whole-model/IEEE/AD theorem.
TAC source 3d7946a3e46233fc0c09e86f60ad2c24c27a1674.
-/
namespace PNDS.HistoryQuery016

abbrev HistoryValues (b : Nat) := Fin b → Fin 88 → Fin 13 → ℝ
abbrev HistoryMask (b : Nat) := Fin b → Fin 88 → Fin 13 → Bool
abbrev PastPrediction (b : Nat) := Fin b → Fin 8 → ℝ

/-- beta=1 SmoothL1, exact-real definition only. -/
def smoothL1 (e : ℝ) : ℝ :=
  if |e| < 1 then e^2 / 2 else |e| - 1/2

/-- Missing targets contribute exactly zero; denominator includes actual observed slots. -/
def historicalLoss {b : Nat} (pred target : PastPrediction b)
    (observed : Fin b → Fin 8 → Bool) : ℝ :=
  let count : ℝ := ∑ i, ∑ j, if observed i j then (1 : ℝ) else 0
  (∑ i, ∑ j, if observed i j then smoothL1 (pred i j - target i j) else 0) /
    max 1 count

/-- Same source forecast objective, requiring positive sd at the model boundary. -/
def forecastLoss {b : Nat} (mu sd target : Fin b → Fin 13 → ℝ)
    (observed : Fin b → Fin 13 → Bool) : ℝ :=
  let count : ℝ := ∑ i, ∑ j, if observed i j then (1 : ℝ) else 0
  (∑ i, ∑ j, if observed i j then
    (4/5 : ℝ) * ((Real.log (2 * Real.pi) + 2 * Real.log (sd i j) +
      ((target i j - mu i j) / sd i j)^2) / 2) +
    (1/5 : ℝ) * |target i j - mu i j| else 0) / max 1 count

/-- No theorem asserted: encoder/write/update/read/forecast are the original
88-step TAC graph. Source correspondence, masks, precision and AD remain OPEN. -/
structure OriginalGraphBoundary (b : Nat) where
  values : HistoryValues b
  observed : HistoryMask b
  age : HistoryValues b
  last : HistoryValues b
  owner : Fin b → Fin 12
  pastClocks : Fin b → Fin 8 → Fin 6 → ℝ
  futureClock : Fin b → Fin 6 → ℝ

end PNDS.HistoryQuery016
