import Mathlib

/-!
LANGSCALE070 (TAC-OSM scaling, source-pinned companion, PARTIAL).

Original first-stage TAC + 48D upper recurrence are held fixed.
Only the language gated FFN hidden width is varied. Original 280D input:
up and gate weights each (280*W + W) and output (256*W + 256).
The hidden-unit dependent cost is 2*(280+1)+256=818 per width,
with original model-owned fixed base intercept 127433 parameters.

This exact Nat arithmetic theorem does not prove full source/gradients,
floating-point behavior, trainability, quality, energy or scalability.
-/
namespace PNDS.LangScale070

def totalParams (width : Nat) : Nat := 127433 + 818 * width

theorem p250k : totalParams 150 = 250133 := by decide
theorem p500k : totalParams 455 = 499623 := by decide
theorem p1000k : totalParams 1067 = 1000239 := by decide

theorem increasing (a b : Nat) (h : a < b) :
    totalParams a < totalParams b := by
  unfold totalParams
  omega

/-- The 48D upper recurrent state does not change with FFN width.
    This is a model-design declaration, not a proof of serialization. -/
def upperStateWidth (_ffnWidth : Nat) : Nat := 48
theorem width_invariant (a b : Nat) : upperStateWidth a = upperStateWidth b := by rfl

/-- Incremental trainable FFN parameter cost per extra hidden unit. -/
theorem extra_units (width increment : Nat) :
    totalParams (width + increment) = totalParams width + 818 * increment := by
  unfold totalParams
  omega

end PNDS.LangScale070
