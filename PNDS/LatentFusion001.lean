import Mathlib

/-!
LATFUSE001 exact-real *local operator* contracts only. The recurrent
neural graph, real data, trained performance, IEEE and AD refinement remain OPEN.
-/
namespace PNDS.LatentFusion001

/-- Chronological diagonal affine recurrence on one coordinate. -/
def affineStep (a b z : ℝ) : ℝ := a * z + b

def compose (first second : ℝ × ℝ) : ℝ × ℝ :=
  (second.1 * first.1, second.1 * first.2 + second.2)

theorem compose_run (first second : ℝ × ℝ) (z : ℝ) :
    affineStep (compose first second).1 (compose first second).2 z =
      affineStep second.1 second.2 (affineStep first.1 first.2 z) := by
  dsimp [affineStep, compose]
  ring

theorem compose_assoc (a b c : ℝ × ℝ) :
    compose (compose a b) c = compose a (compose b c) := by
  cases a with
  | mk aa ab =>
    cases b with
    | mk ba bb =>
      cases c with
      | mk ca cb =>
        dsimp [compose]
        congr 1 <;> ring

/-- Scalar normalized-key delta update, assuming a unit key k=1. -/
def delta (m v beta : ℝ) : ℝ := m + beta * (v - m)

theorem delta_no_write (m v : ℝ) : delta m v 0 = m := by
  simp [delta]

theorem delta_full_write (m v : ℝ) : delta m v 1 = v := by
  simp [delta]

/-- A one-slot write changes only the selected source-authorized index. -/
def slotWrite (slots : Nat → ℝ) (selected : Nat) (beta v : ℝ) : Nat → ℝ :=
  fun i => if i = selected then slots i + beta * (v - slots i) else slots i

theorem unaddressed_is_unchanged (slots : Nat → ℝ) (selected other : Nat)
    (beta v : ℝ) (different : other ≠ selected) :
    slotWrite slots selected beta v other = slots other := by
  simp [slotWrite, different]

theorem selected_zero_write (slots : Nat → ℝ) (selected : Nat) (v : ℝ) :
    slotWrite slots selected 0 v selected = slots selected := by
  simp [slotWrite]

end PNDS.LatentFusion001
