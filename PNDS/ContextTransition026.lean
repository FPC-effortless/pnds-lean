import Mathlib

/-! CONTEXT026: changed key/write/native-update graph only. Parent full graph,
IEEE/AD equivalence, optimizer, memory usefulness and safety are OPEN. -/
namespace PNDS.ContextTransition026

abbrev ByteWindow (n : Nat) := Fin n → Fin 256
abbrev Key := Fin 8 → ℝ
abbrev NativeMemory := Fin 8 → Fin 8 → ℝ
abbrev LagWeights := Fin 3 → Fin 48 → ℝ

abbrev Embedding := Fin 256 → Fin 48 → ℝ
abbrev KeyWeight := Fin 8 → Fin 48 → ℝ

noncomputable def lexical (e : Embedding) (tokens : Nat → Fin 256) (t : Nat) (c : Fin 48) : ℝ :=
  Real.tanh (e (tokens t) c)

noncomputable def contextFeature (e : Embedding) (w : LagWeights)
    (tokens : Nat → Fin 256) (t : Nat) (c : Fin 48) : ℝ :=
  Real.tanh (lexical e tokens t c + ∑ lag : Fin 3,
    if lag.val + 1 ≤ t then Real.tanh (w lag c) * lexical e tokens (t-(lag.val+1)) c else 0)

noncomputable def contextualKey (e : Embedding) (lags : LagWeights) (w : KeyWeight)
    (bias : Key) (tokens : Nat → Fin 256) (t : Nat) : Key :=
  let raw : Key := fun i => (∑ c, w i c * contextFeature e lags tokens t c) + bias i
  let denom := max (1/1000000 : ℝ) (Real.sqrt (∑ i, (raw i)^2))
  fun i => raw i / denom

noncomputable def projection (alpha beta : ℝ) (k : Key) (i j : Fin 8) : ℝ :=
  alpha * ((if i = j then 1 else 0) - beta * k i * k j)

noncomputable def nativeUpdate (m : NativeMemory) (k v : Key) (alpha beta : ℝ)
    (admitted : Bool) : NativeMemory := fun i j =>
  if admitted then (∑ r, m i r * projection alpha beta k r j) + beta * v i * k j else m i j

noncomputable def postwriteRead (m : NativeMemory) (query : Key) : Key :=
  fun i => ∑ j, m i j * query j

/-- The query sees current lexical context; write sees predecessor context. -/
def writePosition (t : Nat) : Option Nat := if 0 < t then some (t-1) else none

theorem first_write_absent : writePosition 0 = none := by simp [writePosition]

theorem write_strictly_past (t : Nat) (ht : 0 < t) : t-1 < t := by omega

theorem no_write_identity (m : NativeMemory) (k v : Key) (a b : ℝ) :
    nativeUpdate m k v a b false = m := by funext i j; simp [nativeUpdate]

/-- Conditional scalar envelope; deriving IEEE tensor norms remains OPEN. -/
theorem envelope_invariant (r cap force bound : ℝ) (hr : r ≤ bound)
    (hc : 0 ≤ cap) (hb : cap * bound + force ≤ bound) :
    cap * r + force ≤ bound := by
  have h := mul_le_mul_of_nonneg_left hr hc
  linarith

theorem native_envelope (r : ℝ) (hr : r ≤ 200 * Real.sqrt 8) :
    (199/200 : ℝ) * r + Real.sqrt 8 ≤ 200 * Real.sqrt 8 := by
  have h := mul_le_mul_of_nonneg_left hr (show (0 : ℝ) ≤ 199/200 by norm_num)
  linarith

/-- Legal interior erase gates .9/.1 and decay .99 need not contract Euclidean norm.
The normalized key (1,1)/sqrt2 induces this rational oblique matrix. -/
theorem oblique_erase_can_amplify :
    (99/100 : ℝ)^2 * (((11/20 : ℝ)*1 - (1/20)*(-1))^2 +
      (-(9/20 : ℝ)*1 + (19/20)*(-1))^2) > (1 : ℝ)^2 + (-1)^2 := by norm_num

end PNDS.ContextTransition026
#print axioms PNDS.ContextTransition026.first_write_absent
#print axioms PNDS.ContextTransition026.write_strictly_past
#print axioms PNDS.ContextTransition026.no_write_identity
#print axioms PNDS.ContextTransition026.envelope_invariant
#print axioms PNDS.ContextTransition026.native_envelope
#print axioms PNDS.ContextTransition026.oblique_erase_can_amplify
