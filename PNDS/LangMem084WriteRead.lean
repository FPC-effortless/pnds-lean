import Mathlib

/-! Partial exact-real source/overwrite/read contracts only.
SHA256, Float32, source-trained key/value encoders, full TAC graph and
learned factual recall are explicitly OPEN; no auxiliary fitting here.
-/
namespace PNDS.LangMem084
abbrev Key := Fin 8 → ℝ
abbrev Memory := Fin 8 → Fin 8 → ℝ

def read (m : Memory) (q : Key) : Key := fun i => ∑ j, m i j * q j
def dot (k q : Key) : ℝ := ∑ j, k j * q j

def overwrite (m : Memory) (k v : Key) (rho : ℝ) : Memory :=
  fun i j => rho * m i j + (v i - rho * read m k i) * k j

theorem read_overwrite (m : Memory) (k v q : Key) (rho : ℝ) :
    read (overwrite m k v rho) q =
      fun i => rho * read m q i + (v i - rho * read m k i) * dot k q := by
  funext i
  simp only [read, overwrite, dot, add_mul, mul_assoc,
    Finset.sum_add_distrib, ← Finset.mul_sum]

theorem unit_key_current_value (m : Memory) (k v : Key) (rho : ℝ)
    (h : dot k k = 1) : read (overwrite m k v rho) k = v := by
  rw [read_overwrite]
  funext i
  simp only [h, mul_one]
  ring

theorem orthogonal_read_preserved (m : Memory) (k v q : Key) (rho : ℝ)
    (h : dot k q = 0) :
    read (overwrite m k v rho) q = fun i => rho * read m q i := by
  rw [read_overwrite]
  simp [h]

def encodeBits (b : Fin 8 → Bool) : Key := fun i => if b i then 1 else -1

theorem signed_identity_injective : Function.Injective encodeBits := by
  intro a b h
  funext i
  have hi := congrFun h i
  cases ha : a i <;> cases hb : b i <;> simp_all [encodeBits]

def sourceUpdate (m : Memory) (k v : Key) (rho beta epsilon : ℝ) : Memory :=
  fun i j => rho * m i j - rho * beta * read m k i * k j + epsilon * v i * k j

def priorEvent (a t : Nat) : Prop := 4 ≤ a ∧ a < t

theorem prior_event_no_future (a t : Nat) (h : priorEvent a t) : a < t := h.2

theorem bank_FP32_payload : 8 * 8 * 4 = (256 : Nat) := by norm_num

end PNDS.LangMem084
#print axioms PNDS.LangMem084.read_overwrite
#print axioms PNDS.LangMem084.unit_key_current_value
#print axioms PNDS.LangMem084.orthogonal_read_preserved
#print axioms PNDS.LangMem084.signed_identity_injective
