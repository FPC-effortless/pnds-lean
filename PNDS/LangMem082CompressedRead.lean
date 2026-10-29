import Mathlib

/-! LANGMEM082 PARTIAL exact-real read and ring contracts.
No theorem establishes trained byte recall, floating-point equivalence or
end-to-end TAC behavior. Old addresses and tagged events are evaluator data.
-/
namespace PNDS.LangMem082
abbrev Key := Fin 8 → ℝ
abbrev Memory := Fin 8 → Fin 8 → ℝ

def read (m : Memory) (q : Key) : Key := fun i => ∑ j, m i j * q j

def add (a b : Memory) : Memory := fun i j => a i j + b i j

theorem read_add (a b : Memory) (q : Key) :
    read (add a b) q = fun i => read a q i + read b q i := by
  funext i
  simp [read, add, add_mul, Finset.sum_add_distrib]

def rankOne (v k : Key) : Memory := fun i j => v i * k j

theorem isolated_read_direction (v k q : Key) :
    read (rankOne v k) q = fun i => v i * (∑ j, k j * q j) := by
  funext i
  simp only [read, rankOne, mul_assoc, Finset.mul_sum]

def squaredNorm (m : Memory) : ℝ := ∑ i, ∑ j, (m i j)^2

def permuteRows (e : Equiv.Perm (Fin 8)) (m : Memory) : Memory :=
  fun i j => m (e i) j

theorem permuteRows_squaredNorm (e : Equiv.Perm (Fin 8)) (m : Memory) :
    squaredNorm (permuteRows e m) = squaredNorm m := by
  unfold squaredNorm permuteRows
  exact Equiv.sum_comp e (fun i => ∑ j, (m i j)^2)

def ringAvailable (next count anchor : Nat) : Prop :=
  next - count ≤ anchor ∧ anchor < next

theorem ring_no_future (next count anchor : Nat)
    (h : ringAvailable next count anchor) : anchor < next := h.2

theorem ring256_capacity : 256 - 16 = (240 : Nat) := by norm_num
theorem ring4160_capacity : 4160 - 16 = (4144 : Nat) := by norm_num

end PNDS.LangMem082

#print axioms PNDS.LangMem082.read_add
#print axioms PNDS.LangMem082.isolated_read_direction
#print axioms PNDS.LangMem082.permuteRows_squaredNorm
#print axioms PNDS.LangMem082.ring_no_future
