import Mathlib

/-! LANGMATH098: actual finite tensor affine action, normalized bank invariants,
bounded readout, and linear-work halving scan. Exact reals only. Neural encoder
IEEE/AD correspondence is a separate source/reference obligation, not assumed
proved by these theorems. No scalar norm-bound premise replaces the bank proof.
-/
namespace PNDS.LANGMATH098
noncomputable section
open scoped BigOperators

abbrev Tensor (ι : Type*) := ι → ℝ
abbrev Affine (ι : Type*) := Tensor ι × Tensor ι
def compose {ι} (first second : Affine ι) : Affine ι :=
  (fun i => second.1 i * first.1 i,
   fun i => second.1 i * first.2 i + second.2 i)
def identity {ι} : Affine ι := (fun _ => 1, fun _ => 0)
def applyAffine {ι} (s : Tensor ι) (a : Affine ι) : Tensor ι :=
  fun i => a.1 i * s i + a.2 i

theorem tensor_compose_assoc {ι} (a b c : Affine ι) :
    compose (compose a b) c = compose a (compose b c) := by
  apply Prod.ext <;> funext i <;> dsimp [compose] <;> ring
theorem tensor_identity_left {ι} (a : Affine ι) : compose identity a = a := by
  apply Prod.ext <;> funext i <;> simp [compose, identity]
theorem tensor_identity_right {ι} (a : Affine ι) : compose a identity = a := by
  apply Prod.ext <;> funext i <;> simp [compose, identity]
theorem tensor_action {ι} (s : Tensor ι) (a b : Affine ι) :
    applyAffine s (compose a b) = applyAffine (applyAffine s a) b := by
  funext i; dsimp [compose, applyAffine]; ring
theorem tensor_nonwrite {ι} (s : Tensor ι) : applyAffine s identity = s := by
  funext i; simp [applyAffine, identity]

def unroll {S I} (step : S → I → S) (s : S) (xs : List I) : S := xs.foldl step s
theorem chunk_equivalence {S I} (step : S → I → S) (s : S) (a b : List I) :
    unroll step s (a ++ b) = unroll step (unroll step s a) b := by
  simp [unroll, List.foldl_append]

def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))
def squash (x : ℝ) : ℝ := 2 * sigmoid (2*x) - 1
theorem sigmoid_range (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  have h := Real.exp_pos (-x)
  constructor
  · unfold sigmoid; positivity
  · unfold sigmoid; apply (div_lt_one (by positivity)).2; linarith
theorem squash_bound (x : ℝ) : |squash x| ≤ 1 := by
  have h := sigmoid_range (2*x)
  apply abs_le.mpr; dsimp [squash]; constructor <;> linarith

def softmax {n : ℕ} (x : Fin (n+1) → ℝ) (j : Fin (n+1)) : ℝ :=
  Real.exp (x j) / ∑ i, Real.exp (x i)
theorem softmax_nonnegative {n} (x : Fin (n+1) → ℝ) (j) : 0 ≤ softmax x j := by
  unfold softmax; positivity
theorem softmax_mass {n} (x : Fin (n+1) → ℝ) : ∑ j, softmax x j = 1 := by
  have hp : 0 < ∑ i : Fin (n+1), Real.exp (x i) := by
    apply Finset.sum_pos
    · intro i _; exact Real.exp_pos (x i)
    · exact Finset.univ_nonempty
  simp only [softmax, ← Finset.sum_div]
  exact div_self (ne_of_gt hp)

structure Bank (keys values : ℕ) where
  mass : Fin keys → ℝ
  memory : Fin values → Fin keys → ℝ
def Bank.Valid {k v} (s : Bank k v) : Prop :=
  (∀ j, 0 ≤ s.mass j) ∧ (∀ i j, |s.memory i j| ≤ s.mass j)
def Bank.zero (k v : ℕ) : Bank k v := ⟨fun _ => 0, fun _ _ => 0⟩
def Bank.update {k v} (s : Bank k v) (a : ℝ)
    (key : Fin k → ℝ) (value : Fin v → ℝ) : Bank k v :=
  ⟨fun j => a*s.mass j+(1-a)*key j,
   fun i j => a*s.memory i j+(1-a)*value i*key j⟩
theorem zero_valid (k v : ℕ) : (Bank.zero k v).Valid := by
  constructor <;> intros <;> simp [Bank.zero]

/-- Coordinate proof for the actual tensor update, not an assumed norm inequality. -/
theorem bank_update_valid {k v} (s : Bank k v) (a : ℝ)
    (key : Fin k → ℝ) (value : Fin v → ℝ)
    (hs : s.Valid) (ha : 0 ≤ a ∧ a ≤ 1)
    (hk : ∀ j, 0 ≤ key j) (hv : ∀ i, |value i| ≤ 1) :
    (s.update a key value).Valid := by
  have hw : 0 ≤ 1-a := by linarith [ha.2]
  constructor
  · intro j; dsimp [Bank.update]
    exact add_nonneg (mul_nonneg ha.1 (hs.1 j)) (mul_nonneg hw (hk j))
  · intro i j
    have hm := abs_le.mp (hs.2 i j)
    have hval := abs_le.mp (hv i)
    have p1 := mul_le_mul_of_nonneg_left hm.1 ha.1
    have p2 := mul_le_mul_of_nonneg_left hm.2 ha.1
    have q1 := mul_le_mul_of_nonneg_right hval.1 (mul_nonneg hw (hk j))
    have q2 := mul_le_mul_of_nonneg_right hval.2 (mul_nonneg hw (hk j))
    apply abs_le.mpr; dsimp [Bank.update]; constructor <;> nlinarith

theorem bank_mass_update {k v} (s : Bank k v) (a : ℝ)
    (key : Fin k → ℝ) (value : Fin v → ℝ)
    (hs : ∑ j, s.mass j ≤ 1) (hk : ∑ j, key j ≤ 1)
    (ha : 0 ≤ a ∧ a ≤ 1) : ∑ j, (s.update a key value).mass j ≤ 1 := by
  have h1 := mul_le_mul_of_nonneg_left hs ha.1
  have h2 := mul_le_mul_of_nonneg_left hk (show 0 ≤ 1-a by linarith [ha.2])
  simp only [Bank.update, Finset.sum_add_distrib, ← Finset.mul_sum]
  nlinarith

def Bank.read {k v} (s : Bank k v) (q : Fin k → ℝ) (epsilon : ℝ) (i : Fin v) : ℝ :=
  (∑ j, s.memory i j*q j) / (epsilon + ∑ j, s.mass j*q j)
theorem bank_read_bound {k v} (s : Bank k v) (q : Fin k → ℝ)
    (epsilon : ℝ) (i : Fin v) (hs : s.Valid) (hq : ∀ j, 0 ≤ q j)
    (he : 0 < epsilon) : |s.read q epsilon i| ≤ 1 := by
  have hd : 0 ≤ ∑ j, s.mass j*q j :=
    Finset.sum_nonneg (fun j _ => mul_nonneg (hs.1 j) (hq j))
  have hp : 0 < epsilon + ∑ j, s.mass j*q j := by linarith
  have hu : (∑ j, s.memory i j*q j) ≤ ∑ j, s.mass j*q j := by
    apply Finset.sum_le_sum; intro j _
    exact mul_le_mul_of_nonneg_right (abs_le.mp (hs.2 i j)).2 (hq j)
  have hl : (∑ j, -(s.mass j*q j)) ≤ ∑ j, s.memory i j*q j := by
    apply Finset.sum_le_sum; intro j _
    have h := mul_le_mul_of_nonneg_right (abs_le.mp (hs.2 i j)).1 (hq j)
    nlinarith
  simp only [Finset.sum_neg_distrib] at hl
  apply abs_le.mpr; unfold Bank.read
  constructor
  · apply (le_div_iff₀ hp).2; nlinarith
  · apply (div_le_iff₀ hp).2; nlinarith

theorem convex_coordinate (h offer a : ℝ) (hh : |h| ≤ 1)
    (ho : |offer| ≤ 1) (ha : 0 ≤ a ∧ a ≤ 1) :
    |a*h+(1-a)*offer| ≤ 1 := by
  have h := abs_le.mp hh; have o := abs_le.mp ho
  have h1 := mul_le_mul_of_nonneg_left h.1 ha.1
  have h2 := mul_le_mul_of_nonneg_left h.2 ha.1
  have o1 := mul_le_mul_of_nonneg_left o.1 (show 0 ≤ 1-a by linarith [ha.2])
  have o2 := mul_le_mul_of_nonneg_left o.2 (show 0 ≤ 1-a by linarith [ha.2])
  apply abs_le.mpr; constructor <;> nlinarith

structure NativeState where
  context : Fin 48 → ℝ
  lexical : Fin 3 → Fin 32 → ℝ
  previousKey : Fin 2 → Fin 8 → ℝ
  banks : Fin 2 → Bank 8 16

/-- Exposed reads are retained separately; comparison adds, rather than erases,
the pair. This elementary injectivity is a deliberate interface guarantee. -/
def exposed {ι} (fast slow mixed : Tensor ι) := (fast, slow, mixed)
theorem exposed_reads_injective {ι} (f s m f' s' m' : Tensor ι)
    (h : exposed f s m = exposed f' s' m') : f = f' ∧ s = s' := by
  simpa [exposed] using (show f = f' ∧ s = s' from ⟨congrArg Prod.fst h, congrArg (fun x => x.2.1) h⟩)

/-- Count of state-width multiply/add units in pair reduction/reconstruction. -/
def scanWork (n : ℕ) : ℕ := if n ≤ 1 then n else n + scanWork (n/2)
termination_by n
decreasing_by omega
theorem scan_work_linear (n : ℕ) : scanWork n ≤ 2*n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    rw [scanWork]
    split_ifs with hn
    · omega
    · have hlt : n/2 < n := by omega
      have hi := ih (n/2) hlt
      omega

end
end PNDS.LANGMATH098

#print axioms PNDS.LANGMATH098.tensor_compose_assoc
#print axioms PNDS.LANGMATH098.tensor_identity_left
#print axioms PNDS.LANGMATH098.tensor_identity_right
#print axioms PNDS.LANGMATH098.tensor_action
#print axioms PNDS.LANGMATH098.tensor_nonwrite
#print axioms PNDS.LANGMATH098.chunk_equivalence
#print axioms PNDS.LANGMATH098.sigmoid_range
#print axioms PNDS.LANGMATH098.squash_bound
#print axioms PNDS.LANGMATH098.softmax_nonnegative
#print axioms PNDS.LANGMATH098.softmax_mass
#print axioms PNDS.LANGMATH098.zero_valid
#print axioms PNDS.LANGMATH098.bank_update_valid
#print axioms PNDS.LANGMATH098.bank_mass_update
#print axioms PNDS.LANGMATH098.bank_read_bound
#print axioms PNDS.LANGMATH098.convex_coordinate
#print axioms PNDS.LANGMATH098.exposed_reads_injective
#print axioms PNDS.LANGMATH098.scan_work_linear
