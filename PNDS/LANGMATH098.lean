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

structure NativeState.Valid (s : NativeState) : Prop where
  context_bounded : ∀ i, |s.context i| ≤ 1
  lexical_bounded : ∀ l i, |s.lexical l i| ≤ 1
  previous_nonnegative : ∀ h j, 0 ≤ s.previousKey h j
  previous_mass : ∀ h, ∑ j, s.previousKey h j ≤ 1
  banks_valid : ∀ h, (s.banks h).Valid
  banks_mass : ∀ h, ∑ j, (s.banks h).mass j ≤ 1

def nativeZero : NativeState :=
  ⟨fun _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ => Bank.zero 8 16⟩
theorem native_zero_valid : nativeZero.Valid := by
  constructor <;> intros <;> simp [nativeZero, Bank.zero, Bank.Valid]

def nativeStep (s : NativeState) (contextOffer : Fin 48 → ℝ)
    (embedding : Fin 32 → ℝ) (keys : Fin 2 → Fin 8 → ℝ)
    (values : Fin 2 → Fin 16 → ℝ) (delta : Fin 48 → ℝ)
    (alpha : Fin 2 → ℝ) : NativeState where
  context := fun i => delta i*s.context i+(1-delta i)*contextOffer i
  lexical := fun l i => if l=0 then s.lexical 1 i else if l=1 then s.lexical 2 i else embedding i
  previousKey := keys
  banks := fun h => (s.banks h).update (alpha h) (s.previousKey h) (values h)

theorem native_step_valid (s : NativeState) (c : Fin 48 → ℝ)
    (e : Fin 32 → ℝ) (keys : Fin 2 → Fin 8 → ℝ)
    (values : Fin 2 → Fin 16 → ℝ) (delta : Fin 48 → ℝ) (alpha : Fin 2 → ℝ)
    (hs : s.Valid) (hc : ∀ i, |c i| ≤ 1) (he : ∀ i, |e i| ≤ 1)
    (hk : ∀ h j, 0 ≤ keys h j) (hm : ∀ h, ∑ j, keys h j ≤ 1)
    (hv : ∀ h i, |values h i| ≤ 1)
    (hd : ∀ i, 0 ≤ delta i ∧ delta i ≤ 1)
    (ha : ∀ h, 0 ≤ alpha h ∧ alpha h ≤ 1) :
    (nativeStep s c e keys values delta alpha).Valid := by
  constructor
  · intro i; exact convex_coordinate _ _ _ (hs.context_bounded i) (hc i) (hd i)
  · intro l i
    dsimp [nativeStep]
    split_ifs
    · exact hs.lexical_bounded 1 i
    · exact hs.lexical_bounded 2 i
    · exact he i
  · exact hk
  · exact hm
  · intro h; exact bank_update_valid _ _ _ _ (hs.banks_valid h) (ha h)
      (hs.previous_nonnegative h) (hv h)
  · intro h; exact bank_mass_update _ _ _ _ (hs.banks_mass h) (hs.previous_mass h) (ha h)

def bankAlpha (h : Fin 2) (raw : ℝ) : ℝ :=
  if h=0 then (0.97:ℝ)+(0.995-0.97)*sigmoid raw
  else (0.995:ℝ)+(0.99995-0.995)*sigmoid raw
theorem bank_alpha_range (h : Fin 2) (raw : ℝ) :
    0 ≤ bankAlpha h raw ∧ bankAlpha h raw ≤ 1 := by
  have hr := sigmoid_range raw
  unfold bankAlpha
  split_ifs <;> constructor <;> nlinarith

/-- All encoder logits may depend on state and current observations. Bounds
follow from the implemented activation ranges rather than assuming bounded
matrix updates. Exact-real squash is tanh's sigmoid identity. -/
structure RawOffer where
  context : Fin 48 → ℝ
  embedding : Fin 32 → ℝ
  keys : Fin 2 → Fin 8 → ℝ
  values : Fin 2 → Fin 16 → ℝ
  decay : Fin 48 → ℝ
  alpha : Fin 2 → ℝ
  allowed : Bool

def encodedStep (s : NativeState) (raw : RawOffer) : NativeState :=
  nativeStep s (fun i => squash (raw.context i))
    (fun i => squash (raw.embedding i)) (fun h => softmax (raw.keys h))
    (fun h i => squash (raw.values h i)) (fun i => sigmoid (-raw.decay i))
    (fun h => if raw.allowed then bankAlpha h (raw.alpha h) else 1)

theorem encoded_step_valid (s : NativeState) (raw : RawOffer) (hs : s.Valid) :
    (encodedStep s raw).Valid := by
  unfold encodedStep
  apply native_step_valid
  · exact hs
  · intro i; exact squash_bound _
  · intro i; exact squash_bound _
  · intro h j; exact softmax_nonnegative _ _
  · intro h; exact le_of_eq (softmax_mass _)
  · intro h i; exact squash_bound _
  · intro i; exact ⟨(sigmoid_range _).1.le, (sigmoid_range _).2.le⟩
  · intro h; split_ifs
    · exact bank_alpha_range _ _
    · norm_num

/-- Full persistent tensor invariant for an arbitrary state/current-input
encoder. It does not prove that an encoder learns information or that AD and
IEEE arithmetic have exact-real behavior. -/
theorem full_persistent_state_invariant {Input : Type*}
    (encoder : NativeState → Input → RawOffer) (s : NativeState)
    (hs : s.Valid) (observations : List Input) :
    (unroll (fun state input => encodedStep state (encoder state input)) s observations).Valid := by
  induction observations generalizing s with
  | nil => simpa [unroll] using hs
  | cons x xs ih =>
    simpa [unroll, List.foldl_cons] using
      ih (encodedStep s (encoder s x)) (encoded_step_valid s (encoder s x) hs)

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

#print axioms PNDS.LANGMATH098.native_zero_valid
#print axioms PNDS.LANGMATH098.native_step_valid
#print axioms PNDS.LANGMATH098.bank_alpha_range
#print axioms PNDS.LANGMATH098.encoded_step_valid
#print axioms PNDS.LANGMATH098.full_persistent_state_invariant
