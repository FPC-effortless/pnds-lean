import Mathlib

/-! Exact-real, shape-indexed equations for the source-pinned LANGCAP103 graphs.
Numerical source correspondence is checked separately in Python. This module
does not prove PyTorch/IEEE implementations, trained capability or long-memory
utility. Positive dimensions and parameter/source correspondence are external
registration obligations. -/

open scoped BigOperators

namespace PNDS.LANGCAP103

abbrev Vec (n : ℕ) := Fin n → ℝ
abbrev Mat (m n : ℕ) := Fin m → Fin n → ℝ

structure Affine (m n : ℕ) where
  weight : Mat m n
  bias : Vec m

def affine {m n : ℕ} (p : Affine m n) (x : Vec n) : Vec m :=
  fun i => (∑ j, p.weight i j * x j) + p.bias i

def cat {n m : ℕ} (x : Vec n) (y : Vec m) : Vec (n+m) :=
  fun i => if h : i.val < n then x ⟨i.val, h⟩
    else y ⟨i.val-n, by omega⟩

noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))
noncomputable def tanh (x : ℝ) : ℝ :=
  (Real.exp (2*x)-1) / (Real.exp (2*x)+1)
noncomputable def silu (x : ℝ) : ℝ := x * sigmoid x
noncomputable def erf (x : ℝ) : ℝ :=
  (2 / Real.sqrt Real.pi) * ∫ t in (0 : ℝ)..x, Real.exp (-(t^2))
noncomputable def gelu (x : ℝ) : ℝ :=
  (x/2) * (1 + erf (x / Real.sqrt 2))

noncomputable def softmax {n : ℕ} (x : Vec n) : Vec n :=
  fun i => Real.exp (x i) / ∑ j, Real.exp (x j)

structure Norm (n : ℕ) where
  gain : Vec n
  bias : Vec n

noncomputable def layerNorm {n : ℕ} (p : Norm n) (x : Vec n) : Vec n :=
  let mean := (∑ j, x j) / (n : ℝ)
  let variance := (∑ j, (x j-mean)^2) / (n : ℝ)
  fun i => p.gain i * ((x i-mean) / Real.sqrt (variance + 1/100000)) + p.bias i

structure GatedHead (d w : ℕ) where
  norm : Norm d
  up : Affine w d
  gate : Affine w d
  output : Affine 256 w

noncomputable def gatedHead {d w : ℕ} (p : GatedHead d w) (x : Vec d) : Vec 256 :=
  let z := layerNorm p.norm x
  affine p.output (fun i => affine p.up z i * silu (affine p.gate z i))

abbrev FeatureDim (d c v : ℕ) := d+c+v+v+v+2

def features {d c v : ℕ} (h : Vec d) (s : Vec c)
    (fast slow third : Vec v) (e : Vec 2) : Vec (FeatureDim d c v) :=
  cat (cat (cat (cat (cat h s) fast) slow) third) e

structure BankState (c k v : ℕ) where
  context : Vec c
  key : Fin 2 → Vec k
  matrix : Fin 2 → Mat v k
  mass : Fin 2 → Vec k

structure BlockParams (d c k v : ℕ) where
  contextInput : Affine c d
  rawDecay : Vec c
  key : Fin 2 → Affine k (d+c)
  query : Fin 2 → Affine k (d+c)
  value : Fin 2 → Affine v (d+c)
  retention : Affine 2 (d+c)
  compare : Affine v (FeatureDim d c v)
  norm : Norm (FeatureDim d c v)
  output : Affine d (FeatureDim d c v)

noncomputable def retentionLow (b : Fin 2) : ℝ := if b.val = 0 then 97/100 else 995/1000
noncomputable def retentionHigh (b : Fin 2) : ℝ := if b.val = 0 then 995/1000 else 99995/100000

def bankMass {k : ℕ} (a : ℝ) (old key : Vec k) : Vec k :=
  fun j => a*old j + (1-a)*key j

def bankMatrix {k v : ℕ} (a : ℝ) (old : Mat v k)
    (value : Vec v) (key : Vec k) : Mat v k :=
  fun i j => a*old i j + (1-a)*value i*key j

noncomputable def block {d c k v : ℕ} (p : BlockParams d c k v)
    (s : BankState c k v) (clock : ℕ) (h : Vec d) : Vec d × BankState c k v :=
  let context : Vec c := fun i =>
    sigmoid (-p.rawDecay i)*s.context i +
      (1-sigmoid (-p.rawDecay i))*tanh (affine p.contextInput h i)
  let input := cat h s.context
  let key : Fin 2 → Vec k := fun b => softmax (affine (p.key b) input)
  let query : Fin 2 → Vec k := fun b => softmax (affine (p.query b) input)
  let value : Fin 2 → Vec v := fun b i => tanh (affine (p.value b) input i)
  let a : Vec 2 := fun b => if clock = 0 then 1 else
    retentionLow b + (retentionHigh b-retentionLow b)*sigmoid (affine p.retention input b)
  let mass : Fin 2 → Vec k := fun b => bankMass (a b) (s.mass b) (s.key b)
  let matrix : Fin 2 → Mat v k := fun b => bankMatrix (a b) (s.matrix b) (value b) (s.key b)
  let evidence : Vec 2 := fun b => ∑ j, mass b j * query b j
  let read : Fin 2 → Vec v := fun b i =>
    (∑ j, matrix b i j * query b j) / (evidence b + 1/10000)
  let fast := read ⟨0, by decide⟩
  let slow := read ⟨1, by decide⟩
  let compare := features h s.context fast slow (fun i => |fast i-slow i|) evidence
  let gate : Vec v := fun i => sigmoid (affine p.compare compare i)
  let mixed : Vec v := fun i => gate i * fast i + (1-gate i)*slow i
  let z := layerNorm p.norm (features h s.context fast slow mixed evidence)
  (fun i => h i + tanh (affine p.output z i), ⟨context, key, matrix, mass⟩)

structure TacState (d c k v L : ℕ) where
  clock : ℕ
  lexical : Fin 3 → Vec d
  layers : Fin L → BankState c k v

structure TacParams (d c k v L w : ℕ) where
  embedding : Mat 256 d
  lag : Fin 3 → Vec d
  blocks : Fin L → BlockParams d c k v
  head : GatedHead d w

noncomputable def tacStep {d c k v L w : ℕ} (p : TacParams d c k v L w)
    (s : TacState d c k v L) (token : Fin 256) : Vec 256 × TacState d c k v L :=
  let e : Vec d := fun i => tanh (p.embedding token i)
  let localInput : Vec d := fun i => tanh (e i + ∑ j : Fin 3,
    tanh (p.lag j i)*s.lexical ⟨2-j.val, by omega⟩ i)
  let result := (List.finRange L).foldl
    (fun (acc : Vec d × (Fin L → BankState c k v)) index =>
      let next := block (p.blocks index) (acc.2 index) s.clock acc.1
      (next.1, fun j => if j = index then next.2 else acc.2 j)) (localInput, s.layers)
  let lexical : Fin 3 → Vec d := fun j =>
    if j.val = 0 then s.lexical ⟨1, by decide⟩
    else if j.val = 1 then s.lexical ⟨2, by decide⟩ else e
  (gatedHead p.head result.1, ⟨s.clock+1, lexical, result.2⟩)

structure GRUCellParams (h : ℕ) where
  resetInput : Affine h h
  resetHidden : Affine h h
  updateInput : Affine h h
  updateHidden : Affine h h
  newInput : Affine h h
  newHidden : Affine h h

noncomputable def gruCell {h : ℕ} (p : GRUCellParams h) (old input : Vec h) : Vec h :=
  let reset : Vec h := fun i => sigmoid (affine p.resetInput input i + affine p.resetHidden old i)
  let update : Vec h := fun i => sigmoid (affine p.updateInput input i + affine p.updateHidden old i)
  let new : Vec h := fun i => tanh (affine p.newInput input i + reset i*affine p.newHidden old i)
  fun i => (1-update i)*new i + update i*old i

structure GRUParams (h L : ℕ) where
  embedding : Mat 256 h
  cells : Fin L → GRUCellParams h
  output : Affine 256 h

noncomputable def gruStep {h L : ℕ} (p : GRUParams h L)
    (s : Fin L → Vec h) (token : Fin 256) : Vec 256 × (Fin L → Vec h) :=
  let result := (List.finRange L).foldl
    (fun (acc : Vec h × (Fin L → Vec h)) index =>
      let next := gruCell (p.cells index) (acc.2 index) acc.1
      (next, fun j => if j = index then next else acc.2 j)) (p.embedding token, s)
  (affine p.output result.1, result.2)

def maskedWeight {T : ℕ} (score : Fin T → ℝ) (time key : Fin T) : ℝ :=
  if key.val ≤ time.val then score key else 0

structure AttentionParams (d heads dh : ℕ) where
  query : Fin heads → Affine dh d
  key : Fin heads → Affine dh d
  value : Fin heads → Affine dh d
  output : Fin heads → Mat d dh
  bias : Vec d

noncomputable def attention {d heads dh T : ℕ} (p : AttentionParams d heads dh)
    (x : Fin T → Vec d) (time : Fin T) : Vec d :=
  let q : Fin heads → Vec dh := fun b => affine (p.query b) (x time)
  let scores : Fin heads → Vec T := fun b key => Real.exp
    ((∑ i, q b i * affine (p.key b) (x key) i) / Real.sqrt (dh : ℝ))
  let weight : Fin heads → Vec T := fun b key =>
    maskedWeight (scores b) time key / ∑ j, maskedWeight (scores b) time j
  let context : Fin heads → Vec dh := fun b i =>
    ∑ key, weight b key * affine (p.value b) (x key) i
  fun i => p.bias i + ∑ b, ∑ j, p.output b i j * context b j

structure TransformerBlock (d heads dh f : ℕ) where
  norm1 : Norm d
  norm2 : Norm d
  attn : AttentionParams d heads dh
  up : Affine f d
  down : Affine d f

noncomputable def transformerBlock {d heads dh f T : ℕ}
    (p : TransformerBlock d heads dh f) (x : Fin T → Vec d) : Fin T → Vec d :=
  let normalized := fun t => layerNorm p.norm1 (x t)
  let residual : Fin T → Vec d := fun t i => x t i + attention p.attn normalized t i
  fun t i => residual t i + affine p.down
    (fun j => gelu (affine p.up (layerNorm p.norm2 (residual t)) j)) i

noncomputable def position {d T : ℕ} (time : Fin T) : Vec d :=
  fun j => let frequency := Real.exp (((2*(j.val/2) : ℕ) : ℝ) * (-Real.log 10000 / (d : ℝ)))
    if j.val % 2 = 0 then Real.sin ((time.val : ℝ)*frequency)
    else Real.cos ((time.val : ℝ)*frequency)

structure TransformerParams (d heads dh f : ℕ) where
  embedding : Mat 256 d
  blocks : List (TransformerBlock d heads dh f)
  norm : Norm d
  output : Affine 256 d

noncomputable def transformer {d heads dh f T : ℕ} (p : TransformerParams d heads dh f)
    (tokens : Fin T → Fin 256) : Fin T → Vec 256 :=
  let initial : Fin T → Vec d := fun t i =>
    p.embedding (tokens t) i * Real.sqrt (d : ℝ) + position t i
  let latent := p.blocks.foldl (fun h layer => transformerBlock layer h) initial
  fun t => affine p.output (layerNorm p.norm (latent t))

def outputs {S I O : Type} (step : S → I → O × S) (s : S) : List I → List O
  | [] => []
  | x :: xs => (step s x).1 :: outputs step (step s x).2 xs

def stateAfter {S I O : Type} (step : S → I → O × S) (s : S) (xs : List I) : S :=
  xs.foldl (fun state input => (step state input).2) s

/-- Structural streaming composition for the explicit causal recurrent graph. -/
theorem stateAfter_append {S I O : Type} (step : S → I → O × S)
    (s : S) (xs ys : List I) :
    stateAfter step s (xs ++ ys) = stateAfter step (stateAfter step s xs) ys := by
  simp [stateAfter, List.foldl_append]

/-- Appending future inputs cannot alter the earlier explicit recurrent outputs. -/
theorem outputs_prefix {S I O : Type} (step : S → I → O × S)
    (s : S) (xs ys : List I) :
    (outputs step s (xs ++ ys)).take xs.length = outputs step s xs := by
  induction xs generalizing s with
  | nil => simp [outputs]
  | cons x xs ih => simp [outputs, ih]

theorem future_mask_zero {T : ℕ} (score : Fin T → ℝ) (time key : Fin T)
    (future : time.val < key.val) : maskedWeight score time key = 0 := by
  simp [maskedWeight, Nat.not_le.mpr future]

theorem bank_mass_nonnegative {k : ℕ} (a : ℝ) (old key : Vec k)
    (ha : 0 ≤ a ∧ a ≤ 1) (ho : ∀ j, 0 ≤ old j) (hk : ∀ j, 0 ≤ key j) :
    ∀ j, 0 ≤ bankMass a old key j := by
  intro j
  exact add_nonneg (mul_nonneg ha.1 (ho j))
    (mul_nonneg (sub_nonneg.mpr ha.2) (hk j))

theorem read_denominator_positive {k : ℕ} (mass query : Vec k) (epsilon : ℝ)
    (hm : ∀ j, 0 ≤ mass j) (hq : ∀ j, 0 ≤ query j) (he : 0 < epsilon) :
    0 < (∑ j, mass j * query j) + epsilon := by
  exact add_pos_of_nonneg_of_pos (Finset.sum_nonneg (fun j _ => mul_nonneg (hm j) (hq j))) he

#print axioms stateAfter_append
#print axioms outputs_prefix
#print axioms future_mask_zero
#print axioms bank_mass_nonnegative
#print axioms read_denominator_positive

end PNDS.LANGCAP103
