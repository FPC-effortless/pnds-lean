import Mathlib

/-!
LANGCACHE104 exact-real mathematical record. This file is a typed operator and
graph-interface formulation, NOT a completed source-equivalence proof.  Python
contains the separately executable full neural graph.  The build/axiom audit,
array-layout lowering, all-source gradients and finite precision remain OPEN.
No trained capability, brain equivalence, or GPU speed follows from this file.
-/

namespace PNDS.LANGCACHE104

open scoped BigOperators

abbrev Vec (n : ℕ) := Fin n → ℝ

structure Geometry where
  width heads keyDim valueDim rank layers ffnDim window : ℕ
  positive : 0 < width ∧ 0 < heads ∧ 0 < keyDim ∧ 0 < valueDim ∧
    0 < rank ∧ 0 < layers ∧ 0 < ffnDim ∧ 0 < window
  evenValue : valueDim % 2 = 0

structure BankState (k v : ℕ) where
  memory : Fin v → Fin k → ℝ
  mass : Vec k

structure BankOffer (k v r : ℕ) where
  query : Vec k
  keys : Fin r → Vec k
  values : Fin r → Vec v
  retention : ℝ
  write : ℝ

/-- Current observed input writes before retrieval; no future byte is used. -/
noncomputable def bankStep {k v r : ℕ} (s : BankState k v)
    (u : BankOffer k v r) : BankState k v :=
  { memory := fun i j => u.retention * s.memory i j +
      u.write / (r : ℝ) * ∑ p, u.values p i * u.keys p j
    mass := fun j => u.retention * s.mass j +
      u.write / (r : ℝ) * ∑ p, u.keys p j }

noncomputable def evidence {k v : ℕ} (s : BankState k v) (q : Vec k) : ℝ :=
  ∑ j, s.mass j * q j

noncomputable def bankRead {k v : ℕ} (s : BankState k v)
    (q : Vec k) (epsilon : ℝ) : Vec v :=
  fun i => (∑ j, s.memory i j * q j) / (evidence s q + epsilon)

def BankCone {k v : ℕ} (s : BankState k v) : Prop :=
  (∀ j, 0 ≤ s.mass j ∧ s.mass j ≤ 1) ∧
  (∀ i j, -s.mass j ≤ s.memory i j ∧ s.memory i j ≤ s.mass j)

/-- A one-coordinate sufficient condition for preserving the positive cone.
    It is deliberately separate from the OPEN tensor/source lifting proof. -/
theorem cone_coordinate (m z a b c : ℝ)
    (hm : -z ≤ m ∧ m ≤ z) (hz : 0 ≤ z ∧ z ≤ 1)
    (ha : 0 ≤ a ∧ a ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1-a)
    (hb : -c ≤ b ∧ b ≤ c) :
    0 ≤ a*z+c ∧ a*z+c ≤ 1 ∧
      -(a*z+c) ≤ a*m+b ∧ a*m+b ≤ a*z+c := by
  have hlow : 0 ≤ a*(m+z) := mul_nonneg ha.1 (by linarith [hm.1])
  have hhigh : 0 ≤ a*(z-m) := mul_nonneg ha.1 (by linarith [hm.2])
  have hmass : 0 ≤ a*z := mul_nonneg ha.1 hz.1
  have hcap : 0 ≤ a*(1-z) := mul_nonneg ha.1 (by linarith [hz.2])
  constructor
  · linarith [hc.1]
  constructor
  · nlinarith [hc.2]
  constructor <;> nlinarith [hb.1, hb.2]

structure Affine where
  a b : ℝ

def compose (later earlier : Affine) : Affine :=
  ⟨later.a*earlier.a, later.a*earlier.b+later.b⟩

def applyAffine (transition : Affine) (s : ℝ) : ℝ := transition.a*s+transition.b

theorem affine_composition (later earlier : Affine) (s : ℝ) :
    applyAffine (compose later earlier) s = applyAffine later (applyAffine earlier s) := by
  dsimp [applyAffine, compose]
  ring

theorem affine_associative (x y z : Affine) :
    compose (compose x y) z = compose x (compose y z) := by
  cases x with
  | mk xa xb =>
    cases y with
    | mk ya yb =>
      cases z with
      | mk za zb =>
        dsimp [compose]
        congr 1 <;> ring

abbrev PairVector (pairs : ℕ) := Fin pairs → (ℝ × ℝ)

noncomputable def rotate (theta : ℝ) (xy : ℝ × ℝ) : ℝ × ℝ :=
  (Real.cos theta*xy.1-Real.sin theta*xy.2,
   Real.sin theta*xy.1+Real.cos theta*xy.2)

/-- The signed rotating lane has a norm-envelope obligation, not BankCone. -/
noncomputable def rotaryStep {p : ℕ} (a : ℝ) (theta : Vec p)
    (offer previous : PairVector p) : PairVector p := fun j =>
  let old := rotate (theta j) (previous j)
  (a*old.1+(1-a)*(offer j).1, a*old.2+(1-a)*(offer j).2)

structure RecentKV (k v : ℕ) where
  key : Vec k
  value : Vec v
  observedAt : ℕ

def legalRecent {k v : ℕ} (clock window : ℕ) (entry : RecentKV k v) : Prop :=
  entry.observedAt ≤ clock ∧ clock < entry.observedAt + window

structure LayerState (g : Geometry) where
  fast slow : Fin g.heads → BankState g.keyDim g.valueDim
  rotary : Fin g.heads → PairVector (g.valueDim / 2)
  recent : Fin g.heads → List (RecentKV g.keyDim g.valueDim)

/-- Named perspectives stay distinct until the learned output projection. -/
structure Perspective (v : ℕ) where
  fast slow difference product recent : Vec v
  rotary : PairVector (v / 2)
  fastEvidence slowEvidence : ℝ

structure StreamState (g : Geometry) where
  owner : String
  clock : ℕ
  layerState : Fin g.layers → LayerState g

structure Linear (input output : ℕ) where
  weight : Fin output → Fin input → ℝ
  bias : Vec output

noncomputable def linear {n m : ℕ} (p : Linear n m) (x : Vec n) : Vec m :=
  fun i => (∑ j, p.weight i j*x j)+p.bias i

noncomputable def sigmoid (x : ℝ) : ℝ := 1/(1+Real.exp (-x))

noncomputable def rmsNorm {n : ℕ} (weight x : Vec n) (epsilon : ℝ) : Vec n :=
  fun i => weight i*x i/Real.sqrt ((∑ j, x j*x j)/(n : ℝ)+epsilon)

noncomputable def swiglu {n f : ℕ} (value gate : Linear n f)
    (output : Linear f n) (x : Vec n) : Vec n :=
  linear output (fun j => linear value x j * linear gate x j * sigmoid (linear gate x j))

/-- Full layer boundary, with source lowering explicitly OPEN.
`memory` must apply pre-RMSNorm, both bank updates/reads, the separate rotary
update, causal position-aware local attention, and exact feature concatenation
specified in FORMULATION.md; it may not inspect future or target observations.
The function slots are an interface, not an equivalence theorem. -/
structure LayerGraph (g : Geometry) where
  memory : ℕ → Bool → Vec g.width → LayerState g →
    (Vec g.width × LayerState g)
  ffn : Vec g.width → Vec g.width

noncomputable def layerOutput {g : Geometry} (graph : LayerGraph g)
    (clock : ℕ) (writeAllowed : Bool) (x : Vec g.width) (s : LayerState g) :
    Vec g.width × LayerState g :=
  let first := graph.memory clock writeAllowed x s
  let scale := 1/Real.sqrt (2*(g.layers : ℝ))
  let residual := fun j => x j + scale*first.1 j
  (fun j => residual j + scale*graph.ffn residual j, first.2)

/-- OPEN statements are data, never new axioms or admitted theorems. -/
def openObligations : List String :=
  ["LEAN_KERNEL_BUILD_AND_AXIOM_AUDIT",
   "RANK_R_TENSOR_CONE_LIFT_AND_NORMALIZED_READ_BOUND",
   "ROTARY_NORM_ENVELOPE_AND_PAIR_SCAN_CORRESPONDENCE",
   "FULL_LAYERGRAPH_TO_ORIGINAL_PYTORCH_EQUIVALENCE",
   "ALL_PARAMETER_AND_INPUT_STATE_GRADIENT_CORRESPONDENCE",
   "FP32_FP64_LONG_HORIZON_ERROR_AND_MIXED_PRECISION",
   "TRAINED_REAL_SOURCE_CAPABILITY_AND_MATCHED_COST"]

end PNDS.LANGCACHE104
