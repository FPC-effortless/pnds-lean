import Mathlib

/-!
TAC-OCR-001, partial computational formulation.

Float arithmetic below is executable Lean semantics, NOT a real-number
stability or PyTorch finite-precision equivalence theorem. The full neural
encoder, learned projections, recurrent source graph, autograd, and frozen
inference lowering remain OPEN. The sole theorem concerns list capacity.
-/
namespace PNDS.TACOCR001

abbrev Vec (n : Nat) := Fin n → Float
abbrev Mat (r c : Nat) := Fin r → Fin c → Float

def sumFloat (xs : List Float) : Float := xs.foldl (· + ·) 0

def dot {n : Nat} (a b : Vec n) : Float :=
  sumFloat (List.ofFn (fun i => a i * b i))

def attentionHead {k d v : Nat} (nonempty : 0 < k)
    (q : Vec d) (keys : Mat k d) (values : Mat k v) : Vec v :=
  let scores : Vec k := fun i => dot q (keys i) / (Float.ofNat d).sqrt
  let largest := (List.ofFn scores).foldl max (scores ⟨0, nonempty⟩)
  let weights : Vec k := fun i => (scores i - largest).exp
  let denominator := sumFloat (List.ofFn weights)
  fun j => sumFloat (List.ofFn (fun i => weights i * values i j)) / denominator

structure PositiveBank (d v : Nat) where
  memory : Mat v d
  mass : Vec d

def bankStep {d v r : Nat} (old : PositiveBank d v)
    (alpha eligibility : Float) (keys : Mat r d) (values : Mat r v) : PositiveBank d v :=
  let weight := (1 - alpha) * eligibility / Float.ofNat r
  { memory := fun i j => alpha * old.memory i j +
      weight * sumFloat (List.ofFn (fun rank => values rank i * keys rank j))
    mass := fun j => alpha * old.mass j +
      weight * sumFloat (List.ofFn (fun rank => keys rank j)) }

def bankRead {d v : Nat} (state : PositiveBank d v) (query : Vec d) : Vec v :=
  fun i => dot (state.memory i) query / (dot state.mass query + 0.0001)

/- Full source lowering and learned tensor operations are deliberately
explicit parameters here, not hidden assumptions asserted as proofs. -/
structure LayerBoundary (width stateSize references : Nat) where
  visual : Vec width → Mat references width → Vec width
  recurrence : Vec width → Vec stateSize → Vec width × Vec stateSize
  feedForward : Vec width → Vec width

def groundedLayer {w s k : Nat} (ops : LayerBoundary w s k)
    (x : Vec w) (memory : Vec s) (refs : Mat k w) : Vec w × Vec s :=
  let (hidden, nextMemory) := ops.recurrence (ops.visual x refs) memory
  (ops.feedForward hidden, nextMemory)

def boundedReferences {α : Type} (budget : Nat) (available : List α) : List α :=
  available.take budget

theorem referenceCountBound {α : Type} (budget : Nat) (available : List α) :
    (boundedReferences budget available).length ≤ budget := by
  unfold boundedReferences
  rw [List.length_take]
  exact Nat.min_le_left _ _

#print axioms referenceCountBound

end PNDS.TACOCR001
