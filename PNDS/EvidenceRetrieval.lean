/- Typed observation-to-output graph for shared-core evidence retrieval.
   Tensor maps are parameters, NOT axiomatized claims about PyTorch floats.
   Existing RetentionBudget handles conditional scalar bounds separately.
   Full concrete NN/finite-precision refinement remains OPEN. -/
namespace PNDS.EvidenceRetrieval
structure State (Evidence Memory : Type) where
  evidence : Evidence
  memory : Memory

structure Graph (Input Evidence Proposal Key Value Schedule Query Memory Output : Type) where
  summarize : Evidence → Input → Evidence
  propose : Evidence → Input → Proposal
  key : Input → Key
  value : Input → Value
  schedule : Input → Proposal → Schedule
  write : Schedule → Key → Value → Memory → Memory
  query : Input → Proposal → Key → Query
  decode : Input → Proposal → Memory → Query → Output

def step (g : Graph I E P K V A Q M O) (valid : Bool)
    (s : State E M) (x : I) : State E M × O :=
  let p := g.propose s.evidence x
  let k := g.key x
  let q := g.query x p k
  let out := g.decode x p s.memory q
  let next := if valid then
    { evidence := g.summarize s.evidence x
      memory := g.write (g.schedule x p) k (g.value x) s.memory }
    else s
  (next, out)

/-- Padding leaves both evidence and associative memory unchanged. -/
theorem invalid_identity (g : Graph I E P K V A Q M O) (s : State E M) (x : I) :
    (step g false s x).1 = s := by rfl

/-- Retrieval uses pre-write memory, independent of the current write result. -/
theorem read_before_write (g : Graph I E P K V A Q M O)
    (valid : Bool) (s : State E M) (x : I) :
    (step g valid s x).2 = g.decode x (g.propose s.evidence x) s.memory
      (g.query x (g.propose s.evidence x) (g.key x)) := by rfl

/-- Query/decoder cannot change the next stored state for fixed update maps. -/
theorem query_does_not_write (g : Graph I E P K V A Q M O)
    (q : I → P → K → Q) (valid : Bool) (s : State E M) (x : I) :
    (step {g with query := q} valid s x).1 = (step g valid s x).1 := by rfl
end PNDS.EvidenceRetrieval
