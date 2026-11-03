/- DRAFT, NOT COMPILED. Necessary lifecycle predicate only, NOT neural lowering.
   Fresh token inequality is a premise. UUID randomness is not proved here. -/
namespace PNDS.TACOCR004Lifecycle
structure Binding where
  signature : String
  generation : String
  owners : List String

def accepts (current refs state : Binding) : Prop :=
  current.generation = refs.generation ∧
  current.generation = state.generation ∧
  current.owners = refs.owners ∧ current.owners = state.owners ∧
  current.signature = state.signature

def renew (current : Binding) (fresh : String)
    (_nonempty : fresh ≠ "") (_different : fresh ≠ current.generation) : Binding :=
  { current with generation := fresh }

theorem staleReferenceRejected (current refs state : Binding) (fresh : String)
    (old : refs.generation = current.generation)
    (different : fresh ≠ current.generation)
    (nonempty : fresh ≠ "") :
    ¬ accepts (renew current fresh nonempty different) refs state := by
  intro h
  exact different (h.1.trans old)

#print axioms staleReferenceRejected
end PNDS.TACOCR004Lifecycle
