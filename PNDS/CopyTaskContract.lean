import Std

/- Finite symbolic task semantics only. No neural graph, IEEE, AD or
   learned-capability theorem is stated by these fixture contracts. -/
namespace CopyTaskContract

abbrev Value := List Nat
abbrev History := List (Nat × Value)

def lookup : History → Nat → Option Value
  | [], _ => none
  | (key, value) :: rest, query =>
      match lookup rest query with
      | some answer => some answer
      | none => if key = query then some value else none

def lastValue : History → Option Value
  | [] => none
  | (_, value) :: rest =>
      match lastValue rest with
      | some answer => some answer
      | none => some value

def SelectionWitness (history : History) (q₁ q₂ : Nat) : Prop :=
  ∃ v₁ v₂, lookup history q₁ = some v₁ ∧ lookup history q₂ = some v₂ ∧ v₁ ≠ v₂

theorem lookup_empty (query : Nat) : lookup [] query = none := rfl

theorem lookup_singleton (key : Nat) (value : Value) :
    lookup [(key, value)] key = some value := by
  simp [lookup]

theorem singleton_key_blind (key : Nat) (value : Value) :
    lookup [(key, value)] key = lastValue [(key, value)] := by
  simp [lookup, lastValue]

theorem latest_write_wins (history : History) (key : Nat) (value : Value) :
    lookup (history ++ [(key, value)]) key = some value := by
  induction history with
  | nil => simp [lookup]
  | cons record rest ih =>
      cases record with
      | mk k v => simp [lookup, ih]

theorem witness_excludes_key_blind (history : History) (q₁ q₂ : Nat)
    (f : History → Option Value) (hw : SelectionWitness history q₁ q₂) :
    ¬ (f history = lookup history q₁ ∧ f history = lookup history q₂) := by
  cases hw with
  | intro v₁ inner =>
    cases inner with
    | intro v₂ conditions =>
      intro both
      have eqSome : some v₁ = some v₂ :=
        (both.1.trans conditions.1).symm.trans (both.2.trans conditions.2.1)
      exact conditions.2.2 (Option.some.inj eqSome)

theorem same_query_no_witness (history : History) (query : Nat) :
    ¬ SelectionWitness history query query := by
  intro hw
  cases hw with
  | intro v₁ inner =>
    cases inner with
    | intro v₂ conditions =>
      exact conditions.2.2 (Option.some.inj (conditions.1.symm.trans conditions.2.1))

theorem two_key_witness : SelectionWitness [(0, [0]), (1, [1])] 0 1 := by
  exact ⟨[0], [1], by simp [lookup], by simp [lookup], by decide⟩

#print axioms lookup_empty
#print axioms lookup_singleton
#print axioms singleton_key_blind
#print axioms latest_write_wins
#print axioms witness_excludes_key_blind
#print axioms same_query_no_witness
#print axioms two_key_witness

end CopyTaskContract
