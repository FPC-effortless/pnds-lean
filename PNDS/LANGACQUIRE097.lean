import Mathlib

/-! LANGACQUIRE097: chronology, masked objectives and local chain rules.

These declarations specify exact-real training semantics. They do not identify
the nonlinear TAC transition with an affine scan or certify PyTorch/IEEE AD.
The source graph, parameter sharing, tensor layouts, finite precision and
actual-source gradients require their separate executable correspondence tests.
No learned capability or useful persistent state follows from these laws.
-/
namespace PNDS.LANGACQUIRE097
noncomputable section
open scoped BigOperators

variable {State Input : Type*}

/-- Apply exactly the observations supplied, in chronological order. -/
def unroll (step : State → Input → State) (initial : State)
    (observations : List Input) : State :=
  observations.foldl step initial

theorem unroll_nil (step : State → Input → State) (initial : State) :
    unroll step initial [] = initial := by
  rfl

/-- A process boundary changes no transition when the complete state is carried.
This is serial chunking, not an associativity assumption about nonlinear steps.
-/
theorem unroll_append (step : State → Input → State) (initial : State)
    (acquisition query : List Input) :
    unroll step initial (acquisition ++ query) =
      unroll step (unroll step initial acquisition) query := by
  simp [unroll, List.foldl_append]

theorem unroll_three_chunks (step : State → Input → State) (initial : State)
    (acquisition delay query : List Input) :
    unroll step initial ((acquisition ++ delay) ++ query) =
      unroll step (unroll step (unroll step initial acquisition) delay) query := by
  rw [unroll_append, unroll_append]

/-- Exact-real categorical negative log likelihood; a target supplies a valid
vocabulary index, so a zero-sized vocabulary cannot be instantiated here. -/
def tokenNLL {vocab : ℕ} (logits : Fin vocab → ℝ)
    (target : Fin vocab) : ℝ :=
  Real.log (∑ j, Real.exp (logits j)) - logits target

def maskedTotal {length : ℕ} (active : Fin length → Bool)
    (losses : Fin length → ℝ) : ℝ :=
  ∑ i, if active i = true then losses i else 0

def maskMass {length : ℕ} (active : Fin length → Bool) : ℝ :=
  ∑ i, if active i = true then 1 else 0

/-- The executable API rejects zero mask mass. The definition is total in Lean;
the nonempty training requirement is stated explicitly in the contract below. -/
def maskedMean {length : ℕ} (active : Fin length → Bool)
    (losses : Fin length → ℝ) : ℝ :=
  maskedTotal active losses / maskMass active

def maskedObjective {length vocab : ℕ}
    (active : Fin length → Bool)
    (logits : Fin length → Fin vocab → ℝ)
    (targets : Fin length → Fin vocab) : ℝ :=
  maskedMean active (fun i => tokenNLL (logits i) (targets i))

/-- Labels at inactive loss positions cannot alter the masked sum, holding the
observed trajectory fixed. This does not remove their observations from state.
-/
theorem masked_total_congr_active {length : ℕ}
    (active : Fin length → Bool) (left right : Fin length → ℝ)
    (h : ∀ i, active i = true → left i = right i) :
    maskedTotal active left = maskedTotal active right := by
  unfold maskedTotal
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : active i = true
  · simp [hi, h i hi]
  · simp [hi]

theorem masked_mean_congr_active {length : ℕ}
    (active : Fin length → Bool) (left right : Fin length → ℝ)
    (h : ∀ i, active i = true → left i = right i) :
    maskedMean active left = maskedMean active right := by
  unfold maskedMean
  rw [masked_total_congr_active active left right h]

/-- An all-position objective has the usual positive normalizer when length>0.
The query-only objective changes supervision, not the unrolled observations. -/
theorem all_active_mean_and_positive_mass {length : ℕ}
    (losses : Fin length → ℝ) (nonempty : 0 < length) :
    0 < maskMass (fun _ : Fin length => true) ∧
    maskedMean (fun _ : Fin length => true) losses =
      (∑ i, losses i) / (length : ℝ) := by
  constructor
  · simpa [maskMass] using (show (0 : ℝ) < (length : ℝ) by exact_mod_cast nonempty)
  · simp [maskedMean, maskedTotal, maskMass]

theorem inactive_loss_irrelevance {length : ℕ}
    (active : Fin length → Bool) (losses : Fin length → ℝ)
    (index : Fin length) (replacement : ℝ) (inactive : active index = false) :
    maskedMean active (Function.update losses index replacement) =
      maskedMean active losses := by
  apply masked_mean_congr_active
  intro i hi
  have distinct : i ≠ index := by
    intro equal
    subst i
    simp [inactive] at hi
  simp [Function.update, distinct, Ne.symm distinct]

/-- Local scalar chain through acquired state. Differentiability assumptions
are explicit; this is not a proof of a model's automatic differentiation.
-/
theorem acquisition_query_chain (acquire query : ℝ → ℝ)
    (parameter acquisitionGradient queryGradient : ℝ)
    (hAcquire : HasDerivAt acquire acquisitionGradient parameter)
    (hQuery : HasDerivAt query queryGradient (acquire parameter)) :
    HasDerivAt (fun p => query (acquire p))
      (queryGradient * acquisitionGradient) parameter := by
  exact hQuery.comp parameter hAcquire

/-- A direct parameter path and an acquisition-mediated path add. The complete
shared-parameter tensor graph needs the separate actual-source gradient audit.
-/
theorem direct_and_acquisition_chain (direct acquire query : ℝ → ℝ)
    (parameter directGradient acquisitionGradient queryGradient : ℝ)
    (hDirect : HasDerivAt direct directGradient parameter)
    (hAcquire : HasDerivAt acquire acquisitionGradient parameter)
    (hQuery : HasDerivAt query queryGradient (acquire parameter)) :
    HasDerivAt (fun p => direct p + query (acquire p))
      (directGradient + queryGradient * acquisitionGradient) parameter := by
  exact hDirect.add (hQuery.comp parameter hAcquire)

end
end PNDS.LANGACQUIRE097

#print axioms PNDS.LANGACQUIRE097.unroll_nil
#print axioms PNDS.LANGACQUIRE097.unroll_append
#print axioms PNDS.LANGACQUIRE097.unroll_three_chunks
#print axioms PNDS.LANGACQUIRE097.masked_total_congr_active
#print axioms PNDS.LANGACQUIRE097.masked_mean_congr_active
#print axioms PNDS.LANGACQUIRE097.all_active_mean_and_positive_mass
#print axioms PNDS.LANGACQUIRE097.inactive_loss_irrelevance
#print axioms PNDS.LANGACQUIRE097.acquisition_query_chain
#print axioms PNDS.LANGACQUIRE097.direct_and_acquisition_chain
