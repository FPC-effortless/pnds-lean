import Mathlib

/-!
TAC-LANG-GRAPH-006-v2: exact-real diagnosis and candidate repair.
Source-specific *scoped* math, NOT full Torch/IEEE/AD equivalence.

Original symmetric-Laplacian equilibrium:
  (5/4 I + 1/4 L) C = G + 1/4 C_old
If the graph's Laplacian has zero total, the unweighted sum/mean
eliminates L. A slot-specific weighted read can observe graph changes.
A two-slot concrete witness demonstrates the distinction. The actual
model has six slots; a zero-sum hypothesis is explicitly required where
the full 6x6 symmetry is not discharged.
-/
namespace PNDS.LanguageWaveGraphRepair
noncomputable section
open Finset

abbrev Concepts := Fin 6 → ℝ

def slotSum (c : Concepts) : ℝ := ∑ i : Fin 6, c i

def slotMean (c : Concepts) : ℝ := slotSum c / 6

def graphRead (weights c : Concepts) : ℝ := ∑ i : Fin 6, weights i * c i

/-- The source six-slot graph is symmetric after pairwise averaging.
This explicit source-shaped Laplacian uses weighted node degrees minus
cross-slot weighted messages, at each latent coordinate. -/
def symmetricLaplacian6
    (edge : Fin 6 → Fin 6 → ℝ) (concept : Concepts) (i : Fin 6) : ℝ :=
  (∑ j : Fin 6, edge i j) * concept i -
      ∑ j : Fin 6, edge i j * concept j

/-- A symmetric edge matrix implies exact zero-total graph diffusion.
Unlike a free zero-sum premise, this discharges the cancellation from
the actual six-node *symmetric* graph construction. -/
theorem symmetric_laplacian6_zero_total
    (edge : Fin 6 → Fin 6 → ℝ) (concept : Concepts)
    (hsym : ∀ i j : Fin 6, edge i j = edge j i) :
    (∑ i : Fin 6, symmetricLaplacian6 edge concept i) = 0 := by
  have hs :
      (∑ i : Fin 6, ∑ j : Fin 6, edge i j * concept i) =
      (∑ i : Fin 6, ∑ j : Fin 6, edge i j * concept j) := by
    calc
      _ = ∑ i : Fin 6, ∑ j : Fin 6, edge j i * concept j := by
        rw [Finset.sum_comm]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        rw [hsym j i]
  unfold symmetricLaplacian6
  rw [Finset.sum_sub_distrib]
  have hfirst :
      (∑ i : Fin 6, (∑ j : Fin 6, edge i j) * concept i) =
      (∑ i : Fin 6, ∑ j : Fin 6, edge i j * concept i) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.sum_mul]
  rw [hfirst, hs]
  ring

/-- Aggregating the equilibrium removes a zero-sum graph interaction.
The total-zero condition follows from a true symmetric graph Laplacian,
but no equivalence to the whole Torch implementation is claimed. -/
theorem unweighted_equilibrium_cancels_graph
    (sumC sumG sumOld sumLap : ℝ)
    (heq : (5/4 : ℝ)*sumC+(1/4 : ℝ)*sumLap =
            sumG+(1/4 : ℝ)*sumOld)
    (hzero : sumLap=0) :
    sumC = (4/5 : ℝ)*sumG+(1/5 : ℝ)*sumOld := by
  rw [hzero] at heq
  ring_nf at heq ⊢
  linarith

/-- Two source-known concept slots with RHS=(1,0).
For 1.25I+.25L and L=[[1,-1],[-1,1]], equilibrium is
(24/35, 4/35); for L=0 it is (4/5,0). Both have identical mean. -/
theorem two_slot_graph_solution :
    (3/2 : ℝ)*(24/35)-(1/4 : ℝ)*(4/35)=1 ∧
    -(1/4 : ℝ)*(24/35)+(3/2 : ℝ)*(4/35)=0 := by
  norm_num

theorem two_slot_old_mean_cancellation :
    ((24/35 : ℝ)+(4/35))/2 =
      ((4/5 : ℝ)+0)/2 := by
  norm_num

/-- Nonuniform weights (3/4,1/4) retain a nonzero graph effect:
graph read =19/35, no-graph read =3/5; difference 2/35.
This explicit counterexample fixes the structural impossibility of
the uniform mean, but does not prove nonzero gradients for all inputs. -/
theorem two_slot_weighted_graph_is_observable :
    (3/4 : ℝ)*(24/35)+(1/4 : ℝ)*(4/35) = 19/35 ∧
    (3/4 : ℝ)*(4/5)+(1/4 : ℝ)*0 = 3/5 ∧
    (19/35 : ℝ) ≠ 3/5 := by
  norm_num

theorem read_from_constant_slots_is_invariant
    (weights : Concepts) (value : ℝ)
    (hunit : slotSum weights=1) :
    graphRead weights (fun _ => value) = value := by
  unfold graphRead
  rw [← Finset.sum_mul]
  simpa [slotSum] using congrArg (fun z : ℝ => z*value) hunit

/-- A strict-past global query is legitimate even when the local chunk
has just ONE token. Its validity depends on absolute time, not length. -/
theorem carried_one_token_old_query
    (start lag : ℕ) (hpast : lag ≤ start) (hlag : 0 < lag) :
    start-lag < start+1 := by
  omega

theorem resumed_query_eligible
    (start offset lag : ℕ)
    (h : lag ≤ start+offset) :
    (start+offset)-lag+lag=start+offset := by
  omega

/-- A normalized key and bounded value are required separately. With
write= (1-alpha)*raw, the scalar affine recurrence is bounded under the
specified contraction/convexity premises. NOT a global matrix-Frobenius
norm or finite precision proof. -/
def coupledScalarStep (old value alpha raw : ℝ) : ℝ :=
  alpha*old+(1-alpha)*raw*value

theorem coupled_scalar_preserves_bound
    (old value alpha raw bound : ℝ)
    (hb : 0 ≤ bound) (ha0 : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hr0 : 0 ≤ raw) (hr1 : raw ≤ 1)
    (ho : |old| ≤ bound) (hv : |value| ≤ bound) :
    |coupledScalarStep old value alpha raw| ≤ bound := by
  have ho' := abs_le.mp ho
  have hv' := abs_le.mp hv
  have hcoef : 0 ≤ (1-alpha)*raw := mul_nonneg (by linarith) hr0
  have hsum : alpha+(1-alpha)*raw ≤ 1 := by
    have h := mul_nonneg (sub_nonneg.mpr ha1) (sub_nonneg.mpr hr1)
    nlinarith
  have h0 := mul_le_mul_of_nonneg_left ho'.1 ha0
  have h1 := mul_le_mul_of_nonneg_left ho'.2 ha0
  have h2 := mul_le_mul_of_nonneg_left hv'.1 hcoef
  have h3 := mul_le_mul_of_nonneg_left hv'.2 hcoef
  apply abs_le.mpr
  unfold coupledScalarStep
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.mpr hsum) hb]

/-- Bounded state is not recoverable memory: zero retention erases all
distinct prior scalar states, even with a coupled write. -/
theorem coupled_scalar_can_erase_all_old_states
    (old1 old2 value raw : ℝ) :
    coupledScalarStep old1 value 0 raw =
      coupledScalarStep old2 value 0 raw := by
  simp [coupledScalarStep]

end
end PNDS.LanguageWaveGraphRepair
