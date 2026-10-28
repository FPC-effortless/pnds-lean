import Mathlib

/-!
GRAPH009: exact-real, scoped algebraic contracts for a FIT-calibrated
Dirichlet-smoothed source count cache and normalized mixture.
Source counterpart: TAC-OSM/bayesian_cache_scan_v5.py and
source_calibrated_tac_v5.py. These do not prove full PyTorch/FP32/AD,
source-authentication, state-queue refinement or language superiority.
-/
namespace PNDS.LanguageRetrieval009
noncomputable section

def posterior (count n tau prior : ℝ) : ℝ :=
  (count+tau*prior)/(n+tau)

def fusion (base retrieved weight : ℝ) : ℝ :=
  (1-weight)*base+weight*retrieved

def rightAffineCompose (left right : ℝ × ℝ) : ℝ × ℝ :=
  (left.1*right.1,left.2*right.1+right.2)

theorem posterior_denom_pos (n tau : ℝ)
    (hn : 0 ≤ n) (ht : 0 < tau) : 0 < n+tau :=
  add_pos_of_nonneg_of_pos hn ht

theorem posterior_numerator_nonneg (count tau prior : ℝ)
    (hc : 0 ≤ count) (ht : 0 ≤ tau) (hp : 0 ≤ prior) :
    0 ≤ count+tau*prior :=
  add_nonneg hc (mul_nonneg ht hp)

theorem posterior_nonnegative (count n tau prior : ℝ)
    (hc : 0 ≤ count) (hn : 0 ≤ n)
    (ht : 0 < tau) (hp : 0 ≤ prior) :
    0 ≤ posterior count n tau prior := by
  unfold posterior
  exact div_nonneg (posterior_numerator_nonneg count tau prior hc (le_of_lt ht) hp)
    (le_of_lt (posterior_denom_pos n tau hn ht))

/-- The count sum equals n and prior categorical mass equals one;
    this is total-mass algebra, not a proof the Python count queue implements
    those premises for arbitrary corrupted states. -/
theorem posterior_total_mass (n tau priorMass : ℝ)
    (hn : 0 ≤ n) (ht : 0 < tau) (hp : priorMass = 1) :
    (n+tau*priorMass)/(n+tau) = 1 := by
  rw [hp, mul_one]
  exact div_self (ne_of_gt (posterior_denom_pos n tau hn ht))

theorem no_hit_returns_base (base retrieved : ℝ) :
    fusion base retrieved 0 = base := by
  simp [fusion]

theorem fusion_nonnegative (base retrieved weight : ℝ)
    (hb : 0 ≤ base) (hr : 0 ≤ retrieved)
    (hw0 : 0 ≤ weight) (hw1 : weight ≤ 1) :
    0 ≤ fusion base retrieved weight := by
  unfold fusion
  exact add_nonneg (mul_nonneg (sub_nonneg.mpr hw1) hb)
    (mul_nonneg hw0 hr)

theorem fusion_total_mass (baseMass retrievalMass weight : ℝ)
    (hb : baseMass = 1) (hr : retrievalMass = 1) :
    fusion baseMass retrievalMass weight = 1 := by
  rw [hb, hr]
  simp [fusion]
  ring

/-- Exact-real right-affine transition composition used by structured
    scan algebra: (P_a,Q_a) ⊗ (P_b,Q_b) = (P_a P_b,Q_a P_b+Q_b).
    This proves associativity for scalar coefficients, not the entire model. -/
theorem right_affine_associative (a b c : ℝ × ℝ) :
    rightAffineCompose (rightAffineCompose a b) c =
      rightAffineCompose a (rightAffineCompose b c) := by
  apply Prod.ext
  · simp only [rightAffineCompose]
    ring
  · simp only [rightAffineCompose]
    ring

end
end PNDS.LanguageRetrieval009
