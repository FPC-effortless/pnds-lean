import Mathlib
import PNDS.UnifiedTACOSM

/-!
Typed, source-pinned per-variant PNDS/TAC-OSM links and extra exact-real
operator specifications. A link is NOT a full proof of the trained network.

The 17 entries are an explicitly partial historic inventory. Mathematical
readiness of the whole graph remains OPEN for every entry here.
-/

namespace PNDS.VariantContracts

inductive Kernel where
  | delta | gated | distribution | median | affine | event
  | temporal | edge | causal | window | motion | qualification
  deriving Repr, DecidableEq

structure VariantLink where
  id : String
  sourceSHA : String
  kernel : Kernel
  deriving Repr

def indexedVariants : List VariantLink :=
  [ ⟨"CORE-DIAGKEY004", "081e534a074001c6a3b2251b3ff31ad20ec4fca8", .delta⟩
  , ⟨"SYSTEM1-ME076", "9690bba1e226a8bc0de139dadb6225933e322875", .gated⟩
  , ⟨"SYSTEM1-ME077A", "971f302f2cfb5d2f43c02884f111ebbacf2bff62", .gated⟩
  , ⟨"SYSTEM1-ME079", "40433af2493e6c64f254e5ef49d3eedac448c5da", .gated⟩
  , ⟨"PHYSICAL-ME080", "afea4dd91830e32c3aec8881d5213e52d853dc2d", .distribution⟩
  , ⟨"PHYSICAL-ME094", "f1d5fd08cb89044a1838eddc55315ab11e7085f6", .median⟩
  , ⟨"PHYSICAL-ME095", "f1d5fd08cb89044a1838eddc55315ab11e7085f6", .affine⟩
  , ⟨"ENTERPRISE-ME069", "65a41dd9b8aa34bc29a570e72c9c7117abcbf74d", .event⟩
  , ⟨"ENTERPRISE-ME073", "166d60d8d897f1f8be5f48410e7d90b8af46048c", .event⟩
  , ⟨"TEMPORAL-V1", "a09b9e21c91cb6c8ed96d5846545aeffe63b3e01", .temporal⟩
  , ⟨"EDGE-V1", "a09b9e21c91cb6c8ed96d5846545aeffe63b3e01", .edge⟩
  , ⟨"COUNTERFACTUAL-V1", "a09b9e21c91cb6c8ed96d5846545aeffe63b3e01", .causal⟩
  , ⟨"ESWAM-ESWAM010", "14e8686ad78b1081532f2735bb98b31d24a8eb7e", .window⟩
  , ⟨"ESWAM-ESWAM012", "d9db3a0ac68d2bbabe3febe6cfe3d40bf322ea1d", .window⟩
  , ⟨"ESWAM-ESWAM013", "cfb2e25cb87112281243dcc48e5ebd4d52c88d83", .window⟩
  , ⟨"ESWAM-ESWAM014", "3117a77ee57d3aca4d7cc27c691ac46d6f7fca7c", .motion⟩
  , ⟨"QUAL-001", "01735e2929449b949db9c83da53fce51e75fd88b", .qualification⟩
  ]

/-- The historic indexed set is finite and has 17 entries, not the entire past. -/
theorem indexed_count : indexedVariants.length = 17 := by
  rfl

/-- Translation of a base cumulative distribution, assuming a true base median. -/
def translatedCDF (base : ℝ → ℝ) (q m y : ℝ) : ℝ :=
  base (y - m + q)

theorem translatedCDF_at_median (base : ℝ → ℝ) (q m : ℝ)
    (hBaseMedian : base q = 1 / 2) :
    translatedCDF base q m m = 1 / 2 := by
  simpa [translatedCDF] using hBaseMedian

/-- Source-valid and record-known intervals. This does not authenticate timestamps. -/
def bitemporalEligible (validStart validEnd recordStart recordEnd v k : ℝ) : Prop :=
  validStart ≤ v ∧ v < validEnd ∧ recordStart ≤ k ∧ k < recordEnd

theorem bitemporal_known_after_record_start
    (validStart validEnd recordStart recordEnd v k : ℝ)
    (h : bitemporalEligible validStart validEnd recordStart recordEnd v k) :
    recordStart ≤ k := by
  exact h.2.2.1

/-- Finite observed event window: weight normalization is NOT assumed here. -/
def weightedWindow {n : ℕ} (weights observed : Fin n → ℝ) : ℝ :=
  ∑ i, weights i * observed i

theorem weightedWindow_zero {n : ℕ} (observed : Fin n → ℝ) :
    weightedWindow (fun _ => 0) observed = 0 := by
  simp [weightedWindow]

/-- A small edge numerical discrepancy is a measured property, not an automatic theorem
    about quantizing/exporting arbitrary TAC-OSM nonlinear graphs. -/
def edgeDiscrepancy (reference quantized : ℝ) : ℝ :=
  |reference - quantized|

theorem edgeDiscrepancy_nonnegative (reference quantized : ℝ) :
    0 ≤ edgeDiscrepancy reference quantized := by
  simp [edgeDiscrepancy]

/-- Explicit external requirement: source observational conditioning is not by
    definition an identified interventional causal distribution. -/
def identifiedCausalEffect (scmAndIdentificationProved : Prop) : Prop :=
  scmAndIdentificationProved

end PNDS.VariantContracts
