import PNDS.HistoryQuery016
import Mathlib

import PNDS.Softmax
import PNDS.PathVerification
import PNDS.Registration
import PNDS.Scaling
import PNDS.Statistics
import PNDS.Causal
import PNDS.UnifiedTACOSM
import PNDS.TACOSMVariantContracts

/- Axiom audit. CI fails if any proof in the PNDS tree depends on an admitted
   goal, and prints the axioms of each top-level theorem so that hidden
   assumptions are visible. -/

#check @PNDS.Softmax.distractorShare_le
#check @PNDS.Softmax.distractorShare_le_one
#check @PNDS.PathVerification.V_P_antitone
#check @PNDS.PathVerification.V_P_tendsto_zero
#check @PNDS.Registration.Register_iff_allGates
#check @PNDS.Registration.bayes_inversion
#check @PNDS.Scaling.distractors_in_topK_eq_recall
#check @PNDS.Scaling.distractors_in_topK_eq
#check @PNDS.Scaling.distractors_in_topK_le_of_recall
#check @PNDS.Scaling.distractors_in_topK_ge_of_missed
#check @PNDS.Scaling.recall_one_iff
#check @PNDS.Scaling.distractors_in_topK_eq_iff_recall_one
#check @PNDS.Statistics.three_seeds_cannot_reach_0p05
#check @PNDS.Statistics.six_seeds_reach_0p05
#check @PNDS.Statistics.minP_oneSided_antitone
#check @PNDS.Statistics.minP_twoSided_antitone

#print axioms PNDS.Softmax.distractorShare_le
#print axioms PNDS.Softmax.distractorShare_le_one
#print axioms PNDS.PathVerification.V_P_antitone
#print axioms PNDS.PathVerification.V_P_tendsto_zero
#print axioms PNDS.Registration.Register_iff_allGates
#print axioms PNDS.Registration.bayes_inversion
#print axioms PNDS.Scaling.distractors_in_topK_eq_recall
#print axioms PNDS.Scaling.distractors_in_topK_eq
#print axioms PNDS.Scaling.distractors_in_topK_le_of_recall
#print axioms PNDS.Scaling.distractors_in_topK_ge_of_missed
#print axioms PNDS.Scaling.recall_one_iff
#print axioms PNDS.Scaling.distractors_in_topK_eq_iff_recall_one
#print axioms PNDS.Statistics.three_seeds_cannot_reach_0p05
#print axioms PNDS.Statistics.six_seeds_reach_0p05
#print axioms PNDS.Statistics.minP_oneSided_antitone
#print axioms PNDS.Statistics.minP_twoSided_antitone

/-! New PNDS foundation v1 operator contracts: explicit kernel axiom audit. -/
#check @PNDS.UnifiedTACOSM.gated_interval
#check @PNDS.UnifiedTACOSM.gated_error
#check @PNDS.UnifiedTACOSM.run_compose
#check @PNDS.UnifiedTACOSM.compose_assoc
#check @PNDS.UnifiedTACOSM.source_end_mass_normalized
#check @PNDS.UnifiedTACOSM.source_end_mass_nonnegative
#check @PNDS.UnifiedTACOSM.quadratic_zero
#check @PNDS.UnifiedTACOSM.synthetic_not_real_qualification
#check @PNDS.UnifiedTACOSM.development_not_independent_qualification
#print axioms PNDS.UnifiedTACOSM.gated_interval
#print axioms PNDS.UnifiedTACOSM.gated_error
#print axioms PNDS.UnifiedTACOSM.run_compose
#print axioms PNDS.UnifiedTACOSM.compose_assoc
#print axioms PNDS.UnifiedTACOSM.source_end_mass_normalized
#print axioms PNDS.UnifiedTACOSM.source_end_mass_nonnegative
#print axioms PNDS.UnifiedTACOSM.quadratic_zero
#print axioms PNDS.UnifiedTACOSM.synthetic_not_real_qualification
#print axioms PNDS.UnifiedTACOSM.development_not_independent_qualification

/-! Per-variant dual-formulation exact-real and indexing checks; source parity OPEN. -/
#check @PNDS.VariantContracts.indexed_count
#check @PNDS.VariantContracts.translatedCDF_at_median
#check @PNDS.VariantContracts.bitemporal_known_after_record_start
#check @PNDS.VariantContracts.weightedWindow_zero
#check @PNDS.VariantContracts.edgeDiscrepancy_nonnegative
#print axioms PNDS.VariantContracts.indexed_count
#print axioms PNDS.VariantContracts.translatedCDF_at_median
#print axioms PNDS.VariantContracts.bitemporal_known_after_record_start
#print axioms PNDS.VariantContracts.weightedWindow_zero
#print axioms PNDS.VariantContracts.edgeDiscrepancy_nonnegative

#check PNDS.HistoryQuery016.historicalLoss
#check PNDS.HistoryQuery016.forecastLoss
#print axioms PNDS.HistoryQuery016.historicalLoss
#print axioms PNDS.HistoryQuery016.forecastLoss
