import Mathlib

import PNDS.Softmax
import PNDS.PathVerification
import PNDS.Registration
import PNDS.Scaling
import PNDS.Statistics
import PNDS.Causal
import PNDS.UnifiedTACOSM
import PNDS.TACOSMVariantContracts
import PNDS.LatentFusion001
import PNDS.LatentFusion002
import PNDS.LatentFusion003
import PNDS.LatentFusion004
import PNDS.LatentFusion007
import PNDS.LatentFusion008
import PNDS.LatentFusion010

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


/-! LATFUSE001: operator-only exact-real audit, full graph still OPEN. -/
#check @PNDS.LatentFusion001.compose_run
#check @PNDS.LatentFusion001.compose_assoc
#check @PNDS.LatentFusion001.delta_no_write
#check @PNDS.LatentFusion001.delta_full_write
#check @PNDS.LatentFusion001.unaddressed_is_unchanged
#check @PNDS.LatentFusion001.selected_zero_write
#print axioms PNDS.LatentFusion001.compose_run
#print axioms PNDS.LatentFusion001.compose_assoc
#print axioms PNDS.LatentFusion001.delta_no_write
#print axioms PNDS.LatentFusion001.delta_full_write
#print axioms PNDS.LatentFusion001.unaddressed_is_unchanged
#print axioms PNDS.LatentFusion001.selected_zero_write


/-! LATFUSE002 local exact-real operator audits; no IEEE/full neural proof. -/
#check @PNDS.LatentFusion002.slow_difference
#check @PNDS.LatentFusion002.slow_identity_at_one
#check @PNDS.LatentFusion002.slow_no_memory_at_zero
#check @PNDS.LatentFusion002.deltaErase_zero
#check @PNDS.LatentFusion002.deltaErase_one
#check @PNDS.LatentFusion002.belief_zero_preserves_query
#check @PNDS.LatentFusion002.retained_scalar_count
#check @PNDS.LatentFusion002.exact_nominal_state_bytes
#print axioms PNDS.LatentFusion002.slow_difference
#print axioms PNDS.LatentFusion002.exact_nominal_state_bytes


/-! LATFUSE003 source-pinned operator-only proof audit; whole-graph FP32, AD and calibration OPEN. -/
#check @PNDS.LatentFusion003.partition_uses_all_256_matrix_values
#check @PNDS.LatentFusion003.exact_original_state_floats
#check @PNDS.LatentFusion003.exact_original_state_bytes
#check @PNDS.LatentFusion003.zero_write_retains_only_alpha
#check @PNDS.LatentFusion003.unselected_head_preserves_old
#check @PNDS.LatentFusion003.headDelta_affine_in_old
#check @PNDS.LatentFusion003.difference_transport
#check @PNDS.LatentFusion003.unselected_index_isolation
#print axioms PNDS.LatentFusion003.headDelta_affine_in_old
#print axioms PNDS.LatentFusion003.difference_transport
#print axioms PNDS.LatentFusion003.unselected_index_isolation
#print axioms PNDS.LatentFusion003.exact_original_state_bytes


/-! LATFUSE004 local two/four-head FP32 capacity and soft/hard exact-real edits. -/
#check @PNDS.LatentFusion004.two_head_preserves_matrix
#check @PNDS.LatentFusion004.four_head_preserves_matrix
#check @PNDS.LatentFusion004.equal_matrix_capacities
#check @PNDS.LatentFusion004.original_carried_values
#check @PNDS.LatentFusion004.original_carried_fp32_nominal_bytes
#check @PNDS.LatentFusion004.untouched_when_unselected
#check @PNDS.LatentFusion004.updates_when_selected
#check @PNDS.LatentFusion004.softBlend_zero
#check @PNDS.LatentFusion004.softBlend_one
#check @PNDS.LatentFusion004.softBlend_difference
#print axioms PNDS.LatentFusion004.two_head_preserves_matrix
#print axioms PNDS.LatentFusion004.original_carried_fp32_nominal_bytes
#print axioms PNDS.LatentFusion004.untouched_when_unselected
#print axioms PNDS.LatentFusion004.softBlend_difference


/-! LATFUSE007 source-only historic event QUERY abstract interface; finite precision/AD still OPEN. -/
#check @PNDS.LatentFusion007.original_memory_still_256_cells
#check @PNDS.LatentFusion007.original_recurrent_bytes
#check @PNDS.LatentFusion007.original_parameter_budget
#check @PNDS.LatentFusion007.abstract_truth_label_does_not_enter_query
#print axioms PNDS.LatentFusion007.original_recurrent_bytes
#print axioms PNDS.LatentFusion007.original_parameter_budget
#print axioms PNDS.LatentFusion007.abstract_truth_label_does_not_enter_query


/-! LATFUSE008 paired source-historical readout exact budget and abstract noassoc-path invariance. -/
#check @PNDS.LatentFusion008.identical_live_params
#check @PNDS.LatentFusion008.original_live_params
#check @PNDS.LatentFusion008.identical_original_state_bytes
#check @PNDS.LatentFusion008.abstract_nonassociative_reader_ignores_past_matrix
#print axioms PNDS.LatentFusion008.identical_live_params
#print axioms PNDS.LatentFusion008.identical_original_state_bytes
#print axioms PNDS.LatentFusion008.abstract_nonassociative_reader_ignores_past_matrix


/-! LATFUSE010 only abstract original exact-real scalar source-write and difference. -/
#check @PNDS.LatentFusion010.single_step_write_bypass_difference
#check @PNDS.LatentFusion010.common_source_suffix_difference
#check @PNDS.LatentFusion010.source_write_common_value_cancels
#check @PNDS.LatentFusion010.original_state_budget_unchanged
#print axioms PNDS.LatentFusion010.single_step_write_bypass_difference
#print axioms PNDS.LatentFusion010.common_source_suffix_difference
#print axioms PNDS.LatentFusion010.original_state_budget_unchanged
