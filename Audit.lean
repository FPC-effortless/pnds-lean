import Mathlib

import PNDS.Softmax
import PNDS.PathVerification
import PNDS.Registration
import PNDS.Scaling
import PNDS.Statistics
import PNDS.Causal
import PNDS.UnifiedTACOSM
import PNDS.TACOSMVariantContracts
import PNDS.NativeMemory
import PNDS.NativeInvariant
import PNDS.ParallelMemory
import PNDS.DualModeMemory
import PNDS.EntityAdmission
import PNDS.TimescaleEvidence
import PNDS.JointAffineTAC
import PNDS.RetentionBudget
import PNDS.ConstraintAdapter
import PNDS.ParallelResidualHead
import PNDS.ParallelTimeAge
import PNDS.SourceAlignedMemory

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


/-! Native memory exact-real source-algebra audit, not full PyTorch source proof. -/
#check @PNDS.NativeMemory.arbitrary_query_decomposition
#check @PNDS.NativeMemory.orthogonal_query_retains_only_decay
#check @PNDS.NativeMemory.unit_address_self_read
#check @PNDS.NativeMemory.unit_address_residual
#check @PNDS.NativeMemory.read_only_no_mutation
#check @PNDS.NativeMemory.shared_key_exact_collision_impossible
#check @PNDS.NativeMemory.projected_exact_write_counterexample
#check @PNDS.NativeMemory.fixed_gate_incremental_sensitivity
#print axioms PNDS.NativeMemory.arbitrary_query_decomposition
#print axioms PNDS.NativeMemory.orthogonal_query_retains_only_decay
#print axioms PNDS.NativeMemory.unit_address_self_read
#print axioms PNDS.NativeMemory.unit_address_residual
#print axioms PNDS.NativeMemory.read_only_no_mutation
#print axioms PNDS.NativeMemory.shared_key_exact_collision_impossible
#print axioms PNDS.NativeMemory.projected_exact_write_counterexample
#print axioms PNDS.NativeMemory.fixed_gate_incremental_sensitivity

#check @PNDS.NativeMemory.row_query_decomposition
#check @PNDS.NativeMemory.row_unit_key_residual
#print axioms PNDS.NativeMemory.row_query_decomposition
#print axioms PNDS.NativeMemory.row_unit_key_residual

/-! Optional civilian constraint adapter: exact scalar QP and planar sampled barrier. -/
#check @PNDS.ConstraintAdapter.projection_feasible
#check @PNDS.ConstraintAdapter.projection_optimal
#check @PNDS.ConstraintAdapter.dual_lower_nonnegative
#check @PNDS.ConstraintAdapter.kkt_stationarity
#check @PNDS.ConstraintAdapter.kkt_complementary
#check @PNDS.ConstraintAdapter.planar_sample_identity
#check @PNDS.ConstraintAdapter.planar_sample_safe
#print axioms PNDS.ConstraintAdapter.projection_feasible
#print axioms PNDS.ConstraintAdapter.projection_optimal
#print axioms PNDS.ConstraintAdapter.dual_lower_nonnegative
#print axioms PNDS.ConstraintAdapter.kkt_stationarity
#print axioms PNDS.ConstraintAdapter.kkt_complementary
#print axioms PNDS.ConstraintAdapter.planar_sample_identity
#print axioms PNDS.ConstraintAdapter.planar_sample_safe

#check @PNDS.ConstraintAdapter.planar_interval_safe
#print axioms PNDS.ConstraintAdapter.planar_interval_safe

/-! Native full-state conditional bounded forward envelope and read-only noncontraction. -/
#check @PNDS.NativeInvariant.mixed_unit_box
#check @PNDS.NativeInvariant.step_preserves_envelope
#check @PNDS.NativeInvariant.repeated_forward_envelope
#check @PNDS.NativeInvariant.no_strict_contraction_of_read_only_memory
#print axioms PNDS.NativeInvariant.mixed_unit_box
#print axioms PNDS.NativeInvariant.step_preserves_envelope
#print axioms PNDS.NativeInvariant.repeated_forward_envelope
#print axioms PNDS.NativeInvariant.no_strict_contraction_of_read_only_memory

#check @PNDS.NativeInvariant.projectedMagnitude_bounds
#print axioms PNDS.NativeInvariant.projectedMagnitude_bounds

#check @PNDS.NativeInvariant.saturated_budget_forbids_nonzero_orthogonal_write
#print axioms PNDS.NativeInvariant.saturated_budget_forbids_nonzero_orthogonal_write

/-! Conditional time-parallel 8×8 affine-memory scan; NOT the full native recurrence. -/
#check @PNDS.ParallelMemory.apply_then
#check @PNDS.ParallelMemory.then_assoc
#check @PNDS.ParallelMemory.apply_identity
#check @PNDS.ParallelMemory.then_identity
#check @PNDS.ParallelMemory.identity_then
#check @PNDS.ParallelMemory.chronological_chunks
#check @PNDS.ParallelMemory.zero_apply
#check @PNDS.ParallelMemory.read_only_no_mutation
#print axioms PNDS.ParallelMemory.apply_then
#print axioms PNDS.ParallelMemory.then_assoc
#print axioms PNDS.ParallelMemory.apply_identity
#print axioms PNDS.ParallelMemory.then_identity
#print axioms PNDS.ParallelMemory.identity_then
#print axioms PNDS.ParallelMemory.chronological_chunks
#print axioms PNDS.ParallelMemory.zero_apply
#print axioms PNDS.ParallelMemory.read_only_no_mutation

#check @PNDS.ParallelMemory.bounded_projection_free_budget
#print axioms PNDS.ParallelMemory.bounded_projection_free_budget

#check @PNDS.DualModeMemory.scan_matches_stream_from
#check @PNDS.DualModeMemory.scan_matches_stream
#check @PNDS.DualModeMemory.stream_resume
#check @PNDS.DualModeMemory.read_only_stream
#check @PNDS.DualModeMemory.append_basis_sum
#check @PNDS.DualModeMemory.context_decomposes_through_statistic
#print axioms PNDS.DualModeMemory.scan_matches_stream
#print axioms PNDS.DualModeMemory.stream_resume
#print axioms PNDS.DualModeMemory.read_only_stream
#print axioms PNDS.DualModeMemory.append_basis_sum

#check @PNDS.EntityAdmission.unrelated_slot_unchanged
#check @PNDS.EntityAdmission.replacement_does_not_copy_evicted_content
#check @PNDS.EntityAdmission.existing_slot_interpolation
#check @PNDS.EntityAdmission.read_only_unchanged
#check @PNDS.EntityAdmission.capacity_full_no_new_slot
#print axioms PNDS.EntityAdmission.unrelated_slot_unchanged
#print axioms PNDS.EntityAdmission.replacement_does_not_copy_evicted_content
#print axioms PNDS.EntityAdmission.existing_slot_interpolation
#print axioms PNDS.EntityAdmission.capacity_full_no_new_slot

#check @PNDS.TimescaleEvidence.bound_unit_interval
#check @PNDS.TimescaleEvidence.state_difference_linear
#check @PNDS.TimescaleEvidence.same_observation_nonexpansive
#check @PNDS.TimescaleEvidence.then_assoc
#check @PNDS.TimescaleEvidence.update_is_affine
#check @PNDS.TimescaleEvidence.one_step_perturbation_bound
#print axioms PNDS.TimescaleEvidence.bound_unit_interval
#print axioms PNDS.TimescaleEvidence.same_observation_nonexpansive
#print axioms PNDS.TimescaleEvidence.then_assoc
#print axioms PNDS.TimescaleEvidence.one_step_perturbation_bound

#check @PNDS.JointAffineTAC.apply_then
#check @PNDS.JointAffineTAC.then_assoc
#check @PNDS.JointAffineTAC.fold_matches_stream
#check @PNDS.JointAffineTAC.stream_resume
#print axioms PNDS.JointAffineTAC.apply_then
#print axioms PNDS.JointAffineTAC.then_assoc
#print axioms PNDS.JointAffineTAC.fold_matches_stream
#print axioms PNDS.JointAffineTAC.stream_resume

#check @PNDS.JointAffineTAC.same_decoder_after_joint_parallel_or_stream
#print axioms PNDS.JointAffineTAC.same_decoder_after_joint_parallel_or_stream

#check @PNDS.RetentionBudget.keep_range
#check @PNDS.RetentionBudget.written_range
#check @PNDS.RetentionBudget.alpha_lower
#check @PNDS.RetentionBudget.alpha_budget
#check @PNDS.RetentionBudget.radius_four_next_bounded
#print axioms PNDS.RetentionBudget.alpha_lower
#print axioms PNDS.RetentionBudget.alpha_budget
#print axioms PNDS.RetentionBudget.radius_four_next_bounded

/-! Exact-real scalar post-scan residual readout; NOT full neural/IEEE proof. -/
#check @PNDS.ParallelResidualHead.zero_gate
#check @PNDS.ParallelResidualHead.zero_innovation
#check @PNDS.ParallelResidualHead.postscan_state_unchanged
#check @PNDS.ParallelResidualHead.additive_difference
#check @PNDS.ParallelResidualHead.bounded_prediction
#print axioms PNDS.ParallelResidualHead.zero_gate
#print axioms PNDS.ParallelResidualHead.postscan_state_unchanged
#print axioms PNDS.ParallelResidualHead.bounded_prediction

/-! PAR-TIME-001: narrowly scoped exact-real clock and prior-write-age contracts. -/
#check @PNDS.ParallelTimeAge.monotone_clock
#check @PNDS.ParallelTimeAge.refusal_keeps_timestamp
#check @PNDS.ParallelTimeAge.admission_updates_timestamp
#check @PNDS.ParallelTimeAge.no_write_age_advances
#check @PNDS.ParallelTimeAge.admitted_write_resets_next_age
#check @PNDS.ParallelTimeAge.age_nonnegative
#check @PNDS.ParallelTimeAge.index_strictly_increases
#print axioms PNDS.ParallelTimeAge.monotone_clock
#print axioms PNDS.ParallelTimeAge.no_write_age_advances
#print axioms PNDS.ParallelTimeAge.age_nonnegative
#print axioms PNDS.ParallelTimeAge.index_strictly_increases

/-! PAR-MEM-010: exact-real local operator contracts only; no IEEE/AD/history skill. -/
#check @PNDS.SourceAlignedMemory.shared_source_event_address
#check @PNDS.SourceAlignedMemory.reject_wrong_source_owner
#check @PNDS.SourceAlignedMemory.reject_future_source_ordinal
#check @PNDS.SourceAlignedMemory.declined_write_is_identity
#check @PNDS.SourceAlignedMemory.admitted_write_is_corrective
#check @PNDS.SourceAlignedMemory.historical_query_does_not_write
#check @PNDS.SourceAlignedMemory.no_source_admission_preserves_all_heads
#print axioms PNDS.SourceAlignedMemory.shared_source_event_address
#print axioms PNDS.SourceAlignedMemory.reject_wrong_source_owner
#print axioms PNDS.SourceAlignedMemory.reject_future_source_ordinal
#print axioms PNDS.SourceAlignedMemory.declined_write_is_identity
#print axioms PNDS.SourceAlignedMemory.admitted_write_is_corrective
#print axioms PNDS.SourceAlignedMemory.historical_query_does_not_write
#print axioms PNDS.SourceAlignedMemory.no_source_admission_preserves_all_heads
