import PNDS.LanguageWave500K
import PNDS.LanguageFPContracts
import PNDS.LanguageLayerNorm
import PNDS.LanguageLossAD
import PNDS.LanguageMemoryAD
import PNDS.LanguageMemoryStability
import PNDS.LanguageActivationAD
import PNDS.LanguageLayerNormAD
import PNDS.LanguageTensorFP
import PNDS.LanguageKeySensitivity
import PNDS.LanguageRepairCandidates
import PNDS.LanguageSelectiveRetrieval
import PNDS.LanguageMultiLagObjective
import PNDS.LanguageGradientGuard

#print axioms PNDS.LanguageCategorical.Z_pos
#print axioms PNDS.LanguageCategorical.probability_pos
#print axioms PNDS.LanguageCategorical.probability_sum_one
#print axioms PNDS.LanguageCategorical.probability_le_one
#print axioms PNDS.LanguageCategorical.nll_nonnegative
#print axioms PNDS.LanguageCategorical.bounded_write_range
#print axioms PNDS.LanguageCategorical.scalar_delta_affine
#print axioms PNDS.LanguageCategorical.scalar_delta_difference
#print axioms PNDS.LanguageCategorical.meanStep_equiv
#print axioms PNDS.LanguageCategorical.modulo_cycle

#print axioms PNDS.LanguageCategorical.routed_write_budget
#print axioms PNDS.LanguageCategorical.meanStep_unit_box

#print axioms PNDS.LanguageFullGraph.step_owner
#print axioms PNDS.LanguageFullGraph.step_counter
#print axioms PNDS.LanguageFullGraph.advance_append
#print axioms PNDS.LanguageFullGraph.token_probability_sum_one
#print axioms PNDS.LanguageFullGraph.prefix_causal
#print axioms PNDS.LanguageFullGraph.advance_owner

#print axioms PNDS.LanguageCategorical.partition_shift
#print axioms PNDS.LanguageCategorical.probability_shift
#print axioms PNDS.LanguageCategorical.nll_shift

#print axioms PNDS.LanguageFullGraph.source_parameter_count_exact
#print axioms PNDS.LanguageFullGraph.streaming_float_count_exact

#print axioms PNDS.LanguageFPContracts.rounded_source_sum_error
#print axioms PNDS.LanguageFPContracts.sum_with_basis_input_error
#print axioms PNDS.LanguageFPContracts.normalized_sum_error
#print axioms PNDS.LanguageFPContracts.ideal_sum_l1_bound

#print axioms PNDS.LanguageLayerNorm.variance248_nonnegative
#print axioms PNDS.LanguageLayerNorm.layernorm_denominator_positive
#print axioms PNDS.LanguageLayerNorm.deployed_epsilon_denominator_positive

#print axioms PNDS.LanguageCategorical.nll_equals_log_partition_minus_logit
#print axioms PNDS.LanguageCategorical.shifted_stable_nll_correct

#print axioms PNDS.LanguageLossAD.partition_hasDerivAt
#print axioms PNDS.LanguageLossAD.log_partition_hasDerivAt
#print axioms PNDS.LanguageLossAD.logitVelocity_as_fraction
#print axioms PNDS.LanguageLossAD.nll_hasDerivAt

#print axioms PNDS.LanguageMemoryAD.aligned_write_refines_kernel
#print axioms PNDS.LanguageMemoryAD.native_write_refines_kernel
#print axioms PNDS.LanguageMemoryAD.source_kernel_coordinate_jvp
#print axioms PNDS.LanguageMemoryAD.native_kernel_coordinate_jvp

#print axioms PNDS.LanguageMemoryStability.aligned_key_energy
#print axioms PNDS.LanguageMemoryStability.source_retention_range
#print axioms PNDS.LanguageMemoryStability.source_beta_range
#print axioms PNDS.LanguageMemoryStability.source_aligned_write_nonexpansive

#print axioms PNDS.LanguageActivationAD.sigmoid_hasDerivAt
#print axioms PNDS.LanguageActivationAD.silu_hasDerivAt
#print axioms PNDS.LanguageActivationAD.scaled_silu_hasDerivAt
#print axioms PNDS.LanguageActivationAD.gated_silu_modulation_hasDerivAt

#print axioms PNDS.LanguageLayerNormAD.normalized_feature_hasDerivAt
#print axioms PNDS.LanguageLayerNormAD.mean248_hasDerivAt

#print axioms PNDS.LanguageLayerNormAD.variance248_hasDerivAt
#print axioms PNDS.LanguageLayerNormAD.norm248_coordinate_hasDerivAt

#print axioms PNDS.LanguageTensorFP.vector_sum_error
#print axioms PNDS.LanguageTensorFP.rounded_sum_error
#print axioms PNDS.LanguageTensorFP.source_row_read_error

#print axioms PNDS.LanguageKeySensitivity.small_key_exact
#print axioms PNDS.LanguageKeySensitivity.small_key_error
#print axioms PNDS.LanguageKeySensitivity.key_amplification_witness

#print axioms PNDS.LanguageRepairCandidates.smooth_denom_pos
#print axioms PNDS.LanguageRepairCandidates.smooth_key_zero
#print axioms PNDS.LanguageRepairCandidates.smooth_denom_ge_tau
#print axioms PNDS.LanguageRepairCandidates.smooth_key_amplitude_bound
#print axioms PNDS.LanguageRepairCandidates.ordinal_reconstructed
#print axioms PNDS.LanguageRepairCandidates.era_residue_injective
#print axioms PNDS.LanguageRepairCandidates.next_cycle_increments_era
#print axioms PNDS.LanguageRepairCandidates.first_era_untouched
#print axioms PNDS.LanguageRepairCandidates.era_fraction_strict_cycle

#print axioms PNDS.LanguageSelectiveRetrieval.source_at_trained_lag
#print axioms PNDS.LanguageSelectiveRetrieval.fallback_for_unqualified_lag
#print axioms PNDS.LanguageSelectiveRetrieval.fallback_is_strictly_positive
#print axioms PNDS.LanguageSelectiveRetrieval.fallback_normalization
#print axioms PNDS.LanguageSelectiveRetrieval.convex_history_baseline_floor
#print axioms PNDS.LanguageSelectiveRetrieval.convex_history_normalization

#print axioms PNDS.LanguageMultiLagObjective.auxiliary_budget_preserved
#print axioms PNDS.LanguageMultiLagObjective.paired_objective_average
#print axioms PNDS.LanguageMultiLagObjective.paired_objective_hasDerivAt
#print axioms PNDS.LanguageMultiLagObjective.paired_objective_nonnegative
#print axioms PNDS.LanguageMultiLagObjective.paired_historical_queries_are_past

#print axioms PNDS.LanguageGradientGuard.conflict_overlap_becomes_zero
#print axioms PNDS.LanguageGradientGuard.nonconflicting_overlap_is_unchanged
#print axioms PNDS.LanguageGradientGuard.guarded_overlap_nonnegative
#print axioms PNDS.LanguageGradientGuard.guarded_total_alignment

/-! Prospective 500K wave/CED typed and scalar contract, partial only. -/
#check @PNDS.LanguageWave500K.candidate_within_new_band
#check @PNDS.LanguageWave500K.persistent_state_520
#check @PNDS.LanguageWave500K.right_affine_associative
#check @PNDS.LanguageWave500K.count_strictly_advances
#print axioms PNDS.LanguageWave500K.right_affine_associative

#check @PNDS.LanguageWave500K.wave_rotation_norm_sq
#check @PNDS.LanguageWave500K.damped_wave_nonexpansive
#check @PNDS.LanguageWave500K.pair_concept_sum_preserved
#check @PNDS.LanguageWave500K.pair_concept_difference
#check @PNDS.LanguageWave500K.pair_concept_energy_nonincrease
#print axioms PNDS.LanguageWave500K.wave_rotation_norm_sq
#print axioms PNDS.LanguageWave500K.damped_wave_nonexpansive
#print axioms PNDS.LanguageWave500K.pair_concept_energy_nonincrease

/-! 500K historical-source old-state contraction and overwrite limits. -/
#check @PNDS.LanguageHistory500K.known_past_ordinal_correct
#check @PNDS.LanguageHistory500K.old_state_difference_one_step
#check @PNDS.LanguageHistory500K.old_state_difference_nonexpansive
#check @PNDS.LanguageHistory500K.full_corrective_write_erases_old
#check @PNDS.LanguageHistory500K.old_state_collision
#print axioms PNDS.LanguageHistory500K.old_state_difference_nonexpansive
#print axioms PNDS.LanguageHistory500K.old_state_collision
