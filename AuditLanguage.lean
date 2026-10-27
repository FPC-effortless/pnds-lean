import PNDS.LanguageFPContracts
import PNDS.LanguageLayerNorm
import PNDS.LanguageLossAD
import PNDS.LanguageMemoryAD
import PNDS.LanguageMemoryStability
import PNDS.LanguageActivationAD
import PNDS.LanguageLayerNormAD
import PNDS.LanguageTensorFP

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
