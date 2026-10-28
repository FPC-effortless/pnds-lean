import Mathlib
import PNDS.LanguageWave500K
import PNDS.LanguageHistory500K
import PNDS.LanguageWaveGraphRepair
import PNDS.LanguageMemory007
import PNDS.LanguageSourceSlots008
import PNDS.LanguageRetrieval009
import PNDS.LanguageQuery010
import PNDS.LanguageReplay011

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
import PNDS.SourceAlignedVerification
import PNDS.SourceAlignedModel
import PNDS.SourceAlignedScan
import PNDS.SourceAlignedExecution
import PNDS.SourceAlignedDifferentiation
import PNDS.SourceAlignedRoundoff
import PNDS.SourceAlignedPrograms
import PNDS.LanguageCategorical
import PNDS.LanguageFullGraph
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

/-- The version of the document this formalization targets. -/
def docVersion : String := "v0.2"

/-!
# Root file: the PNDS formalization.

Imports the modules corresponding to the sections of
`PNDS_Mathematical_Research_Representation_v0.2.docx` that contain provable content.

Sections imported: §5 (`Softmax`), §10 (`PathVerification`), §12/§14
(`Registration`), §18 (`Scaling`), §20 (`Statistics`).

Sections *not* imported as theorems: §9 (`Causal`) is imported in `Causal.lean` as an
explicit specification gap, and §24 is theorem-candidates only.
-/

/-! Foundation v1 is versioned separately from the original v0.2 document. -/
def unifiedFoundationVersion : String := "1.0.0"
