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
import PNDS.System1Head
import PNDS.System1VelocityAnchor
import PNDS.System1GradientStop
import PNDS.System1ProjectedBudget
import PNDS.System1FixedSlow
import PNDS.System1MultiTimescale

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
