import Mathlib

import PNDS.Softmax
import PNDS.PathVerification
import PNDS.Registration
import PNDS.Scaling
import PNDS.Statistics
import PNDS.Causal

/-- Axiom/`sorry` audit. CI fails if any proof in the PNDS tree depends on
    `sorry`, and prints the axioms of each top-level theorem so that hidden
    assumptions are visible. -/

#check @PNDS.Softmax.distractorShare_le
#check @PNDS.Softmax.distractorShare_le_one
#check @PNDS.PathVerification.V_P_antitone
#check @PNDS.PathVerification.V_P_tendsto_zero
#check @PNDS.Registration.Register_iff_allGates
#check @PNDS.Registration.P_contam_eq
#check @PNDS.Statistics.three_seeds_cannot_reach_0p05
#check @PNDS.Statistics.six_seeds_reach_0p05
#check @PNDS.Statistics.minP_antitone

#print axioms PNDS.Softmax.distractorShare_le
#print axioms PNDS.Softmax.distractorShare_le_one
#print axioms PNDS.PathVerification.V_P_antitone
#print axioms PNDS.PathVerification.V_P_tendsto_zero
#print axioms PNDS.Registration.Register_iff_allGates
#print axioms PNDS.Registration.P_contam_eq
#print axioms PNDS.Statistics.three_seeds_cannot_reach_0p05
#print axioms PNDS.Statistics.six_seeds_reach_0p05
#print axioms PNDS.Statistics.minP_antitone
