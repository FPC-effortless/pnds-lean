import Mathlib

/-!
LANGHEAD093 scoped exact-real mathematical facts for one coordinate of the
new TAC-OSM direct tied byte embedding prediction head.

Original TAC source commit de954a177aa5bf5bcd162acf7bd5797fb5bc75cf
and source research/mathematical-engineering/langhead093/model093.py.

THIS IS ONLY A SCALAR DECODER FACT. The complete original 100k TAC recurrent
graph, 32D finite-precision PyTorch AD, embedding training, source-to-implementation
equivalence, actual model dynamics and empirical language generalization remain OPEN.
-/
namespace PNDS.LANGHEAD093

/-- Direct Euclidean distance contribution for a single embedding coordinate. -/
def negDistance (gain x y : ℝ) : ℝ := -gain * (y - x)^2

/-- Tied dot-product plus candidate-embedding squared-norm contribution. -/
def tiedDotNorm (gain x y : ℝ) : ℝ := gain * (2*x*y - y*y)

/-- Any reference-token squared term cancels in a difference of candidate scores. -/
theorem candidate_difference_identity (gain x y z : ℝ) :
    negDistance gain x y - negDistance gain x z =
    tiedDotNorm gain x y - tiedDotNorm gain x z := by
  dsimp [negDistance, tiedDotNorm]
  ring

/-- A candidate-independent constant leaves every pairwise logit difference unchanged. -/
theorem common_shift_invariance (a b c : ℝ) :
    (a+c)-(b+c)=a-b := by
  ring

end PNDS.LANGHEAD093
