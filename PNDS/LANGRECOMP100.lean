import Mathlib

/-! Deterministic retained-input replay and conditional parameter derivatives.
The NN graph/state proof is inherited from LANGTRAIN099. These theorems do NOT
formalize PyTorch's checkpoint engine, floating point, RNG, AdamW or peak memory.
-/
namespace PNDS.LANGRECOMP100
noncomputable section

def unroll {P S I : Type*} (step : P → S → I → S) (p : P) (s : S)
    (xs : List I) : S := xs.foldl (step p) s

theorem checkpoint_replay_exact {P S I : Type*} (step : P → S → I → S)
    (p : P) (s : S) (prefix suffix : List I) :
    unroll step p (unroll step p s prefix) suffix =
      unroll step p s (prefix ++ suffix) := by
  simp [unroll, List.foldl_append]

def directLoss {P S I : Type*} (step : P → S → I → S) (loss : P → S → ℝ)
    (s : S) (prefix suffix : List I) : P → ℝ :=
  fun p => loss p (unroll step p s (prefix ++ suffix))
def replayLoss {P S I : Type*} (step : P → S → I → S) (loss : P → S → ℝ)
    (s : S) (prefix suffix : List I) : P → ℝ :=
  fun p => loss p (unroll step p (unroll step p s prefix) suffix)

theorem parameter_loss_equal {P S I : Type*} (step : P → S → I → S)
    (loss : P → S → ℝ) (s : S) (prefix suffix : List I) :
    replayLoss step loss s prefix suffix = directLoss step loss s prefix suffix := by
  funext p
  simp [replayLoss, directLoss, checkpoint_replay_exact]

theorem conditional_derivative_equal {P S I : Type*} [NormedAddCommGroup P]
    [NormedSpace ℝ P] (step : P → S → I → S) (loss : P → S → ℝ)
    (s : S) (prefix suffix : List I) (p : P) (derivative : P →L[ℝ] ℝ)
    (h : HasFDerivAt (directLoss step loss s prefix suffix) derivative p) :
    HasFDerivAt (replayLoss step loss s prefix suffix) derivative p := by
  rw [parameter_loss_equal]
  exact h

theorem frechet_derivative_equal {P S I : Type*} [NormedAddCommGroup P]
    [NormedSpace ℝ P] (step : P → S → I → S) (loss : P → S → ℝ)
    (s : S) (prefix suffix : List I) (p : P) :
    fderiv ℝ (replayLoss step loss s prefix suffix) p =
      fderiv ℝ (directLoss step loss s prefix suffix) p := by
  rw [parameter_loss_equal]

end
end PNDS.LANGRECOMP100
#print axioms PNDS.LANGRECOMP100.checkpoint_replay_exact
#print axioms PNDS.LANGRECOMP100.parameter_loss_equal
#print axioms PNDS.LANGRECOMP100.conditional_derivative_equal
#print axioms PNDS.LANGRECOMP100.frechet_derivative_equal
