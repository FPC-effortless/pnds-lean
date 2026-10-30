import PNDS.LangMem086Autonomous

/-! Exact-real partial087 read attribution. Entire neural graph Float32/AD,
semantic usefulness and independent source qualification remain OPEN. -/
namespace PNDS.LangMem087
open PNDS.LangMem086

def oldComponent (m : Memory) (k : Key) (rho eta : ℝ) : Memory :=
  fun i j => rho*m i j-eta*rho*read m k i*k j

def currentComponent (k v : Key) (eta : ℝ) : Memory :=
  fun i j => eta*v i*k j

theorem residual_decomposition (m : Memory) (k v : Key) (rho eta : ℝ) :
    residualUpdate m k v rho eta =
      fun i j => oldComponent m k rho eta i j+currentComponent k v eta i j := by
  funext i j
  simp only [residualUpdate, oldComponent, currentComponent]
  ring

theorem zero_old_retains_current_write (k v : Key) (rho eta : ℝ) :
    residualUpdate (fun _ _ => 0) k v rho eta = currentComponent k v eta := by
  funext i j
  simp [residualUpdate, read, currentComponent]

theorem duration_target_counts :
    160*4*256=163840 ∧ 640*4*256=655360 ∧ 1280*4*256=(1310720 : Nat) := by norm_num
end PNDS.LangMem087
#print axioms PNDS.LangMem087.residual_decomposition
#print axioms PNDS.LangMem087.zero_old_retains_current_write
