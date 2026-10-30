import PNDS.LacunaRetrieval003

/-! Exact-real algebra only. Numerical projection, full NN IEEE/AD remain OPEN. -/
namespace PNDS.LacunaReadNorm004
open PNDS.LangMem086
abbrev normSq := PNDS.LacunaRetrieval003.squaredNorm
def inner (x y : Key) : ℝ := ∑ i, x i*y i

theorem norm_sum (x y : Key) :
    normSq (fun i => x i+y i)=normSq x+normSq y+2*inner x y := by
  unfold normSq PNDS.LacunaRetrieval003.squaredNorm inner
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem inner_scaled (a b : ℝ) (x y : Key) :
    inner (fun i => a*x i) (fun i => b*y i)=a*b*inner x y := by
  unfold inner
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem norm_scaled (a : ℝ) (x : Key) :
    normSq (fun i => a*x i)=a*a*normSq x := by
  simpa only [PNDS.LacunaRetrieval003.scaled] using
    PNDS.LacunaRetrieval003.squaredNorm_scaled a x

theorem unit_tangent_radius (a r : ℝ) (u z : Key)
    (hu : normSq u=1) (hz : normSq z=1) (huz : inner u z=0) :
    normSq (fun i => a*u i+r*z i)=a*a+r*r := by
  rw [norm_sum, norm_scaled, norm_scaled, inner_scaled, hu, hz, huz]
  ring

theorem same_history_norm_and_projection_preserve_total (h candidate c : Key)
    (hn : normSq candidate=normSq h) (hp : inner candidate c=inner h c) :
    normSq (fun i => candidate i+c i)=normSq (fun i => h i+c i) := by
  rw [norm_sum, norm_sum, hn, hp]

theorem both_norms_fix_projection (h candidate c : Key)
    (hn : normSq candidate=normSq h)
    (ht : normSq (fun i => candidate i+c i)=normSq (fun i => h i+c i)) :
    inner candidate c=inner h c := by
  rw [norm_sum, norm_sum, hn] at ht
  linarith

end PNDS.LacunaReadNorm004
#print axioms PNDS.LacunaReadNorm004.unit_tangent_radius
#print axioms PNDS.LacunaReadNorm004.same_history_norm_and_projection_preserve_total
#print axioms PNDS.LacunaReadNorm004.both_norms_fix_projection
