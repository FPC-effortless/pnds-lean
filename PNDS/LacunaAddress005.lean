import PNDS.LacunaReadNorm004

/-! Conditional exact-real read algebra; basis construction/IEEE/full NN OPEN. -/
namespace PNDS.LacunaAddress005
open PNDS.LangMem086
open PNDS.LacunaReadNorm004

def mixedRead (f h c : Key) (a b : ℝ) : Key :=
  fun i => a*f i+b*(h i+c i)

theorem inner_add_right (x y z : Key) :
    inner x (fun i => y i+z i)=inner x y+inner x z := by
  unfold PNDS.LacunaReadNorm004.inner
  simp only [mul_add, Finset.sum_add_distrib]

theorem inner_symmetry (x y : Key) : inner x y=inner y x := by
  unfold PNDS.LacunaReadNorm004.inner
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem mixed_norm_preserved (f h t c : Key) (a b : ℝ)
    (hn : normSq t=normSq h) (hc : inner t c=inner h c)
    (hf : inner t f=inner h f) :
    normSq (mixedRead f t c a b)=normSq (mixedRead f h c a b) := by
  have hs : normSq (fun i => t i+c i)=normSq (fun i => h i+c i) :=
    same_history_norm_and_projection_preserve_total h t c hn hc
  have ht : inner f t=inner f h := by
    rw [inner_symmetry f t, inner_symmetry f h, hf]
  have hi : inner f (fun i => t i+c i)=inner f (fun i => h i+c i) := by
    rw [inner_add_right, inner_add_right, ht]
  unfold mixedRead
  rw [norm_sum (fun i => a*f i) (fun i => b*(t i+c i)),
    norm_sum (fun i => a*f i) (fun i => b*(h i+c i)),
    norm_scaled a f, norm_scaled b (fun i => t i+c i),
    norm_scaled b (fun i => h i+c i),
    inner_scaled a b f (fun i => t i+c i),
    inner_scaled a b f (fun i => h i+c i), hs, hi]

def spanRead (fixed tangent : Key) (radius : ℝ) : Key :=
  fun i => fixed i+radius*tangent i

theorem tangent_projection_preserved (fixed tangent axis : Key) (r : ℝ)
    (hz : inner tangent axis=0) :
    inner (spanRead fixed tangent r) axis=inner fixed axis := by
  unfold spanRead PNDS.LacunaReadNorm004.inner at *
  calc
    (∑ i, (fixed i+r*tangent i)*axis i)=
        (∑ i, fixed i*axis i)+r*(∑ i, tangent i*axis i) := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = ∑ i, fixed i*axis i := by rw [hz]; ring

end PNDS.LacunaAddress005
#print axioms PNDS.LacunaAddress005.mixed_norm_preserved
#print axioms PNDS.LacunaAddress005.tangent_projection_preserved
