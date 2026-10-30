import PNDS.LangMem087Historical

/-! Partial exact-real P0 formulas, not a learned retrieval/quality theorem. -/
namespace PNDS.LacunaSurvival001
open PNDS.LangMem086

def transport (m : Memory) (k : Key) (rho eta : ℝ) : Memory :=
  PNDS.LangMem087.oldComponent m k rho eta

theorem transport_add (a b : Memory) (k : Key) (rho eta : ℝ) :
    transport (fun i j => a i j+b i j) k rho eta =
      fun i j => transport a k rho eta i j+transport b k rho eta i j := by
  funext i j
  simp only [transport, PNDS.LangMem087.oldComponent, PNDS.LangMem086.read,
    add_mul, Finset.sum_add_distrib]
  ring

theorem original_update_decomposition (m : Memory) (k v : Key) (rho eta : ℝ) :
    residualUpdate m k v rho eta =
      fun i j => transport m k rho eta i j+
        PNDS.LangMem087.currentComponent k v eta i j :=
  PNDS.LangMem087.residual_decomposition m k v rho eta

def erase (m : Memory) (e : Key) (gamma : ℝ) : Memory :=
  fun i j => m i j-gamma*PNDS.LangMem086.read m e i*e j

theorem erase_zero (m : Memory) (e : Key) : erase m e 0=m := by
  funext i j
  simp [erase]

abbrev ProbVector := Fin 256 → ℝ

def mixProbability (p r : ProbVector) (g : ℝ) : ProbVector :=
  fun i => (1-g)*p i+g*r i

theorem mix_zero (p r : ProbVector) : mixProbability p r 0=p := by
  funext i
  simp [mixProbability]

theorem mix_nonnegative (p r : ProbVector) (g : ℝ)
    (hp : ∀ i, 0 ≤ p i) (hr : ∀ i, 0 ≤ r i) (hg : 0 ≤ g) (hg1 : g ≤ 1) :
    ∀ i, 0 ≤ mixProbability p r g i := by
  intro i
  exact add_nonneg (mul_nonneg (sub_nonneg.mpr hg1) (hp i)) (mul_nonneg hg (hr i))

theorem mix_normalized (p r : ProbVector) (g : ℝ)
    (hp : (∑ i, p i)=1) (hr : (∑ i, r i)=1) :
    (∑ i, mixProbability p r g i)=1 := by
  simp only [mixProbability, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hp, hr]
  ring

theorem registered_payloads :
    24+25*9+7=256 ∧ 1920+8+1928+2=3858 ∧
    3858≤4096 ∧ 896/4=(224 : Nat) := by norm_num
end PNDS.LacunaSurvival001
#print axioms PNDS.LacunaSurvival001.transport_add
#print axioms PNDS.LacunaSurvival001.mix_normalized
