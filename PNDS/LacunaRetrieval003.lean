import PNDS.LacunaSurvival001

/-! Conditional read operators only. Full source, softmax numerics and NN AD OPEN. -/
namespace PNDS.LacunaRetrieval003
open PNDS.LangMem086

def scaled (c : ℝ) (v : Key) : Key := fun i => c*v i
def squaredNorm (v : Key) : ℝ := ∑ i, v i*v i

theorem squaredNorm_scaled (c : ℝ) (v : Key) :
    squaredNorm (scaled c v) = c*c*squaredNorm v := by
  unfold squaredNorm scaled
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

def conditionalRead (old current : Memory) (address nativeQuery : Key) : Key :=
  fun i => read old address i + read current nativeQuery i

theorem native_read_decomposition (old current : Memory) (q : Key) :
    conditionalRead old current q q = read (fun i j => old i j+current i j) q := by
  funext i
  simp [conditionalRead, PNDS.LangMem086.read, add_mul, Finset.sum_add_distrib]

def weightedValue {n : Nat} (weights : Fin n → ℝ) (values : Fin n → Key) : Key :=
  fun i => ∑ j, weights j*values j i

theorem weightedValue_constant {n : Nat} (w : Fin n → ℝ) (v : Key)
    (hw : (∑ j, w j)=1) : weightedValue w (fun _ => v)=v := by
  funext i
  simp only [weightedValue, ← Finset.sum_mul, hw, one_mul]

theorem bank_budget : 8*8*4=(256 : Nat) ∧ 8+248=(256 : Nat) ∧
    1920+8+256=(2184 : Nat) := by norm_num
end PNDS.LacunaRetrieval003
#print axioms PNDS.LacunaRetrieval003.squaredNorm_scaled
#print axioms PNDS.LacunaRetrieval003.native_read_decomposition
#print axioms PNDS.LacunaRetrieval003.weightedValue_constant
