import Mathlib

/-! Partial086 typed memory/read and exact count contracts.
Full compact source neural graph, Float32 and automatic differentiation remain
OPEN. No theorem establishes empirical recall or trained language quality.
-/
namespace PNDS.LangMem086
abbrev Key := Fin 8 → ℝ
abbrev Memory := Fin 8 → Fin 8 → ℝ

def read (m : Memory) (q : Key) : Key := fun i => ∑ j, m i j * q j

def residualUpdate (m : Memory) (k v : Key) (rho eta : ℝ) : Memory :=
  fun i j => rho*m i j + eta*(v i-rho*read m k i)*k j

def mixRead (a b : Memory) (q : Key) (w : ℝ) : Key :=
  fun i => w*read a q i+(1-w)*read b q i

theorem mixRead_no_external_address (a b : Memory) (q : Key) (w : ℝ) :
    mixRead a b q w = fun i => w*read a q i+(1-w)*read b q i := rfl

theorem update_read_unit (m : Memory) (k v : Key) (rho eta : ℝ)
    (h : (∑ j, k j*k j)=1) :
    read (residualUpdate m k v rho eta) k =
      fun i => (1-eta)*rho*read m k i+eta*v i := by
  funext i
  simp only [read, residualUpdate, add_mul, mul_assoc,
    Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [h]
  ring

def activeParameters (ffnWidth : Nat) : Nat := 46189+578*ffnWidth

theorem registered_active_count : activeParameters 94 = 100521 := by norm_num [activeParameters]
theorem within_required_band : 95000 ≤ activeParameters 94 ∧ activeParameters 94 ≤ 105000 := by
  norm_num [activeParameters]

def stateFloats : Nat := 64+64+128+8+48+32+96+8+32

theorem registered_state_payload : 4*stateFloats=1920 := by norm_num [stateFloats]
theorem fixed_code_buffer_payload : 256*8*4=(8192 : Nat) := by norm_num

def predecisionPrediction {Byte State Output : Type}
    (start : State) (advance : State → Byte → State) (predict : State → Output)
    (observed : List Byte) : Output := predict (observed.foldl advance start)

theorem appended_future_excluded {Byte State Output : Type}
    (start : State) (advance : State → Byte → State) (predict : State → Output)
    (observed : List Byte) (future : Byte) :
    predecisionPrediction start advance predict ((observed++[future]).take observed.length) =
    predecisionPrediction start advance predict observed := by simp

end PNDS.LangMem086
#print axioms PNDS.LangMem086.update_read_unit
#print axioms PNDS.LangMem086.appended_future_excluded
