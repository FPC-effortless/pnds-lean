import Mathlib

/-!
GRAPH008: scoped exact-real finite source-slot memory mathematics.
Source: TAC-OSM research/language/flow_500k/source_slot_memory_v4.py,
owner-bound, ordinal-addressed, bounded 512-slot bank. The full Torch
program, FP32 bounds, autograd/AdamW, model behavior, and source
authentication are NOT proved by these theorems.

This bank expands state capacity. It is not 520-float compressed recall.
-/
namespace PNDS.LanguageSourceSlots008
noncomputable section

abbrev Store := Fin 512 → ℝ

def writeSlot (store : Store) (address : Fin 512) (value : ℝ) : Store :=
  fun i => if i = address then value else store i

def readSlot (store : Store) (address : Fin 512) : ℝ :=
  store address

def authorizedCarry (requestOwner stateOwner : String) : Prop :=
  requestOwner ≠ "" ∧ requestOwner = stateOwner

/-- The currently admitted source overwrites its designated exact slot. -/
theorem source_write_exact (store : Store) (i : Fin 512) (v : ℝ) :
    readSlot (writeSlot store i v) i = v := by
  simp [readSlot, writeSlot]

/-- A different live slot is algebraically unaffected by any new write. -/
theorem source_write_no_cross_slot_interference
    (store : Store) (written other : Fin 512) (v : ℝ)
    (hne : other ≠ written) :
    readSlot (writeSlot store written v) other = readSlot store other := by
  simp [readSlot, writeSlot, hne]

/-- Exact source value survives one unrelated write. Induction over a
    sequence of distinct slot addresses yields finite-horizon survival. -/
theorem source_read_survives_different_write
    (store : Store) (source next : Fin 512) (v w : ℝ)
    (hne : source ≠ next) :
    readSlot (writeSlot (writeSlot store source v) next w) source = v := by
  simp [readSlot, writeSlot, hne]

/-- Store remains bounded if each source latent is bounded. -/
theorem source_write_bound
    (store : Store) (address : Fin 512) (v : ℝ)
    (hs : ∀ i, |store i| ≤ 1) (hv : |v| ≤ 1) :
    ∀ i, |writeSlot store address v i| ≤ 1 := by
  intro i
  by_cases hi : i = address
  · subst i
    simpa [writeSlot] using hv
  · simpa [writeSlot, hi] using hs i

theorem owner_mismatch_rejected (current stored : String)
    (h : current ≠ stored) : ¬ authorizedCarry current stored := by
  intro ha
  exact h ha.2

/-- Source-known historical queries have to refer to strictly prior
    available positions; the caller must enforce the hypothesis. -/
theorem strict_past_is_not_future (source current : ℕ)
    (hpast : source < current) : ¬ current ≤ source := by
  omega

/-- Two ordinals fewer than 512 apart cannot alias modulo 512.
    This is what permits a finite 512-slot live window. -/
theorem live_window_has_no_address_collision (i j : ℕ)
    (hij : i < j) (hgap : j - i < 512) :
    i % 512 ≠ j % 512 := by
  omega


/-- The probability-level retrieval mixture cannot create negative mass
    when the base and copy distributions have nonnegative coordinates. -/
theorem retrieval_mixture_nonnegative
    (p q lam : ℝ)
    (hp : 0 ≤ p) (hq : 0 ≤ q) (hl0 : 0 ≤ lam) (hl1 : lam ≤ 1) :
    0 ≤ (1-lam)*p+lam*q := by
  exact add_nonneg (mul_nonneg (sub_nonneg.mpr hl1) hp) (mul_nonneg hl0 hq)

/-- Mixing two complete normalized categorical distributions preserves
    their total probability mass at one; this is a scalar total-mass
    contract, not a full network softmax/source equivalence theorem. -/
theorem retrieval_mixture_normalized
    (totalBase totalCopy lam : ℝ)
    (hb : totalBase = 1) (hc : totalCopy = 1) :
    (1-lam)*totalBase+lam*totalCopy = 1 := by
  rw [hb,hc]
  ring

end
end PNDS.LanguageSourceSlots008
