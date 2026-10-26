import PNDS.RoleSpecializedFFN

/-!
ME109 residual decoder boundary. All parent affine heads remain present.
Availability is appended explicitly; normalized ages already occupy features
88..93. This exact-real module proves preservation at this boundary, conditional
on using the same deterministic inherited decoder and the unchanged scan.
It does not formalize the entire source decoder, its derivatives or IEEE rules.
-/
namespace PNDS.ResidualRoleFFN

open PNDS.RoleSpecializedFFN

structure Residual (out : ℕ) where
  input : Linear 12 108
  output : Linear out 12

noncomputable def context (phi : Fin 102 → ℝ) (available : Fin 6 → Bool) :
    Fin 108 → ℝ := fun i =>
  if h : i.val < 102 then phi ⟨i.val, h⟩
  else if available ⟨i.val - 102, by omega⟩ then 1 else 0

noncomputable def residualApply {out : ℕ} (r : Residual out) (x : Fin 108 → ℝ) :
    Fin out → ℝ :=
  linearApply r.output (fun j => Real.tanh (linearApply r.input x j))

noncomputable def corrected {out : ℕ} (parent : Linear out 102) (r : Residual out)
    (phi : Fin 102 → ℝ) (available : Fin 6 → Bool) : Fin out → ℝ :=
  fun i => linearApply parent phi i + residualApply r (context phi available) i

def zeroOutput {out : ℕ} (r : Residual out) : Prop :=
  (∀ i j, r.output.weight i j = 0) ∧ (∀ i, r.output.bias i = 0)

theorem residual_zero_output {out : ℕ} (r : Residual out) (h : zeroOutput r)
    (x : Fin 108 → ℝ) : residualApply r x = fun _ => 0 := by
  funext i
  simp [residualApply, linearApply, h.1, h.2]

theorem corrected_parent_identity {out : ℕ} (parent : Linear out 102)
    (r : Residual out) (h : zeroOutput r) (phi : Fin 102 → ℝ)
    (available : Fin 6 → Bool) : corrected parent r phi available = linearApply parent phi := by
  funext i
  simp [corrected, residual_zero_output r h]

structure ParentHeads where
  current : Linear 12 102
  width : Linear 12 102
  median : Linear 6 102

structure ResidualHeads where
  current : Residual 12
  width : Residual 12
  median : Residual 6

abbrev RawOutputs := (Fin 12 → ℝ) × (Fin 12 → ℝ) × (Fin 6 → ℝ)

noncomputable def parentRaw (p : ParentHeads) (phi : Fin 102 → ℝ) : RawOutputs :=
  (linearApply p.current phi, linearApply p.width phi, linearApply p.median phi)

noncomputable def residualRaw (p : ParentHeads) (r : ResidualHeads)
    (phi : Fin 102 → ℝ) (available : Fin 6 → Bool) : RawOutputs :=
  (corrected p.current r.current phi available,
   corrected p.width r.width phi available,
   corrected p.median r.median phi available)

theorem all_raw_parent_identity (p : ParentHeads) (r : ResidualHeads)
    (hc : zeroOutput r.current) (hw : zeroOutput r.width) (hm : zeroOutput r.median)
    (phi : Fin 102 → ℝ) (available : Fin 6 → Bool) :
    residualRaw p r phi available = parentRaw p phi := by
  simp [residualRaw, parentRaw, corrected_parent_identity _ _ hc,
    corrected_parent_identity _ _ hw, corrected_parent_identity _ _ hm]

/-- Deterministic-decoder congruence, explicitly conditional on an identical
decoder and state. This is not a claimed full-source decoder formalization. -/
theorem final_forecast_parent_identity {Forecast : Type} (decode : RawOutputs → Forecast)
    (p : ParentHeads) (r : ResidualHeads)
    (hc : zeroOutput r.current) (hw : zeroOutput r.width) (hm : zeroOutput r.median)
    (phi : Fin 102 → ℝ) (available : Fin 6 → Bool) :
    decode (residualRaw p r phi available) = decode (parentRaw p phi) := by
  rw [all_raw_parent_identity p r hc hw hm]

/-- Dedicated width and median residuals do not write current raw outputs. -/
theorem current_role_independent (p : ParentHeads) (r s : ResidualHeads)
    (h : r.current = s.current) (phi : Fin 102 → ℝ) (available : Fin 6 → Bool) :
    (residualRaw p r phi available).1 = (residualRaw p s phi available).1 := by
  simp [residualRaw, h]

/-- This is a weight-dependent role bound: no fixed bound on a learned raw
correction is asserted. Hidden activation boundedness is an explicit premise. -/
theorem residual_coordinate_bound {out : ℕ} (r : Residual out) (x : Fin 108 → ℝ)
    (i : Fin out) (h : ∀ j, |Real.tanh (linearApply r.input x j)| ≤ 1) :
    |residualApply r x i| ≤ |r.output.bias i| + ∑ j, |r.output.weight i j| := by
  unfold residualApply linearApply
  apply le_trans (abs_add _ _)
  apply add_le_add_left
  apply le_trans (Finset.abs_sum_le_sum_abs _ _)
  apply Finset.sum_le_sum
  intro j _
  rw [abs_mul]
  simpa using mul_le_mul_of_nonneg_left (h j) (abs_nonneg (r.output.weight i j))

def parameterCount : ℕ := 100349 + 3 * (108 * 12 + 12) + (12 + 1) * (12 + 12 + 6)

theorem parameter_count_exact : parameterCount = 104663 := by norm_num [parameterCount]

end PNDS.ResidualRoleFFN
