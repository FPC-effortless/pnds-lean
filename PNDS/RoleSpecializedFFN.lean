import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import PNDS.SensorGraph105

/-!
ME107 role-specialized native TAC decoder contract. The inherited recurrent
scan and raw decoder are reused unchanged, while current belief, median and
width roles use separate 102→16→output tanh FFNs. This is an exact-real Lean
formulation; correspondence to trained source, finite precision, gradients and
whole-model empirical behavior remain OPEN.
-/
namespace PNDS.RoleSpecializedFFN

open PNDS.SensorGraph105

/-- A dense affine layer with source-style output-by-input indexing. -/
structure Linear (out inp : ℕ) where
  weight : Fin out → Fin inp → ℝ
  bias : Fin out → ℝ

/-- A two-layer tanh FFN; tanh is applied to the hidden layer. -/
structure TanhFFN (out : ℕ) where
  hidden : Linear 16 102
  output : Linear out 16

noncomputable def linearApply {out inp : ℕ} (p : Linear out inp) (x : Fin inp → ℝ) : Fin out → ℝ :=
  fun i => p.bias i + ∑ j, p.weight i j * x j

noncomputable def ffnApply {out : ℕ} (p : TanhFFN out) (x : Fin 102 → ℝ) : Fin out → ℝ :=
  linearApply p.output (fun h => Real.tanh (linearApply p.hidden x h))

/-- The native state transition scan is exactly the inherited ME105 scan. -/
noncomputable def inheritedScan {T : ℕ} (p : Parameters) (o : Observation T) : Trace :=
  SensorGraph105.scan p o

theorem inheritedScan_eq {T : ℕ} (p : Parameters) (o : Observation T) :
    inheritedScan p o = SensorGraph105.scan p o := rfl

/-- Independent hard-assigned output roles. Each declaration owns its own
parameter tensors; no role writes another role's output. -/
structure RoleHeads where
  reconstruction12 : TanhFFN 12
  width12 : TanhFFN 12
  median6 : TanhFFN 6

noncomputable def reconstructionOutput (h : RoleHeads) (phi : Fin 102 → ℝ) : Fin 12 → ℝ :=
  ffnApply h.reconstruction12 phi

noncomputable def widthOutput (h : RoleHeads) (phi : Fin 102 → ℝ) : Fin 12 → ℝ :=
  ffnApply h.width12 phi

noncomputable def medianOutput (h : RoleHeads) (phi : Fin 102 → ℝ) : Fin 6 → ℝ :=
  ffnApply h.median6 phi

/-- Backbone tensors retained from ME105. The three replaced dense heads do
not occur here, so they cannot contribute parameters to this role-specific
forward. `asInheritedParameters` inserts zero placeholders solely to call the
inherited scan and raw decoder, which do not read those fields. -/
structure BackboneParameters where
  encoder : Affine 48 49
  admission : Affine 13 48
  fast : Affine 176 144
  slow : Affine 176 184
  address : Affine 8 48
  content : Affine 8 48
  gates : Affine 2 224
  query : Affine 8 224
  readout1 : Affine 132 232
  readout2 : Affine 12 132
  retrieved1 : Affine 32 8
  retrieved2 : Affine 12 32
  stationEmbedding : Fin 12 → Fin 4 → ℝ
  latentCoeff : Fin 3 → Fin 6 → ℝ
  response : V6
  fitFloorRate : V6
  targetMean : V6
  targetStd : V6

def zeroAffine (m n : ℕ) : Affine m n :=
  { weight := fun _ _ => 0, bias := fun _ => 0 }

def asInheritedParameters (b : BackboneParameters) : Parameters :=
  { encoder := b.encoder, admission := b.admission, fast := b.fast, slow := b.slow,
    address := b.address, content := b.content, gates := b.gates, query := b.query,
    readout1 := b.readout1, readout2 := b.readout2,
    retrieved1 := b.retrieved1, retrieved2 := b.retrieved2,
    current := zeroAffine 12 102, width := zeroAffine 12 102,
    median := zeroAffine 6 102, stationEmbedding := b.stationEmbedding,
    latentCoeff := b.latentCoeff, response := b.response, fitFloorRate := b.fitFloorRate,
    targetMean := b.targetMean, targetStd := b.targetStd }

/-- The parent 102-feature representation and dimensions used at the decoder
boundary. This records type/shape only, not a verified source tensor mapping. -/
noncomputable def decoderInput {T : ℕ} (o : Observation T) (t : Fin T) (s : State) (rb : Fin 8 → ℝ) :
    Fin 102 → ℝ := feature102 o t s rb

/-- Current-event belief from the dedicated 12-output reconstruction role.
The first six outputs shift location by `2*tanh`; the last six define the
positive softplus variance term. The inherited missingness, age and confidence
terms are unchanged. -/
noncomputable def roleCurrentBelief {T : ℕ} (b : BackboneParameters)
    (h : RoleHeads) (o : Observation T) (t : Fin T) (s : State)
    (phi : Fin 102 → ℝ) (anchor : V6) : Belief := by
  let p := asInheritedParameters b
  let raw := reconstructionOutput h phi
  exact {
    location := fun c => anchor c + 2 * Real.tanh (raw ⟨c.val, by omega⟩)
    variance := fun c =>
      let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
      let ageHours := Real.exp (o.age t ⟨c.val, by omega⟩ * Real.log 257) - 1
      let confidence := min 1 (max 0 (s.sourceMass ⟨c.val, by omega⟩))
      let latent :=
        (softplus (p.latentCoeff 0 c) + softplus (p.latentCoeff 1 c) * (ageHours / 24)) *
          (1 + softplus (p.latentCoeff 2 c) * (1 - confidence))
      (1/400:ℝ) + (softplus (raw ⟨6+c.val, by omega⟩))^2 + missing * latent }

/-- Role-specialized protected output. Median and left/right width raw values
come only from their dedicated FFNs; floors, observed-channel clamp, and scale
transform follow the inherited equations. -/
noncomputable def roleProtect (h : RoleHeads) (phi : Fin 102 → ℝ)
    (baseLocation baseScale floor : V6) (observed : Fin 6 → Bool) : Forecast := by
  let medianRaw := medianOutput h phi
  let widthRaw := widthOutput h phi
  exact {
    location := fun c => baseLocation c + (1/2:ℝ) * Real.tanh (medianRaw c)
    leftScale := fun c => max (Real.sqrt (floor c))
      (max (if observed c then 0 else (4/5:ℝ) * baseScale c)
        (baseScale c * Real.exp (Real.log 4 * Real.tanh (widthRaw ⟨c.val, by omega⟩))))
    rightScale := fun c => max (Real.sqrt (floor c))
      (max (if observed c then 0 else (4/5:ℝ) * baseScale c)
        (baseScale c * Real.exp (Real.log 4 * Real.tanh (widthRaw ⟨6+c.val, by omega⟩))))
    floorVariance := floor, features := phi }

/-- The role-specific split scales retain the inherited source floor. These
are exact-real lower bounds; they do not assert calibration or coverage. -/
theorem roleProtect_left_floor_lower_bound (h : RoleHeads) (phi : Fin 102 → ℝ)
    (loc scale floor : V6) (obs : Fin 6 → Bool) (c : Fin 6) :
    Real.sqrt (floor c) ≤ (roleProtect h phi loc scale floor obs).leftScale c := by
  simp only [roleProtect]
  exact le_max_left _ _

theorem roleProtect_right_floor_lower_bound (h : RoleHeads) (phi : Fin 102 → ℝ)
    (loc scale floor : V6) (obs : Fin 6 → Bool) (c : Fin 6) :
    Real.sqrt (floor c) ≤ (roleProtect h phi loc scale floor obs).rightScale c := by
  simp only [roleProtect]
  exact le_max_left _ _

/-- When a channel is missing, each protected split scale is also at least the
inherited 80-percent parent-scale floor. -/
theorem roleProtect_left_missing_floor_lower_bound (h : RoleHeads) (phi : Fin 102 → ℝ)
    (loc scale floor : V6) (obs : Fin 6 → Bool) (c : Fin 6) (hc : obs c = false) :
    (4/5:ℝ) * scale c ≤ (roleProtect h phi loc scale floor obs).leftScale c := by
  simp only [roleProtect]
  rw [hc]
  exact le_trans (le_max_left _ _) (le_max_right _ _)

theorem roleProtect_right_missing_floor_lower_bound (h : RoleHeads) (phi : Fin 102 → ℝ)
    (loc scale floor : V6) (obs : Fin 6 → Bool) (c : Fin 6) (hc : obs c = false) :
    (4/5:ℝ) * scale c ≤ (roleProtect h phi loc scale floor obs).rightScale c := by
  simp only [roleProtect]
  rw [hc]
  exact le_trans (le_max_left _ _) (le_max_right _ _)

/-- Complete role-specific graph from the inherited pre-event decoder trace.
The scan and decodeRaw are inherited unchanged; all three replaced raw roles
are evaluated from role-owned FFNs. -/
noncomputable def forwardRoleFromTrace {T : ℕ} (b : BackboneParameters)
    (h : RoleHeads) (o : Observation T) (t : Fin T) (tr : Trace) : Forecast := by
  let p := asInheritedParameters b
  let rb := memRead tr.finalState.memory (normalize8 (affine p.address tr.lastEncoded))
  let phi := feature102 o t tr.finalState rb
  let raw := decodeRaw p tr
  let anchor : V6 := fun c =>
    if o.observed t ⟨c.val, by omega⟩ then safeInput o t ⟨c.val, by omega⟩
    else sourceMean tr.finalState ⟨c.val, by omega⟩
  let baseLocation : V6 := fun c => anchor c + 2 * Real.tanh (raw c)
  let inheritedScale : V6 := fun c => (1/20:ℝ) + softplus (raw ⟨6+c.val, by omega⟩)
  let belief := roleCurrentBelief b h o t tr.finalState phi anchor
  let response : V6 := fun c => 1+(1/4:ℝ)*Real.tanh (p.response c)
  let location : V6 := fun c =>
    let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
    baseLocation c + response c * (belief.location c-anchor c) * missing
  let floor : V6 := fun c =>
    let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
    let ageHours := Real.exp (o.age t ⟨c.val, by omega⟩ * Real.log 257) - 1
    p.fitFloorRate c * missing * ageHours/24
  let modelVariance : V6 := fun c =>
    let missing := if o.observed t ⟨c.val, by omega⟩ then 0 else 1
    inheritedScale c^2 + response c^2 * belief.variance c * missing
  let parentScale : V6 := fun c => Real.sqrt (max (modelVariance c) (floor c))
  exact roleProtect h phi location parentScale floor (fun c => o.observed t ⟨c.val, by omega⟩)

/-- Probability-law operations are inherited verbatim because the role FFNs
only change the Forecast parameters, not the split-normal/lognormal law. -/
noncomputable def roleRawQuantile (b : BackboneParameters) (law : NormalLaw)
    (f : Forecast) (c : Fin 6) (probability : ℝ) : ℝ :=
  SensorGraph105.rawQuantile law (asInheritedParameters b) f c probability

noncomputable def roleRawCDF (b : BackboneParameters) (law : NormalLaw)
    (f : Forecast) (c : Fin 6) (y : ℝ) : ℝ :=
  SensorGraph105.rawCDF law (asInheritedParameters b) f c y

noncomputable def roleNllElement (b : BackboneParameters) (law : NormalLaw)
    (f : Forecast) (c : Fin 6) (y : ℝ) : ℝ :=
  SensorGraph105.nllElement law (asInheritedParameters b) f c y

theorem roleRawQuantile_inherited (b : BackboneParameters) (law : NormalLaw)
    (f : Forecast) (c : Fin 6) (probability : ℝ) :
    roleRawQuantile b law f c probability =
      SensorGraph105.rawQuantile law (asInheritedParameters b) f c probability := rfl

theorem roleRawCDF_inherited (b : BackboneParameters) (law : NormalLaw)
    (f : Forecast) (c : Fin 6) (y : ℝ) :
    roleRawCDF b law f c y = SensorGraph105.rawCDF law (asInheritedParameters b) f c y := rfl

theorem roleNllElement_inherited (b : BackboneParameters) (law : NormalLaw)
    (f : Forecast) (c : Fin 6) (y : ℝ) :
    roleNllElement b law f c y =
      SensorGraph105.nllElement law (asInheritedParameters b) f c y := rfl

/-- End-to-end role-specific forecast. The native TAC chronological state scan
is inherited; `last` and `hlast` retain the inherited API contract. -/
noncomputable def forwardRole {T : ℕ} (b : BackboneParameters) (h : RoleHeads)
    (o : Observation T) (last : Fin T) (hlast : ∀ i : Fin T, i ≤ last) : Forecast := by
  have _coversHistory := hlast
  let p := asInheritedParameters b
  exact forwardRoleFromTrace b h o last (SensorGraph105.scan p o)

theorem role_output_dimensions :
    (Fintype.card (Fin 12), Fintype.card (Fin 12), Fintype.card (Fin 6)) = (12, 12, 6) := by
  norm_num

/-- Existing repository count plus the explicitly requested FFN head deltas. -/
def roleSpecializedParameterCount : ℕ :=
  100349 + (1750 - 618) + 2 * (1852 - 1236)

theorem roleSpecializedParameterCount_exact : roleSpecializedParameterCount = 102713 := by
  norm_num [roleSpecializedParameterCount]

/-! A generic finite real-vector protection operator. It is intentionally
independent of the old ME106 hard-coded parameter index and does not claim to
be an optimizer implementation or gradient of the role-specific network. -/
variable {n : ℕ}

abbrev VecR := Fin n → ℝ

def dot (x y : Fin n → ℝ) : ℝ := ∑ i, x i * y i
def normSq (x : Fin n → ℝ) : ℝ := dot x x
def sub (x y : Fin n → ℝ) : Fin n → ℝ := fun i => x i - y i
def scale (a : ℝ) (x : Fin n → ℝ) : Fin n → ℝ := fun i => a * x i

noncomputable def protectedDisplacement (g delta : Fin n → ℝ) : Fin n → ℝ :=
  if normSq g > 0 ∧ dot g delta > 0 then
    sub delta (scale (dot g delta / normSq g) g)
  else delta

theorem normSq_nonneg (x : Fin n → ℝ) : normSq x ≥ 0 := by
  unfold normSq dot
  apply Finset.sum_nonneg
  intro i hi
  exact mul_self_nonneg _

theorem dot_sub_projection_zero (g delta : Fin n → ℝ) (hn : normSq g ≠ 0) :
    dot g (sub delta (scale (dot g delta / normSq g) g)) = 0 := by
  simp only [dot, sub, scale]
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  have hs : (∑ i, g i * (dot g delta / normSq g * g i)) =
      (dot g delta / normSq g) * normSq g := by
    calc
      _ = ∑ i, (dot g delta / normSq g) * (g i * g i) := by
        apply Finset.sum_congr rfl
        intro i hi
        ring
      _ = (dot g delta / normSq g) * normSq g := by
        rw [← Finset.mul_sum]
        rfl
  have hs' : (∑ i, g i * ((∑ j, g j * delta j) / normSq g * g i)) =
      ((∑ j, g j * delta j) / normSq g) * normSq g := by
    simpa [dot] using hs
  rw [hs']
  unfold normSq dot at hn ⊢
  field_simp

theorem protected_dot_nonpos (g delta : Fin n → ℝ) (hn : normSq g > 0) :
    dot g (protectedDisplacement g delta) ≤ 0 := by
  by_cases hd : dot g delta > 0
  · have hzero := dot_sub_projection_zero g delta (ne_of_gt hn)
    simp [protectedDisplacement, hn, hd, hzero]
  · simp [protectedDisplacement, hn, hd]
    exact le_of_not_gt hd

theorem zero_protected_coordinate_unchanged (g delta : Fin n → ℝ) (i : Fin n)
    (hi : g i = 0) : protectedDisplacement g delta i = delta i := by
  by_cases h : normSq g > 0 ∧ dot g delta > 0
  · simp [protectedDisplacement, h, sub, scale, hi]
  · simp [protectedDisplacement, h]

theorem feasible_displacement_unchanged (g delta : Fin n → ℝ)
    (h : dot g delta ≤ 0) : protectedDisplacement g delta = delta := by
  have hnot : ¬ dot g delta > 0 := not_lt_of_ge h
  simp [protectedDisplacement, hnot]

end PNDS.RoleSpecializedFFN
