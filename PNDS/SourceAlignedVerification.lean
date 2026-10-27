import PNDS.SourceAlignedMemory
import PNDS.ParallelMemory
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases

/-!
PAR-MEM-010, exact-real verification beyond the original local contracts.
All schedules are fixed before the memory scan. No assertion here identifies
real arithmetic with IEEE execution, or an authorized ordinal with an observed
record. See foundation/v1/PAR-MEM-010-VERIFICATION.md for the source boundary.
-/
namespace PNDS.SourceAlignedVerification

open PNDS.SourceAlignedMemory
open PNDS.ParallelMemory

def rowP (key : Address) (alpha beta : ℝ) : Matrix (Fin 8) (Fin 8) ℝ :=
  fun i j => alpha * ((if i = j then 1 else 0) - beta * key i * key j)

def rowQ (key : Address) (value beta : ℝ) : Address :=
  fun j => beta * value * key j

/-- The right-affine convention agrees with the source corrective write.
In particular alpha multiplies the OLD prediction inside the residual. -/
theorem row_affine_correspondence (row key : SourceRow)
    (value alpha beta : ℝ) (j : Fin 8) :
    (∑ i, row i * rowP key alpha beta i j) + rowQ key value beta j =
      writeRow row key value alpha beta true j := by
  have hexpand : (∑ i, row i * rowP key alpha beta i j) =
      alpha * row j - alpha * beta * rowRead row key * key j := by
    simp only [rowP]
    calc
      _ = (∑ i : Fin 8, ((if i = j then alpha * row i else 0) -
          alpha * beta * (row i * key i) * key j)) := by
        apply Finset.sum_congr rfl
        intro i _
        split_ifs <;> ring
      _ = _ := by
        simp [Finset.sum_sub_distrib, Finset.mul_sum, Finset.sum_mul, rowRead]
  rw [hexpand]
  simp only [rowQ, writeRow, Bool.true_eq, ↓reduceIte]
  ring

/-- Change of a read at ANY query, including competing old addresses. -/
theorem read_after_write (row key query : SourceRow) (value alpha beta : ℝ) :
    rowRead (writeRow row key value alpha beta true) query =
      alpha * rowRead row query +
        beta * (value - alpha * rowRead row key) * rowRead key query := by
  simp only [rowRead, writeRow, Bool.true_eq, ↓reduceIte]
  simp_rw [add_mul]
  rw [Finset.sum_add_distrib]
  simp only [mul_assoc, ← Finset.mul_sum]

/-- Key normalization is an explicit hypothesis, not assumed for zero or
epsilon-normalized keys. -/
theorem unit_key_residual (row key : SourceRow) (value alpha beta : ℝ)
    (hunit : rowRead key key = 1) :
    value - rowRead (writeRow row key value alpha beta true) key =
      (1 - beta) * (value - alpha * rowRead row key) := by
  rw [read_after_write, hunit]
  ring

/-- Normalization with epsilon can yield ||k||<1; this general identity
remains valid in that case and at k=0. -/
theorem arbitrary_key_residual (row key : SourceRow) (value alpha beta : ℝ) :
    value - rowRead (writeRow row key value alpha beta true) key =
      (1 - beta * rowRead key key) * (value - alpha * rowRead row key) := by
  rw [read_after_write]
  ring

theorem unit_key_error_nonincrease (row key : SourceRow) (value alpha beta : ℝ)
    (hunit : rowRead key key = 1) (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1) :
    |value - rowRead (writeRow row key value alpha beta true) key| ≤
      |value - alpha * rowRead row key| := by
  rw [unit_key_residual row key value alpha beta hunit, abs_mul]
  have hh : |1 - beta| ≤ 1 := by rw [abs_of_nonneg (by linarith)]; linarith
  nlinarith [abs_nonneg (value - alpha * rowRead row key)]

/-- Orthogonal old keys avoid correction interference but still undergo
forgetting by alpha. Soft routing alone does not give orthogonality. -/
theorem orthogonal_read_retention (row key query : SourceRow)
    (value alpha beta : ℝ) (horth : rowRead key query = 0) :
    rowRead (writeRow row key value alpha beta true) query =
      alpha * rowRead row query := by
  rw [read_after_write, horth]
  ring

/-- Two value halves, each with eight physical rows, cover all 16 rows. -/
def rowIndex (block : Fin 2) (r : Fin 8) : Fin 16 :=
  ⟨8 * block.val + r.val, by omega⟩

theorem rowIndex_bijective : Function.Bijective (fun br : Fin 2 × Fin 8 =>
    rowIndex br.1 br.2) := by
  constructor
  · intro a b h
    have hv := congrArg Fin.val h
    apply Prod.ext <;> apply Fin.ext <;> simp only [rowIndex] at hv <;> omega
  · intro v
    refine ⟨(⟨v.val / 8, by omega⟩, ⟨v.val % 8, by omega⟩), ?_⟩
    apply Fin.ext
    simp only [rowIndex]
    omega

abbrev BlockState := Fin 2 → Fin 2 → Matrix (Fin 8) (Fin 8) ℝ
abbrev BlockSchedule := Fin 2 → Fin 2 → AffinePair 8

def splitMemory (m : SourceMatrix) : BlockState :=
  fun h b r j => m h (rowIndex b r) j

def writeMatrix (m : SourceMatrix) (keys : Fin 2 → Address)
    (value : Fin 16 → ℝ) (alpha beta : Fin 2 → ℝ) (admitted : Bool) :
    SourceMatrix :=
  fun h r => writeRow (m h r) (keys h) (value r) (alpha h) (beta h) admitted

def blockSchedule (keys : Fin 2 → Address) (value : Fin 16 → ℝ)
    (alpha beta : Fin 2 → ℝ) (admitted : Bool) : BlockSchedule :=
  fun h b => if admitted then
    ⟨rowP (keys h) (alpha h) (beta h),
     fun r j => rowQ (keys h) (value (rowIndex b r)) (beta h) j⟩
    else identity 8

def blockApply (a : BlockSchedule) (s : BlockState) : BlockState :=
  fun h b => applyPair (a h b) (s h b)

def blockThen (a b : BlockSchedule) : BlockSchedule :=
  fun h r => thenPair (a h r) (b h r)

def blockIdentity : BlockSchedule := fun _ _ => identity 8

/-- All four 8x8 blocks implement the two 16x8 source matrices, including
the non-admission identity. This is an index/equation theorem, not a proof
of Torch's concrete permute/reshape implementation. -/
theorem four_block_write_correspondence (m : SourceMatrix)
    (keys : Fin 2 → Address) (value : Fin 16 → ℝ)
    (alpha beta : Fin 2 → ℝ) (admitted : Bool) :
    blockApply (blockSchedule keys value alpha beta admitted) (splitMemory m) =
      splitMemory (writeMatrix m keys value alpha beta admitted) := by
  funext h b r j
  cases admitted with
  | false => simp [blockApply, blockSchedule, apply_identity, splitMemory,
      writeMatrix, writeRow]
  | true =>
    exact row_affine_correspondence (m h (rowIndex b r)) (keys h)
      (value (rowIndex b r)) (alpha h) (beta h) j

theorem block_apply_then (a b : BlockSchedule) (s : BlockState) :
    blockApply (blockThen a b) s = blockApply b (blockApply a s) := by
  funext h r
  exact apply_then (a h r) (b h r) (s h r)

theorem block_then_assoc (a b c : BlockSchedule) :
    blockThen (blockThen a b) c = blockThen a (blockThen b c) := by
  funext h r
  exact then_assoc (a h r) (b h r) (c h r)

/-- Every chronological prefix, from arbitrary initial state and accumulator.
This is the scan law for noncommuting right-affine memory operators. -/
theorem block_fold_matches_stream (events : List BlockSchedule)
    (acc : BlockSchedule) (s : BlockState) :
    blockApply (events.foldl blockThen acc) s =
      events.foldl (fun state event => blockApply event state) (blockApply acc s) := by
  induction events generalizing acc with
  | nil => rfl
  | cons e es ih =>
    simpa only [List.foldl_cons, block_apply_then] using ih (blockThen acc e)

theorem block_resume (first second : List BlockSchedule) (s : BlockState) :
    (first ++ second).foldl (fun state event => blockApply event state) s =
      second.foldl (fun state event => blockApply event state)
        (first.foldl (fun state event => blockApply event state) s) := by
  simp only [List.foldl_append]

inductive ScheduleTree where
  | empty : ScheduleTree
  | leaf : BlockSchedule → ScheduleTree
  | fork : ScheduleTree → ScheduleTree → ScheduleTree

def treeEvents : ScheduleTree → List BlockSchedule
  | .empty => []
  | .leaf a => [a]
  | .fork a b => treeEvents a ++ treeEvents b

def treeSummary : ScheduleTree → BlockSchedule
  | .empty => blockIdentity
  | .leaf a => a
  | .fork a b => blockThen (treeSummary a) (treeSummary b)

/-- Every binary grouping of contiguous events has the streaming semantics.
The tree may be balanced, unbalanced, or contain empty padding subtrees. -/
theorem ordered_tree_matches_stream (tree : ScheduleTree) (s : BlockState) :
    blockApply (treeSummary tree) s =
      (treeEvents tree).foldl (fun old event => blockApply event old) s := by
  induction tree generalizing s with
  | empty =>
    funext h r
    exact apply_identity (s h r)
  | leaf a => rfl
  | fork a b iha ihb =>
    simp only [treeSummary, treeEvents, block_apply_then, List.foldl_append]
    rw [iha, ihb]

theorem same_decoder_after_ordered_tree {Output : Type} (tree : ScheduleTree)
    (s : BlockState) (decoder : BlockState → Output) :
    decoder (blockApply (treeSummary tree) s) =
      decoder ((treeEvents tree).foldl (fun old event => blockApply event old) s) := by
  exact congrArg decoder (ordered_tree_matches_stream tree s)

def squaredNorm (x : SourceRow) : ℝ := ∑ j, x j * x j

/-- Exact energy identity for the homogeneous state difference. It exposes
the hypotheses needed for stability instead of assuming a norm bound. -/
theorem homogeneous_energy (row key : SourceRow) (alpha beta : ℝ) :
    squaredNorm (writeRow row key 0 alpha beta true) =
      alpha^2 * (squaredNorm row -
        beta*(2-beta*squaredNorm key)*(rowRead row key)^2) := by
  have expand : squaredNorm (writeRow row key 0 alpha beta true) =
      alpha^2*squaredNorm row -
      2*alpha^2*beta*rowRead row key*rowRead row key +
      alpha^2*beta^2*(rowRead row key)^2*squaredNorm key := by
    unfold squaredNorm
    simp only [writeRow, Bool.true_eq, ↓reduceIte]
    calc
      _ = ∑ j : Fin 8, (alpha^2*(row j*row j) -
          2*alpha^2*beta*rowRead row key*(row j*key j) +
          alpha^2*beta^2*(rowRead row key)^2*(key j*key j)) := by
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = _ := by
        simp [Finset.sum_add_distrib, Finset.sum_sub_distrib,
              ← Finset.mul_sum, rowRead]
  rw [expand]
  ring

theorem homogeneous_nonexpansive (row key : SourceRow) (alpha beta : ℝ)
    (hk : squaredNorm key ≤ 1) (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1) :
    squaredNorm (writeRow row key 0 alpha beta true) ≤ alpha^2*squaredNorm row := by
  rw [homogeneous_energy]
  have hkn : 0 ≤ squaredNorm key := by
    apply Finset.sum_nonneg
    intro j _
    exact mul_self_nonneg (key j)
  have hprod : beta*squaredNorm key ≤ 1 := by nlinarith
  have hcoef : 0 ≤ beta*(2-beta*squaredNorm key) :=
    mul_nonneg hb0 (by linarith)
  have hterm : 0 ≤ beta*(2-beta*squaredNorm key)*(rowRead row key)^2 :=
    mul_nonneg hcoef (sq_nonneg _)
  exact mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg alpha)

/-- With the SAME write value/schedule, old-state perturbations evolve by
the homogeneous contraction. Different schedules require additional terms. -/
theorem difference_write (row other key : SourceRow) (value alpha beta : ℝ) :
    writeRow row key value alpha beta true - writeRow other key value alpha beta true =
      writeRow (row-other) key 0 alpha beta true := by
  funext j
  have hread : rowRead (row-other) key = rowRead row key-rowRead other key := by
    simp [rowRead, sub_mul, Finset.sum_sub_distrib]
  simp only [Pi.sub_apply, writeRow, Bool.true_eq, ↓reduceIte, hread]
  ring

theorem write_nonexpansive (row other key : SourceRow) (value alpha beta : ℝ)
    (hk : squaredNorm key ≤ 1) (ha0 : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1) :
    squaredNorm (writeRow row key value alpha beta true -
                 writeRow other key value alpha beta true) ≤ squaredNorm (row-other) := by
  rw [difference_write]
  have hb := homogeneous_nonexpansive (row-other) key alpha beta hk hb0 hb1
  have hd : 0 ≤ squaredNorm (row-other) := by
    apply Finset.sum_nonneg
    intro j _
    exact mul_self_nonneg _
  have ha2 : alpha^2 ≤ 1 := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right ha2 hd]

/-- The actual integer-residue code has a finite period, independently of
the learned rotations, normalization and floating Fourier evaluation. -/
def ordinalResidues (n : ℕ) : Fin 4 → ℕ :=
  ![n % 7, n % 13, n % 29, n % 59]

theorem ordinal_encoding_periodic (n : ℕ) :
    ordinalResidues (n + 155701) = ordinalResidues n := by
  funext i
  fin_cases i <;> simp [ordinalResidues, Nat.add_mod]

/-- Any deterministic address/read router built ONLY from these residues
aliases at this period, even if its trainable maps are otherwise arbitrary. -/
theorem deterministic_encoding_collision {Output : Type}
    (encoder : (Fin 4 → ℕ) → Output) (n : ℕ) :
    encoder (ordinalResidues (n + 155701)) = encoder (ordinalResidues n) := by
  rw [ordinal_encoding_periodic]

theorem period_events_distinct (n : ℕ) : n + 155701 ≠ n := by omega

/-- The same returned content cannot equal two different true contents.
This is a representational limitation, irrespective of training. -/
theorem aliased_query_cannot_recover_distinct_values (readout : ℕ → ℝ)
    (n : ℕ) (v w : ℝ) (hread : readout (n + 155701) = readout n)
    (hvw : v ≠ w) : ¬ (readout n = v ∧ readout (n + 155701) = w) := by
  rintro ⟨hv, hw⟩
  exact hvw (hv.symm.trans (hread.symm.trans hw))

/-- A chronology guard alone cannot attest membership of observed records. -/
theorem chronology_does_not_prove_observation :
    ∃ last query : ℕ, ∃ observed : Set ℕ,
      authorizedOldQuery 1 1 last query ∧ query ∉ observed := by
  exact ⟨2, 1, ∅, ⟨rfl, by omega⟩, by simp⟩

/-- Generic local-roundoff propagation. hstep is a stated numerical model;
this theorem does not establish it for any IEEE backend. -/
theorem perturbation_step (oldError nextError lipschitz localError : ℝ)
    (hstep : nextError ≤ lipschitz * oldError + localError)
    (hL : lipschitz ≤ 1) (_hL0 : 0 ≤ lipschitz) (he : 0 ≤ oldError) :
    nextError ≤ oldError + localError := by nlinarith

/-- Under a nonexpansive per-event map, local error accumulates at most
linearly. The scan and recurrence have different local error budgets. -/
theorem accumulated_error (err : ℕ → ℝ) (eps initial : ℝ)
    (hzero : err 0 ≤ initial)
    (hstep : ∀ n, err (n+1) ≤ err n + eps) (n : ℕ) :
    err n ≤ initial + n * eps := by
  induction n with
  | zero => simpa using hzero
  | succ n ih =>
    have hs := hstep n
    push_cast
    nlinarith

end PNDS.SourceAlignedVerification
