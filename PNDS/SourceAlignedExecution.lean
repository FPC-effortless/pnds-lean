import PNDS.SourceAlignedModel
import PNDS.SourceAlignedScan

/-!
Full default-model exact-real execution theorem. The prepass prepares the
actual learned schedules using only clocks, source inputs and basis sums.
Native and source-aligned memories then execute as a product affine scan.
Reconstruction includes every persistent field and the complete nonlinear
decoder. This is mathematical execution semantics, not a Torch interpreter.
-/
namespace PNDS.SourceAlignedExecution

noncomputable section

open PNDS.SourceAlignedMemory PNDS.SourceAlignedModel
open PNDS.SourceAlignedVerification PNDS.ParallelMemory

def eraseMemory (s : State) : State :=
  {s with memory := 0, aligned := fun _ _ _ => 0}

def joinMemory (s : BlockState) : SourceMatrix := fun h v j =>
  s h ⟨v.val/8, by omega⟩ ⟨v.val%8, by omega⟩ j

theorem join_split (s : SourceMatrix) : joinMemory (splitMemory s) = s := by
  funext h v j
  have hv : rowIndex ⟨v.val/8, by omega⟩ ⟨v.val%8, by omega⟩ = v := by
    apply Fin.ext
    simp only [rowIndex]
    omega
  simp only [joinMemory, splitMemory, hv]

theorem split_join (s : BlockState) : splitMemory (joinMemory s) = s := by
  funext h b r j
  have hb : (8*b.val+r.val)/8 = b.val := by omega
  have hr : (8*b.val+r.val)%8 = r.val := by omega
  simp only [splitMemory, joinMemory, rowIndex]
  congr 2

abbrev Memories := Matrix (Fin 8) (Fin 8) ℝ × BlockState
abbrev JointSchedule := AffinePair 8 × BlockSchedule

def memories (s : State) : Memories := (s.memory,splitMemory s.aligned)

def restore (context : State) (m : Memories) : State :=
  {context with memory := m.1, aligned := joinMemory m.2}

theorem restore_erase (s : State) : restore (eraseMemory s) (memories s) = s := by
  simp only [restore, eraseMemory, memories, join_split]

theorem prepare_from_context (w : Parameters) (s : State) (x : Input) :
    prepare w (eraseMemory s) x = prepare w s x :=
  prepare_memory_independent w s x 0 (fun _ _ _ => 0)

theorem prepare_context_congr (w : Parameters) (s c : State) (x : Input)
    (h : eraseMemory s = eraseMemory c) : prepare w s x = prepare w c x := by
  rw [← prepare_from_context w s x, ← prepare_from_context w c x, h]

theorem erase_advance (s : State) (x : Input) (p : Prepared) :
    eraseMemory (advance s x p) = eraseMemory (advance (eraseMemory s) x p) := by rfl

def contextStep (w : Parameters) (s : State) (x : Input) : State :=
  eraseMemory (advance s x (prepare w s x))

theorem context_step_correspondence (w : Parameters) (s : State) (x : Input) :
    eraseMemory (step w s x).2 = contextStep w (eraseMemory s) x := by
  simp only [step, prepare_from_context, contextStep]
  exact erase_advance s x (prepare w s x)

theorem context_step_congr (w : Parameters) (s c : State) (x : Input)
    (h : eraseMemory s = eraseMemory c) : contextStep w s x = contextStep w c x := by
  have hp := prepare_context_congr w s c x h
  simp only [contextStep, hp]
  rw [erase_advance s, erase_advance c, h]

def contextRun (w : Parameters) (s : State) (events : List Input) : State :=
  events.foldl (contextStep w) (eraseMemory s)

theorem context_run_correspondence (w : Parameters) (s : State) (events : List Input) :
    eraseMemory (runState w s events) = contextRun w s events := by
  induction events generalizing s with
  | nil => rfl
  | cons x xs ih =>
    simp only [runState, List.foldl_cons]
    rw [← runState, ih]
    simp only [contextRun, List.foldl_cons, context_step_correspondence]

/-- Schedule preparation advances only the erased context. In particular,
neither memory needs to be sequentially evaluated to prepare these events. -/
def prepass (w : Parameters) (context : State) : List Input → List (Input × Prepared)
  | [] => []
  | x::xs => (x,prepare w context x) :: prepass w (contextStep w context x) xs

def scheduledRun (s : State) : List (Input × Prepared) → State
  | [] => s
  | (x,p)::xs => scheduledRun (advance s x p) xs

theorem prepass_matches_full_run (w : Parameters) (s c : State) (events : List Input)
    (h : eraseMemory s = eraseMemory c) :
    scheduledRun s (prepass w c events) = runState w s events := by
  induction events generalizing s c with
  | nil => rfl
  | cons x xs ih =>
    have hp := prepare_context_congr w s c x h
    simp only [prepass, scheduledRun, runState, List.foldl_cons]
    change scheduledRun (advance s x (prepare w c x))
      (prepass w (contextStep w c x) xs) = runState w (step w s x).2 xs
    rw [← hp]
    apply ih
    change eraseMemory (advance s x (prepare w s x)) =
      eraseMemory (contextStep w c x)
    have hc := context_step_congr w s c x h
    simpa only [contextStep, eraseMemory] using hc

def nativeSchedule (x : Input) (p : Prepared) : AffinePair 8 :=
  if x.admitted then
    ⟨rowP p.nativeKey p.nativeAlpha p.nativeBeta,
      fun r j => rowQ p.nativeKey (p.nativeValue r) p.nativeBeta j⟩
  else identity 8

def eventSchedule (event : Input × Prepared) : JointSchedule :=
  (nativeSchedule event.1 event.2,
   blockSchedule event.2.sourceKey event.2.sourceValue event.2.sourceAlpha
     event.2.sourceBeta event.1.admitted)

def jointApply (a : JointSchedule) (s : Memories) : Memories :=
  (applyPair a.1 s.1,blockApply a.2 s.2)

def jointThen (a b : JointSchedule) : JointSchedule :=
  (thenPair a.1 b.1,blockThen a.2 b.2)

def jointIdentity : JointSchedule := (identity 8,blockIdentity)

theorem joint_apply_then (a b : JointSchedule) (s : Memories) :
    jointApply (jointThen a b) s = jointApply b (jointApply a s) := by
  apply Prod.ext
  · exact apply_then a.1 b.1 s.1
  · exact block_apply_then a.2 b.2 s.2

theorem joint_then_assoc (a b c : JointSchedule) :
    jointThen (jointThen a b) c = jointThen a (jointThen b c) := by
  apply Prod.ext
  · exact then_assoc a.1 b.1 c.1
  · exact block_then_assoc a.2 b.2 c.2

theorem joint_identity_apply (s : Memories) : jointApply jointIdentity s = s := by
  apply Prod.ext
  · exact apply_identity s.1
  · funext h b
    exact apply_identity (s.2 h b)

theorem advance_memory_correspondence (s : State) (x : Input) (p : Prepared) :
    memories (advance s x p) = jointApply (eventSchedule (x,p)) (memories s) := by
  apply Prod.ext
  · funext r j
    cases hx : x.admitted with
    | false => simp [memories, advance, jointApply, eventSchedule, nativeSchedule, hx,
        writeRow, applyPair, identity]
    | true =>
      simpa only [memories, advance, jointApply, eventSchedule, nativeSchedule, hx,
          Bool.true_eq, ↓reduceIte, applyPair, Matrix.add_apply, Matrix.mul_apply] using
        (row_affine_correspondence (s.memory r) p.nativeKey
          (p.nativeValue r) p.nativeAlpha p.nativeBeta j).symm
  · exact (four_block_write_correspondence s.aligned p.sourceKey p.sourceValue
        p.sourceAlpha p.sourceBeta x.admitted).symm

theorem scheduled_memory_fold (events : List (Input × Prepared)) (s : State) :
    memories (scheduledRun s events) =
      (events.map eventSchedule).foldl (fun old e => jointApply e old) (memories s) := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    rcases e with ⟨x,p⟩
    simp only [scheduledRun, List.map_cons, List.foldl_cons]
    rw [ih, advance_memory_correspondence]

theorem joint_fold_matches_stream (events : List JointSchedule) (acc : JointSchedule)
    (s : Memories) : jointApply (events.foldl jointThen acc) s =
      events.foldl (fun old e => jointApply e old) (jointApply acc s) := by
  induction events generalizing acc with
  | nil => rfl
  | cons e es ih =>
    simpa only [List.foldl_cons, joint_apply_then] using ih (jointThen acc e)

/-- Whole-state reconstruction after independently preparing schedules and
scanning the two native/aligned stores. All nonlinear maps are the actual
typed graph from SourceAlignedModel, with arbitrary learned parameters. -/
theorem full_model_state_after_scan (w : Parameters) (s : State) (events : List Input) :
    restore (contextRun w s events)
      (jointApply (((prepass w (eraseMemory s) events).map eventSchedule).foldl
        jointThen jointIdentity) (memories s)) = runState w s events := by
  have hp := prepass_matches_full_run w s (eraseMemory s) events (by rfl)
  have hm := scheduled_memory_fold (prepass w (eraseMemory s) events) s
  rw [hp] at hm
  rw [joint_fold_matches_stream, joint_identity_apply, ← hm,
      ← context_run_correspondence, restore_erase]

theorem full_prediction_after_scan (w : Parameters) (s : State)
    (past : List Input) (current : Input) :
    let reconstructed := restore (contextRun w s past)
      (jointApply (((prepass w (eraseMemory s) past).map eventSchedule).foldl
        jointThen jointIdentity) (memories s))
    decode w reconstructed current (prepare w (contextRun w s past) current) =
      (step w (runState w s past) current).1 := by
  dsimp only
  rw [full_model_state_after_scan, ← context_run_correspondence, prepare_from_context]
  rfl

open PNDS.SourceAlignedScan

/-- The balanced source-shaped upsweep/downsweep returns every inclusive
native+aligned memory prefix used by the full graph above. -/
theorem full_model_blelloch_prefixes {d : ℕ} (tree : FullTree JointSchedule d) :
    downsweep jointThen jointIdentity tree =
      prefixes jointThen jointIdentity (leaves tree) :=
  blelloch_all_prefixes jointThen joint_then_assoc tree jointIdentity

end
end PNDS.SourceAlignedExecution
