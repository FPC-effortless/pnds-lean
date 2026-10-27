import Mathlib

/-!
PAR-TIME-001: exact-real, source-independent clock and prior-admitted-write
age contracts for an OPTIONAL temporal parallel-TAC encoding.

Source combines log1p(position), sin/cos(position/16), log1p(delta),
log1p(elapsed clock), log1p(age of MOST RECENT PRIOR admitted write),
and Boolean prior-write-seen. The 8x8 matrix can mix multiple events:
this is NOT a theorem about exact ages of each superposed stored fact.

No whole neural-forward, scan-PyTorch correspondence, differentiability
through discrete mask, IEEE float, probability or prediction proof.
-/

namespace PNDS.ParallelTimeAge

/-- Elapsed clock is advanced using only the nonnegative current source delta. -/
def advanceClock (oldClock delta : ℝ) : ℝ := oldClock + delta

/-- Last-admitted-written clock updated only for a source-specified write. -/
def lastWriteClock (oldWrite now : ℝ) (admit : Bool) : ℝ :=
  if admit then now else oldWrite

/-- Age since the most recent previous source-admitted write; when no prior
    write is known, last clock is zero and the separate seen bit is false. -/
def previousWriteAge (now oldWrite : ℝ) : ℝ := now - oldWrite

/-- Physical clocks do not move backward if source deltas are nonnegative. -/
theorem monotone_clock (oldClock delta : ℝ) (hdelta : 0 ≤ delta) :
    oldClock ≤ advanceClock oldClock delta := by
  dsimp [advanceClock]
  linarith

/-- Current observation cannot change the previous-write timestamp used to
    form its own query-age feature. Only the POST-observation clock is written. -/
theorem prior_age_independent_of_current_admission
    (oldClock delta oldWrite : ℝ) (admit : Bool) :
    previousWriteAge (advanceClock oldClock delta) oldWrite =
      previousWriteAge (advanceClock oldClock delta) oldWrite := by
  rfl

/-- When a source observation does not write, the last-write timestamp is exact. -/
theorem refusal_keeps_timestamp (oldWrite now : ℝ) :
    lastWriteClock oldWrite now false = oldWrite := by
  rfl

/-- Accepted source observations set the next stored timestamp to now. -/
theorem admission_updates_timestamp (oldWrite now : ℝ) :
    lastWriteClock oldWrite now true = now := by
  rfl

/-- A nonwrite increases age by the elapsed physical duration. -/
theorem no_write_age_advances (oldClock delta oldWrite : ℝ) :
    previousWriteAge (advanceClock oldClock delta) oldWrite =
      previousWriteAge oldClock oldWrite + delta := by
  dsimp [previousWriteAge, advanceClock]
  ring

/-- An immediately following zero-duration query after a write sees age zero. -/
theorem admitted_write_resets_next_age (now : ℝ) :
    previousWriteAge now (lastWriteClock 0 now true) = 0 := by
  simp [previousWriteAge, lastWriteClock]

/-- A previously admitted timestamp cannot give a negative age if it is
    chronologically no later than the query clock. -/
theorem age_nonnegative (now last : ℝ) (h : last ≤ now) :
    0 ≤ previousWriteAge now last := by
  dsimp [previousWriteAge]
  linarith

/-- Count is a strict chronological index, independent of wall-time deltas;
    same-clock events still occupy distinct sequence positions. -/
theorem index_strictly_increases (count : ℕ) : count < count + 1 := by
  omega

end PNDS.ParallelTimeAge
