import Mathlib

/-!
PAR-MEM-010: source-identity-aligned **operator-level** TAC memory.
This formalizes only selected source-known address, provenance admission,
read-only query, and one row of the corrective gated-delta write.

The address code in original Torch uses four source-known ordinal Fourier
periods, then one shared trainable map for WRITE and READ. The theorem
assumes its function is deterministic; it does NOT prove IEEE Fourier
rounding or that distinct source events receive distinct keys.

The real implementation stores two 16-by-8 matrices and internally scans
four 8-by-8 affine blocks. No theorem here establishes whole-graph
training, source recall, finite-precision AD, probability calibration or
the correctness of actual learned projections. Original checkpoints
remain unchanged.
-/

namespace PNDS.SourceAlignedMemory

abbrev Address := Fin 8 → ℝ
abbrev SourceRow := Fin 8 → ℝ
abbrev SourceMatrix := Fin 2 → Fin 16 → Fin 8 → ℝ

/-- Caller-supplied source event ordinal mapped to an address without
    consulting the old event measurement. Shared across writer and reader. -/
def sourceKey (code : ℕ → Address) (ordinal : ℕ) : Address :=
  code ordinal

def sourceWriteKey (code : ℕ → Address) (ordinal : ℕ) : Address :=
  sourceKey code ordinal

def sourceReadKey (code : ℕ → Address) (ordinal : ℕ) : Address :=
  sourceKey code ordinal

/-- Identical source event identity and shared code yield identical keys. -/
theorem shared_source_event_address (code : ℕ → Address) (ordinal : ℕ) :
    sourceWriteKey code ordinal = sourceReadKey code ordinal := by
  rfl

/-- The permitted historical read must match source ownership, and may
    refer only to the current or older ordinal of that source. -/
def authorizedOldQuery
    (recordOwner queryOwner lastOrdinal queryOrdinal : ℕ) : Prop :=
  recordOwner = queryOwner ∧ queryOrdinal ≤ lastOrdinal

theorem reject_wrong_source_owner (recordOwner queryOwner lastOrdinal queryOrdinal : ℕ)
    (h : recordOwner ≠ queryOwner) :
    ¬ authorizedOldQuery recordOwner queryOwner lastOrdinal queryOrdinal := by
  intro allowed
  exact h allowed.1

theorem reject_future_source_ordinal (recordOwner queryOwner lastOrdinal queryOrdinal : ℕ)
    (h : lastOrdinal < queryOrdinal) :
    ¬ authorizedOldQuery recordOwner queryOwner lastOrdinal queryOrdinal := by
  intro allowed
  exact (Nat.not_le_of_gt h) allowed.2

/-- One physical output row of a 16x8 value-by-address store. -/
def rowRead (row key : SourceRow) : ℝ :=
  ∑ j : Fin 8, row j * key j

/-- Corrective gated-delta write for a row, or EXACT source non-admission
    identity. The real matrix may have two heads x16 rows and shares key. -/
def writeRow (row key : SourceRow) (value alpha beta : ℝ)
    (admitted : Bool) : SourceRow :=
  if admitted then
    fun j => alpha * row j +
      beta * (value - alpha * rowRead row key) * key j
  else row

/-- Missing/not-authorized source event cannot alter any matrix row. -/
theorem declined_write_is_identity
    (row key : SourceRow) (value alpha beta : ℝ) :
    writeRow row key value alpha beta false = row := by
  rfl

/-- Gated correction is applied to the value residual, not merely appended. -/
theorem admitted_write_is_corrective
    (row key : SourceRow) (value alpha beta : ℝ) (j : Fin 8) :
    writeRow row key value alpha beta true j =
      alpha * row j + beta * (value - alpha * rowRead row key) * key j := by
  rfl

/-- Historical read operations return values but no modified state. -/
def readonlyHistoricalQuery (memory : SourceMatrix)
    (key : Address) (head row : Fin 2) (valueRow : Fin 16) : ℝ × SourceMatrix :=
  (rowRead (memory head valueRow) key, memory)

theorem historical_query_does_not_write (memory : SourceMatrix)
    (key : Address) (head : Fin 2) (valueRow : Fin 16) :
    (readonlyHistoricalQuery memory key head valueRow).2 = memory := by
  rfl

/-- A rejected event never changes the independent two-head memory. -/
def sourceAdmittedUpdate (memory newMemory : SourceMatrix)
    (admit : Bool) : SourceMatrix :=
  if admit then newMemory else memory

theorem no_source_admission_preserves_all_heads
    (memory newMemory : SourceMatrix) :
    sourceAdmittedUpdate memory newMemory false = memory := by
  rfl

end PNDS.SourceAlignedMemory
