import PNDS.SourceAlignedVerification

/-!
Source-shaped inclusive Blelloch scan: a balanced chronological upsweep,
incoming exclusive prefixes on the downsweep, and inclusive leaves. The proofs
cover every leaf, rather than only the final tree summary. Arithmetic is exact;
backend rounding is handled by the separate numerical contracts.
-/
namespace PNDS.SourceAlignedScan

universe u
variable {A : Type u}

def prefixes (op : A → A → A) (acc : A) : List A → List A
  | [] => []
  | x::xs => op acc x :: prefixes op (op acc x) xs

theorem prefixes_length (op : A → A → A) (acc : A) (xs : List A) :
    (prefixes op acc xs).length = xs.length := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih => simp [prefixes, ih]

theorem prefixes_append (op : A → A → A) (acc : A) (xs ys : List A) :
    prefixes op acc (xs++ys) =
      prefixes op acc xs ++ prefixes op (xs.foldl op acc) ys := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih => simp [prefixes, ih]

inductive FullTree (A : Type u) : ℕ → Type u
  | leaf : A → FullTree A 0
  | node {d : ℕ} : FullTree A d → FullTree A d → FullTree A (d+1)

def leaves : {d : ℕ} → FullTree A d → List A
  | _, .leaf a => [a]
  | _, .node a b => leaves a ++ leaves b

def upsweep (op : A → A → A) : {d : ℕ} → FullTree A d → A
  | _, .leaf a => a
  | _, .node a b => op (upsweep op a) (upsweep op b)

def downsweep (op : A → A → A) (incoming : A) : {d : ℕ} → FullTree A d → List A
  | _, .leaf a => [op incoming a]
  | _, .node a b => downsweep op incoming a ++
      downsweep op (op incoming (upsweep op a)) b

theorem leaves_length {d : ℕ} (tree : FullTree A d) : (leaves tree).length = 2^d := by
  induction tree with
  | leaf a => simp [leaves]
  | node a b iha ihb => simp [leaves, iha, ihb, pow_succ, Nat.mul_two]

theorem upsweep_fold (op : A → A → A)
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c))
    {d : ℕ} (tree : FullTree A d) (incoming : A) :
    (leaves tree).foldl op incoming = op incoming (upsweep op tree) := by
  induction tree generalizing incoming with
  | leaf a => rfl
  | node a b iha ihb =>
    simp only [leaves, List.foldl_append, upsweep]
    rw [iha, ihb, hassoc]

/-- Exact semantics of every inclusive leaf produced by the source's
upsweep/downsweep order, for arbitrary noncommuting associative operators. -/
theorem blelloch_all_prefixes (op : A → A → A)
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c))
    {d : ℕ} (tree : FullTree A d) (incoming : A) :
    downsweep op incoming tree = prefixes op incoming (leaves tree) := by
  induction tree generalizing incoming with
  | leaf a => rfl
  | node a b iha ihb =>
    simp only [downsweep, leaves, prefixes_append]
    rw [iha, ihb, upsweep_fold op hassoc]

/-- Cropping a padded scan cannot alter any original chronological prefix. -/
theorem crop_padding (op : A → A → A) (incoming : A) (xs padding : List A) :
    (prefixes op incoming (xs++padding)).take xs.length = prefixes op incoming xs := by
  rw [prefixes_append, ← prefixes_length op incoming xs]
  simp

theorem padded_blelloch (op : A → A → A)
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c))
    {d : ℕ} (tree : FullTree A d) (incoming : A) (xs padding : List A)
    (hl : leaves tree = xs++padding) :
    (downsweep op incoming tree).take xs.length = prefixes op incoming xs := by
  rw [blelloch_all_prefixes op hassoc, hl, crop_padding]

/-- A masked leaf is the identity. This is the exact-real equation behind
the source's last-admitted-prefix gather; byte preservation itself is selection. -/
theorem nonwrite_prefix_identity (op : A → A → A) (identity : A)
    (hr : ∀ a, op a identity = a) (incoming : A) (rest : List A) :
    prefixes op incoming (identity::rest) = incoming :: prefixes op incoming rest := by
  simp [prefixes, hr]

open PNDS.SourceAlignedVerification

theorem source_blocks_blelloch {d : ℕ} (tree : FullTree BlockSchedule d)
    (incoming : BlockSchedule) :
    downsweep blockThen incoming tree = prefixes blockThen incoming (leaves tree) :=
  blelloch_all_prefixes blockThen block_then_assoc tree incoming

/-- Flattening [B,T,H=2,half=2,K] by permute(0,2,3,1,4) gives
(((b*2+h)*2+half)*T+t)*8+j. Reassembly restores value row 8*half+r. -/
def flatScheduleOffset (T b h half t j : ℕ) : ℕ :=
  (((b*2+h)*2+half)*T+t)*8+j

def flatBlockOffset (T b h half t r j : ℕ) : ℕ :=
  ((((b*2+h)*2+half)*T+t)*8+r)*8+j

def sourceMatrixOffset (T b t h v j : ℕ) : ℕ :=
  ((((b*T+t)*2+h)*16+v)*8+j)

theorem schedule_batch_index (b h half : ℕ) :
    (b*2+h)*2+half = b*4+h*2+half := by omega

theorem reassembly_value_row (half r : ℕ) (hh : half < 2) (hr : r < 8) :
    8*half+r < 16 ∧ (8*half+r)/8 = half ∧ (8*half+r)%8 = r := by omega

theorem reassembly_offset (T b t h half r j : ℕ) :
    sourceMatrixOffset T b t h (8*half+r) j =
      ((b*T+t)*256+h*128+half*64+r*8+j) := by
  unfold sourceMatrixOffset
  ring

end PNDS.SourceAlignedScan
