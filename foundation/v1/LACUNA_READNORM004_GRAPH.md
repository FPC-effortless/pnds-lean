# LACUNA-READNORM004 — paired read constraint, PARTIAL

Parent original086graph and003conditional consumer patch are unchanged. Native
input observed int64bytes[B,T], original owner-scoped state1920B tensors and
logicalclock, original100521parameters. Query/gates/fast read/current write all
come from the frozen original trajectory. All counterfactual next states are
discarded. Source receipts and prior failed utility gates remain unchanged.

At a single prediction clock, h=Hq:[8], c=Cq:[8], v:[8]. The new evaluator-only
operator computes u=c/||c||, a=dot(h,u), hp=h-au, r=||hp||, vp=v-dot(v,u)u,
and T(v)=au+r vp/||vp||. Explicit epsilon1e-12 and original-history fallback
flags handle zero/nearzero/collinear inputs. Use float64 projection, float32
total slow read before original source consumers (fast,last8; slow,last8;
norm,96:104). Fixed original byte-code readout reports historical-byte changes.
The paired historical-only baseline exactly reuses003. Actual-source local VJP
tests compose projection→cast→original decoder; no optimizer or parameter fit.

Lean proves squared norm expansion, scaled-inner-product and orthonormal-frame
radius algebra, and that preserving historical squared norm and current-write
projection preserves total squared norm (with converse). Python independently
implements the exact branching numerical projection. Constructed tests are
engineering evidence; exact-real algebra does not establish IEEE equivalence.

The constraints preserve norms of historical and total **slow** reads. They do
not preserve the norm of the mixed fast/slow decoder vector; record this remaining
change rather than claiming all decoder activation magnitudes are controlled.
Projection can remove useful value components and restricts available directions;
a negative result is specific to these frozen decoder interventions.

Already exposed003DEV documents, same226source/query/bin cells across3models;
no new unseen cohort. No fit, GPU, TEST or architecture promotion. Actual evaluator
also retains raw ring and decomposition workspaces, so equal256Bbank payloads do
not establish equal-total-state autonomous architecture efficiency.

OPEN: complete numerical branch/sqrt/division proof, full IEEE roundoff bounds,
all-parameter NN AD and source equivalence, global information-flow theorem,
independent qualification, canonical historical registry migration and any
autonomous retrieval capability. Paired source checks and no-fit evidence do
not reopen earlier failed gates or authorize neural training.

## Original measured evidence (not a theorem)

TAC execution e2a997d882e536dc706c21c88a8219ad57458b5d,
run41010825969/artifact12957059683 completed all integrity checks. Same226source
query/bin cells repeated across3checkpoints=678measurements. All original003
historical-only NLL scores reproduced exactly. Historical and total slow-read
norms held within registered numerical tolerances; no fallback rows occurred.
One new exploratory gate passes: evaluator-supplied key/raw-history at8–31bytes,
gain0.005613 against original but only0.00007078 against wrong key. Seed1702
prefers the wrong-key control. No older or autonomous-query condition passes.
This is exposed DEV with5weakref short-age cells, not independent generalization.
Mixed fast/slow norm can still change; old failed gates/training stop remain.
