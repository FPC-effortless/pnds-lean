# LANGMEM087 typed graph and historical-content attribution — partial proofs

Stage A inherits LANGMEM086_GRAPH.md's complete typed input→output neural
specification without changing its trained equations, timing or dimensions.
Source compact_tac.py SHA7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1.
Model input[B,T]int64 observed bytes, owner andclock; output[B,T,256]next-byte
logits and480FP32/1920B tensorstate. Clock8B, ownerUTF8, shared100521FP32
parameters402084B andfixedcodebuffer8192B separate. No externalkey/label API.

The full encoder/position/lexical→autonomousquery/writer/value→fast8x8/slow8x8
→learnedmix→aligned2x8x8/wave4x2/basis48/concepts4x8/upper32→94SwiGLU256
route and input-only auxiliary state transitions are EXACT086. No read-before-
write training, dimensions, read-bias, write-cap, dropout or supervision change.

Same AdamW trajectory: original sealed160 fullmodel+optimizer+RNG, continue
same4x256 FIT samplerto640 then1280. Original160 rawcheckpointSHA pinned
perseed. Duration targetcounts163840/655360/1310720 differ intentionally;
no equal-budget architecture superiority. New shared3584targetsourcepositions
percheckpoint afterall9seals. New DEV source is not a training input.

Evaluation-only diagnostic graph copies full original forward. Default must
match original bitwise logits andallstate. At a frozen original prior state,
for observed e_t, eta=.01sigmoid(g_s(e_t)), rho=.995+.004sigmoid(s), original
keyk_(t−1), sourcevaluev_t and autonomousq_t:
H=rho*S_old−eta*rho*(S_old k)k^T;
C=eta*v*k^T;
S_new=H+C.
Original model readsS_new*q. Counterfactual readoldzero=C*q preserves current
write but removes all historicalmatrixcontributions includingthe oldstate
residualpredictionpart. Currentwritezero=H*q; oldpermute usesP*S_old inH
and retainsoriginalC. Matrixwrite itself remainsoriginalS_new for allcontrols.
Other priorstates/gates/query/mix/valuefixed. Only downstreamupper differs;
its counterfactual nextstate discarded. No persistentstateerasure or fitted
fast-onlybaseline here. These are immediate read interventions, possiblyOOD.

Lean module LangMem087Historical proves exact-real entrywise H+C identity and
zero-old→C, plus exact duration counts. Python reference independently executes
split and four read controls. TAC parity compares actual diagnostic forward
readvalues/allcomponents inFloat64 plus fourlocaloldmatrix derivatives. It
also retains original086fast/slowoperatorparity. Unit tests actualforward
bitwise normalgraph/currentonlyequivalence andimmutable storedstate.

Proof scope is PARTIAL. Full model IEEE/AD/noninterference, learned content
sufficiency, semanticrecall, long-range contribution age decomposition,
independent reward/goal/ACL certification andcanonicalregistrymerge OPEN.
Positive immediate oldstate lesionNLLdelta cannot establish retrieval of a
particular historical byte or externally qualified language improvement.
Two newRustowners sharetargets acrossallseeds/durations; no i.i.d. byteclaim.
