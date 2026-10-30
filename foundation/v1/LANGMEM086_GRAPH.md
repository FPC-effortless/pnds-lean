# LANGMEM086 complete typed graph record with partial proofs

Original implementation: compact_tac.py in TAC LANGMEM086, new100521-parameter
model trained from scratch. Existing078500k weights never modified. This
record defines the full source shape/timing route, while executable Python
and Lean cover only named residual/read/count kernels. **Whole neural graph
source Float32/AD equivalence is OPEN**, not certified by these kernels.

Input observed bytes [B,T] int64, owner identity for state separation only,
clock n. Prediction logits[B,T,256] float32 for next bytes, state per-owner.
Only source observations through current t enter each output. No labels,
provided historical address, byte index oracle, reward, candidate answer or
external source-history table enters model forward.

State: fast8x8, slow8x8, aligned2x8x8, wave4x2, basis48, concepts4x8,
lexical3x32, previous writer8, upper32 =480 floats/1920B. Clock8B and owner
UTF8 metadata separate; no persisted logits. Shared fixed bytecode8192B and
learned weights402084B. Full process/allocator workspace is separate.

Encoding: learned embedding256x32 and position7→32 -> tanh e32; lexical tanh
embedding with learned three lag weights -> local32. Writer normalize(W32→8
local). Current query normalize(Q64→8[local,prior_upper32]); separate from
writer and learned using source next-byte loss. Value normalize(signed8-bit
bytecode+.1tanh(V32→8 e)). The historical writer from prior observation is
used for current observed byte's value write; source first byte skips pair.

Fast: rho_f=.97+.02sigmoid(g0(e)), residual v-rho_f F k,
eta_f=min(1,.35/(norm(residual)+1e-9))sigmoid(g1(e)); update rho_f F+eta_f
residual outer k. Slow rho_s=.995+.004sigmoid(s), eta_s=.01sigmoid(g_s(e)),
update rho_s S+eta_s(v-rho_s S k) outer k. Learned64→2 softmax mixtures of
both current matrices read at model q yield8D native read. All update paths
are differentiable almost everywhere; exact clip/clamp boundary and Float32
error/AD proof OPEN. New next-byte task trains these paths end to end.

Other native TAC routes: two learned ordinal rotated8D time keys update2x8x8
aligned matrices; prior aligned read8. Four2D damped waves and basis48 sum
are updated causally; basis mean and prior wave8 enter48D fast processing,
then48D slow processing. Evidence [e32,fast48,slow48,aligned8,wave8]→32,
reshape4x8. Learned q/k concept projections8x8 produce symmetrized nonnegative
adjacency and Laplacian L4x4. Quadratic equilibrium C=(1.25I+.25L)^-1
(G+.25prior_concepts), pooled8. Separate input-only concept seed update
small learned convex gate. Combine fast48+slow48+native8+aligned8+wave8+
pooled8+e32=160; LayerNorm; upper input160→32, prior-gated recurrent read
32→160 residual; upper observe bounded32. Final94-wide SwiGLU output256
plus concept8→256. Optimizer cross-entropy against actual next source byte
AFTER predicting. Full source operator definitions and parameters pinned in
reciprocal manifest; no mathematical claim that loss implies semantic recall.

Lean proves exact-real unit-key residual read interpolation and integer
parameter/state counts; generic observed-prefix exclusion is a specification,
not arbitrary source-level noninterference. Python independently implements
clip/residual/mix equations. TAC source/reference numerical and local-memory
gradient checks are named separately. Full model masks, source IEEE/AD,
serialized memory ownership enforcement, semantic retrieval, external
qualification and coordinated canonical registry merge remain OPEN.
