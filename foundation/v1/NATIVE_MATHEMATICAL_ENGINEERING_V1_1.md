# PNDS native TAC-OSM dynamical engineering — extension v1.1

**Formal scope:** mathematically exact unprojected gated-delta memory update.
**Source:** FPC-effortless/TAC-OSM commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8, src/tac_osm/aligned_associative_tac.py.
**Status:** source-grounded partial mathematical research; complete nonlinear/float model proof still OPEN.
**Implementation connection:** https://github.com/FPC-effortless/TAC-OSM/pull/389.

The 17-version historical registry and all negative evidence records remain unchanged. This report is additive and cannot change a variant's evidence tier or claim a new model fit. The mathematical procedure is inspired by the engineering discipline of selective SSMs and Mamba, not an attempt to replace TAC-OSM with either architecture.

## I. Typed transition and observation cut

For the pinned native model:
- Current encoded event u has 48 coordinates; learned fast state f and slow state s each have 88 coordinates; learned key k and value v each have eight coordinates; memory M is 8 by 8.
- The same key identifier determines the learned address at write and query time.
- The causal read r=M_previous k occurs before f and s update.
- Learned gates alpha and beta depend on the new f/s; the memory write follows their updates.
- Source role kind=2 bypasses memory mutation; read-only does not mean decaying the old state.
- A writable update is radial Frobenius projected to radius four.

Thus one input step is an ordered nonlinear operator composition, not an affine scan. Any proposed scan must prove that its per-step summary can be constructed without depending on the state to be scanned. Existing ME095 affine-filter laws apply to that subfilter only.

## II. Exact associative-read decomposition, before projection

For an arbitrary finite number of key dimensions n and one value row m in R^n, define

R(m,k)=sum_i m_i k_i,
W_i(m,k,v;alpha,beta)=alpha m_i+
    beta (v-alpha R(m,k)) k_i.

The Lean theorem NativeMemory.row_query_decomposition establishes for any q:

R(W(m,k,v;alpha,beta),q)
   = alpha R(m,q) + beta (v-alpha R(m,k)) R(k,q).

This is a genuine arbitrary finite-dimension theorem, so n=8 is a valid instance. Repeating the same row algebra for eight value coordinates gives the preprojection 8x8 relation in exact real arithmetic. Lean also proves:

R(k,k)=1 implies v-R(W(m,k,v),k)
               =(1-beta)(v-alpha R(m,k)).

Unit-length keys are necessary for that simplified identity. The actual PyTorch epsilon-normalized address is not guaranteed to be unit when its source embedding approaches zero. No theorem here characterizes all key embeddings encountered during fitting.

Let the original query q differ from the current write key. The interference term is proportional to key overlap R(k,q) and the innovation v-alpha R(m,k). Orthogonal writes still multiply the previous query by alpha, because read-only bypass and writable retention are different semantics. Even perfect abstract beta=1 correction is no longer guaranteed once radius projection is active.

## III. Conditioning and projection: limitations that matter

For the source matrix update U = alpha M + beta(v-alpha Mk)k^T, the actual writable output is

M_next = U / max(1, ||U||_F / 4).

The projection bounds matrix magnitude, not associative information content or retention. The exact unprojected residual theorem may fail after clipping.

Concrete adverse case: old M=0, unit k=(1,0,...), v=(5,0,...), alpha=beta=1. Before projection the self-read is 5. After source-style radius-4 projection it is 4. This demonstrates why an unqualified perfect-correction theorem would be false for the actual model.

For independently addressed old query q, the unprojected step is

r_next(q)=alpha r_old(q)+beta e <k,q>,

where e=v-alpha Mk. Iterating gives a scalar affine unroll *once the actual time-dependent terms are known*. It does not make the native recurrent graph parallelizable because both gates and innovation depend on the state path. When q is orthogonal to all new writable addresses, its stored read is multiplied by each alpha. Source imposes alpha<=0.9999 per writable event, so useful long memory cannot be inferred from a small scalar state bound.

Read-only exact preservation is formalized separately: the original matrix is unchanged, including when the learned alpha could otherwise be below one.

## IV. State-space and stability engineering without changing model family

For the full learned fast/slow/association state z, use a nonlinear time-indexed transition

z_{t+1}=F_theta(z_t,o_t,a_{<t},dt,role,entity).

A meaningful proof requires:
- strict observation-time causality and masking, including actual action lags;
- bounded-input/bounded-state claims in a declared domain;
- derivative growth of the actual nonlinear transition, not just a scalar gate;
- finite-precision discontinuities at projection/normalization boundaries;
- read-only memory invariant and conditions for identity separation;
- stable but useful persistence under source-observed, real long histories.

For a state-dependent gated coordinate h'=(1-g(h,u))h+g(h,u)c(h,u), the derivative includes

d h'/d h = 1-g + g(dc/dh)+(c-h)(dg/dh).

The proof of contraction for a fixed g and fixed c does not establish contraction for the actual learned recurrence. A candidate certification may use weighted Jacobian or Lyapunov inequalities for an adaptive fast subspace while preserving explicitly protected fact/identity memory. This remains an unproved hypothesis until evaluated against the complete source graph and tested without training/test leakage.

## V. Computational-graph and gradient classification

Classify each node separately:
1. Parallel-legal: per-observation embedding, batch/entity masks with no cross-actor coupling, independent horizon/coordinate projection, fixed finite-window reductions.
2. Scan-legal only with derivation: an input-only affine transition whose (A_t,B_t) composition is associative; this holds for ME095's *affine subfilter* under its assumptions.
3. Sequential: fast state feeds slow state; both feed gates; memory read and later write depend on prior state, and a state-refinement or looped decoder consumes its previous output.

Proof that operator composition is associative is **necessary but not sufficient** for a floating-point speedup: verify its gradient and rounding behavior against source and account for memory traffic and recomputation. The Mamba papers motivate this engineering standard but do not confer it on a different nonlinear graph.

## VI. Evidence/proof status matrix

| Requirement | New or existing result | Remaining gap |
|---|---|---|
| Scalar fixed gated boundedness | prior PNDS UnifiedTACOSM Lean proof | whole learned Jacobian, temporal compounding |
| Generic unprojected key retrieval/interference | new NativeMemory Lean theorem over arbitrary finite n | projected actual 8x8 numerical/gradient equivalence |
| Same-key read residual, unit norm | new Lean theorem with explicit unit premise | runtime epsilon normalization and learned beta |
| Read-only identity | new Boolean mask Lean fact + finite-array Python test | complete source import/role/missingness integration |
| Colliding same-key targets | exact logical obstruction | useful memory capacity/retention on authentic entities |
| Norm projection | precise mathematical reference and clipping counterexample | whole-model tensor FP error certificate |
| Learned real source parity | companion Torch-vs-PNDS test candidate, original source commit pinned | full native fast/slow gradients and original checkpoint replay |
| Real long-term memory/intelligence | NONE established by this change | untouched authentic longitudinal cases, original source-factual baseline |

Authoritative trained scientific failures are unchanged: strict DIAGKEY004 raw logit replay, ME069 vs matched GRU, ME076 vs persistence, physical ME094 quiet regression, ESWAM014 low-speed no-regression. Never promote a proven operator equality to an empirical win.

## VII. Definition of rigor for the next full-model version

A mathematical release certificate must include an exact typed state, read/write/retention graph, source and model hashes, Lean complete operator lemmas and explicit open assumptions, Python reference, the original Torch compiled/serial forward and backward verification with finite-precision tolerances, critical-path FLOP/activation accounting, and genuine real-source original-checkpoint evidence. No proof of action effects without identified intervention data. No claim of independent generalization without untouched authentic test cohorts.

This is a measurable path toward an SSM/Mamba level of *methodological* rigor while keeping TAC-OSM's distinct persistent, associative and decision-oriented architecture.
