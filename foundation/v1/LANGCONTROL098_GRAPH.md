# LANGCONTROL098: coefficient-controlled prefix-supervision contrast

This is an objective-only follow-up to TAC-OSM PR #665 / PNDS PR #96. It keeps
all four original inference graphs and their 100521 TAC / 100661 GRU / 100404
SSM / 100484 Transformer active parameter counts. The architecture, typed state,
causal input-output graph and acquisition-gradient paths are inherited from
`LANGACQUIRE097_GRAPH.md`. This record adds loss semantics; it does not substitute
LANGMATH098's new contextual-bank model for the original TAC086 graph.

## Source boundary and typed loss graph

Training receives original byte episodes A (acquisition), Q (query) and Y
(future target), forming R=A||Q||Y, model inputs X=R[:-1] and targets Z=R[1:].
The actual model receives only byte IDs `Long[B,T]`, owned causal state and
legitimate stream metadata. Future labels, domain IDs and masks are loss-only.
The first Y target is predicted after the last Q input, with ordinary causal
teacher forcing only during training. Evaluation must generate its own bytes.

The loss receives logits `Float[B,T,256]`, targets `Long[B,T]`, valid/query
Boolean masks `[B,T]`, and one loss-only domain label per episode. End padding
is contiguous. Every episode has at least one future and one prefix target;
both lexical and ARC domains are present. Future masks cannot include padding.
Define P_i=valid_i \\ Y_i. For each registered minibatch let n_d be its number
of episodes from domain d. Domain weights are fixed at 1/2.

L_lambda = sum_d (1/2) * mean_(i in d) [mean_(t in Y_i) CE_it
                                    + lambda * mean_(t in P_i) CE_it]

Lambda is exactly 0 (local receipt label `query`) or 1 (label `full`). The local
labels are retained for runner compatibility; `full` no longer means a global
mean over all valid tokens. Neither arm divides by 1+lambda, active token count
or surviving coefficient mass. At fixed logits, each future coefficient is
(1/2)/(n_d*|Y_i|) in both arms; prefix coefficients are lambda*(1/2)/(n_d*|P_i|),
and padding coefficients are zero. Consequently total coefficient mass is 1
for lambda=0 and 2 for lambda=1. Both arms compute the complete CE array and
use the same observed sequence geometry. Their explicit answer coefficients
are equal; their total gradient magnitudes and nonlinear optimizer effects need
not be. AdamW/clipping statistics require empirical reporting.

## Scope of the two formulations

`PNDS/LANGCONTROL098.lean` formalizes token coefficients, fixed future weights,
zero prefix/padding weights, positivity under positive count premises, the exact
prefix-only objective contrast, zero-weight direct loss irrelevance, and an
explicitly assumed scalar derivative rule before optimizer transformation.
It does not prove that production masks/counts are correct: actual-source
validation and the source-to-reference tests supply that separate evidence.

`langcontrol098_reference.py` independently computes coefficient arrays,
stable cross entropy and analytic dL/dlogits using NumPy. Tests use unequal
segment lengths and unequal domain cardinalities, scalar episode averaging,
finite differences, prefix/padding poisoning and malformed input rejection.
The reference neither calls the source Torch loss nor reconstructs a model.
It inherits the fixed-coefficient objective semantics already present in
PNDS PR #97's NumPy `controlled_loss`; that prior implementation concerns a
different new inference architecture and supplies no learned results here.

The original serial chunk and acquisition-chain laws remain in
`PNDS/LANGACQUIRE097.lean`; later target gradients may reach earlier state even
when prefix logits have no direct loss. This experiment isolates adding prefix
supervision at fixed future coefficients. It does not uniquely isolate the
existence of acquisition learning or establish the foundations of intelligence.

## Remaining obligations

Actual source commit and per-file digests must be pinned in
`langcontrol098-source-link.json` and reciprocal TAC metadata before any new fit.
Full tensor/source-equivalent model composition, IEEE floating point, all-source
AD certification and canonical main/master integration remain OPEN. Learned
capability, retention utility, transfer superiority, resource advantage and
independent qualification are not inferred from these scoped equations/tests.
