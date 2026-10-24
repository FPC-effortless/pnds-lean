# Unified TAC-OSM mathematical computational-graph specification (v1)

## 1. Typed world-state interface, NOT one forced architecture

For each research lane \(\ell\), define an independent state space \(\mathcal S_\ell\), observation \(o_t\), known-at-time action/context \(a_{<t}\), time increment \(\Delta t\), and transition

\[
S_{t+1}^{(\ell)}=F_{\theta,\ell}(S_t^{(\ell)},o_t,a_{<t},\Delta t).
\]

Readouts may describe an event/END distribution, real physical next-state distribution, memory retrieval, or a model-independent governance predicate. These spaces and objectives cannot be conflated.

- **Core native:** \(S=(f,s,M)\): nonlinear fast/slow recurrence and bounded 8×8 gated-delta associative state, updated on observation; DIAGKEY004 is an aligned write-query *synthetic immediate-recall* candidate, not a proven physical state predictor. A query can be read-only.
- **System 1:** native fast-only physically conditioned recurrence and movement/still mixture for ME077, not the DIAGKEY004 slow/association stack. Past issued actions are observational conditioning, not \(do(a)\).
- **Physical:** native 8-frame state plus future horizon-conditioned multimodal probabilistic readouts (ME080–094). ME094 distribution translation aligns its stated median with its probability distribution; no proof of improved physical prediction follows.
- **ME095 proposal:** adjunct diagonally gated **affine** state filter; only the filter's transition is scan-associative, NOT the full nonlinear fast/slow/associative graph.
- **Enterprise:** native \(f,s,M\), activity, continuation and positive/zero-duration stochastic event-time law. Absorbing **end of source record (EOR)** is not verified business completion. Free-running particles approximate multi-step marginals.
- **ES-WAM:** per source-identified actor, observed 256-frame input windows, selective fast/slow aggregations and entity-relative physical correction. Its window-fed attention-like readout is not infinitely carried state. No arbitrary actor ID induction or verified intervention model.
- **Temporal:** valid-time/record-time reconstruction with authenticated as-known-at eligibility, interface only as neural model.
- **Counterfactual:** \(P(y\mid s,a)\neq P(y\mid s,do(a))\) without causal identification; existing advisory interface is not an identified intervention law.
- **Edge-Lite:** same semantics under declared exact/approx numerical transformation and measured hardware execution; it is NOT a uniquely trained fourth recurrent state.

## 2. Shared proofs by *narrow*, explicit operator assumptions

### Gated scalar memory and delta error

\[
h'= (1-g)h+gv,\quad g\in[0,1].
\]

For \(h,v\in[-B,B]\), \(h'\in[-B,B]\). At fixed target \(v\),

\[
v-h'=(1-g)(v-h).
\]

The bound prevents explosion but for repeated \(g>0\) may destroy older information; no guarantee of useful long-range retention is implied.

The core matrix candidate is

\[
D=\alpha M,\quad
M'=\Pi_{\|M\|_F\le R}\{D+\beta(v-Dk)k^\top\}.
\]

Its projection radius bounds matrix magnitude, **not** information-theoretic capacity, interference, or retrieval error. Formalize true retention only with assumptions on address similarity, write admission, decay, finite precision and query-key alignment.

### Affine filter and parallel scan

\[
k'=Ak+B,\quad (A_2,B_2)\circ(A_1,B_1)
=(A_2 A_1,A_2B_1+B_2).
\]

Composition is associative over real arithmetic; serial execution and composing fixed affine steps agree. For an input-conditioned affine filter these steps must be computed from inputs **without depending on the state being scanned**. The ME095 filter meets that form under its registered design, while the nonlinear fast/slow update generally does not. Floating-point summation order can change results and gradients.

### Event source continuation

\[
P(\mathrm{EOR})=1-s,\quad
P(a,\Delta t)=s\,p(a)\,q(\Delta t\mid a).
\]

For \(s\in[0,1]\), normalized activity distribution and normalized activity-conditional duration densities, the complete one-step law is normalized. This does not confer calibrated probabilities or semantic verified completion.

### Coherent physical prediction

If \(B(q)=1/2\) and \(F(y)=B(y-m+q)\), then \(F(m)=1/2\) **under exact continuous-distribution assumptions**. ME094's realization requires approximate root solving and tested implicit differentiation. The theorem cannot be carried over to a different decoder without the premises.

### Motion selection

With \(c>0\), \(\gamma(v)=v^2/(v^2+c)\) obeys \(\gamma(0)=0\); at nonzero noisy speed the residual is not automatically zero. ESWAM014's explicit zero-speed fallback does **not** prove near-stationary no-regression.

## 3. Whole-graph obligations before any model capability claim

1. Record the exact source-tree/version hash, typed input dimensions, initial states, memory writes, output probabilities, known-at-time action flow and missingness.
2. Formally analyze each operator and its assumptions, then try composing whole-graph bounds and time-index constraints.
3. Classify work as pointwise parallel, scan-parallel affine, or sequential nonlinear; profile real FLOPs and memory, not just symbolic complexity.
4. Add Python and Lean numerical correspondence (forward, backward, finite precision tolerances, edge cases, CPU/GPU parity), with unmodified failure gates.
5. Fit new models only with preregistered authentic source, real matched ~100k learned parameters and equal data/compute; independent replay and untouched site/entity TEST.
6. Evaluate relevant baselines and actual downstream outcome costs; do not promote synthetic recall or exposed DEV wins to general intelligence.

This document is an operator-level scientific *contract*, not a claim that a complete full-model Lean proof has already been delivered.
