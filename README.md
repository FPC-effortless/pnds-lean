PNDS - Lean 4 Formalization
============================

Formalization of the mathematical representation in
`PNDS_Mathematical_Research_Representation_v0.2.docx` (§3, §5, §10, §12, §14, §18, §20).

What is actually proved here
----------------------------

Every theorem below is a real `theorem` with a complete proof — no `sorry`, no `admit`,
no `sorry`-bearing intermediate. Each corresponds to a numbered claim in the document.

| File | Statement | Doc ref |
| ---- | --------- | ------- |
| `PNDS/Softmax.lean` | the softmax selection-margin bound | §5 |
| `PNDS/PathVerification.lean` | `V_P = Π v_i` is the joint iff the `v_i` are conditional; the `0.99^n` decay | §10 |
| `PNDS/Registration.lean` | admission iff all four gates; Bayes inversion of `P_contam` vs `P_false_admit` | §12, §14 |
| `PNDS/Scaling.lean` | the §18 distractor-count identity `|TopK \ R| = K - ρ`, the recall-1 corollary `= K - r`, and the control showing the `≤ K - r` bound is a recall consequence, not a cardinality one | §18 |
| `PNDS/Statistics.lean` | sign-test minimum attainable p; 3 seeds cannot reach `p < 0.05` | §20 |

What is deliberately **not** proved
-----------------------------------

The following are named as `sorry` so that `#print axioms` and the CI gate fail loudly,
rather than the build quietly passing on unverified content.

* `§9 do(·)` on internal substrate nodes — the document states no SCM, so the
  interventional target is not well defined (see `PNDS/Causal.lean`).
* `§24` theorem candidates — the document itself labels these "theorem candidates,
  not established theorems".
* Any transition/invariant on the `ρ ∈ {E,P,R,Q,T}` status alphabet — the document
  states an admission condition only, no transition function, no retention rule.
* **The §18 scaling claim itself** — `C(R ∪ I_N) → C(R)` as `|I_N| → ∞`. What
  `PNDS/Scaling.lean` proves is the *combinatorial core*: `|TopK \ R| = K - ρ` where
  `ρ` is the recall, hence `|TopK \ R| = K - r` at recall 1. The document's asymptotic
  claim is a statement about the *scoring function* (the index must exclude distractors
  strongly enough that `K` stays bounded), and the recall hypothesis that would make it
  true is *not* a consequence of cardinality — see `distractors_in_topK_ge_of_missed`,
  which shows the bound fails outright once a relevant item is missed. No limit, no
  asymptotic, and no bound involving `N` is proved anywhere in this repo.
* **The §5 → §18 bridge** — the selection-margin bound of §5 constrains *softmax mass*,
  not *top-`K` membership*. Turning "distractor share `≤ ε`" into "recall `≥ r`" is a
  probabilistic statement requiring a model of the scores, which the document does not
  supply. `PNDS/Softmax.lean` and `PNDS/Scaling.lean` are therefore independent results;
  no theorem here connects them.

Build
-----

```bash
lake build
```

CI
--

`.github/workflows/lean.yml` builds on `ubuntu-latest` with Lean from `elan-actions/setup-lean`.
The workflow is **fast and cheap** (no matrix, no `--wfail`): a clean Lean build of this
size takes well under 10 minutes on a GitHub-hosted runner.

Empirical status is separate from this repo
-------------------------------------------

**Nothing in this repository is evidence for an empirical claim.** This repo contains
formal theorems about the *document's own definitions*. A proof here says the definitions
entail the stated conclusion; it says nothing about whether any trained model satisfies the
definitions, or whether the conclusion holds empirically.

In particular, the C5 "cost reduction" hypothesis — that routing cost is sublinear in
history length, or that accuracy is preserved as `N` grows at bounded `K` — is **not**
established by `PNDS/Scaling.lean`, and must not be presented as if it were. The module
docstring of `PNDS/Scaling.lean` states this as well. The measured scaling results the
project has (a six-point pilot on retrieval top-1 accuracy at 6 / 12 / 24 candidates) are
recorded in the experiment repository, are explicitly *not* a scaling result, and are
unchanged by anything proved here.
