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

## Unified TAC-OSM mathematical research foundation (v1.0.0, October 24 2026)

The existing PNDS v0.2 document formalization **remains unchanged**.
The new [source-commit-pinned unified foundation](foundation/v1/README.md)
is the mathematical-engineering home for **all TAC-OSM research lanes**:
Core/DIAGKEY, System 1, full physical, Enterprise, Temporal, Edge-Lite,
Counterfactual and ES-WAM. It uses a versioned registry, append-only
positive/negative findings, explicit archival gaps, proof obligations and
original-source cross-references. **TAC-OSM remains the only source of the
actual model source files, original data, fitted checkpoint bytes and results.**

- [Versioned architecture/variants](foundation/v1/registry.json)
- [Mathematical operator graph](foundation/v1/ARCHITECTURE.md)
- [Proof/qualification obligations](foundation/v1/PROOF_OBLIGATIONS.md)
- [Append-only original research evidence ledger](foundation/v1/evidence-ledger.jsonl)
- [Historical gaps rather than false completeness](foundation/v1/archive-gaps.json)
- [Prior original PNDS math and open formal branches](foundation/v1/prior-math.json)
- [New exact-real Lean operator proofs](PNDS/UnifiedTACOSM.lean)

```bash
python3 scripts/validate_foundation.py --strict
python3 -m unittest discover -s tests -p 'test_foundation.py'
lake build PNDS Audit
```

A Lean proof about exact affine scans or scalar convex updates is **not** an
end-to-end proof of native PyTorch floating-point operation, real-world model
intelligence or causality. Historical negative outcomes never disappear when
new improvements are appended.
