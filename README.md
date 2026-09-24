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
| `PNDS/Scaling.lean` | a `K`-independent distractor-exceedance lower bound | §18 |
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
