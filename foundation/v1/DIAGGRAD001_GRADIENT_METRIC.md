# DIAGGRAD001: diagnostic supplement to ME106

This adds no trained variant, forward function, state, weights, loss or optimizer
change. ME106's original reciprocal source entry remains authoritative and
unchanged. Actual diagnostic source was frozen at TAC-OSM `2d5edb8c`; inherited
model/checkpoints at `e6ac736a`, branch `research/diaggrad001-loss-flow-2026-10-26`.
Full findings live in TAC's `research/mathematical-engineering/diaggrad001/RESULTS.md`.

`PNDS/GradientMetricCounterexample.lean` and
`foundation/v1/gradient_metric_reference.py` express the same exact arithmetic:
p=(1,1), u=(2,-2), <p,u>=0; combined g=(3,-1) satisfies <p,g>=2>0.
D=diag(1,100) gives <p,Dg>=-97, so the negative scaled update has point dot+97.
This proves a limitation of a Euclidean gradient-direction guarantee under unequal
coordinate scaling. It is not the original trained model's measured behavior.

All four Lean declarations compile; axioms are propext, Classical.choice and
Quot.sound only. The exact-rational Python test checks identical constants.
The inherited ME106 proof correctly excludes AdamW descent. No proof is weakened
or replaced; whole-neural, Gaussian-law and IEEE refinement remain OPEN.

Actual FIT-only probes: all15 frozen ME106 models, 180 batches across8/64/256h,
two origin populations and two repeat batches. No new fits/optimizer steps,
REPORT or protected numerical data access. NLL/point global negativecos0/180;
point/interval139/180; saved-AdamW hypothetical point-uphill34/180. Pure group-rate
SGD global uphill0/180 in the actual probes, so the exact counterexample must
not be described as an observed group-rate reversal in those models.

The diagnostic's PyTorch test separately checks read-only saved-moment AdamW
algebra against a real optimizer step on two toy parameters. That is numerical
unit evidence, not a formal AdamW implementation or a real model training step.
The actual shadow direction is evaluated in FP64; finite-step loss changes and
causal attribution of failed model qualification are not measured.
