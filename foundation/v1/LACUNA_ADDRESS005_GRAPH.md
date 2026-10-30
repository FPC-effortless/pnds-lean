# LACUNA-ADDRESS005 — fresh-source conditional address test, PARTIAL

Original100521parameter086computational graph, original640checkpoints, owner
state1920Btensors+logicalclock; no training. Full inherited graph is pinned by
source SHA7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1.
Observation→embedding/position/lexical→writer/query/gates→fast/slow/aligned/wave
/basis/concepts→native/features/evidence→norm/upper/FFN→256logits is unchanged.
All counterfactual states are discarded; only the original path advances.

Evaluator inputs h=Hq,c=Cq,f=fastread,v=candidate:[8]. Two-pass float64
Gram–Schmidt yields Q[8,r],r<=2 spanning current and fast directions. Projection
removes Q components from h/v; T(v)=fixedpart(h)+norm(perp(h))*normalize(perp(v)).
Zero/nearzero/dependent axes and zero radii/candidates have explicit flags and
no useful-direction gate can pass if a compared control falls back. All shapes
and finite values are checked. Resulting total slow read T(v)+c is castfloat32;
the actual mixed source vector is patched at fast/slow/norm consumers as003.

Fixing historical norm and dot products against c and f fixes total slow and
mixed fast/slow norms. Lean proves exact-real mixed-norm equality under those
hypotheses and preservation of an axis projection for an orthogonal tangent.
Independent list Python implements basis construction and projection. Actual
source tests check all3norms before/after nativecast, identity/zero read, prefix
future poison and local composed projection→decoder VJP/FD. No global NN proof.

Six prospectively frozen original PSF/Rust documents,3perpublisher; no labels
select cohort or controls. Primary supplied-key/raw correct-minus-wrong NLL
age8–31threshold0.001perpublisher,20distinct source cells perowner, positive
eachowner/seed, plus meaningful shuffled/irrelevant/random-direction controls.
Native query and generic random controls are also scored at EVERY query clock,
including no-match positions. Key selection is evaluated separately against a
uniform prefix-only key. Seeds repeat source outcomes, never source sample size.

Native slow bank256B, raw reference ring256B (8clock+248bytes); evaluator stores
reconstructed keys/values and conditional coefficients as separately charged
workspace. No equal-full-state autonomous model efficiency or deployment claim.

OPEN: complete basis/radius/normalization finite-precision proof, full NN IEEE/AD,
all-parameter gradient/source equivalence, global noninterference, independent
qualification and legacy canonical-index migration. Exact algebra/constructed
tests cannot establish predictive capability. Old failed gates stayunchanged;
no result automatically authorizes a trained candidate.
