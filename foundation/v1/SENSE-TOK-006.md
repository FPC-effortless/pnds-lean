# SENSE-TOK-006 preparation: partial mathematical source record

TAC-OSM draft PR #675 owns the actual source and no-fit evidence. The exact
implementation commit and file digests are in the prospective study record
in `variant-math-links.json`, mirrored into TAC's reciprocal manifest.

This preparation is listed separately from the 17 immutable historical
variant links. It is not silently added to their original shared-kernel
coverage or relabelled as mathematically complete.

The Lean and executable Python files formulate three limited objects:

1. Selected code lookup and fixed-key value permutation. Jointly permuting
   the code values and inversely relabelling the selected ID preserves the
   representation. The selected q is a premise. Encoder argmax uniqueness,
   tie handling and actual source execution are not proved by this identity.
2. Realized regret: selected horizon BPB minus min(BYTE BPB, TOKEN BPB).
   Nonnegativity does not establish expected decision quality. Horizon
   availability, target alignment and teacher-forced continuation are checked
   in the TAC evaluator, not proved by a scalar identity here.
3. Dense BYTE's declared embedding/GRU/readout parameter count, 100,688.
   This does not prove active-gradient coverage or full training parity.

TAC's dense BYTE graph has observed previous IDs in {BOS,0..255}, 32-wide
embeddings, a 118-wide GRU and a 256-way affine head on concatenated state
and current-byte embedding. S004/S005 preserve their original 110-wide
recurrent/pending state and byte-local decoder. The four derived codebook
controls change forcing, gate training status or exogenous routing only;
their native numerical/gradient/source correspondence remains OPEN.

Whole-model Lean/Python formulation, original-source forward/backward and
finite-precision correspondence, explicit source-specific probe adapters,
serial/parallel equivalence and full cost accounting remain OPEN. No theorem,
unit check or status record grants training, sealed-TEST or promotion authority.
There are zero new model fits and no capability or learned token result.

Python verification: `python -m unittest discover -s tests -v`.
Lean verification uses the repository's Lean 4.14/mathlib toolchain and
`lake build`; this local environment had no installed toolchain, so no local
Lean build or axiom-inspection pass is claimed. CI results must be checked.
