# CSL-125 mathematical reference — PARTIAL, NOT FORMALLY QUALIFIED

Paired with TAC-OSM CSL-125 draft PR #711 at exact implementation commit ed9913d33da0cd9d94dc985f946ebe673799d41e and SHA-256 0bf0ba779cf570084179675dd83a8b2a423867275282a129359a5f5a5828e85b.

The Python/NumPy reference covers serial EMA and causal bounded GQA selection with explicit KV groups and absolute positions. The Lean module proves only two discrete causal-window properties and the fixed 123699534 <= 125000000 integer budget, without sorry. It does not model the entire source, learned tokenization, finite-precision recurrences, parameter gradient parity, multi-entity state, data, decoding, training or real-world outcomes.

Local pilot source checks used a CSL-001 compatibility fixture; no independent PNDS Lean build or full source correspondence was run. Canonical reciprocal manifests remain unchanged; append a reviewed variant row to both only after exact correspondence checks and approval. No 125M training or bypass of the TAC-OSM 100k policy is authorized.
