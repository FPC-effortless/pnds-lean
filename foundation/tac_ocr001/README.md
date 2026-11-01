# TAC-OCR-001 mathematical record

This is a **partial** formulation of the vision-grounded OCR decoder graph.
TAC code owns image recognition and execution evidence. No trained OCR
capability, architecture superiority, or full-source mathematical proof is
claimed here.

The independent scalar Python reference covers one visual cross-attention
head, including stabilized softmax. Tests compare it to the actual SDPA
path and one selected finite-difference derivative. The inherited full TAC
reference is the separately pinned LANGCACHE104 record. New bridge/core
composition and the optional TorchScript lowering are covered by numerical
fixtures only; complete independent whole-model source correspondence stays
OPEN.

`PNDS/TACOCR001.lean` defines typed vector/matrix interfaces, executable
Float attention and positive-bank formulas, an explicit layer interface,
and a scoped reference-count theorem. It does not prove floating-point
stability, pooling information retention, or learned transcription fidelity.
The list-capacity theorem is not a theorem about every PyTorch allocation.

See `SOURCE_MAP.json` for exact TAC implementation and mathematical hashes.
Tests and compilation receipts must retain their actual status.
