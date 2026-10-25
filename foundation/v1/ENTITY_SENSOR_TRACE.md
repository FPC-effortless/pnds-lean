# Source-aligned TAC-Temporal kernel and numerical refinement

ME100 uses one causal per-channel affine sum/mass trace: S'=(1-beta)S+beta*z, C'=(1-beta)C+beta, |z|<=1, 0<=beta<=1. A missing channel sets beta=0. PNDS/SourceAlignedTrace.lean proves the invariant |S|<=C<=1, no-observation identity, associative affine composition and its action. This is an exact-real kernel proof, not a proof of all neural nonlinearities or IEEE arithmetic.

The independent serial evaluator in entity_sensor_reference.py evaluates source traces, rank-one associative delta memory, fast/slow feature maps, retrieval and final location/scale from actual neural weights without calling the production scan. Its source is byte-identical to TAC scripts/me100_graph_reference.py. The TAC unit test compares full outputs, state and all parameter gradients in FP32/FP64. Original real-data fitted weights will be preserved in TAC; replay will additionally check their numerical refinement on FIT observations.

Declared primary state is152FP32values, with integer owner/count/clock and source-age bookkeeping charged separately. Gates remain input/prefix-evidence-conditioned; reads do not feed back into the same matrix scan. Natural missing source cells are not proof of delayed arrival or counterfactual control. Exact chronological algebra does not establish predictive superiority.
