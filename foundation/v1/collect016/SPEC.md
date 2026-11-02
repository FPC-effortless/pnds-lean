# COLLECT016 typed graph and credit contract

Original fit source: TAC-OSM `2209e1328e3b42db4dd48cd702a21a55ffad24af`,
draft PR https://github.com/FPC-effortless/TAC-OSM/pull/700.
Lineage: COLLECT014, COLLECT012 and actual COLLECT011/TAC110 source, not a new
state-space-model rewrite. CPU FP32 fits; numerical fixtures also use FP64.

Input is byte IDs Long[B,T], B=2 and T=384 for training, split into three 128-byte
calls. Output is Float[B,T,256] next-byte logits. Ordered unique owner strings,
state/config signatures and synchronized integer clocks are required. Labels and
copy-query masks enter loss selection only; neither is a model argument or write
mask. All observations are causally observed before the corresponding next-byte
prediction. TEST is unavailable.

Four width-16 layers contain native fast/slow associative states and mass,
rotary field, FFN, and a mid-stack 32-slot SSA memory. These states are not
collapsed into the phase matrix. The phase lane has 4 heads with key/value
dimensions 16. Learned content/role projections produce normalized sine/cosine
keys; value projections are separate. Hard bank selection is one-hot argmax in
the forward path, with an explicitly declared softmax straight-through surrogate
in reverse AD. This is not the ordinary derivative of hard argmax.

For each head, the inherited operator is:

```
z=P k; leverage=k^T z; u=z/max(leverage,1e-8)
g=sigmoid(admit(x))*write_mask*route
S'=S+g*(v-S k)*u^T
P'=sym(P-g*z*z^T/(1+g*leverage))
output=(S' k)*route
```

At g=0 both state tensors are frozen exactly by a conditional selection. This
normalized write is NOT exact recursive least squares. The source validates
finite, symmetric positive definite incoming P. There is no general long-horizon
IEEE stability proof. The reference reproduces floor and conditional semantics.

The full_bptt and detach_128 graphs have identical forward equations. At boundaries
the latter explicitly stops VJPs through every carried state tensor. Connected
full BPTT merely exposes the 384-byte path; it does not guarantee a useful or
well-conditioned gradient. Both reset between independent episodes, with no
weight update inside an episode.

The current_only phase lane computes route*sigmoid(admit)*write_mask*route*v.
It removes content/role matrices and all historical phase reads/writes; constant
placeholder phase state preserves the existing state API. Native other memory
paths remain. FFN dimension rises from 57 to 78 to reallocate removed weights to
active computation: 99,907 versus 99,971 parameters. All compatible initializer
tensors and original FFN slices are shared. This is not a pure shape-identical
intervention or a history-free model. It is a separately trained capacity-matched
alternative; measured FLOPs parity remains OPEN.

The scalar loss is natural-byte CE + .25*task-sequence CE + answer-byte CE +
.01*mean Switch balancing. Every step has one real source window and one
constructed temporary-key copy episode with verbatim source quotes. Correct
source offsets are retained. TRAIN and DEV bytes and first-character key families
are disjoint. Four answer bytes are supervised; future answers never enter the
prefix used to predict the first answer byte. Training absence occurs in 1/5
steps; overwrite in 1/3. Ordinary-text DEV is a separate primary source measure;
the copy interface is constructed mechanism evidence.

Python reference code expands the whole native/phase/SSA/FFN output graph using
inherited independent-expression TAC104/110 references and does not call model,
block or memory forward methods. Finite fixtures check values, all carried fields
and parameter surrogate gradients in FP32/FP64, plus masks, chunk boundaries,
owners and future poison. These checks are finite engineering evidence.

Lean `PNDS/COLLECT016.lean` provides exact-real local current-value/loss definitions
and a typed causal recurrence interface. Its stopGradientPair is a model of local
AD value/tangent behavior, not a theorem that identity has zero real derivative.
Two reflexive definitional lemmas do NOT prove actual graph/source equivalence.
Full Lean graph, kernel build/axiom inspection, IEEE analysis, full AD proof,
scientific generalization and attention equivalence remain OPEN. No `sorry` is
introduced to disguise those obligations. Python/source success cannot close them.

Source/dataset/parameter/journal/checkpoint audits and actual fit results live in
TAC-OSM. A declaration, successful compilation or copied hash is not independent
authentication of model intelligence, real-world utility or source provenance.
