# Lacuna frozen FIT P0: full timing record, partial formulations

TAC source c5df300cf0d565ba7f9c9622a31821ba812333df, PR637, original run40997601362.
Native trained graph is UNCHANGED LANGMEM086_GRAPH.md,100521parameters,
480FP32numbers/1920B tensorstate +8Bclock, one8x8 fast/slow pair. Its original
value/key/query/write/read/concept/upper/decoder timing and owners are inherited.
Original087640 checkpoints are replayed, never fitted. Actor forward accepts
observed current byte and its own state. Future targets only enter scorer.

Telemetry D_j has type diagnostic_history×8×8 Float64, outside deployed model.
Each observed current sourcebyte creates C_j=eta_j*v_j*k_(j-1)^T. Given original
future trajectory keys/gains, transport D <- rho(D-eta(Dk)k^T). Their sum
matches actual slow S. Exact-real additivity/decomposition Lean proofs apply
to these CONDITIONALLY FIXED coefficients. Full intervention feedback through
future queries/gates is not included. Fast clipping depends on previous F and
is not an unconditional affine recurrence.

Forensic supplied historical key/read of full S and isolated D are diagnostics,
not autonomous model inputs or a byte decoder head. Rank-one isolated value
identity is structurally retained up to scalar; this is not byte recovery from
compressed S. Old-event removal patches only original prior S and recomputes
normal current write; wrong events are age-local, exclude requested byte from
wrong4bytecontext and have matched contribution norm. Other priorstate fixed;
newcounterfactual state discarded. AD/FD checks concern local predicted-class
logprob sensitivity on actual source; no optimizer step or `.backward`.

Lookback uses genuine raw-ring1928B and observed fourbytecontext counts.
Full reference distribution mixes native+ring at.5 only if count>=2. Entropy,
current-query/previous-key overlap proxy, and predetermined random-offset
policies make exactlyone query per4observedbytes. Last step forced if none
selected earlier; no future-observation matching. Shared static FITprior2048B
is outside perownerpayload. Native1928+ring1928+flags2=3858B logicalpayload,
not1928B integrated memory. Python metadata/workspace/RSS separate.

Separate packed context buffers are256B each:24header,25x9B slots,7unusedbytes
charged. FIFO/reservoir/diversity are fixed causal component baselines, not
learned swaprounding or eviction-to-matrix conservation. Reference Kalman P
usesFloat64 cov512B plus separate referenceM512B; positivity tested only on
actual FIT-key traces. No new neural covariance graph is integrated. Literal
bracket audit is not a parser, syntax labels or model input.

Lean: transport additivity, source update split, zero erasure, normalized
nonnegative convex probability mixture, integerpayload/querycounts only.
Python independently implements named kernels. Existing087 reference checks
actual Float32 slowstep at real clocks. Numerical source masserror1.14e-7,
stepreference6.89e-9; local FD discrepancies<=3.90e-4 at prereg2e-3tolerance.
Full neural IEEE/AD, optimized kernels, whole-source parallelscan, semantics,
learned gate/slot policy, global budget qualification and independent science
remain OPEN. Outcomes show retained numerical traces but no preregistered
consistent positive old-event NLL utility. No theorem upgrades that negative
FIT mechanism result or stops future redesigned architectures in general.
