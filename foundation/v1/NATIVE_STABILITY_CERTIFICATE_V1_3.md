# TAC-OSM native forward-state stability and retention certificate v1.3

**Scope:** formal conditional boundedness for original native fast/slow/associative state; exact-real scalar norm bound; source-tested implementation diagnostics. **Not:** a global contraction/ISS/Lyapunov proof, useful long-memory guarantee, trained model result or general field-world stability certificate.

Pinned original implementation: [TAC-OSM DIAGKEY004 source](https://github.com/FPC-effortless/TAC-OSM/blob/081e534a074001c6a3b2251b3ff31ad20ec4fca8/src/tac_osm/aligned_associative_tac.py). Companion source tests [TAC-OSM PR #389](https://github.com/FPC-effortless/TAC-OSM/pull/389).

## Source-level facts and explicit mathematical assumptions

This exact model initializes \(f_0,s_0=0\) with 88 real coordinates each, and \(M_0=0\) with 8×8 values. For each event:
1. Old \(M_{t-1} k_t\) contributes to fast and slow candidates. This read is performed before the write.
2. The fast target is tanh-bounded and mixed with \(f_{t-1}\) at a sigmoid gate in (0,1).
3. The slow target is tanh-bounded and mixed with \(s_{t-1}\) at a gate in (0,1/4).
4. A writable event updates and globally Frobenius-projects the associative matrix into radius 4; a read-only event preserves \(M\) exactly. Read-only does **not** freeze the fast/slow registers.

In exact arithmetic, with the true source target/gate/radial properties, every forward step preserves

\[
 \lVert f_t\rVert_\infty\le1,\quad
 \lVert s_t\rVert_\infty\le1,\quad
 \lVert M_t\rVert_F\le4.
\]

The new Lean \`PNDS/NativeInvariant.lean\` proves bounded scalar convex mixing, an 88+88+8×8 **source-shaped complete-register forward-envelope implication** and composition for arbitrarily many steps, assuming bounded candidates/gates and a radial projection with per-entry postcondition. It also proves the positive-radius scalar clipping norm bound for \(\|U\|_F / \max(1,\|U\|_F/4)\). Full equality of the floating-point tensor norm and exact-real projection is an explicit additional numerical obligation; a proof of the abstract source-shaped theorem is not a proof of all original PyTorch kernels.

No learned parameter norm or time-uniform Jacobian bound is assumed in the boundedness theorem. Bounded outputs **cannot** establish small sensitivities to inputs/history. It is mathematically incorrect to call this a global contraction proof.

## A rigorous limit to full-state contraction

When the observed event is role 2 (read-only), the actual source enforces \(M'=M\) for all valid \(M\), regardless of changes to fast and slow registers.

Consequently, choosing two states with identical fast/slow and different memory registers leaves a memory separation of unchanged magnitude after one read-only event. Under a product norm which includes that memory difference without cancellation, no \(\rho<1\) uniformly bounds the ratio of new to old **whole-state** distances. Lean proves the exact read-only scalar identity and its incompatibility with a universal strict contraction in the preserved memory coordinates. The PyTorch whole-state numerical-JVP check independently exposes the identity memory block.

This is **not a flaw that must be removed**. A useful world-state model needs protected slow facts, yet should permit robust fast-state adaptation. The correct prospective research target is *partially stable/contractive adaptive fast dynamics plus explicit protected memory invariants*, not "contract everything".

## Global budget projection couples otherwise orthogonal facts

For an unprojected writable update,

\[
 U = \alpha M+\beta(v-\alpha Mk)k^\top,\quad
 Uq=\alpha Mq+\beta(v-\alpha Mk)\langle k,q\rangle.
\]

When \(k\cdot q=0\), the added innovation cannot directly change the old read at q, but retention still multiplies it by \(\alpha\).

However, the actual source then applies the **same global Frobenius factor to every matrix entry**. An exact admissible boundary example uses \(\|M\|_F=4\), \(M_{00}=4\), a unit write key \(k=e_1\) orthogonal to query \(q=e_0\), unit value \(v=e_0\), \(\beta=1\) and \(\alpha=0.9999\). The unprojected matrix has entries \(U_{00}=3.9996\), \(U_{01}=1\). Its norm \(\sqrt{3.9996^2+1}>4\), so projection rescales **both** columns.

The old read is approximately 3.88055 rather than 3.9996. This is a numerical mechanism/counterexample using an actually admissible \(\|v\|_2=\|k\|_2=1\); it is not a measurement of trained recall. The source test reproduces that coupling.

Therefore the previous orthogonal-keys-retain-only-decay theorem is **preprojection only**, and cannot be lifted to the implemented global Frobenius projection without an added no-clipping condition or a certified error term.

## Scientifically meaningful next design decision

One larger, capability-bearing hypothesis deserves evaluation rather than many disconnected tweaks: preserve entity-aligned old facts through a *separately budgeted* memory identity/retention path, while controlling new writable updates and an adaptive fast component. Propose candidate budgets, protective gates and interference limits **before** touching model weights. Never claim this is a working improvement from a formal proof alone.

For any actual prospective training study use original authentic observations with identifiable entities, state changes, actions and verified outcomes; compare to native TAC, useful structured entity memory and matched neural controls under identical data/parameter/training-compute/tuning/split conditions. CPU elapsed time is an efficiency measurement, not a fairness gate. The original known memory and physical/stationary performance failures must remain visible.

## Release checklist / current status

| Property | Status |
|---|---|
| Scalar convex gating/candidate boundedness in real arithmetic | Lean conditional theorem |
| 88+88 register and 8×8 memory componentwise envelope | Lean theorem under explicit projection/gate assumptions |
| Multistep preservation under same premises | Lean induction theorem |
| Source radial norm scaling (scalar norm) | Lean exact-real scalar bound |
| Actual source forward upper bounds, masking, JVP and orthogonal clipping case | PyTorch mechanism checks in TAC PR #389 |
| Full PyTorch tensor floating-point refinement / complete nonlinear stability | OPEN |
| Jacobian contraction of entire native state | **Not possible uniformly under exact read-only memory preservation** |
| Retrieval accuracy after authentic long distractor histories | OPEN |
| Trained real-world predictive advantage | Not established by this mathematics |

This certificate must remain separate from the 17-variant original experimental ledger, cannot certify a different physical or enterprise architecture by inheritance, and cannot be used to upgrade a synthetic mechanism to real-world capability.
