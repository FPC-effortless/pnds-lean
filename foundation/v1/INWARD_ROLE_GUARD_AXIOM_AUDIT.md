# Inward role guard Lean axiom audit

The audit module is `PNDS/InwardRoleGuardAudit.lean`. It prints axioms for
both new theorems. Run after building the imported module:

```sh
ELAN_HOME=/tmp/elan /tmp/elan/bin/lake build PNDS.InwardRoleGuard
ELAN_HOME=/tmp/elan /tmp/elan/bin/lake env lean PNDS/InwardRoleGuardAudit.lean
```

Output for both `PNDS.InwardRoleGuard.inwardProjection_dot_le` and
`PNDS.InwardRoleGuard.rounded_dot_nonpos`:

```text
[propext, Classical.choice, Quot.sound]
```

No task-local axiom, `sorry`, or `admit` occurs in either proof. The listed
axioms are Lean's standard foundational axioms. This kernel audit covers only
these exact-real/scalar statements; it does not certify the runtime's floating
point error envelope.
