# Schema v0.1.0

This directory contains the canonical `v0.1.0` artifact schemas.

The intended scope of this first version is:

- `manifest.schema.json`
- `raw_trace_event.schema.json`
- `derived_event.schema.json`
- `motif_ledger.schema.json`
- `contingency.schema.json`
- `layout.schema.json`

The first implementation target is aligned with [`docs/v0.1.0_proposal.md`](/Users/noahlyons/dev/inferoscope/docs/v0.1.0_proposal.md).

## Notes

- `raw_trace_event.schema.json` should define the `token_complete` event shape used by replay-first capture.
- `manifest.schema.json`, `raw_trace_event.schema.json`, and `layout.schema.json` are the stable `v0.1.0` test targets.
- `derived_event.schema.json`, `motif_ledger.schema.json`, and `contingency.schema.json` are intentionally provisional in `v0.1.0`: they define a machine-checkable envelope while leaving internal motif payload semantics open.
- Replay-critical invariants that span fields or files should be enforced by semantic validation in addition to JSON Schema.
- `layout.schema.json` should remain separate from trace schemas so geometry can evolve without rewriting runs.
- `v0.1.0` should preserve the full routing probability vector in raw trace artifacts for auditability.
