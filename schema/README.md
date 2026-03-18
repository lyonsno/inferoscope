# Schema

This directory is the canonical home for versioned artifact schemas used by `inferoscope`.

## Source Of Truth

The canonical schema definitions should live here as versioned JSON Schema files, not only as Python or TypeScript runtime types.

Implementation-local adapters are fine, but they should follow the contracts defined under `schema/`.

## Layout

Each schema version gets its own directory:

```text
schema/
  v0.1.0/
```

Within a version directory, keep raw, derived, and support artifacts separate instead of collapsing everything into one monolithic schema.

## Compatibility

- Raw trace schemas and derived schemas are versioned independently in their payloads.
- Breaking field changes should increment the schema version.
- Derived outputs should also carry a derivation version because motif logic may evolve without changing the raw trace contract.

## Planned First Files

The initial `v0.1.0` schema set is expected to cover:

- `manifest.schema.json`
- `raw_trace_event.schema.json`
- `derived_event.schema.json`
- `motif_ledger.schema.json`
- `contingency.schema.json`
- `layout.schema.json`

See [`schema/v0.1.0/README.md`](/Users/noahlyons/dev/inferoscope/schema/v0.1.0/README.md) for the first version scope.
