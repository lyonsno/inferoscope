# Next Steps

Updated: 2026-03-24

## Current Read

The repo now has the capture-side path in place:

- raw event builders
- manifest and layout builders
- schema and semantic validation
- replay-bundle write/load helpers
- PyTorch recorder helpers
- OLMoE capture bridge
- bundle inspection CLI

What is still missing is the first replay-side layer that turns stable raw bundles into provisional derived artifacts and, after that, a first honest replay viewer.

## Near-Term Backlog

1. `Next`: add a tiny provisional derivation surface that converts a loaded run bundle into schema-valid `derived.ndjson`, `motif_ledger.json`, and `contingency.json` artifacts.
2. `After that`: add a replay-oriented surface that consumes raw-plus-derived bundles and emits deterministic frames or state for a viewer.
3. `Then`: build the first ugly-but-honest viewer with replay controls and easy access back to raw routing values.

## Item 1 Boundary

The first derivation step should stay deliberately small:

- input: a loaded run bundle
- output: one derived-event stream plus minimal ledger and contingency payloads
- derived-event summary shape for the first slice: `derived_payload.global_motif.motif_id` plus `derived_payload.layer_motifs[{layer_index, motif_id}]`
- matching rule: exact routing-fingerprint identity only for the first pass
- scope: deterministic per-run motif ids, no cross-run identity claims
- non-goals: semantic labeling, live streaming, polished UX, or ambitious clustering

This narrows the broader `docs/v0.1.0_proposal.md` motifing story on purpose. The immediate goal is a deterministic baseline that can later grow into conservative similarity-threshold clustering without changing the provisional artifact envelopes.
