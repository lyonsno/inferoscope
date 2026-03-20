# Bundle Profiles

Updated: 2026-03-20

## Why This Exists

`inferoscope` needs strong replay guarantees for `v0.1.0`, but it will likely also want clipped replay windows and a live or live-mimicking mode later.

The important design decision is to avoid weakening the current replay bundle contract just to make future modes fit. Instead, bundle semantics should be explicit about which kind of artifact is being validated.

## Current Decision

For `v0.1.0`, the stable replay contract is a `full_run` bundle:

- raw `token_index` values are unique,
- raw `token_index` values are ordered,
- raw `token_index` values are contiguous,
- raw `token_index` values start at `0`,
- derived events must not repeat `token_index` values,
- and derived events may be a subset of the raw token sequence while derivation remains provisional.

This keeps deterministic replay honest for the first implementation target.

## Planned Bundle Profiles

### `full_run`

Use this for frozen end-to-end captures that are meant to support replay, audit, and durable comparison.

Expected invariants:

- raw events form the exact sequence `0..N-1`,
- raw events are unique and ordered,
- derived events are unique,
- derived events refer only to tokens present in the raw trace,
- layout and support artifacts share the same `run_id`,
- and the bundle is suitable for deterministic replay.

This is the current `v0.1.0` semantic target.

### `contiguous_segment`

Use this for replayable clipped excerpts or rolling windows that mimic live behavior without pretending to be a full run.

Expected invariants:

- raw events are unique and ordered,
- raw events are contiguous,
- raw events may start at `K > 0`,
- the segment should declare its token span explicitly,
- derived events are unique,
- and derived events refer only to tokens present in the segment raw trace.

This is the intended future home for replayable windows or a "real-time mimic" mode.

### `live_stream_session`

Use this for append-only or incomplete in-flight state.

This should not quietly reuse the `full_run` contract with a few checks removed. A live session is a different artifact class with different completeness expectations.

Likely properties:

- raw events may still need to be ordered and unique,
- support artifacts may be absent or incomplete,
- derivation may lag behind raw capture,
- and the session may be valid for interactive viewing without being valid as a frozen replay bundle.

## What Not To Make First-Class

Do not treat arbitrary sparse token slices as primary replay bundles.

Sparse slices may still be useful as analysis products, but they are a poor core replay artifact because they weaken ordering guarantees and make missing context harder to reason about.

If the viewer needs a live-looking subset of a run, prefer a contiguous segment over a sparse sample.

## Validator Evolution

Near term, the semantic validator can remain strict for `v0.1.0` while being refactored internally around sequence-profile helpers.

Examples of acceptable evolution paths:

- keep `validate_run_bundle_semantics()` as the strict `full_run` validator and add `validate_segment_bundle_semantics()` later,
- or make `validate_run_bundle_semantics(..., profile="full_run")` explicit while preserving the same strict default.

The important thing is not the API spelling. The important thing is that the profile is explicit and the `full_run` guarantees are not silently diluted.

## Future Manifest Direction

When `contiguous_segment` becomes a real artifact contract, the manifest should declare that directly.

Likely future fields:

- `bundle_kind: "full_run" | "contiguous_segment"`
- `token_span.start_index`
- `token_span.end_index_exclusive`

Then validation can remain exact:

- `full_run` means the raw trace must cover `0..N-1`,
- `contiguous_segment` means the raw trace must cover `K..K+N-1`,
- and live or append-only state can remain a separate artifact class rather than a weakened replay bundle.

## Viewer Implication

The current replay-first viewer contract still stands.

If the project later adds a live or live-mimicking mode, the honest representation is:

- one-token-delayed display,
- backed by completed-token events,
- over a contiguous segment or append-only stream,
- rather than pretending that an arbitrary sparse slice is a coherent replay artifact.
