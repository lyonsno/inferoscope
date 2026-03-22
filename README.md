# inferoscope

Replay-first telemetry and artifact contracts for inspecting sparse Mixture-of-Experts routing.

`inferoscope` is a Python toolkit for capturing completed-token MoE routing events, validating them into durable run bundles, and preserving a clean boundary between raw model behavior and any later interpretation layered on top.

It is aimed at the part of the stack that usually gets hand-waved away: the layer between model internals and the tools people really use to debug, compare, replay, and eventually visualize what happened.

## What Problem This Solves

Model-internal tooling often fails in one of two ways:

- it is too live and ephemeral to replay or audit later
- it jumps too quickly to interpretation and loses the raw substrate

`inferoscope` takes the opposite approach.

It treats a completed token event as the primary object, records the full routing state needed for replay, and validates those artifacts aggressively so downstream analysis can start from something trustworthy.

That makes it useful when you want to:

- capture sparse MoE routing behavior as durable artifacts
- build analysis or visualization layers without coupling them to one runtime
- compare runs without relying on transient in-memory state
- keep derived interpretations honest by preserving the raw basis nearby

## Core Ideas

- **Replay first**
  The durable artifact is the product. Live streaming can come later.
- **Completed tokens, not partial updates**
  A `token_complete` event is the atomic unit for `v0.1.0`.
- **Raw and derived stay separate**
  Raw routing traces are stable contracts. Derived motif-style outputs remain explicitly provisional.
- **Contracts matter**
  JSON Schema handles shape/version validity, while semantic validation handles replay-critical invariants.
- **Auditability is not optional**
  If a later viewer or derivation layer claims a pattern, the route back to raw routing values should still exist.

## What Exists Today

The current repo is strongest as a contract and capture layer.

Today it includes:

- builders for raw token-complete routing events
- builders for bundle manifests and layout artifacts
- strict bundle write/load helpers
- package-local JSON Schema validation
- semantic validation for replay-critical cross-field and cross-file invariants
- PyTorch-friendly adapters for tensor-like router logits
- a tested `v0.1.0` replay-bundle contract

This is enough to use `inferoscope` as a foundation for:

- model instrumentation experiments
- offline replay pipelines
- artifact validation and debugging
- future visualization or motif-derivation layers

## Current Scope

`inferoscope` is early, but the core artifact path is already real.

In scope now:

- capture-side builders
- artifact schemas
- semantic validation
- replay-bundle file I/O
- PyTorch-friendly bridging for layer capture inputs

Not in scope yet:

- a polished public viewer
- model-specific forward-hook integrations
- a stable derived-motif contract beyond provisional envelopes
- packaging/publishing polish for general installation workflows

For now, the expected workflow is to use the repo directly from a checkout.

## Run Bundle Shape

Each captured run is stored under:

```text
runs/<run_id>/
```

The `v0.1.0` bundle shape is:

```text
manifest.json
raw_trace.ndjson
layout.json
derived.ndjson        # optional, provisional
motif_ledger.json     # optional, provisional
contingency.json      # optional, provisional
```

This separation is intentional:

- `manifest.json` records provenance, configuration, and expected artifact versions
- `raw_trace.ndjson` preserves completed-token routing events
- `layout.json` keeps viewer geometry independent from capture data
- optional derived/support artifacts can evolve without contaminating the raw trace contract

## Quick Example

If you're working from a checkout instead of an installed package, run this from the repo root or set `PYTHONPATH=/path/to/inferoscope` first.

The example below builds a minimal one-token run bundle, validates it on write and load, and uses a temporary bundle root so it is safe to rerun:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from inferoscope.extraction import (
    MoELayerCaptureInput,
    build_layer_grid_layout,
    build_manifest,
    build_token_complete_event,
    load_run_bundle,
    write_run_bundle,
)

manifest = build_manifest(
    run_id="demo-run",
    created_at="2026-03-21T12:00:00Z",
    model_id="allenai/OLMoE-1B-7B-0125",
    tokenizer_id="allenai/OLMoE-1B-7B-0125",
    prompt_text="hello",
    derivation_version="motifs/v0.1.0-alpha",
    derivation_config_id="motifs/default-alpha",
    generation_config={"max_new_tokens": 8},
)

raw_event = build_token_complete_event(
    run_id="demo-run",
    token_index=0,
    token_id=42,
    token_text=" world",
    context_length=5,
    decode_start_ms=10.0,
    decode_end_ms=25.0,
    layer_inputs=[
        MoELayerCaptureInput(
            layer_index=0,
            router_logits=[2.0, 1.0, 0.0, -1.0],
            num_active_experts=2,
        )
    ],
)

layout = build_layer_grid_layout(
    run_id="demo-run",
    layout_id="default-grid",
    layer_expert_counts=[(0, 4)],
)

with TemporaryDirectory() as bundle_root:
    run_dir = write_run_bundle(Path(bundle_root), manifest, [raw_event], layout)
    bundle = load_run_bundle(run_dir)

    print(run_dir)
    print(bundle["raw_events"][0]["layers"][0]["topk_indices"])
```

That creates and validates a bundle under a temporary directory, so you can paste and rerun the example without cleaning up a previous `run_id`.

## PyTorch OLMoE Bridge Example

For generated-token callbacks, `inferoscope` also exposes a recorder-oriented bridge through `PyTorchRunBundleRecorder` and `record_olmoe_generated_token`.

This is useful when your model loop already emits router logits per layer and you want to convert each completed token into a validated replay bundle event:

```python
from tempfile import TemporaryDirectory

from inferoscope.extraction import (
    PyTorchRunBundleRecorder,
    load_run_bundle,
    record_olmoe_generated_token,
)


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


recorder = PyTorchRunBundleRecorder(
    run_id="demo-olmoe-run",
    created_at="2026-03-22T12:00:00Z",
    model_id="allenai/OLMoE-1B-7B-0125",
    tokenizer_id="allenai/OLMoE-1B-7B-0125",
    prompt_text="hello",
    derivation_version="motifs/v0.1.0-alpha",
    derivation_config_id="motifs/default-alpha",
    generation_config={"max_new_tokens": 4},
    capture_config={"adapter": "olmoe"},
)

record_olmoe_generated_token(
    recorder,
    token_id=42,
    token_text=" hello",
    context_length=5,
    decode_start_ms=10.0,
    decode_end_ms=20.0,
    router_logits_by_layer={
        1: FakeTensor([1.0, 0.0]),
        0: FakeTensor([2.0, 1.0, 0.0, -1.0]),
    },
    num_active_experts=1,
)

with TemporaryDirectory() as bundle_root:
    run_dir = recorder.write_bundle(bundle_root)
    bundle = load_run_bundle(run_dir)

    print(bundle["raw_events"][0]["token_index"])
    print([layer["layer_index"] for layer in bundle["raw_events"][0]["layers"]])
```

## Why Replay-First Matters

A replay-first design has a few practical advantages:

- it makes bugs easier to reproduce
- it keeps visualization work downstream of the artifact contract instead of intertwined with runtime capture
- it enables stricter validation and clearer provenance
- it makes comparisons and derivations easier to reason about later

This repo is deliberately opinionated here. The goal is not to emit the loosest possible internal trace. The goal is to emit artifacts that are robust enough to become dependable infrastructure.

## Project Layout

- `inferoscope/extraction/`
  Capture-side builders, PyTorch adapters, and bundle I/O.
- `inferoscope/validation/`
  Schema and semantic validation helpers.
- `schema/`
  Canonical versioned JSON Schemas for artifact contracts.
- `docs/`
  Design notes, proposal docs, and bundle-profile direction.
- `tests/`
  Contract and regression coverage for the current behavior.

## Documentation

- [`docs/v0.1.0_proposal.md`](docs/v0.1.0_proposal.md)
  Project direction, artifact goals, and the first viewer/replay target.
- [`docs/bundle_profiles.md`](docs/bundle_profiles.md)
  Why `v0.1.0` uses a strict replay profile and how future bundle kinds may evolve.
- [`schema/README.md`](schema/README.md)
  How schema validation and semantic validation split responsibilities.
- [`schema/v0.1.0/README.md`](schema/v0.1.0/README.md)
  Notes for the first schema version.

## Running Tests

From the repo root:

```bash
python -m unittest discover -s tests -v
```

## Near-Term Direction

The near-term goal is not just to collect more telemetry.

The near-term goal is to make sparse-model behavior legible without washing out the structure:

1. capture exact completed-token routing events
2. validate them aggressively
3. keep the raw contract stable
4. layer replay, inspection, and provisional derivation on top

If you care about model-internal tooling that is rigorous enough to build on, that is the niche `inferoscope` is trying to fill.
