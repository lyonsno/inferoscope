"""High-level PyTorch-friendly run-bundle recording helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .bundle import build_layer_grid_layout, build_manifest
from .pytorch import PyTorchMoELayerCaptureInput, build_token_complete_event_from_pytorch


class PyTorchRunBundleRecorder:
    """Accumulate PyTorch-style token captures and write a validated replay bundle."""

    def __init__(
        self,
        *,
        run_id: str,
        created_at: str,
        model_id: str,
        tokenizer_id: str,
        prompt_text: str,
        derivation_version: str,
        derivation_config_id: str,
        seed: int | None = None,
        generation_config: dict[str, Any] | None = None,
        capture_config: dict[str, Any] | None = None,
        layout_id: str = "default-grid",
    ) -> None:
        self._manifest = build_manifest(
            run_id=run_id,
            created_at=created_at,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            prompt_text=prompt_text,
            derivation_version=derivation_version,
            derivation_config_id=derivation_config_id,
            seed=seed,
            generation_config=dict(generation_config) if generation_config is not None else None,
            capture_config=dict(capture_config) if capture_config is not None else None,
        )
        self._layout_id = layout_id
        self._raw_events: list[dict[str, Any]] = []
        self._layer_expert_counts: dict[int, int] = {}

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the manifest that will be written for this run."""

        return deepcopy(self._manifest)

    @property
    def raw_events(self) -> list[dict[str, Any]]:
        """Return the recorded raw events accumulated so far."""

        return deepcopy(self._raw_events)

    def record_token_complete(
        self,
        *,
        token_index: int,
        token_id: int,
        token_text: str,
        context_length: int,
        decode_start_ms: float,
        decode_end_ms: float,
        layer_inputs: list[PyTorchMoELayerCaptureInput],
    ) -> dict[str, Any]:
        """Build and store one completed-token raw event."""

        event = build_token_complete_event_from_pytorch(
            run_id=self._manifest["run_id"],
            token_index=token_index,
            token_id=token_id,
            token_text=token_text,
            context_length=context_length,
            decode_start_ms=decode_start_ms,
            decode_end_ms=decode_end_ms,
            layer_inputs=layer_inputs,
        )

        proposed_counts = dict(self._layer_expert_counts)
        for layer in event["layers"]:
            layer_index = layer["layer_index"]
            num_total_experts = layer["num_total_experts"]
            previous = proposed_counts.get(layer_index)
            if previous is not None and previous != num_total_experts:
                raise ValueError(
                    f"layer {layer_index} num_total_experts must stay constant across the run."
                )
            proposed_counts[layer_index] = num_total_experts

        self._layer_expert_counts = proposed_counts
        self._raw_events.append(event)
        return deepcopy(event)

    def build_layout(self) -> dict[str, Any]:
        """Build the deterministic layer-grid layout inferred from recorded events."""

        if not self._layer_expert_counts:
            raise ValueError("cannot build a layout before recording at least one token_complete event")

        return build_layer_grid_layout(
            run_id=self._manifest["run_id"],
            layout_id=self._layout_id,
            layer_expert_counts=sorted(self._layer_expert_counts.items()),
        )

    def write_bundle(self, root_dir: str | Path) -> Path:
        """Write the recorded run as a validated replay bundle."""

        if not self._raw_events:
            raise ValueError("cannot write a run bundle before recording at least one token_complete event")

        from .run_bundle_io import write_run_bundle

        return write_run_bundle(
            root_dir,
            deepcopy(self._manifest),
            deepcopy(self._raw_events),
            self.build_layout(),
        )
