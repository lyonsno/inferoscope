"""OLMoE-specific capture adapters built on inferoscope extraction primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .pytorch import PyTorchMoELayerCaptureInput

if TYPE_CHECKING:
    from .pytorch_recorder import PyTorchRunBundleRecorder


def build_olmoe_layer_inputs(
    *,
    router_logits_by_layer: Mapping[int, Any],
    num_active_experts: int,
) -> list[PyTorchMoELayerCaptureInput]:
    """Convert OLMoE layer router logits into sorted recorder layer inputs."""

    if not isinstance(router_logits_by_layer, Mapping):
        raise TypeError(
            "router_logits_by_layer must be a mapping from integer layer indices to router logits."
        )
    if not isinstance(num_active_experts, int) or isinstance(num_active_experts, bool):
        raise TypeError("num_active_experts must be an integer")
    if num_active_experts < 1:
        raise ValueError("num_active_experts must be at least 1")

    layer_inputs: list[PyTorchMoELayerCaptureInput] = []
    for layer_index, router_logits in router_logits_by_layer.items():
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("router_logits_by_layer keys must be integer layer_index values")
        if layer_index < 0:
            raise ValueError("router_logits_by_layer layer_index values must be non-negative")
        layer_inputs.append(
            PyTorchMoELayerCaptureInput(
                layer_index=layer_index,
                router_logits=router_logits,
                num_active_experts=num_active_experts,
            )
        )

    if not layer_inputs:
        raise ValueError("router_logits_by_layer must not be empty")

    return sorted(layer_inputs, key=lambda layer: layer.layer_index)


def record_olmoe_generated_token(
    recorder: "PyTorchRunBundleRecorder",
    *,
    token_id: int,
    token_text: str,
    context_length: int,
    decode_start_ms: float,
    decode_end_ms: float,
    router_logits_by_layer: Mapping[int, Any],
    num_active_experts: int,
) -> dict[str, Any]:
    """Record one generated token from OLMoE-style layer router logits."""

    record_generated_token = getattr(recorder, "record_generated_token", None)
    if not callable(record_generated_token):
        raise TypeError("recorder must implement record_generated_token")

    return record_generated_token(
        token_id=token_id,
        token_text=token_text,
        context_length=context_length,
        decode_start_ms=decode_start_ms,
        decode_end_ms=decode_end_ms,
        layer_inputs=build_olmoe_layer_inputs(
            router_logits_by_layer=router_logits_by_layer,
            num_active_experts=num_active_experts,
        ),
    )
