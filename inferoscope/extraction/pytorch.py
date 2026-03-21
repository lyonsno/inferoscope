"""PyTorch-friendly adapters for inferoscope raw-trace capture."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .raw_trace import MoELayerCaptureInput, build_token_complete_event


@dataclass(frozen=True)
class PyTorchMoELayerCaptureInput:
    """Capture input for one sparse MoE layer using tensor-like router logits."""

    layer_index: int
    router_logits: Any
    num_active_experts: int


def _coerce_router_logits(values: Any) -> list[float]:
    tensor_like = values.detach() if hasattr(values, "detach") else values
    tensor_like = tensor_like.cpu() if hasattr(tensor_like, "cpu") else tensor_like
    if hasattr(tensor_like, "tolist"):
        tensor_like = tensor_like.tolist()

    if isinstance(tensor_like, (str, bytes, bytearray)) or not isinstance(tensor_like, Sequence):
        raise TypeError("router_logits must be a 1D sequence or tensor-like object with tolist()")

    coerced = list(tensor_like)
    if any(isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) for item in coerced):
        raise ValueError("router_logits must be a 1D sequence of numeric values")

    try:
        return [float(item) for item in coerced]
    except (TypeError, ValueError) as exc:
        raise TypeError("router_logits must contain only numeric values") from exc


def _to_raw_trace_input(layer_input: PyTorchMoELayerCaptureInput) -> MoELayerCaptureInput:
    return MoELayerCaptureInput(
        layer_index=layer_input.layer_index,
        router_logits=_coerce_router_logits(layer_input.router_logits),
        num_active_experts=layer_input.num_active_experts,
    )


def build_token_complete_event_from_pytorch(
    *,
    run_id: str,
    token_index: int,
    token_id: int,
    token_text: str,
    context_length: int,
    decode_start_ms: float,
    decode_end_ms: float,
    layer_inputs: Sequence[PyTorchMoELayerCaptureInput],
) -> dict:
    """Build a raw token-complete event from tensor-like layer captures."""

    return build_token_complete_event(
        run_id=run_id,
        token_index=token_index,
        token_id=token_id,
        token_text=token_text,
        context_length=context_length,
        decode_start_ms=decode_start_ms,
        decode_end_ms=decode_end_ms,
        layer_inputs=[_to_raw_trace_input(layer_input) for layer_input in layer_inputs],
    )
