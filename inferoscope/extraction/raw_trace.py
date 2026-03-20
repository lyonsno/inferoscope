"""Raw trace extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class MoELayerCaptureInput:
    """Capture input for one sparse MoE layer on one completed token."""

    layer_index: int
    router_logits: Sequence[float]
    num_active_experts: int


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("router_logits must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("router_logits must contain only finite values")

    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    denom = sum(exps)
    return [value / denom for value in exps]


def build_moe_layer_trace(
    layer_index: int, router_logits: Sequence[float], *, num_active_experts: int
) -> dict:
    """Build a raw-trace layer payload from finite router logits."""

    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    num_total_experts = len(router_logits)
    if num_total_experts == 0:
        raise ValueError("router_logits must not be empty")
    if num_active_experts < 1:
        raise ValueError("num_active_experts must be at least 1")
    if num_active_experts > num_total_experts:
        raise ValueError("num_active_experts must not exceed the number of experts")

    router_probs = _softmax(router_logits)
    ranked_indices = sorted(
        range(num_total_experts),
        key=lambda index: (-router_probs[index], index),
    )
    topk_indices = ranked_indices[:num_active_experts]
    topk_probs = [router_probs[index] for index in topk_indices]
    entropy = -sum(prob * math.log(prob) for prob in router_probs if prob > 0.0)
    normalized_entropy = entropy / math.log(num_total_experts) if num_total_experts > 1 else 0.0
    top1_prob = topk_probs[0]
    top1_top2_margin = topk_probs[0] - topk_probs[1] if len(topk_probs) >= 2 else topk_probs[0]

    return {
        "layer_index": layer_index,
        "layer_kind": "moe",
        "num_total_experts": num_total_experts,
        "num_active_experts": num_active_experts,
        "router_probs": router_probs,
        "topk_indices": topk_indices,
        "topk_probs": topk_probs,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top1_prob": top1_prob,
        "top1_top2_margin": top1_top2_margin,
    }


def build_token_complete_event(
    *,
    run_id: str,
    token_index: int,
    token_id: int,
    token_text: str,
    context_length: int,
    decode_start_ms: float,
    decode_end_ms: float,
    layer_inputs: Sequence[MoELayerCaptureInput],
) -> dict:
    """Build a raw token_complete event from schema-valid capture inputs."""

    if not run_id:
        raise ValueError("run_id must not be empty")
    if token_index < 0:
        raise ValueError("token_index must be non-negative")
    if context_length < 0:
        raise ValueError("context_length must be non-negative")
    if not layer_inputs:
        raise ValueError("layer_inputs must not be empty")
    if decode_end_ms < decode_start_ms:
        raise ValueError("decode_end_ms must be greater than or equal to decode_start_ms")
    layer_indices = [layer_input.layer_index for layer_input in layer_inputs]
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("duplicate layer_index values are not allowed")

    layers = sorted(
        (
            build_moe_layer_trace(
                layer_input.layer_index,
                layer_input.router_logits,
                num_active_experts=layer_input.num_active_experts,
            )
            for layer_input in layer_inputs
        ),
        key=lambda layer: layer["layer_index"],
    )

    return {
        "event_type": "token_complete",
        "schema_version": "raw/v0.1.0",
        "run_id": run_id,
        "token_index": token_index,
        "token_id": token_id,
        "token_text": token_text,
        "context_length": context_length,
        "timing_ms": {
            "decode_start": decode_start_ms,
            "decode_end": decode_end_ms,
            "decode_duration": decode_end_ms - decode_start_ms,
        },
        "layers": layers,
    }
