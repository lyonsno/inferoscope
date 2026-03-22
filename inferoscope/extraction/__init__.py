"""Extraction helpers for inferoscope capture artifacts."""

from .bundle import build_layer_grid_layout, build_manifest
from .olmoe import build_olmoe_layer_inputs, record_olmoe_generated_token
from .pytorch import PyTorchMoELayerCaptureInput, build_token_complete_event_from_pytorch
from .pytorch_recorder import PyTorchRunBundleRecorder
from .raw_trace import MoELayerCaptureInput, build_moe_layer_trace, build_token_complete_event

__all__ = [
    "MoELayerCaptureInput",
    "PyTorchMoELayerCaptureInput",
    "PyTorchRunBundleRecorder",
    "build_layer_grid_layout",
    "build_manifest",
    "build_moe_layer_trace",
    "build_olmoe_layer_inputs",
    "record_olmoe_generated_token",
    "build_token_complete_event",
    "build_token_complete_event_from_pytorch",
    "load_run_bundle",
    "write_run_bundle",
]


def __getattr__(name: str):
    if name in {"load_run_bundle", "write_run_bundle"}:
        from .run_bundle_io import load_run_bundle, write_run_bundle

        exports = {
            "load_run_bundle": load_run_bundle,
            "write_run_bundle": write_run_bundle,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
