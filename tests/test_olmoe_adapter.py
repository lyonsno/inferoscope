import tempfile
import unittest

import inferoscope.extraction as extraction


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class OLMoEAdapterTests(unittest.TestCase):
    def test_extraction_exports_olmoe_capture_helpers(self) -> None:
        self.assertTrue(
            hasattr(extraction, "build_olmoe_layer_inputs"),
            "inferoscope.extraction must export build_olmoe_layer_inputs",
        )
        self.assertTrue(
            hasattr(extraction, "record_olmoe_generated_token"),
            "inferoscope.extraction must export record_olmoe_generated_token",
        )

    def build_recorder(self) -> extraction.PyTorchRunBundleRecorder:
        return extraction.PyTorchRunBundleRecorder(
            run_id="demo-run",
            created_at="2026-03-22T12:00:00Z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
            generation_config={"max_new_tokens": 2},
            capture_config={"adapter": "olmoe"},
        )

    def build_olmoe_layer_inputs(self, *, required_behavior: str, **kwargs):
        build_fn = getattr(extraction, "build_olmoe_layer_inputs", None)
        self.assertIsNotNone(
            build_fn,
            f"inferoscope.extraction must export build_olmoe_layer_inputs to {required_behavior}",
        )
        return build_fn(**kwargs)

    def record_olmoe_generated_token(self, recorder, *, required_behavior: str, **kwargs):
        record_fn = getattr(extraction, "record_olmoe_generated_token", None)
        self.assertIsNotNone(
            record_fn,
            f"inferoscope.extraction must export record_olmoe_generated_token to {required_behavior}",
        )
        return record_fn(recorder, **kwargs)

    def write_and_load(self, recorder) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = recorder.write_bundle(tmpdir)
            return extraction.load_run_bundle(run_dir)

    def test_build_olmoe_layer_inputs_sorts_layers_and_applies_shared_topk(self) -> None:
        layer_inputs = self.build_olmoe_layer_inputs(
            required_behavior="convert OLMoE router logits into sorted recorder inputs",
            router_logits_by_layer={
                3: [0.0, 2.0, 1.0],
                1: [1.0, 0.0],
            },
            num_active_experts=2,
        )

        self.assertEqual(
            layer_inputs,
            [
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=1,
                    router_logits=[1.0, 0.0],
                    num_active_experts=2,
                ),
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=3,
                    router_logits=[0.0, 2.0, 1.0],
                    num_active_experts=2,
                ),
            ],
        )

    def test_record_olmoe_generated_token_writes_valid_bundle(self) -> None:
        recorder = self.build_recorder()

        first_event = self.record_olmoe_generated_token(
            recorder,
            required_behavior="record an OLMoE-generated token through the recorder bridge",
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
        second_event = self.record_olmoe_generated_token(
            recorder,
            required_behavior="advance token indices across OLMoE-generated tokens",
            token_id=43,
            token_text=" world",
            context_length=6,
            decode_start_ms=21.0,
            decode_end_ms=30.0,
            router_logits_by_layer={
                0: FakeTensor([0.5, 1.5, 0.0, -0.5]),
                1: FakeTensor([0.1, 0.9]),
            },
            num_active_experts=1,
        )

        self.assertEqual(first_event["token_index"], 0)
        self.assertEqual(second_event["token_index"], 1)

        bundle = self.write_and_load(recorder)

        raw_events = bundle["raw_events"]
        self.assertEqual(len(raw_events), 2)

        first_raw_event = raw_events[0]
        self.assertEqual(first_raw_event["run_id"], "demo-run")
        self.assertEqual(first_raw_event["token_index"], 0)
        self.assertEqual(first_raw_event["token_id"], 42)
        self.assertEqual(first_raw_event["token_text"], " hello")
        self.assertEqual(first_raw_event["context_length"], 5)
        self.assertEqual([layer["layer_index"] for layer in first_raw_event["layers"]], [0, 1])
        self.assertEqual(first_raw_event["layers"][0]["num_total_experts"], 4)
        self.assertEqual(first_raw_event["layers"][1]["num_total_experts"], 2)
        self.assertEqual(first_raw_event["layers"][0]["topk_indices"], [0])
        self.assertEqual(first_raw_event["layers"][1]["topk_indices"], [0])

        second_raw_event = raw_events[1]
        self.assertEqual(second_raw_event["run_id"], "demo-run")
        self.assertEqual(second_raw_event["token_index"], 1)
        self.assertEqual(second_raw_event["token_id"], 43)
        self.assertEqual(second_raw_event["token_text"], " world")
        self.assertEqual(second_raw_event["context_length"], 6)
        self.assertEqual([layer["layer_index"] for layer in second_raw_event["layers"]], [0, 1])
        self.assertEqual(second_raw_event["layers"][0]["num_total_experts"], 4)
        self.assertEqual(second_raw_event["layers"][1]["num_total_experts"], 2)
        self.assertEqual(second_raw_event["layers"][0]["topk_indices"], [1])
        self.assertEqual(second_raw_event["layers"][1]["topk_indices"], [1])
        self.assertEqual(
            bundle["layout"],
            extraction.build_layer_grid_layout(
                run_id="demo-run",
                layout_id="default-grid",
                layer_expert_counts=[(0, 4), (1, 2)],
            ),
        )

    def test_build_olmoe_layer_inputs_rejects_non_mapping_router_logits(self) -> None:
        with self.assertRaisesRegex(TypeError, r"router_logits_by_layer|mapping|dict"):
            self.build_olmoe_layer_inputs(
                required_behavior="reject malformed router_logits_by_layer values",
                router_logits_by_layer=[FakeTensor([1.0, 0.0])],
                num_active_experts=1,
            )

    def test_build_olmoe_layer_inputs_rejects_non_integer_layer_index(self) -> None:
        with self.assertRaisesRegex(TypeError, r"layer|index|integer"):
            self.build_olmoe_layer_inputs(
                required_behavior="reject non-integer OLMoE layer indices",
                router_logits_by_layer={"0": [1.0, 0.0]},
                num_active_experts=1,
            )

    def test_build_olmoe_layer_inputs_rejects_negative_layer_index(self) -> None:
        with self.assertRaisesRegex(ValueError, r"layer_index|non-negative|>= 0"):
            self.build_olmoe_layer_inputs(
                required_behavior="reject negative OLMoE layer indices at adapter construction time",
                router_logits_by_layer={-1: [1.0, 0.0]},
                num_active_experts=1,
            )

    def test_build_olmoe_layer_inputs_rejects_non_positive_num_active_experts(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_active_experts"):
            self.build_olmoe_layer_inputs(
                required_behavior="reject non-positive shared top-k size",
                router_logits_by_layer={0: [1.0, 0.0]},
                num_active_experts=0,
            )

    def test_build_olmoe_layer_inputs_rejects_empty_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, r"router_logits_by_layer.*must not be empty"):
            self.build_olmoe_layer_inputs(
                required_behavior="reject empty OLMoE layer mappings",
                router_logits_by_layer={},
                num_active_experts=1,
            )

    def test_record_olmoe_generated_token_rejects_non_positive_num_active_experts(self) -> None:
        recorder = self.build_recorder()

        with self.assertRaisesRegex(ValueError, "num_active_experts"):
            self.record_olmoe_generated_token(
                recorder,
                required_behavior="reject invalid shared top-k size in recorder bridge",
                token_id=42,
                token_text=" hello",
                context_length=5,
                decode_start_ms=10.0,
                decode_end_ms=20.0,
                router_logits_by_layer={0: FakeTensor([2.0, 1.0])},
                num_active_experts=0,
            )

    def test_record_olmoe_generated_token_rejects_empty_layer_mapping_before_recorder_call(
        self,
    ) -> None:
        class SpyRecorder:
            def __init__(self) -> None:
                self.called = False

            def record_generated_token(self, **kwargs):  # pragma: no cover - must not be reached
                self.called = True
                raise AssertionError(
                    "record_generated_token should not be called for empty router_logits_by_layer"
                )

        recorder = SpyRecorder()

        with self.assertRaisesRegex(ValueError, r"router_logits_by_layer.*must not be empty"):
            self.record_olmoe_generated_token(
                recorder,
                required_behavior=(
                    "reject empty OLMoE layer mappings through recorder bridge "
                    "before recorder invocation"
                ),
                token_id=42,
                token_text=" hello",
                context_length=5,
                decode_start_ms=10.0,
                decode_end_ms=20.0,
                router_logits_by_layer={},
                num_active_experts=1,
            )
        self.assertFalse(recorder.called)

    def test_record_olmoe_generated_token_rejects_non_callable_recorder_method(self) -> None:
        class BadRecorder:
            record_generated_token = 123

        with self.assertRaisesRegex(TypeError, "recorder must implement record_generated_token"):
            self.record_olmoe_generated_token(
                BadRecorder(),
                required_behavior="fail with explicit adapter error on non-callable recorder method",
                token_id=42,
                token_text=" hello",
                context_length=5,
                decode_start_ms=10.0,
                decode_end_ms=20.0,
                router_logits_by_layer={0: FakeTensor([2.0, 1.0])},
                num_active_experts=1,
            )

    def test_record_olmoe_generated_token_accepts_torch_tensors_when_available(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            self.skipTest("torch is not available in this test environment")

        recorder = self.build_recorder()
        event = self.record_olmoe_generated_token(
            recorder,
            required_behavior="handle real torch.Tensor router logits across adapter and bundle I/O",
            token_id=42,
            token_text=" hello",
            context_length=5,
            decode_start_ms=10.0,
            decode_end_ms=20.0,
            router_logits_by_layer={
                2: torch.tensor([0.0, 2.0, 1.0]),
                0: torch.tensor([2.0, 1.0, 0.0, -1.0]),
            },
            num_active_experts=2,
        )

        self.assertEqual(event["token_index"], 0)
        self.assertEqual([layer["layer_index"] for layer in event["layers"]], [0, 2])
        self.assertEqual(event["layers"][0]["num_total_experts"], 4)
        self.assertEqual(event["layers"][1]["num_total_experts"], 3)
        self.assertEqual(event["layers"][0]["topk_indices"], [0, 1])
        self.assertEqual(event["layers"][1]["topk_indices"], [1, 2])

        bundle = self.write_and_load(recorder)
        persisted_event = bundle["raw_events"][0]
        self.assertEqual(persisted_event["token_id"], 42)
        self.assertEqual([layer["layer_index"] for layer in persisted_event["layers"]], [0, 2])
        self.assertEqual(persisted_event["layers"][0]["topk_indices"], [0, 1])
        self.assertEqual(persisted_event["layers"][1]["topk_indices"], [1, 2])


if __name__ == "__main__":
    unittest.main()
