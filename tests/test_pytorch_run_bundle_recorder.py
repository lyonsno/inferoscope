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


class PyTorchRunBundleRecorderTests(unittest.TestCase):
    def test_extraction_exports_pytorch_run_bundle_recorder(self) -> None:
        self.assertTrue(
            hasattr(extraction, "PyTorchRunBundleRecorder"),
            "inferoscope.extraction must export PyTorchRunBundleRecorder",
        )

    def build_recorder(self, *, required_behavior: str):
        recorder_cls = getattr(extraction, "PyTorchRunBundleRecorder", None)
        self.assertIsNotNone(
            recorder_cls,
            f"inferoscope.extraction must export PyTorchRunBundleRecorder to {required_behavior}",
        )
        return recorder_cls(
            run_id="demo-run",
            created_at="2026-03-22T12:00:00Z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
            generation_config={"max_new_tokens": 2},
        )

    def first_token_kwargs(self) -> dict:
        return {
            "token_index": 0,
            "token_id": 42,
            "token_text": " hello",
            "context_length": 5,
            "decode_start_ms": 10.0,
            "decode_end_ms": 20.0,
            "layer_inputs": [
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=1,
                    router_logits=FakeTensor([1.0, 0.0]),
                    num_active_experts=1,
                ),
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=0,
                    router_logits=FakeTensor([2.0, 1.0, 0.0, -1.0]),
                    num_active_experts=2,
                ),
            ],
        }

    def second_token_kwargs(self) -> dict:
        return {
            "token_index": 1,
            "token_id": 43,
            "token_text": " world",
            "context_length": 6,
            "decode_start_ms": 21.0,
            "decode_end_ms": 30.0,
            "layer_inputs": [
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=0,
                    router_logits=FakeTensor([0.5, 1.5, 0.0, -0.5]),
                    num_active_experts=2,
                ),
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=1,
                    router_logits=FakeTensor([0.1, 0.9]),
                    num_active_experts=1,
                ),
            ],
        }

    def manual_nonzero_token_kwargs(self) -> dict:
        token_kwargs = self.first_token_kwargs()
        token_kwargs["token_index"] = 7
        token_kwargs["token_id"] = 77
        token_kwargs["token_text"] = " manual-seven"
        token_kwargs["context_length"] = 12
        token_kwargs["decode_start_ms"] = 100.0
        token_kwargs["decode_end_ms"] = 120.0
        return token_kwargs

    def first_generated_token_kwargs(self) -> dict:
        token_kwargs = self.first_token_kwargs()
        del token_kwargs["token_index"]
        return token_kwargs

    def second_generated_token_kwargs(self) -> dict:
        token_kwargs = self.second_token_kwargs()
        del token_kwargs["token_index"]
        return token_kwargs

    def inconsistent_generated_token_kwargs(self) -> dict:
        return {
            "token_id": 43,
            "token_text": " world",
            "context_length": 6,
            "decode_start_ms": 21.0,
            "decode_end_ms": 30.0,
            "layer_inputs": [
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=0,
                    router_logits=FakeTensor([2.0, 1.0, 0.0, -1.0, -2.0]),
                    num_active_experts=2,
                ),
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=1,
                    router_logits=FakeTensor([0.1, 0.9]),
                    num_active_experts=1,
                ),
            ],
        }

    def record_generated_token(self, recorder, *, required_behavior: str, **kwargs) -> dict:
        self.assertTrue(
            hasattr(recorder, "record_generated_token"),
            f"PyTorchRunBundleRecorder must implement record_generated_token to {required_behavior}",
        )
        return recorder.record_generated_token(**kwargs)

    def write_and_load(self, recorder) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = recorder.write_bundle(tmpdir)
            return extraction.load_run_bundle(run_dir)

    def test_pytorch_run_bundle_recorder_writes_valid_bundle_with_inferred_layout(self) -> None:
        recorder = self.build_recorder(
            required_behavior="write exact manifest/raw-event/layout bundle artifacts",
        )

        recorder.record_token_complete(**self.first_token_kwargs())
        recorder.record_token_complete(**self.second_token_kwargs())

        bundle = self.write_and_load(recorder)

        self.assertEqual(
            bundle["manifest"],
            extraction.build_manifest(
                run_id="demo-run",
                created_at="2026-03-22T12:00:00Z",
                model_id="allenai/OLMoE-1B-7B-0125",
                tokenizer_id="allenai/OLMoE-1B-7B-0125",
                prompt_text="hello",
                derivation_version="motifs/v0.1.0-alpha",
                derivation_config_id="motifs/default-alpha",
                generation_config={"max_new_tokens": 2},
            ),
        )
        self.assertEqual(
            bundle["raw_events"],
            [
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    **self.first_token_kwargs(),
                ),
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    **self.second_token_kwargs(),
                ),
            ],
        )
        self.assertEqual(
            bundle["layout"],
            extraction.build_layer_grid_layout(
                run_id="demo-run",
                layout_id="default-grid",
                layer_expert_counts=[(0, 4), (1, 2)],
            ),
        )

    def test_pytorch_run_bundle_recorder_rejects_inconsistent_layer_expert_counts_before_successful_write(
        self,
    ) -> None:
        recorder = self.build_recorder(
            required_behavior="reject inconsistent layer expert counts before a successful bundle write",
        )
        recorder.record_token_complete(**self.first_token_kwargs())

        inconsistent_second_token = {
            "token_index": 1,
            "token_id": 43,
            "token_text": " world",
            "context_length": 6,
            "decode_start_ms": 21.0,
            "decode_end_ms": 30.0,
            "layer_inputs": [
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=0,
                    router_logits=FakeTensor([2.0, 1.0, 0.0, -1.0, -2.0]),
                    num_active_experts=2,
                ),
                extraction.PyTorchMoELayerCaptureInput(
                    layer_index=1,
                    router_logits=FakeTensor([0.1, 0.9]),
                    num_active_experts=1,
                )
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            r"(layer 0.*num_total_experts|num_total_experts.*layer 0)",
        ):
            recorder.record_token_complete(**inconsistent_second_token)

        bundle = self.write_and_load(recorder)
        self.assertEqual([event["token_index"] for event in bundle["raw_events"]], [0])

    def test_pytorch_run_bundle_recorder_manifest_property_returns_a_defensive_copy(self) -> None:
        recorder = self.build_recorder(
            required_behavior="protect internal manifest state from caller mutation",
        )
        recorder.record_token_complete(**self.first_token_kwargs())

        manifest_view = recorder.manifest
        manifest_view["run_id"] = "other-run"

        bundle = self.write_and_load(recorder)
        self.assertEqual(bundle["manifest"]["run_id"], "demo-run")

    def test_pytorch_run_bundle_recorder_raw_events_property_returns_defensive_copies(self) -> None:
        recorder = self.build_recorder(
            required_behavior="protect recorded raw events from caller mutation through raw_events",
        )
        recorder.record_token_complete(**self.first_token_kwargs())

        raw_events_view = recorder.raw_events
        raw_events_view[0]["token_text"] = "mutated"

        bundle = self.write_and_load(recorder)
        self.assertEqual(
            bundle["raw_events"],
            [
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    **self.first_token_kwargs(),
                )
            ],
        )

    def test_pytorch_run_bundle_recorder_record_token_complete_returns_a_defensive_copy(self) -> None:
        recorder = self.build_recorder(
            required_behavior="protect recorded raw events from caller mutation through the returned event",
        )
        event = recorder.record_token_complete(**self.first_token_kwargs())
        event["run_id"] = "other-run"

        bundle = self.write_and_load(recorder)
        self.assertEqual(
            bundle["raw_events"],
            [
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    **self.first_token_kwargs(),
                )
            ],
        )

    def test_pytorch_run_bundle_recorder_deep_copies_nested_manifest_configs(self) -> None:
        recorder_cls = getattr(extraction, "PyTorchRunBundleRecorder", None)
        self.assertIsNotNone(
            recorder_cls,
            "inferoscope.extraction must export PyTorchRunBundleRecorder to snapshot nested config values",
        )

        generation_config = {"sampling": {"temperature": 0.7}}
        capture_config = {"hooks": {"router": "full-router-probs"}}
        recorder = recorder_cls(
            run_id="demo-run",
            created_at="2026-03-22T12:00:00Z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
            generation_config=generation_config,
            capture_config=capture_config,
        )

        generation_config["sampling"]["temperature"] = 1.1
        capture_config["hooks"]["router"] = "topk-only"

        recorder.record_token_complete(**self.first_token_kwargs())
        bundle = self.write_and_load(recorder)

        self.assertEqual(
            bundle["manifest"]["generation_config"],
            {"sampling": {"temperature": 0.7}},
        )
        self.assertEqual(
            bundle["manifest"]["capture_config"],
            {"hooks": {"router": "full-router-probs"}},
        )

    def test_pytorch_run_bundle_recorder_record_generated_token_auto_assigns_token_indices(
        self,
    ) -> None:
        recorder = self.build_recorder(
            required_behavior="provide an auto-indexing generation callback surface",
        )

        first_event = self.record_generated_token(
            recorder,
            required_behavior="auto-assign token indices for generation callbacks",
            **self.first_generated_token_kwargs(),
        )
        second_event = self.record_generated_token(
            recorder,
            required_behavior="advance token indices across generation callbacks",
            **self.second_generated_token_kwargs(),
        )

        self.assertEqual(first_event["token_index"], 0)
        self.assertEqual(second_event["token_index"], 1)

        bundle = self.write_and_load(recorder)
        self.assertEqual(
            bundle["raw_events"],
            [
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    token_index=0,
                    **self.first_generated_token_kwargs(),
                ),
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    token_index=1,
                    **self.second_generated_token_kwargs(),
                ),
            ],
        )

    def test_pytorch_run_bundle_recorder_record_generated_token_failed_append_does_not_advance_index(
        self,
    ) -> None:
        recorder = self.build_recorder(
            required_behavior="keep generation callback state usable after a failed append",
        )
        self.record_generated_token(
            recorder,
            required_behavior="record the first generated token",
            **self.first_generated_token_kwargs(),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"(layer 0.*num_total_experts|num_total_experts.*layer 0)",
        ):
            self.record_generated_token(
                recorder,
                required_behavior="reject inconsistent expert counts without advancing token indices",
                **self.inconsistent_generated_token_kwargs(),
            )

        second_event = self.record_generated_token(
            recorder,
            required_behavior="resume token indexing after a failed append",
            **self.second_generated_token_kwargs(),
        )

        self.assertEqual(second_event["token_index"], 1)

        bundle = self.write_and_load(recorder)
        self.assertEqual([event["token_index"] for event in bundle["raw_events"]], [0, 1])

    def test_pytorch_run_bundle_recorder_record_generated_token_continues_after_manual_tokens(
        self,
    ) -> None:
        recorder = self.build_recorder(
            required_behavior="keep generated token indices contiguous after manual recording",
        )
        recorder.record_token_complete(**self.first_token_kwargs())

        generated_event = self.record_generated_token(
            recorder,
            required_behavior="continue token indices after manual record_token_complete calls",
            **self.second_generated_token_kwargs(),
        )

        self.assertEqual(generated_event["token_index"], 1)

        bundle = self.write_and_load(recorder)
        self.assertEqual([event["token_index"] for event in bundle["raw_events"]], [0, 1])

    def test_pytorch_run_bundle_recorder_record_generated_token_rejects_noncontiguous_manual_state(
        self,
    ) -> None:
        recorder = self.build_recorder(
            required_behavior="reject auto-indexing after noncontiguous manual recording",
        )
        recorder.record_token_complete(**self.manual_nonzero_token_kwargs())

        with self.assertRaisesRegex(
            ValueError,
            r"^record_generated_token requires existing token_index values to be contiguous starting at 0\.$",
        ):
            self.record_generated_token(
                recorder,
                required_behavior="reject auto-indexing after a non-zero manual record_token_complete call",
                **self.second_generated_token_kwargs(),
            )

        self.assertEqual(
            [event["token_index"] for event in recorder.raw_events],
            [7],
        )

    def test_pytorch_run_bundle_recorder_record_generated_token_returns_a_defensive_copy(
        self,
    ) -> None:
        recorder = self.build_recorder(
            required_behavior="protect recorded raw events from caller mutation through generated-token callbacks",
        )

        event = self.record_generated_token(
            recorder,
            required_behavior="return a defensive copy from record_generated_token",
            **self.first_generated_token_kwargs(),
        )
        event["run_id"] = "other-run"

        bundle = self.write_and_load(recorder)
        self.assertEqual(
            bundle["raw_events"],
            [
                extraction.build_token_complete_event_from_pytorch(
                    run_id="demo-run",
                    token_index=0,
                    **self.first_generated_token_kwargs(),
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
