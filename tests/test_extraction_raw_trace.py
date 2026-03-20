import math
import unittest

from inferoscope.extraction.raw_trace import (
    MoELayerCaptureInput,
    build_moe_layer_trace,
    build_token_complete_event,
)


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    denom = sum(exps)
    return [value / denom for value in exps]


class BuildMoeLayerTraceTests(unittest.TestCase):
    def test_build_moe_layer_trace_derives_probs_topk_and_metrics_from_logits(self) -> None:
        logits = [2.0, 1.0, 0.0, -1.0]
        expected_probs = softmax(logits)

        layer = build_moe_layer_trace(3, logits, num_active_experts=2)

        self.assertEqual(layer["layer_index"], 3)
        self.assertEqual(layer["layer_kind"], "moe")
        self.assertEqual(layer["num_total_experts"], 4)
        self.assertEqual(layer["num_active_experts"], 2)
        self.assertEqual(layer["topk_indices"], [0, 1])
        self.assertEqual(len(layer["router_probs"]), 4)
        for actual, expected in zip(layer["router_probs"], expected_probs):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(layer["topk_probs"][0], expected_probs[0])
        self.assertAlmostEqual(layer["topk_probs"][1], expected_probs[1])
        expected_entropy = -sum(prob * math.log(prob) for prob in expected_probs)
        self.assertAlmostEqual(layer["entropy"], expected_entropy)
        self.assertAlmostEqual(
            layer["normalized_entropy"],
            expected_entropy / math.log(4),
        )
        self.assertAlmostEqual(layer["top1_prob"], expected_probs[0])
        self.assertAlmostEqual(layer["top1_top2_margin"], expected_probs[0] - expected_probs[1])

    def test_build_moe_layer_trace_breaks_ties_by_expert_index(self) -> None:
        layer = build_moe_layer_trace(0, [0.0, 0.0, 0.0], num_active_experts=2)

        self.assertEqual(layer["topk_indices"], [0, 1])

    def test_build_moe_layer_trace_rejects_num_active_experts_above_expert_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_active_experts"):
            build_moe_layer_trace(0, [0.0, 1.0], num_active_experts=3)

    def test_build_moe_layer_trace_rejects_negative_layer_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_index"):
            build_moe_layer_trace(-1, [0.0, 1.0], num_active_experts=1)

    def test_build_moe_layer_trace_uses_topk_probs_for_top1_top2_margin(self) -> None:
        logits = [2.0, 1.0, 0.0]
        expected_probs = softmax(logits)

        layer = build_moe_layer_trace(0, logits, num_active_experts=2)

        self.assertAlmostEqual(
            layer["top1_top2_margin"],
            expected_probs[0] - expected_probs[1],
        )

    def test_build_moe_layer_trace_rejects_non_finite_logits(self) -> None:
        for bad_value in [float("inf"), float("-inf"), float("nan")]:
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    build_moe_layer_trace(0, [bad_value, 0.0], num_active_experts=1)


class BuildTokenCompleteEventTests(unittest.TestCase):
    def test_build_token_complete_event_sorts_layers_and_computes_duration(self) -> None:
        event = build_token_complete_event(
            run_id="run-001",
            token_index=7,
            token_id=42,
            token_text="hello",
            context_length=99,
            decode_start_ms=10.0,
            decode_end_ms=25.5,
            layer_inputs=[
                MoELayerCaptureInput(layer_index=2, router_logits=[0.0, 1.0], num_active_experts=1),
                MoELayerCaptureInput(layer_index=0, router_logits=[2.0, 1.0], num_active_experts=1),
            ],
        )

        self.assertEqual(event["event_type"], "token_complete")
        self.assertEqual(event["schema_version"], "raw/v0.1.0")
        self.assertEqual(event["run_id"], "run-001")
        self.assertEqual(event["token_index"], 7)
        self.assertEqual(event["token_id"], 42)
        self.assertEqual(event["token_text"], "hello")
        self.assertEqual(event["context_length"], 99)
        self.assertEqual(
            event["timing_ms"],
            {
                "decode_start": 10.0,
                "decode_end": 25.5,
                "decode_duration": 15.5,
            },
        )
        self.assertEqual([layer["layer_index"] for layer in event["layers"]], [0, 2])
        self.assertEqual(event["layers"][0]["topk_indices"], [0])
        self.assertEqual(event["layers"][1]["topk_indices"], [1])

    def test_build_token_complete_event_rejects_empty_run_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "run_id"):
            build_token_complete_event(
                run_id="",
                token_index=7,
                token_id=42,
                token_text="hello",
                context_length=99,
                decode_start_ms=10.0,
                decode_end_ms=25.5,
                layer_inputs=[
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0],
                        num_active_experts=1,
                    )
                ],
            )

    def test_build_token_complete_event_rejects_empty_layer_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_inputs"):
            build_token_complete_event(
                run_id="run-001",
                token_index=7,
                token_id=42,
                token_text="hello",
                context_length=99,
                decode_start_ms=10.0,
                decode_end_ms=25.5,
                layer_inputs=[],
            )

    def test_build_token_complete_event_rejects_negative_token_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "token_index"):
            build_token_complete_event(
                run_id="run-001",
                token_index=-1,
                token_id=42,
                token_text="hello",
                context_length=99,
                decode_start_ms=10.0,
                decode_end_ms=25.5,
                layer_inputs=[
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0],
                        num_active_experts=1,
                    )
                ],
            )

    def test_build_token_complete_event_rejects_negative_context_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length"):
            build_token_complete_event(
                run_id="run-001",
                token_index=7,
                token_id=42,
                token_text="hello",
                context_length=-5,
                decode_start_ms=10.0,
                decode_end_ms=25.5,
                layer_inputs=[
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0],
                        num_active_experts=1,
                    )
                ],
            )

    def test_build_token_complete_event_rejects_negative_decode_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "decode_end_ms"):
            build_token_complete_event(
                run_id="run-001",
                token_index=7,
                token_id=42,
                token_text="hello",
                context_length=99,
                decode_start_ms=25.5,
                decode_end_ms=10.0,
                layer_inputs=[
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0],
                        num_active_experts=1,
                    )
                ],
            )

    def test_build_token_complete_event_rejects_duplicate_layer_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_token_complete_event(
                run_id="run-001",
                token_index=7,
                token_id=42,
                token_text="hello",
                context_length=99,
                decode_start_ms=10.0,
                decode_end_ms=25.5,
                layer_inputs=[
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0],
                        num_active_experts=1,
                    ),
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[0.0, 1.0],
                        num_active_experts=1,
                    ),
                ],
            )


if __name__ == "__main__":
    unittest.main()
