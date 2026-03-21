import unittest

from inferoscope.extraction import (
    PyTorchMoELayerCaptureInput,
    build_token_complete_event_from_pytorch,
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


class BuildTokenCompleteEventFromPyTorchTests(unittest.TestCase):
    def build_event(self, router_logits) -> dict:
        return build_token_complete_event_from_pytorch(
            run_id="run-001",
            token_index=0,
            token_id=42,
            token_text="hello",
            context_length=5,
            decode_start_ms=10.0,
            decode_end_ms=25.0,
            layer_inputs=[
                PyTorchMoELayerCaptureInput(
                    layer_index=0,
                    router_logits=router_logits,
                    num_active_experts=1,
                )
            ],
        )

    def test_build_token_complete_event_from_pytorch_accepts_tensor_like_router_logits(self) -> None:
        event = build_token_complete_event_from_pytorch(
            run_id="run-001",
            token_index=0,
            token_id=42,
            token_text="hello",
            context_length=5,
            decode_start_ms=10.0,
            decode_end_ms=25.0,
            layer_inputs=[
                PyTorchMoELayerCaptureInput(
                    layer_index=1,
                    router_logits=FakeTensor([0.0, 1.0]),
                    num_active_experts=1,
                ),
                PyTorchMoELayerCaptureInput(
                    layer_index=0,
                    router_logits=FakeTensor([2.0, 1.0]),
                    num_active_experts=1,
                ),
            ],
        )

        self.assertEqual(event["event_type"], "token_complete")
        self.assertEqual([layer["layer_index"] for layer in event["layers"]], [0, 1])
        self.assertEqual(event["layers"][0]["topk_indices"], [0])
        self.assertEqual(event["layers"][1]["topk_indices"], [1])

    def test_build_token_complete_event_from_pytorch_rejects_non_1d_tensor_like_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "1D"):
            build_token_complete_event_from_pytorch(
                run_id="run-001",
                token_index=0,
                token_id=42,
                token_text="hello",
                context_length=5,
                decode_start_ms=10.0,
                decode_end_ms=25.0,
                layer_inputs=[
                    PyTorchMoELayerCaptureInput(
                        layer_index=0,
                        router_logits=FakeTensor([[2.0, 1.0]]),
                        num_active_experts=1,
                    )
                ],
            )

    def test_build_token_complete_event_from_pytorch_rejects_non_sequence_logits(self) -> None:
        with self.assertRaisesRegex(TypeError, "1D sequence"):
            self.build_event(42)

    def test_build_token_complete_event_from_pytorch_rejects_string_and_bytes_logits(self) -> None:
        for router_logits in ["0123", b"0123", bytearray(b"0123")]:
            with self.subTest(router_logits=type(router_logits).__name__):
                with self.assertRaisesRegex(TypeError, "1D sequence"):
                    self.build_event(router_logits)

    def test_build_token_complete_event_from_pytorch_rejects_non_numeric_logits(self) -> None:
        with self.assertRaisesRegex(TypeError, "numeric values"):
            self.build_event(FakeTensor(["oops", 1.0]))


if __name__ == "__main__":
    unittest.main()
