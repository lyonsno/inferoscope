import tempfile
import unittest

import inferoscope
import inferoscope.extraction as extraction


class DerivationPipelineTests(unittest.TestCase):
    def build_bundle(self) -> dict:
        manifest = extraction.build_manifest(
            run_id="run-001",
            created_at="2026-03-24T12:00:00Z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
        )
        raw_events = [
            extraction.build_token_complete_event(
                run_id="run-001",
                token_index=0,
                token_id=42,
                token_text=" hello",
                context_length=5,
                decode_start_ms=10.0,
                decode_end_ms=20.0,
                layer_inputs=[
                    extraction.MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0, 0.0, -1.0],
                        num_active_experts=2,
                    ),
                    extraction.MoELayerCaptureInput(
                        layer_index=1,
                        router_logits=[1.0, 0.0],
                        num_active_experts=1,
                    ),
                ],
            ),
            extraction.build_token_complete_event(
                run_id="run-001",
                token_index=1,
                token_id=43,
                token_text=" world",
                context_length=6,
                decode_start_ms=21.0,
                decode_end_ms=30.0,
                layer_inputs=[
                    extraction.MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0, 0.0, -1.0],
                        num_active_experts=2,
                    ),
                    extraction.MoELayerCaptureInput(
                        layer_index=1,
                        router_logits=[1.0, -0.25],
                        num_active_experts=1,
                    ),
                ],
            ),
            extraction.build_token_complete_event(
                run_id="run-001",
                token_index=2,
                token_id=44,
                token_text=" hello",
                context_length=7,
                decode_start_ms=31.0,
                decode_end_ms=40.0,
                layer_inputs=[
                    extraction.MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0, 0.0, -1.0],
                        num_active_experts=2,
                    ),
                    extraction.MoELayerCaptureInput(
                        layer_index=1,
                        router_logits=[1.0, 0.0],
                        num_active_experts=1,
                    ),
                ],
            ),
        ]
        layout = extraction.build_layer_grid_layout(
            run_id="run-001",
            layout_id="default-grid",
            layer_expert_counts=[(0, 4), (1, 2)],
        )
        return {
            "manifest": manifest,
            "raw_events": raw_events,
            "layout": layout,
            "derived_events": None,
            "motif_ledger": None,
            "contingency": None,
        }

    def build_loaded_bundle(self) -> dict:
        bundle = self.build_bundle()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = extraction.write_run_bundle(
                tmpdir,
                bundle["manifest"],
                bundle["raw_events"],
                bundle["layout"],
            )
            return extraction.load_run_bundle(run_dir)

    def derive_bundle_artifacts(self, bundle: dict, *, required_behavior: str) -> dict:
        derive_fn = getattr(inferoscope, "derive_bundle_artifacts", None)
        self.assertIsNotNone(
            derive_fn,
            f"inferoscope must export derive_bundle_artifacts to {required_behavior}",
        )
        return derive_fn(bundle)

    def test_inferoscope_exports_bundle_derivation_surface(self) -> None:
        self.assertTrue(
            hasattr(inferoscope, "derive_bundle_artifacts"),
            "inferoscope must export derive_bundle_artifacts",
        )

    def test_derive_bundle_artifacts_emits_schema_valid_optional_artifacts(self) -> None:
        bundle = self.build_loaded_bundle()

        artifacts = self.derive_bundle_artifacts(
            bundle,
            required_behavior="derive schema-valid optional artifacts from a loaded run bundle",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = extraction.write_run_bundle(
                tmpdir,
                bundle["manifest"],
                bundle["raw_events"],
                bundle["layout"],
                derived_events=artifacts["derived_events"],
                motif_ledger=artifacts["motif_ledger"],
                contingency=artifacts["contingency"],
            )
            round_tripped = extraction.load_run_bundle(run_dir)

        self.assertEqual(len(round_tripped["derived_events"]), 3)
        self.assertEqual(
            [event["token_index"] for event in round_tripped["derived_events"]],
            [0, 1, 2],
        )
        self.assertEqual(
            round_tripped["motif_ledger"]["derivation_version"],
            bundle["manifest"]["derivation_version"],
        )
        self.assertEqual(
            round_tripped["contingency"]["derivation_config_id"],
            bundle["manifest"]["derivation_config_id"],
        )
        self.assertEqual(
            sorted(round_tripped["derived_events"][0]["derived_payload"]),
            ["global_motif", "layer_motifs"],
        )

    def test_derive_bundle_artifacts_uses_exact_fingerprint_identity_for_first_pass(self) -> None:
        bundle = self.build_loaded_bundle()

        artifacts = self.derive_bundle_artifacts(
            bundle,
            required_behavior=(
                "assign stable local motif ids by exact per-layer routing fingerprint "
                "and stable global motif ids by exact whole-token fingerprint"
            ),
        )

        first_payload = artifacts["derived_events"][0]["derived_payload"]
        second_payload = artifacts["derived_events"][1]["derived_payload"]
        third_payload = artifacts["derived_events"][2]["derived_payload"]

        first_layer_motifs = {
            item["layer_index"]: item["motif_id"] for item in first_payload["layer_motifs"]
        }
        second_layer_motifs = {
            item["layer_index"]: item["motif_id"] for item in second_payload["layer_motifs"]
        }
        third_layer_motifs = {
            item["layer_index"]: item["motif_id"] for item in third_payload["layer_motifs"]
        }

        self.assertEqual(
            sorted(first_layer_motifs),
            [0, 1],
        )
        self.assertEqual(
            sorted(second_layer_motifs),
            [0, 1],
        )
        self.assertEqual(
            sorted(third_layer_motifs),
            [0, 1],
        )
        self.assertEqual(first_layer_motifs[0], second_layer_motifs[0])
        self.assertNotEqual(first_layer_motifs[1], second_layer_motifs[1])
        self.assertEqual(first_layer_motifs[0], third_layer_motifs[0])
        self.assertEqual(first_layer_motifs[1], third_layer_motifs[1])
        self.assertNotEqual(
            first_payload["global_motif"]["motif_id"],
            second_payload["global_motif"]["motif_id"],
        )
        self.assertEqual(
            first_payload["global_motif"]["motif_id"],
            third_payload["global_motif"]["motif_id"],
        )
