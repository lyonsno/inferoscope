import unittest

from inferoscope.extraction.bundle import build_layer_grid_layout, build_manifest


class BuildManifestTests(unittest.TestCase):
    def test_build_manifest_emits_v0_1_0_artifact_versions_and_metadata(self) -> None:
        manifest = build_manifest(
            run_id="run-001",
            created_at="2026-03-20T12:00:00Z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
            seed=7,
            generation_config={"max_new_tokens": 16},
            capture_config={"trace_level": "full-router-probs"},
        )

        self.assertEqual(manifest["schema_version"], "manifest/v0.1.0")
        self.assertEqual(manifest["run_id"], "run-001")
        self.assertEqual(manifest["created_at"], "2026-03-20T12:00:00Z")
        self.assertEqual(manifest["model_id"], "allenai/OLMoE-1B-7B-0125")
        self.assertEqual(manifest["tokenizer_id"], "allenai/OLMoE-1B-7B-0125")
        self.assertEqual(manifest["prompt"], {"text": "hello"})
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(manifest["generation_config"], {"max_new_tokens": 16})
        self.assertEqual(manifest["capture_config"], {"trace_level": "full-router-probs"})
        self.assertEqual(manifest["derivation_version"], "motifs/v0.1.0-alpha")
        self.assertEqual(manifest["derivation_config_id"], "motifs/default-alpha")
        self.assertEqual(
            manifest["artifact_versions"],
            {
                "raw_event_schema_version": "raw/v0.1.0",
                "derived_event_schema_version": "derived/v0.1.0-provisional",
                "layout_schema_version": "layout/v0.1.0",
                "motif_ledger_schema_version": "motif_ledger/v0.1.0-provisional",
                "contingency_schema_version": "contingency/v0.1.0-provisional",
            },
        )

    def test_build_manifest_rejects_empty_run_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "run_id"):
            build_manifest(
                run_id="",
                created_at="2026-03-20T12:00:00Z",
                model_id="allenai/OLMoE-1B-7B-0125",
                tokenizer_id="allenai/OLMoE-1B-7B-0125",
                prompt_text="hello",
                derivation_version="motifs/v0.1.0-alpha",
                derivation_config_id="motifs/default-alpha",
            )

    def test_build_manifest_rejects_non_datetime_created_at(self) -> None:
        with self.assertRaisesRegex(ValueError, "created_at"):
            build_manifest(
                run_id="run-001",
                created_at="not-a-datetime",
                model_id="allenai/OLMoE-1B-7B-0125",
                tokenizer_id="allenai/OLMoE-1B-7B-0125",
                prompt_text="hello",
                derivation_version="motifs/v0.1.0-alpha",
                derivation_config_id="motifs/default-alpha",
            )

    def test_build_manifest_rejects_non_rfc3339_created_at(self) -> None:
        with self.assertRaisesRegex(ValueError, "created_at"):
            build_manifest(
                run_id="run-001",
                created_at="2026-03-20 12:00:00+00:00",
                model_id="allenai/OLMoE-1B-7B-0125",
                tokenizer_id="allenai/OLMoE-1B-7B-0125",
                prompt_text="hello",
                derivation_version="motifs/v0.1.0-alpha",
                derivation_config_id="motifs/default-alpha",
            )

    def test_build_manifest_accepts_lowercase_rfc3339_created_at(self) -> None:
        manifest = build_manifest(
            run_id="run-001",
            created_at="2026-03-20t12:00:00z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
        )

        self.assertEqual(manifest["created_at"], "2026-03-20t12:00:00z")

    def test_build_manifest_accepts_rfc3339_leap_second_created_at(self) -> None:
        manifest = build_manifest(
            run_id="run-001",
            created_at="1990-12-31T23:59:60Z",
            model_id="allenai/OLMoE-1B-7B-0125",
            tokenizer_id="allenai/OLMoE-1B-7B-0125",
            prompt_text="hello",
            derivation_version="motifs/v0.1.0-alpha",
            derivation_config_id="motifs/default-alpha",
        )

        self.assertEqual(manifest["created_at"], "1990-12-31T23:59:60Z")

    def test_build_manifest_rejects_impossible_rfc3339_leap_second(self) -> None:
        with self.assertRaisesRegex(ValueError, "created_at"):
            build_manifest(
                run_id="run-001",
                created_at="2026-03-20T12:34:60Z",
                model_id="allenai/OLMoE-1B-7B-0125",
                tokenizer_id="allenai/OLMoE-1B-7B-0125",
                prompt_text="hello",
                derivation_version="motifs/v0.1.0-alpha",
                derivation_config_id="motifs/default-alpha",
            )

    def test_build_manifest_rejects_invalid_rfc3339_offset_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "created_at"):
            build_manifest(
                run_id="run-001",
                created_at="2026-03-20T12:00:00+00:60",
                model_id="allenai/OLMoE-1B-7B-0125",
                tokenizer_id="allenai/OLMoE-1B-7B-0125",
                prompt_text="hello",
                derivation_version="motifs/v0.1.0-alpha",
                derivation_config_id="motifs/default-alpha",
            )


class BuildLayerGridLayoutTests(unittest.TestCase):
    def test_build_layer_grid_layout_emits_deterministic_positions(self) -> None:
        layout = build_layer_grid_layout(
            run_id="run-001",
            layout_id="default-grid",
            layer_expert_counts=[(0, 4), (2, 2)],
        )

        self.assertEqual(layout["schema_version"], "layout/v0.1.0")
        self.assertEqual(layout["run_id"], "run-001")
        self.assertEqual(layout["layout_id"], "default-grid")
        self.assertEqual(layout["layout_kind"], "layer-grid")
        self.assertEqual([layer["layer_index"] for layer in layout["layers"]], [0, 2])

        layer0_positions = layout["layers"][0]["positions"]
        self.assertEqual(
            layer0_positions,
            [
                {"expert_index": 0, "x": 0.0, "y": 0.0, "z": 0.0},
                {"expert_index": 1, "x": 1.0, "y": 0.0, "z": 0.0},
                {"expert_index": 2, "x": 0.0, "y": 1.0, "z": 0.0},
                {"expert_index": 3, "x": 1.0, "y": 1.0, "z": 0.0},
            ],
        )
        layer2_positions = layout["layers"][1]["positions"]
        self.assertEqual(layer2_positions[0]["z"], 2.0)
        self.assertEqual(layer2_positions[1]["z"], 2.0)

    def test_build_layer_grid_layout_rejects_duplicate_layer_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_layer_grid_layout(
                run_id="run-001",
                layout_id="default-grid",
                layer_expert_counts=[(0, 4), (0, 2)],
            )

    def test_build_layer_grid_layout_rejects_negative_layer_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_index"):
            build_layer_grid_layout(
                run_id="run-001",
                layout_id="default-grid",
                layer_expert_counts=[(-1, 2)],
            )

    def test_build_layer_grid_layout_rejects_non_positive_expert_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expert_count"):
            build_layer_grid_layout(
                run_id="run-001",
                layout_id="default-grid",
                layer_expert_counts=[(0, 0)],
            )


if __name__ == "__main__":
    unittest.main()
