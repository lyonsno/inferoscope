import unittest

from inferoscope.extraction import (
    MoELayerCaptureInput,
    build_layer_grid_layout,
    build_manifest,
    build_token_complete_event,
)
from inferoscope.validation import validate_artifact_schema, validate_run_bundle_schema


def make_valid_manifest() -> dict:
    return build_manifest(
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


def make_valid_raw_event() -> dict:
    return build_token_complete_event(
        run_id="run-001",
        token_index=0,
        token_id=42,
        token_text="hello",
        context_length=5,
        decode_start_ms=10.0,
        decode_end_ms=25.0,
        layer_inputs=[
            MoELayerCaptureInput(
                layer_index=0,
                router_logits=[2.0, 1.0, 0.0, -1.0],
                num_active_experts=2,
            )
        ],
    )


def make_valid_layout() -> dict:
    return build_layer_grid_layout(
        run_id="run-001",
        layout_id="default-grid",
        layer_expert_counts=[(0, 4)],
    )


class SchemaValidationBehaviorTests(unittest.TestCase):
    def test_validate_artifact_schema_validates_single_raw_event(self) -> None:
        raw_event = make_valid_raw_event()
        raw_event["layers"][0]["unexpected"] = True

        issues = validate_artifact_schema("raw_event", raw_event, location="raw_event")

        self.assertIn("raw_event.layers[0].unexpected is not allowed by the schema", issues)

    def test_validate_artifact_schema_rejects_unknown_artifact_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported artifact schema"):
            validate_artifact_schema("unknown", {})

    def test_validate_run_bundle_schema_reports_type_mismatch(self) -> None:
        manifest = make_valid_manifest()
        manifest["run_id"] = 7

        issues = validate_run_bundle_schema(manifest, [make_valid_raw_event()], make_valid_layout())

        self.assertIn("manifest.run_id must have schema type 'string'", issues)

    def test_validate_run_bundle_schema_reports_enum_violation_in_referenced_raw_layer(self) -> None:
        raw_event = make_valid_raw_event()
        raw_event["layers"][0]["layer_kind"] = "dense"

        issues = validate_run_bundle_schema(make_valid_manifest(), [raw_event], make_valid_layout())

        self.assertIn("raw_events[0].layers[0].layer_kind must be one of ['moe']", issues)

    def test_validate_run_bundle_schema_reports_additional_property_in_referenced_raw_layer(self) -> None:
        raw_event = make_valid_raw_event()
        raw_event["layers"][0]["unexpected"] = True

        issues = validate_run_bundle_schema(make_valid_manifest(), [raw_event], make_valid_layout())

        self.assertIn("raw_events[0].layers[0].unexpected is not allowed by the schema", issues)

    def test_validate_run_bundle_schema_reports_missing_required_property_in_referenced_layout_position(
        self,
    ) -> None:
        layout = make_valid_layout()
        del layout["layers"][0]["positions"][0]["z"]

        issues = validate_run_bundle_schema(make_valid_manifest(), [make_valid_raw_event()], layout)

        self.assertIn("layout.layers[0].positions[0].z is required", issues)

    def test_validate_run_bundle_schema_reports_min_items_in_referenced_layout_positions(self) -> None:
        layout = make_valid_layout()
        layout["layers"][0]["positions"] = []

        issues = validate_run_bundle_schema(make_valid_manifest(), [make_valid_raw_event()], layout)

        self.assertIn("layout.layers[0].positions must contain at least 1 item(s)", issues)


if __name__ == "__main__":
    unittest.main()
