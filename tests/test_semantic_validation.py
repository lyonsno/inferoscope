import copy
import unittest

from inferoscope.validation.semantic import (
    ValidationIssue,
    validate_layout_semantics,
    validate_raw_event_semantics,
    validate_run_bundle_semantics,
)


def make_valid_raw_event() -> dict:
    return {
        "event_type": "token_complete",
        "schema_version": "raw/v0.1.0",
        "run_id": "run-001",
        "token_index": 0,
        "token_id": 42,
        "token_text": "hello",
        "context_length": 5,
        "timing_ms": {
            "decode_start": 10.0,
            "decode_end": 25.0,
            "decode_duration": 15.0,
        },
        "layers": [
            {
                "layer_index": 0,
                "layer_kind": "moe",
                "num_total_experts": 4,
                "num_active_experts": 2,
                "router_probs": [0.6, 0.3, 0.1, 0.0],
                "topk_indices": [0, 1],
                "topk_probs": [0.6, 0.3],
                "entropy": 0.8979457248567797,
                "normalized_entropy": 0.647730922119161,
                "top1_prob": 0.6,
                "top1_top2_margin": 0.3,
            }
        ],
    }


def make_valid_manifest() -> dict:
    return {
        "schema_version": "manifest/v0.1.0",
        "run_id": "run-001",
        "created_at": "2026-03-18T12:00:00Z",
        "model_id": "allenai/OLMoE-1B-7B-0125",
        "tokenizer_id": "allenai/OLMoE-1B-7B-0125",
        "prompt": {"text": "hello"},
        "seed": 7,
        "generation_config": {
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "capture_config": {
            "trace_level": "full-router-probs",
            "device": "mps",
        },
        "artifact_versions": {
            "raw_event_schema_version": "raw/v0.1.0",
            "derived_event_schema_version": "derived/v0.1.0-provisional",
            "layout_schema_version": "layout/v0.1.0",
            "motif_ledger_schema_version": "motif_ledger/v0.1.0-provisional",
            "contingency_schema_version": "contingency/v0.1.0-provisional",
        },
        "derivation_version": "motifs/v0.1.0-alpha",
        "derivation_config_id": "motifs/default-alpha",
    }


def make_valid_layout() -> dict:
    return {
        "schema_version": "layout/v0.1.0",
        "run_id": "run-001",
        "layout_id": "default-grid",
        "layout_kind": "layer-grid",
        "layers": [
            {
                "layer_index": 0,
                "positions": [
                    {"expert_index": 0, "x": 0.0, "y": 0.0, "z": 0.0},
                    {"expert_index": 1, "x": 1.0, "y": 0.0, "z": 0.0},
                    {"expert_index": 2, "x": 0.0, "y": 1.0, "z": 0.0},
                    {"expert_index": 3, "x": 1.0, "y": 1.0, "z": 0.0},
                ],
            }
        ],
    }


def make_valid_derived_event() -> dict:
    return {
        "event_type": "token_derived",
        "schema_version": "derived/v0.1.0-provisional",
        "run_id": "run-001",
        "token_index": 0,
        "derivation_version": "motifs/v0.1.0-alpha",
        "derivation_config_id": "motifs/default-alpha",
        "derived_payload": {},
    }


class SemanticValidationApiTests(unittest.TestCase):
    def test_validation_issue_dataclass_is_constructible(self) -> None:
        issue = ValidationIssue(
            scope="raw_event",
            code="example",
            message="example message",
        )

        self.assertEqual(issue.scope, "raw_event")
        self.assertEqual(issue.code, "example")
        self.assertEqual(issue.message, "example message")


class RawEventSemanticValidationTests(unittest.TestCase):
    def test_validate_raw_event_semantics_accepts_valid_event(self) -> None:
        issues = validate_raw_event_semantics(make_valid_raw_event())
        self.assertEqual(issues, [])

    def test_validate_raw_event_semantics_reports_router_probs_length_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["router_probs"] = [0.6, 0.3, 0.1]

        issues = validate_raw_event_semantics(event)

        self.assertIn("router_probs_length_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_topk_index_out_of_bounds(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["topk_indices"] = [0, 5]

        issues = validate_raw_event_semantics(event)

        self.assertIn("topk_index_out_of_bounds", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_decode_duration_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["timing_ms"]["decode_duration"] = 14.0

        issues = validate_raw_event_semantics(event)

        self.assertIn("decode_duration_mismatch", {issue.code for issue in issues})

class LayoutSemanticValidationTests(unittest.TestCase):
    def test_validate_layout_semantics_reports_duplicate_expert_index(self) -> None:
        layout = make_valid_layout()
        layout["layers"][0]["positions"][1]["expert_index"] = 0

        issues = validate_layout_semantics(layout)

        self.assertIn("duplicate_expert_index", {issue.code for issue in issues})


class RunBundleSemanticValidationTests(unittest.TestCase):
    def test_validate_run_bundle_semantics_reports_layout_expert_count_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        layout["layers"][0]["positions"].pop()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("layout_expert_count_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_requires_matching_derivation_metadata_across_bundle(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event()]
        derived_events[0]["derivation_config_id"] = "motifs/other-alpha"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
            motif_ledger={
                "schema_version": "motif_ledger/v0.1.0-provisional",
                "run_id": "run-001",
                "derivation_version": "motifs/v0.1.0-alpha",
                "derivation_config_id": "motifs/default-alpha",
                "ledger_payload": {},
            },
            contingency={
                "schema_version": "contingency/v0.1.0-provisional",
                "run_id": "run-001",
                "derivation_version": "motifs/v0.1.0-alpha",
                "derivation_config_id": "motifs/default-alpha",
                "contingency_payload": {},
            },
        )

        self.assertIn("derivation_config_mismatch", {issue.code for issue in issues})


if __name__ == "__main__":
    unittest.main()
